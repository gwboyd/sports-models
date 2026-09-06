"""Submit current-slate refreshes through the existing one-time Scheduler path."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import logging
from typing import Any
from uuid import uuid4

import boto3
from botocore.config import Config

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.model_patterns.expected_points.write_policy import is_aws_lambda_runtime
from src.sports.football.schedule_coordinator import (
    ScheduledUpdatePlan,
    ensure_update_schedule,
    load_schedule,
    scheduler_configuration,
    select_plannable_week,
)
from src.utils.db.sports_models_db import (
    confirm_manual_model_update_schedule,
    get_expected_points_picks,
    get_model_update_job,
    get_model_update_job_result,
    insert_manual_model_update,
    model_key,
)


LOGGER = logging.getLogger(__name__)
# Conservative bound covering the configured one-hour Scheduler delivery window,
# one-hour Lambda event age, and final 15-minute executions/retry overhead.
OUTCOME_UNCONFIRMED_AFTER = timedelta(hours=3)
TERMINAL_STATUSES = {"completed", "cancelled", "missed"}


def job_status(row: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Expose the recorded state; never pretend a stale claim proves success/failure."""

    now = now or datetime.now(timezone.utc)
    result = dict(row)
    result["status_url"] = f"/model-update-jobs/{row['run_key']}"
    result["outcome_unconfirmed"] = (
        row["status"] not in TERMINAL_STATUSES
        and now > row["scheduled_for"] + OUTCOME_UNCONFIRMED_AFTER
    )
    if result["outcome_unconfirmed"]:
        result["message"] = "Outcome unconfirmed; inspect training logs and failure queues before requesting another run."
    elif row["status"] == "planned" and row["scheduled_for"] <= now:
        result["outcome_unconfirmed"] = True
        result["message"] = "Schedule acceptance was not recorded and the delivery time has passed; inspect AWS before requesting another run."
    elif row["status"] == "failed":
        result["message"] = "The last attempt failed; AWS may still retry this run."
    if row.get("update_id") is not None:
        result["data"] = get_model_update_job_result(row["league"], row["update_id"])
    return result


def submit_manual_update(
    league: ExpectedPointsLeague,
    client_name: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Register one immutable manual plan and confirm its AWS schedule exists.

    Local callers cannot dispatch production work, even with AWS credentials.
    Existing keys are resolved before feed access so replay never selects a new week.
    """

    if not is_aws_lambda_runtime():
        raise PermissionError("Manual update scheduling is available only through the deployed AWS API")
    digest = hashlib.sha256((idempotency_key or uuid4().hex).encode()).hexdigest()
    run_key = f"api:{league.value}:{digest}"
    row = get_model_update_job(run_key)
    if row is None:
        # Validate infrastructure configuration before creating a database plan.
        scheduler_configuration()
        games = load_schedule(league, now=datetime.now(timezone.utc), timeout=5)
        picks = get_expected_points_picks(league, latest=True)
        now = datetime.now(timezone.utc)
        week_plan = select_plannable_week(league, games=games, latest_picks=picks, now=now)
        if week_plan is None or not any(
            not game.completed and game.kickoff > now for game in week_plan.games
        ):
            return {
                "status": "not_scheduled", "league": league.value,
                "message": "No eligible upcoming slate is available under the current model-week policy.",
            }
        first_game = week_plan.games[0]
        # Round up to a full minute at least 60 seconds away: Scheduler has minute precision.
        scheduled_for = (now + timedelta(minutes=2)).replace(second=0, microsecond=0)
        row = insert_manual_model_update({
            "run_key": run_key,
            "model_key": model_key(league).value,
            "league": league.value,
            "season": first_game.season,
            "week": first_game.week,
            "scheduled_for": scheduled_for,
            "aws_schedule_name": f"sports-models-{league.value}-manual-{digest[:32]}",
            "reason": "On-demand current-slate refresh",
            "client_name": client_name,
        })

    # A replay never recreates a delivered/deleted schedule or changes its time.
    # Planned rows may recover a failed or uncertain create only before delivery.
    if row["status"] != "planned" or row["scheduled_for"] <= datetime.now(timezone.utc):
        return job_status(row)

    plan = ScheduledUpdatePlan(
        league=league,
        season=row["season"], week=row["week"], window_key="manual",
        game_date=None, scheduled_for=row["scheduled_for"], kickoff_at=None,
        run_key=run_key, aws_schedule_name=row["aws_schedule_name"], reason=row["reason"],
    )
    scheduler_client = boto3.client(
        "scheduler", config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1}),
    )
    # Exceptions leave an immutable planned row. A same-key retry reconciles with
    # AWS, including when creation succeeded but the HTTP response was lost.
    LOGGER.info("Confirming manual schedule run_key=%s", run_key)
    ensure_update_schedule(scheduler_client, plan=plan, **scheduler_configuration())
    confirm_manual_model_update_schedule(run_key)
    LOGGER.info("Manual update scheduled run_key=%s league=%s season=%s week=%s client=%s",
                run_key, league.value, row["season"], row["week"], row["client_name"])
    return job_status(get_model_update_job(run_key))


def get_manual_update_status(run_key: str) -> dict[str, Any] | None:
    row = get_model_update_job(run_key)
    return job_status(row) if row else None
