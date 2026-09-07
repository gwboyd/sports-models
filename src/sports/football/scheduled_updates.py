"""Direct Lambda entry point for scheduled expected-points updates."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.model_patterns.expected_points.write_policy import is_aws_lambda_runtime
from src.sports.football.cfb.expected_points import update_picks as cfb_update_picks
from src.sports.football.expected_points_schemas import UpdatePicksRequest
from src.sports.football.nfl.expected_points import update_picks as nfl_update_picks


LOGGER = logging.getLogger(__name__)
SCHEDULED_UPDATE_JOB = "expected_points_update"
SCHEDULED_CLIENT_NAME = "aws-scheduler"


def run_scheduled_expected_points_update(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one automatic or manual update without the HTTP/API Gateway adapter.

    Exceptions intentionally propagate so Lambda asynchronous retries and its
    on-failure destination can observe a failed model update.
    """

    if event.get("job") != SCHEDULED_UPDATE_JOB:
        raise ValueError(f"Unsupported scheduled job: {event.get('job')!r}")

    try:
        league = ExpectedPointsLeague(str(event["league"]).lower())
    except (KeyError, ValueError) as exc:
        raise ValueError("Scheduled update requires league='nfl' or league='cfb'") from exc

    season = _positive_int(event.get("season"), "season")
    week = _positive_int(event.get("week"), "week")
    run_key = str(event.get("run_key") or "")
    if not run_key.startswith(("aws-scheduler:", "api:")):
        raise ValueError("Scheduled run_key must start with 'aws-scheduler:' or 'api:'")

    manages_registered_plan = is_aws_lambda_runtime()
    client_name = SCHEDULED_CLIENT_NAME
    manual = run_key.startswith("api:")
    if manual and not manages_registered_plan:
        raise PermissionError("Manual scheduled jobs require the deployed AWS trainer")
    if manages_registered_plan:
        from src.utils.db.sports_models_db import (
            cancel_obsolete_manual_update,
            claim_scheduled_model_update,
            get_expected_points_picks,
            get_model_update_job,
            is_scheduled_model_update_completed,
            set_scheduled_model_update_status,
        )

        if manual:
            plan = get_model_update_job(run_key)
            if plan is None or (
                plan["trigger_source"], plan["league"], plan["season"], plan["week"]
            ) != ("api", league.value, season, week):
                raise ValueError("Manual event does not match its registered plan")
            client_name = plan["client_name"]

        claim_status = claim_scheduled_model_update(league, run_key, season, week)
        if claim_status != "claimed":
            if claim_status == "missing":
                raise ValueError(f"Scheduled run has no registered plan: {run_key}")
            if claim_status == "running":
                # A previous invocation may have timed out before releasing its
                # claim. Returning success here would consume the AWS retry.
                raise RuntimeError(f"Scheduled run still has an active claim: {run_key}")
            LOGGER.info("Skipping scheduled run %s with status=%s", run_key, claim_status)
            return {
                "status": "skipped",
                "league": league.value,
                "season": season,
                "week": week,
                "reason": f"scheduled run is {claim_status}",
                "run_key": run_key,
            }

    request = UpdatePicksRequest(season=season, week=week)
    LOGGER.info(
        "Starting scheduled %s update for season=%s week=%s",
        league.value.upper(),
        season,
        week,
    )
    runner = _update_runner(league)
    try:
        if manual:
            latest_picks = get_expected_points_picks(league, latest=True)
            if any((int(pick["season"]), int(pick["week"])) > (season, week) for pick in latest_picks):
                cancel_obsolete_manual_update(run_key)
                LOGGER.info("Cancelled obsolete manual run %s", run_key)
                return {"status": "cancelled", "run_key": run_key,
                        "reason": "A newer model week has already been published"}
        LOGGER.info(
            "Executing update run_key=%s client=%s model_version=%s source_git_sha=%s",
            run_key, client_name, os.getenv(f"{league.value.upper()}_EXPECTED_POINTS_VERSION"),
            os.getenv("SOURCE_GIT_SHA"),
        )
        result = runner(
            request,
            client_name,
            run_key=run_key,
        )
        if manages_registered_plan:
            if result.get("database_updated") is not True:
                raise RuntimeError(
                    "Scheduled model update completed without writing to the database"
                )
            if not is_scheduled_model_update_completed(league, run_key):
                raise RuntimeError(
                    "Scheduled model update was not linked to its update-history row"
                )
    except Exception as exc:
        if manages_registered_plan:
            if is_scheduled_model_update_completed(league, run_key):
                LOGGER.warning(
                    "Recovered scheduled run %s after post-persistence failure: %s",
                    run_key,
                    exc,
                )
                return {
                    "status": "success",
                    "league": league.value,
                    "season": season,
                    "week": week,
                    "run_key": run_key,
                    "recovered": True,
                }
            set_scheduled_model_update_status(run_key, "failed", error=str(exc))
        raise
    LOGGER.info(
        "Completed scheduled %s update for season=%s week=%s",
        league.value.upper(),
        season,
        week,
    )
    return {
        "status": "success",
        "league": league.value,
        "season": season,
        "week": week,
        "run_key": run_key,
        "data": result,
    }


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Scheduled update {name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"Scheduled update {name} must be positive")
    return parsed


def _update_runner(
    league: ExpectedPointsLeague,
) -> Callable[..., dict]:
    if league is ExpectedPointsLeague.NFL:
        return nfl_update_picks.main
    return cfb_update_picks.main
