"""Direct Lambda entry point for scheduled expected-points updates."""

from __future__ import annotations

import logging
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
    """Run one coordinator-created update without the HTTP/API Gateway adapter.

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
    if not run_key.startswith("aws-scheduler:"):
        raise ValueError("Scheduled run_key must start with 'aws-scheduler:'")

    manages_registered_plan = is_aws_lambda_runtime()
    if manages_registered_plan:
        from src.utils.db.sports_models_db import (
            claim_scheduled_model_update,
            is_scheduled_model_update_completed,
            set_scheduled_model_update_status,
        )

        claim_status = claim_scheduled_model_update(league, run_key, season, week)
        if claim_status != "claimed":
            if claim_status == "missing":
                raise ValueError(f"Scheduled run has no registered plan: {run_key}")
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
        result = runner(
            request,
            SCHEDULED_CLIENT_NAME,
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
