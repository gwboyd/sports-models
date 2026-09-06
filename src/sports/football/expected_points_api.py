from __future__ import annotations

from dataclasses import dataclass
import logging
from uuid import uuid4
from fastapi import APIRouter, Body, Header, HTTPException, Response

from src.model_patterns.expected_points.reporting import get_result_stats
from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football.expected_points_schemas import (
    CFBPickResponse,
    GameResult,
    PickResponse,
    PickResultsData,
    PickResultsResponse,
    CurrentSlateUpdateRequest,
    ModelUpdateJobResponse,
)
from src.sports.football.manual_updates import get_manual_update_status, submit_manual_update
from src.utils.db.sports_models_db import get_expected_points_picks, get_expected_points_results


logger = logging.getLogger(__name__)
jobs = APIRouter()


@jobs.get("/model-update-jobs/{run_key}", response_model=ModelUpdateJobResponse,
          response_model_exclude_none=True, tags=["Model updates"])
def get_update_job(run_key: str):
    try:
        result = get_manual_update_status(run_key)
    except Exception as exc:
        logger.exception("Model update status read failed")
        raise HTTPException(status_code=503, detail="Model update status is temporarily unavailable") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Model update job not found")
    return result


@dataclass
class ExpectedPointsRouters:
    picks: APIRouter
    results: APIRouter
    update: APIRouter


def build_expected_points_routers(
    league: ExpectedPointsLeague,
) -> ExpectedPointsRouters:
    tag = league.value.upper()
    picks_router = APIRouter()
    results_router = APIRouter()
    update_router = APIRouter()
    pick_response_model = CFBPickResponse if league is ExpectedPointsLeague.CFB else PickResponse

    @picks_router.get(
        f"/{league.value}-picks",
        response_model=list[pick_response_model],
        tags=[tag],
    )
    def get_picks():
        try:
            rows = get_expected_points_picks(league, latest=True)
        except Exception as exc:
            logger.exception("%s picks database read failed", tag)
            raise HTTPException(status_code=500, detail=f"Database read failed: {exc}") from exc
        if not rows:
            raise HTTPException(status_code=404, detail="No picks found for the latest week.")
        return [pick_response_model(**row) for row in rows]

    @results_router.get(
        f"/{league.value}-pick-results",
        response_model=PickResultsResponse,
        tags=[tag],
    )
    def get_pick_results():
        try:
            rows = get_expected_points_results(league)
        except Exception as exc:
            logger.exception("%s results database read failed", tag)
            raise HTTPException(status_code=500, detail=f"Database read failed: {exc}") from exc
        if not rows:
            raise HTTPException(status_code=404, detail="No pick results found.")
        return PickResultsResponse(
            data=PickResultsData(**get_result_stats(rows)),
            games=[GameResult(**row) for row in rows],
        )

    @update_router.post(
        f"/{league.value}-update-picks",
        response_model=ModelUpdateJobResponse,
        response_model_exclude_none=True,
        status_code=202,
        responses={200: {"model": ModelUpdateJobResponse}},
        tags=[tag],
    )
    def update_picks(
        response: Response,
        request_body: CurrentSlateUpdateRequest | None = Body(default=None),
        client_name: str = Header(..., min_length=1, max_length=100, pattern=r"^\S(?:[^\r\n]*\S)?$",
                                  description="Identifier for the requesting entity"),
        idempotency_key: str | None = Header(default=None, min_length=1, max_length=200,
                                             pattern=r"^[A-Za-z0-9._:-]+$"),
    ):
        if client_name == "notebook":
            raise HTTPException(status_code=422, detail="client-name 'notebook' is reserved for interactive notebook execution")
        idempotency_key = idempotency_key or uuid4().hex
        response.headers["Idempotency-Key"] = idempotency_key
        try:
            result = submit_manual_update(league, client_name, idempotency_key)
            response.status_code = (
                200 if result["status"] in {"completed", "cancelled", "missed", "not_scheduled"}
                or result.get("outcome_unconfirmed") else 202
            )
            if result.get("status_url"):
                response.headers["Location"] = result["status_url"]
            return result
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("%s update scheduling failed", tag)
            raise HTTPException(
                status_code=503,
                detail="Scheduling could not be confirmed; retry with the returned Idempotency-Key",
                headers={"Idempotency-Key": idempotency_key},
            ) from exc

    return ExpectedPointsRouters(picks=picks_router, results=results_router, update=update_router)
