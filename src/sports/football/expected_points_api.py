from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from fastapi import APIRouter, Header, HTTPException

from src.model_patterns.expected_points.reporting import get_result_stats
from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football.expected_points_schemas import (
    CFBPickResponse,
    GameResult,
    PickResponse,
    PickResultsData,
    PickResultsResponse,
    UpdatePicksRequest,
    UpdatePicksResponse,
)
from src.utils.db.sports_models_db import get_expected_points_picks, get_expected_points_results


logger = logging.getLogger(__name__)


@dataclass
class ExpectedPointsRouters:
    picks: APIRouter
    results: APIRouter
    update: APIRouter


def build_expected_points_routers(
    league: ExpectedPointsLeague,
    update_runner: Callable[[UpdatePicksRequest, str], dict],
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
        response_model=UpdatePicksResponse,
        response_model_exclude_none=True,
        tags=[tag],
    )
    def update_picks(
        request_body: UpdatePicksRequest,
        client_name: str = Header(..., description="Identifier for the requesting entity"),
    ):
        try:
            result = update_runner(request_body, client_name)
            return {"status": "success", "message": "Update process completed", "data": result}
        except Exception as exc:
            logger.exception("%s update process failed", tag)
            raise HTTPException(status_code=500, detail=f"Update process failed: {exc}") from exc

    return ExpectedPointsRouters(picks=picks_router, results=results_router, update=update_router)
