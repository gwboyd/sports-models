"""HTTP schemas shared by the NFL and CFB expected-points APIs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PickResponse(BaseModel):
    season: int
    week: str
    home_team: str
    away_team: str
    home_score_pred: float
    away_score_pred: float
    spread_pred: float
    spread_line: float
    spread_play: str
    spread_win_prob: float
    spread_lock: int
    total_pred: float
    total_line: float
    total_play: str
    total_win_prob: float
    total_lock: int
    game_id: str
    year_week: str
    date_time: str = Field(..., pattern=r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}")
    write_time: str


class CFBPickResponse(PickResponse):
    home_conference: str | None = None
    away_conference: str | None = None


class GameResult(BaseModel):
    season: int
    week: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_score_pred: float
    away_score_pred: float
    spread_pred: float
    spread_line: float
    true_spread: float
    spread_play: str
    spread_win_prob: float
    spread_lock: int
    correct_spread_play: str | None = None
    spread_win: int | None = None
    total_pred: float
    total_line: float
    true_total: float
    total_play: str
    total_win_prob: float
    total_lock: int
    correct_total_play: str | None = None
    total_win: int | None = None
    year_week: str
    game_id: str
    date_time: str


class PickResultsData(BaseModel):
    predicted_games: int
    spread_wins: int
    spread_losses: int
    spread_pushes: int
    spread_win_pct: float
    spread_lock_predictions: int
    spread_lock_wins: int
    spread_lock_losses: int
    spread_lock_pushes: int
    spread_lock_win_pct: float
    total_wins: int
    total_losses: int
    total_pushes: int
    total_win_pct: float
    total_lock_predictions: int
    total_lock_wins: int
    total_lock_losses: int
    total_lock_pushes: int
    total_lock_win_pct: float


class PickResultsResponse(BaseModel):
    data: PickResultsData
    games: list[GameResult]


class UpdatePicksRequest(BaseModel):
    """Internal runner inputs; HTTP submissions resolve their own current slate."""

    season: int
    week: int
    allow_non_aws_write: bool = False


class CurrentSlateUpdateRequest(BaseModel):
    """An optional empty body; explicit weeks and local-write flags are not HTTP inputs."""

    model_config = ConfigDict(extra="forbid")


class ModelUpdateJobResult(BaseModel):
    id: int
    model_version: str
    source_git_sha: str | None = None
    write_time: datetime
    runtime: float
    picks_num: int
    pick_changes: int
    play_changes: int
    updates_skipped: int


class ModelUpdateJobResponse(BaseModel):
    status: str
    league: str
    run_key: str | None = None
    model_key: str | None = None
    season: int | None = None
    week: int | None = None
    trigger_source: str | None = None
    client_name: str | None = None
    scheduled_for: datetime | None = None
    created_at: datetime | None = None
    claimed_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    attempt_count: int = 0
    update_id: int | None = None
    last_error: str | None = None
    message: str | None = None
    status_url: str | None = None
    outcome_unconfirmed: bool = False
    data: ModelUpdateJobResult | None = None


__all__ = [
    "CurrentSlateUpdateRequest",
    "ModelUpdateJobResponse",
    "ModelUpdateJobResult",
    "CFBPickResponse",
    "GameResult",
    "PickResponse",
    "PickResultsData",
    "PickResultsResponse",
    "UpdatePicksRequest",
]
