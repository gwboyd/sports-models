"""HTTP schemas shared by the NFL and CFB expected-points APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class UpdatePicksData(BaseModel):
    write_time: str
    week: int
    season: int
    environment: str
    client_name: str
    runtime: float
    pick_changes: int
    pick_changes_games: list[str]
    play_changes: int
    play_changes_games: list[str]
    updates_skipped: int
    picks_num: int
    database_updated: bool


class UpdatePicksResponse(BaseModel):
    status: str
    message: str
    data: UpdatePicksData | None = None


class UpdatePicksRequest(BaseModel):
    season: int
    week: int


__all__ = [
    "CFBPickResponse",
    "GameResult",
    "PickResponse",
    "PickResultsData",
    "PickResultsResponse",
    "UpdatePicksData",
    "UpdatePicksRequest",
    "UpdatePicksResponse",
]
