from pydantic import BaseModel
from typing import List, Optional


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
    correct_spread_play: Optional[str]
    spread_win: Optional[int]
    total_pred: float
    total_line: float
    true_total: float
    total_play: str
    total_win_prob: float
    total_lock: int
    correct_total_play: Optional[str]
    total_win: Optional[int]
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
    games: List[GameResult]