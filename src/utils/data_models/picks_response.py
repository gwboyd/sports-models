from pydantic import BaseModel, Field
from typing import Optional

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
    spread_lock: int  # Assuming 0 or 1 as integer values
    total_pred: float
    total_line: float
    total_play: str
    total_win_prob: float
    total_lock: int  # Assuming 0 or 1 as integer values
    game_id: str
    year_week: str
    date_time: str = Field(..., pattern=r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}")