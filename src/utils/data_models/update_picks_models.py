from pydantic import BaseModel
from typing import List, Optional

class UpdatePicksData(BaseModel):
    write_time: str
    week: int
    season: int
    environment: str
    client_name: str
    runtime: float
    pick_changes: int
    pick_changes_games: List[str]
    play_changes: int
    play_changes_games: List[str]
    updates_skipped: int
    picks_num: int
    database_updated: bool


class UpdatePicksResponse(BaseModel):
    status: str
    message: str
    data: Optional[UpdatePicksData] = None

class UpdatePicksRequest(BaseModel):
    season: int
    week: int

