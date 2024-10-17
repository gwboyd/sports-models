from pydantic import BaseModel
from typing import List, Dict

class NBAFirstBasketPick(BaseModel):
    date: str
    player_name: str
    sportsbook: str
    odds: float
    units: float
    bankroll: int