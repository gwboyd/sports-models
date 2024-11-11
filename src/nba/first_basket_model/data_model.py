from pydantic import BaseModel
from typing import List, Dict


class NBAFirstBasketPick(BaseModel):
    date: str
    player_name: str
    team: str
    fb_model_prob: float
    fb_model_odds: float
    odds: float
    sportsbook: str
    units: float


# class NBAFirstBasketPick(BaseModel):
#     date: str
#     player_name: str
#     odds: float
#     sportsbook: str
#     units: float
#     bankroll: float