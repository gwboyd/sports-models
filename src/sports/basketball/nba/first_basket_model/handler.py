from fastapi import APIRouter, HTTPException
from typing import List

from src.sports.basketball.nba.first_basket_model.data_model import NBAFirstBasketPick
from src.utils.db.sports_models_db import (
    get_nba_first_basket_picks,
    replace_nba_first_basket_picks,
)


pick_upload = APIRouter()
picks = APIRouter()


@pick_upload.post("/nba-first-basket-upload", tags=["NBA"])
def nba_first_basket_upload(data: List[NBAFirstBasketPick]):
    try:
        row_count = replace_nba_first_basket_picks([item.dict() for item in data])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database write failed: {exc}") from exc

    return {"message": "Data uploaded successfully", "row_count": row_count}


@picks.get("/nba-first-basket-picks", response_model=List[NBAFirstBasketPick], tags=["NBA"])
def get_picks():

    try:
        results = get_nba_first_basket_picks()

    except Exception as e:
        log_msg = f"Error occurred during database read: {str(e)}"
        raise HTTPException(status_code=500, detail=log_msg)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found")
    
    
    return [NBAFirstBasketPick(**item) for item in results]
