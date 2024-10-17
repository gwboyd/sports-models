from fastapi import APIRouter, HTTPException
from src.utils.dynamo_functions import dynamodb, scan_table
from typing import List

from src.nba.first_basket_model.dynamo_functions import new_picks

from src.nba.first_basket_model.data_model import NBAFirstBasketPick


pick_upload = APIRouter()
picks = APIRouter()

picks_table = dynamodb.Table('nba_first_basket_picks')


@pick_upload.post("/nba-first-basket-upload", tags=["NBA"])
def nba_first_basket_upload(data: List[NBAFirstBasketPick]):

    new_picks(picks_table, {'PartitionKey': 'date', 'SortKey': 'player_name'}, data)
    
    return {"message": "Data uploaded successfully", "row_count": len(data)}


@picks.get("/nba-first-basket-picks", response_model=List[NBAFirstBasketPick], tags=["NBA"])
def get_picks():

    try:
        results = scan_table(picks_table)

    except Exception as e:
        log_msg = f"Error occurred during DynamoDB scan: {str(e)}"
        raise HTTPException(status_code=500, detail=log_msg)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found")
    
    
    return [NBAFirstBasketPick(**item) for item in results]


