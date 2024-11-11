from fastapi import APIRouter, HTTPException, status, Header
import logging
from src.utils.dynamo_functions import dynamodb, scan_table
from src.nfl.utils.expected_points_functions import get_result_stats
from typing import List

from src.nfl.nfl_expected_points import update_picks

from src.utils.data_models.picks_response import PickResponse
from src.utils.data_models.picks_results_response import GameResult, PickResultsData, PickResultsResponse
from src.utils.data_models.update_picks_models import UpdatePicksResponse, UpdatePicksRequest


picks = APIRouter()
pick_results = APIRouter()
update = APIRouter()


picks_table = dynamodb.Table('nfl_expected_points_picks')
results_table = dynamodb.Table('nfl_expected_points_results')

logger = logging.getLogger(__name__)
logger.setLevel("INFO")



#####
@picks.get("/nfl-picks", response_model=List[PickResponse], tags=["NFL"])
def get_picks():

    try:
        results = scan_table(picks_table)

    except Exception as e:
        log_msg = f"Error occurred during DynamoDB scan: {str(e)}"
        raise HTTPException(status_code=500, detail=log_msg)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found for the specified week.")
    
    
    return [PickResponse(**item) for item in results]

@pick_results.get("/nfl-pick-results", response_model=PickResultsResponse, tags=["NFL"])
def get_pick_results():
    
    results = scan_table(results_table)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found for the specified week.")
    
    stats = get_result_stats(results)
    games = [GameResult(**game) for game in results]

    return PickResultsResponse(data = PickResultsData(**stats), games = games)

@update.post("/nfl-update-picks", tags=["NFL"],response_model=UpdatePicksResponse, response_model_exclude_none=True)
def train_model_and_update_picks(
    request_body: UpdatePicksRequest,
    client_name: str = Header(..., description="Identifier for the requesting entity (lambda, jake, etc)")
):
    if client_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'client_name' is required"
        )
    
    try:
        result = update_picks.main(request_body = request_body, client_name = client_name)
        return {"status": "success", "message": "Update process completed", "data": result}

    except Exception as e:
        logger.error(f"Update process failed: {str(e)}")
        return {"status": "error", "message": "Update process failed"}
