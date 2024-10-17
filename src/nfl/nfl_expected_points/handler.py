from fastapi import APIRouter, HTTPException
from src.utils.dynamo_functions import dynamodb, scan_table
from src.nfl.utils.expected_points_functions import get_result_stats
from typing import List


from src.utils.data_models.picks_response import PickResponse
from src.utils.data_models.picks_results_response import GameResult, PickResultsData, PickResultsResponse


picks = APIRouter()
pick_results = APIRouter()


picks_table = dynamodb.Table('nfl_expected_points_picks')
results_table = dynamodb.Table('nfl_expected_points_results')



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
def get_picks():
    
    results = scan_table(results_table)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found for the specified week.")
    
    stats = get_result_stats(results)
    games = [GameResult(**game) for game in results]

    return PickResultsResponse(data = PickResultsData(**stats), games = games)