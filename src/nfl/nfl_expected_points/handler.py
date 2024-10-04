from fastapi import APIRouter, HTTPException
import boto3
from boto3.dynamodb.conditions import Key

picks = APIRouter()

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  
picks_table = dynamodb.Table('nfl_expected_points_picks')

def scan_table(table):
    try:
        response = table.scan()
        return response.get('Items', [])
    
    except Exception as e:
        log_msg = f"Error occurred during DynamoDB scan: {str(e)}"
        raise HTTPException(status_code=500, detail=log_msg)


@picks.get("/nfl-picks")
def get_picks():
    
    results = scan_table(picks_table)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found for the specified week.")
    
    ordered_columns = [
        'season', 'week', 'home_team', 'away_team', 'home_score_pred', 'away_score_pred', 
        'spread_pred', 'spread_line', 'spread_play', 'spread_win_prob', 'spread_lock', 
        'total_pred', 'total_line', 'total_play', 'total_win_prob', 'total_lock', 'game_id', 'year_week'
    ]
    
    ordered_results = [
        {col: item.get(col, None) for col in ordered_columns} for item in results
    ]
    
    return ordered_results
    return results
