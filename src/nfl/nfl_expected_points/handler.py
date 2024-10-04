from fastapi import APIRouter, HTTPException
from src.utils.dynamo_functions import dynamodb, scan_table
from src.nfl.utils.expected_points_functions import get_result_stats

picks = APIRouter()
pick_results = APIRouter()


picks_table = dynamodb.Table('nfl_expected_points_picks')
results_table = dynamodb.Table('nfl_expected_points_results')



@picks.get("/nfl-picks")
def get_picks():

    try:
        results = scan_table(picks_table)
        
    except Exception as e:
        log_msg = f"Error occurred during DynamoDB scan: {str(e)}"
        raise HTTPException(status_code=500, detail=log_msg)
    
    if not results:
        raise HTTPException(status_code=404, detail="No picks found for the specified week.")
    
    ordered_columns = [
        'season', 'week', 'home_team', 'away_team', 'home_score_pred', 'away_score_pred', 
        'spread_pred', 'spread_line', 'spread_play', 'spread_win_prob', 'spread_lock', 
        'total_pred', 'total_line', 'total_play', 'total_win_prob', 'total_lock', 'game_id', 'year_week', 'date_time'
    ]
    
    ordered_results = [
        {col: item.get(col, None) for col in ordered_columns} for item in results
    ]
    
    return ordered_results

@pick_results.get("/nfl-pick-results")
def get_picks():
    
    games = scan_table(results_table)
    
    if not games:
        raise HTTPException(status_code=404, detail="No picks found for the specified week.")
    
    ordered_columns = ['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'home_score',
          'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'spread_win_prob' , 'spread_lock', 
          'correct_spread_play', 'spread_win', 'total_pred', 'total_line', 'true_total', 'total_play', 'total_win_prob', 
          'total_lock', 'correct_total_play', 'total_win', 'year_week','game_id','date_time']
    
    ordered_games = [
        {col: item.get(col, None) for col in ordered_columns} for item in games
    ]

    return_obj = {
        "data": get_result_stats(games),
        "games": ordered_games
    }
    
    return return_obj
