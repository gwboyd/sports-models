
from fastapi import APIRouter, Header
from fastapi import FastAPI, Request, HTTPException

import os
import json

picks = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))

picks_file_path = os.path.join(current_dir, 'picks.json')


with open(picks_file_path, 'r') as f:
    picks_json_data = json.load(f)

@picks.get("/nfl-picks")
def get_picks():
    try:
        return picks_json_data
    
    except Exception as e:
        # Log or handle the error, then raise an HTTPException
        log_msg = f"Error occurred: {str(e)}"
        raise HTTPException(status_code=500, detail=log_msg) 
