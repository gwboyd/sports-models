import os
import uvicorn
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Depends, HTTPException, status
from dotenv import load_dotenv
from mangum import Mangum
import json

from src.nfl.nfl_expected_points import handler as nfl_expected_points_handler

load_dotenv()


app = FastAPI(
    title="Will's Sports Models",
    description="Models to predict sports outcomes",
    version="0.0.1",
    contact={"name": "Will", "email": "willboyd970@gmail.com"},
)

API_KEY = os.getenv("API_KEY")
api_keys = [API_KEY]

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

@app.get("/")
def index():
    return JSONResponse(status_code=200, content={"msg": "ok"})


@app.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"msg": "ok"})

def api_key_auth(api_key: str = Depends(api_key_header)):
    if os.getenv("LOCALHOST") == "True":
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )

    if api_key.strip() not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
def unicorn_exception_handler(request: Request, exc: UnicornException):
    result_info = {
        "url": exc.name,
        "valid_request": "False",
        "error_message": "err: {0}".format(exc),
    }
    result_response = {"meta": result_info, "result": {}}
    return JSONResponse(status_code=400, content=result_response)

app.include_router(nfl_expected_points_handler.picks, dependencies=[Depends(api_key_auth)])



if __name__ == "__main__":
    uvicorn.config.logger.error("Started server...")
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", "3000")), access_log=False
    )


handler = Mangum(app)

# uvicorn main:app --host 0.0.0.0 --port 3000 --reload --log-level warning


