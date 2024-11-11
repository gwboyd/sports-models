import os
import uvicorn
import logging
import sys
import hashlib
import secrets
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Depends, HTTPException, status
from dotenv import load_dotenv
from mangum import Mangum
from typing import List


from src.nfl.nfl_expected_points import handler as nfl_expected_points_handler
from src.nba.first_basket_model import handler as nba_first_basket_handler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    stream=sys.stderr
)

app = FastAPI(
    title="Will's Sports Models",
    description="Models to predict sports outcomes",
    version="0.1.0",
    contact={"name": "Will"},
    openapi_url="/openapi.json",
)


ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
FRONT_END_API_KEY = os.getenv("FRONT_END_API_KEY")
READ_API_KEY = os.getenv("READ_API_KEY")
NBA_API_KEY = os.getenv("NBA_API_KEY")
AWS_API_KEY = os.getenv("AWS_API_KEY")

def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()

API_KEYS = {
    hash_key(ADMIN_API_KEY): ["admin"],
    hash_key(FRONT_END_API_KEY): ["read"],
    hash_key(READ_API_KEY): ["read"],
    hash_key(NBA_API_KEY): ["nba", "read"],
    hash_key(AWS_API_KEY): ["admin"]
}

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

@app.get("/", tags=["default"])
def index():
    return JSONResponse(status_code=200, content={"msg": "ok"})


@app.get("/health", tags=["default"])
def health_check():
    return JSONResponse(status_code=200, content={"msg": "ok"})

def api_key_auth(api_key: str = Depends(api_key_header)):
    if os.getenv("LOCALHOST") == "True":
        return ["admin", "read", "nba"]
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    hashed_input_key = hash_key(api_key.strip())
    for stored_hashed_key in API_KEYS:
        if secrets.compare_digest(hashed_input_key, stored_hashed_key):
            return API_KEYS[stored_hashed_key]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
    )


def require_permission(*required_permissions: str):
    async def permission_dependency(
        request: Request,
        permissions: List[str] = Depends(api_key_auth)
    ):
        # Admin users have access to all endpoints
        if "admin" in permissions:
            return

        # Check if the user's permissions include any of the required permissions
        if any(permission in permissions for permission in required_permissions):
            return
        
        # If none of the above conditions are met, deny access
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this endpoint."
        )
    return permission_dependency
    


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

app.include_router(nfl_expected_points_handler.picks, dependencies=[Depends(require_permission("read"))])
app.include_router(nfl_expected_points_handler.pick_results, dependencies=[Depends(require_permission("read"))])
app.include_router(nfl_expected_points_handler.update, dependencies=[Depends(require_permission())])

app.include_router(nba_first_basket_handler.pick_upload, dependencies=[Depends(require_permission("nba"))])
app.include_router(nba_first_basket_handler.picks, dependencies=[Depends(require_permission("nba","read"))])




if __name__ == "__main__":
    uvicorn.config.logger.error("Started server...")
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", "3000")), access_log=False
    )


handler = Mangum(app)

# uvicorn main:app --host 0.0.0.0 --port 3000 --reload --log-level warning
# sam local invoke "FastAPILambdaFunction"


