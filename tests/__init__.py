from fastapi.testclient import TestClient

from main import app

test_client = TestClient(app)