import pytest
import sys
import os

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.api.app import app

# Tesst client fixture for testing FastAPI endpoints
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

# Auth token fixture for authenticated endpoints
@pytest.fixture(scope="session")
def auth_token(client):
    response = client.post("/auth/login", json={"username": "pavan", "password": "secure123"})
    token = response.json().get("access_token")
    return token

# Auth headers fixture for authenticated requests
@pytest.fixture(scope="session")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}
