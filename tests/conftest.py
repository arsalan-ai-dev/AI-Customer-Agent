import pytest
from fastapi.testclient import TestClient
from app.fastapi_app import app


@pytest.fixture(scope="module")
def test_client():
    """Provides an isolated HTTP Test Client for FastAPI endpoints."""
    with TestClient(app) as client:
        yield client