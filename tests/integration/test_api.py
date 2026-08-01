import pytest
from fastapi.testclient import TestClient
from fastapi_app import app

client = TestClient(app)


def test_health_check():
    """Verify health check endpoint returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint_empty_payload():
    """Verify validation handling for empty query input."""
    response = client.post("/api/v1/chat", json={"question": "", "session_id": "test_session"})
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]


def test_chat_endpoint_valid_query():
    """Verify end-to-end chat response with session ID."""
    payload = {
        "question": "Hello, what services do you provide?",
        "session_id": "test_integration_session_1"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["session_id"] == "test_integration_session_1"
    assert len(data["answer"]) > 0


def test_chat_endpoint_multi_turn_memory():
    """Verify that conversation history persists across turns in the same session."""
    session_id = "test_memory_session"

    # Turn 1
    turn1_payload = {
        "question": "My order number is #99882.",
        "session_id": session_id
    }
    resp1 = client.post("/api/v1/chat", json=turn1_payload)
    assert resp1.status_code == 200

    # Turn 2 (Follow-up requiring memory of Turn 1)
    turn2_payload = {
        "question": "What was my order number again?",
        "session_id": session_id
    }
    resp2 = client.post("/api/v1/chat", json=turn2_payload)
    assert resp2.status_code == 200
    assert "99882" in resp2.json()["answer"]