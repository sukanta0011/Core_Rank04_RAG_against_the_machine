import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_ask_question(client):
    payload = {
        "question": "How do I configure the vLLM server?",
        "k": 3
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "resource_locations" in data
    assert len(data["resource_locations"]) <= 3
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0

    print(f"Verified Answer: {data['answer']}...")


def test_ask_question_invalid_k(client):
    """Test that k=0 returns a 422 Unprocessable Entity error."""
    # 1. Arrange: Send k=0 (which violates Field(gt=0))
    payload = {
        "question": "How to config vLLM?",
        "k": 0
    }

    # 2. Act
    response = client.post("/ask", json=payload)

    # 3. Assert
    # 422 is the standard FastAPI code for validation errors
    assert response.status_code == 422

    data = response.json()
    # Check that the error message specifically mentions 'k'
    assert data["detail"][0]["loc"] == ["body", "k"]
    assert "greater than 0" in data["detail"][0]["msg"]
