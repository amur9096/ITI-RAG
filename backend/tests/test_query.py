import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """
    Test 1: Health check endpoint returns 200 with component details.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "vector_store" in data["components"]
    assert "ollama_llm" in data["components"]


def test_query_happy_path(client: TestClient):
    """
    Test 2: Valid question to /query returns 200, answer, and sources list (Happy Path).
    """
    payload = {"question": "What are the four Coffman conditions for a deadlock?"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "source_citations" in data
    assert isinstance(data["source_citations"], list)
    assert "execution_time_ms" in data


def test_query_invalid_empty_string(client: TestClient):
    """
    Test 3: Empty string question returns HTTP 422 Unprocessable Entity.
    """
    payload = {"question": ""}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_invalid_whitespace_only(client: TestClient):
    """
    Test 4: Whitespace-only question returns HTTP 422 Unprocessable Entity.
    """
    payload = {"question": "     "}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_missing_body(client: TestClient):
    """
    Test 5: Missing request body returns HTTP 422.
    """
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_invalid_json_structure(client: TestClient):
    """
    Test 6: Invalid payload structure returns HTTP 422.
    """
    payload = {"unrelated_field": 12345}
    response = client.post("/query", json=payload)
    assert response.status_code == 422
