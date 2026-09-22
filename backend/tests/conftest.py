import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure root workspace is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.main import app
from backend.app.services.retrieval import RetrievalService
from backend.app.services.generation import GenerationService


@pytest.fixture(scope="session")
def client():
    """
    TestClient fixture for FastAPI application.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """
    Ensures tests can run reliably without requiring an external Ollama daemon or active GPU.
    """
    # Mock Ollama generation response if Ollama is offline
    def mock_generate(*args, **kwargs):
        return {
            "response": "According to the Operating Systems handout, the four Coffman conditions are Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait. [Document: cs101_operating_systems_concurrency.pdf, Page: 2]"
        }

    # Mock Ollama tags health check
    def mock_tags(*args, **kwargs):
        return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr("ollama.Client.generate", mock_generate, raising=False)
    monkeypatch.setattr("ollama.Client.list", mock_tags, raising=False)
