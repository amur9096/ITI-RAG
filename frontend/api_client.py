import os
import logging
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60.0"))


class APIClientError(Exception):
    """Custom exception class for frontend API communication errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RAGApiClient:
    """
    HTTP Client for communicating with the FastAPI RAG backend.
    """
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Checks backend connectivity and component health.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    return {"connected": True, "data": res.json()}
                else:
                    return {
                        "connected": False,
                        "error": f"Backend returned status {res.status_code}",
                        "status_code": res.status_code
                    }
        except httpx.ConnectError:
            return {
                "connected": False,
                "error": f"Could not connect to backend at {self.base_url}. Make sure FastAPI is running (`uvicorn app.main:app`)."
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def query(self, question: str) -> Dict[str, Any]:
        """
        Sends a question to the /query endpoint and returns the grounded answer with sources.
        """
        clean_q = question.strip()
        if not clean_q:
            raise APIClientError("Question cannot be empty.")

        url = f"{self.base_url}/query"
        payload = {"question": clean_q}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 422:
                    error_detail = response.json().get("detail", "Invalid input format.")
                    raise APIClientError(f"Validation Error: {error_detail}", status_code=422)
                else:
                    raise APIClientError(
                        f"Server error ({response.status_code}): {response.text}",
                        status_code=response.status_code
                    )

        except httpx.ConnectError:
            raise APIClientError(
                f"Connection failed: Unable to reach backend server at '{self.base_url}'. "
                f"Please verify the FastAPI backend is running."
            )
        except httpx.TimeoutException:
            raise APIClientError(
                f"Request timed out after {self.timeout}s while waiting for Ollama/backend response."
            )
        except APIClientError:
            raise
        except Exception as e:
            raise APIClientError(f"Unexpected error communicating with API: {str(e)}")
