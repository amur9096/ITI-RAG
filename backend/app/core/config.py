import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings loaded from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", ".env"),
            ".env",
            "backend/.env",
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "RAG-Powered Document Assistant API"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Ollama LLM Settings (kept for compatibility)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # Google Gemini API Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"

    # llama-server (llama.cpp) OpenAI-compatible endpoint
    LLAMA_SERVER_URL: str = "http://127.0.0.1:11434"

    # ChromaDB & Embeddings Settings
    CHROMA_PATH: str = "data/vector_store"
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.2

    # CORS Settings
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:3000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    def get_absolute_chroma_path(self) -> str:
        """
        Resolves the absolute path to the vector store directory.
        Handles both backend-relative execution and root-relative execution.
        """
        if os.path.isabs(self.CHROMA_PATH):
            return self.CHROMA_PATH
        
        # Check backend/data/vector_store vs data/vector_store
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backend_dir = os.path.dirname(current_dir)
        candidate1 = os.path.join(backend_dir, self.CHROMA_PATH)
        candidate2 = os.path.join(os.path.dirname(backend_dir), self.CHROMA_PATH)
        
        if os.path.exists(candidate1):
            return candidate1
        if os.path.exists(candidate2):
            return candidate2
        return candidate1


settings = Settings()
