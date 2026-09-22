import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.utils.logging_config import setup_logging
from backend.app.api.routes.query import router as query_router
from backend.app.services.retrieval import RetrievalService
from backend.app.services.generation import GenerationService

# Setup application logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    Pre-loads heavy resources (ChromaDB vector store, SentenceTransformer model,
    Ollama client connection) ONCE at application startup, avoiding per-request overhead.
    """
    logger.info("Initializing RAG application components...")
    
    # Initialize Vector Retrieval Service
    retrieval_service = RetrievalService.get_instance()
    retrieval_service.initialize()

    # Initialize Ollama LLM Generation Service
    generation_service = GenerationService.get_instance()
    generation_service.initialize()

    logger.info("RAG application startup complete and ready to serve requests.")
    yield
    logger.info("Shutting down RAG application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise-grade RAG Document Assistant API powered by ChromaDB, Sentence Transformers, and Ollama.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception during request processing: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred while processing your query.",
            "error_type": type(exc).__name__
        }
    )

# Include API Routers
app.include_router(query_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to the RAG Document Assistant API. Visit /docs for OpenAPI documentation.",
        "health_endpoint": "/health",
        "query_endpoint": "/query"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
