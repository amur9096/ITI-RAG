import time
import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceItem,
    HealthResponse,
    ComponentHealth
)
from backend.app.services.retrieval import RetrievalService
from backend.app.services.generation import GenerationService
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query & Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health and Component Diagnostics",
    description="Returns health status of the API, vector store, and Ollama connection."
)
async def health_check():
    """
    Diagnostic health endpoint checking vector store and Ollama readiness.
    """
    retrieval_service = RetrievalService.get_instance()
    generation_service = GenerationService.get_instance()

    retrieval_stats = retrieval_service.get_stats()
    ollama_health = generation_service.check_health()

    is_healthy = retrieval_service.is_ready

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        app_name=settings.PROJECT_NAME,
        version="1.0.0",
        components={
            "vector_store": ComponentHealth(
                status="ready" if retrieval_service.is_ready else "uninitialized",
                details=retrieval_stats
            ),
            "ollama_llm": ComponentHealth(
                status=ollama_health.get("status", "unknown"),
                details=ollama_health
            )
        }
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask Question against Document Corpus",
    description="Performs semantic retrieval against the vector store and generates a grounded response with citations."
)
async def process_query(request: QueryRequest):
    """
    Main RAG pipeline endpoint:
    1. Validates user question
    2. Embeds question & retrieves top relevant document chunks
    3. Prompts local LLM with grounded context
    4. Returns synthesized answer with verified source citations
    """
    start_time = time.time()
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty."
        )

    retrieval_service = RetrievalService.get_instance()
    generation_service = GenerationService.get_instance()

    # Step 1: Retrieval
    retrieved_chunks = retrieval_service.retrieve(query=question, top_k=settings.TOP_K)

    # Step 2: Generation
    answer = generation_service.generate_answer(question=question, retrieved_chunks=retrieved_chunks)

    # Step 3: Format Sources
    sources: list[SourceItem] = []
    citations: list[str] = []

    for chunk in retrieved_chunks:
        src = SourceItem(
            document=chunk.get("document", "Unknown"),
            page=chunk.get("page", 1),
            chunk_id=chunk.get("chunk_id", ""),
            score=chunk.get("score"),
            snippet=chunk.get("content", "")[:250]
        )
        sources.append(src)
        cit_str = src.to_citation_str()
        if cit_str not in citations:
            citations.append(cit_str)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        source_citations=citations,
        retrieved_chunks_count=len(retrieved_chunks),
        execution_time_ms=elapsed_ms
    )
