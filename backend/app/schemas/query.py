from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """
    User query request schema.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user query or question to be answered from retrieved documents.",
        examples=["What are the four Coffman conditions for a deadlock?"]
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return clean


class SourceItem(BaseModel):
    """
    Metadata representation of a retrieved document chunk.
    """
    document: str = Field(..., description="Source filename or document title")
    page: Optional[int] = Field(None, description="Page number of the citation")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    score: Optional[float] = Field(None, description="Similarity or relevance score")
    snippet: Optional[str] = Field(None, description="Excerpt of the retrieved chunk")

    def to_citation_str(self) -> str:
        """Returns string representation: 'document.pdf - Page 2'"""
        if self.page is not None:
            return f"{self.document} — Page {self.page}"
        return self.document


class QueryResponse(BaseModel):
    """
    Grounded RAG answer response schema.
    """
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded answer synthesized by Ollama LLM")
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="List of cited source chunks used to generate the answer"
    )
    source_citations: List[str] = Field(
        default_factory=list,
        description="Convenient formatted citation strings (e.g. ['doc.pdf — Page 4'])"
    )
    retrieved_chunks_count: int = Field(0, description="Total number of chunks retrieved")
    execution_time_ms: Optional[float] = Field(None, description="Total request execution time in ms")


class ComponentHealth(BaseModel):
    """Status details for individual pipeline components."""
    status: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """
    System health check response schema.
    """
    status: str = Field(..., description="'healthy' or 'degraded'")
    app_name: str
    version: str
    components: Dict[str, ComponentHealth]
