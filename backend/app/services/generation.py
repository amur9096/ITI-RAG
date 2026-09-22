import logging
from typing import List, Dict, Any, Tuple
import ollama
import httpx
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional, document-grounded AI assistant.
Your job is to answer questions strictly and accurately based on the provided document excerpts (Context).

RULES:
1. Answer the question using ONLY the facts directly mentioned in the Context.
2. Do NOT extrapolate, speculate, or introduce external knowledge.
3. If the provided context does not contain sufficient information to answer the question, state clearly and concisely:
   "I could not find this information in the provided documents."
4. Always cite your sources at the end of relevant points or at the bottom of your response in the format: [Document: <doc_name>, Page: <page_num>].
5. Maintain a professional, clear, and objective tone.
"""


class GenerationService:
    """
    Service responsible for constructing grounded prompts and querying the local Ollama LLM.
    """
    _instance = None

    def __init__(self):
        self.client: Optional[ollama.Client] = None
        self.is_ready: bool = False

    @classmethod
    def get_instance(cls) -> "GenerationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """
        Initializes the Ollama client and tests connectivity.
        """
        try:
            logger.info(f"Configuring Ollama client for host: {settings.OLLAMA_HOST}")
            self.client = ollama.Client(host=settings.OLLAMA_HOST)
            # Connectivity check
            self.check_health()
        except Exception as e:
            logger.warning(f"Ollama client initialization warning: {e}")

    def check_health(self) -> Dict[str, Any]:
        """
        Pings the Ollama daemon and checks whether the configured model is available.
        """
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{settings.OLLAMA_HOST}/api/tags")
                if res.status_code == 200:
                    models_data = res.json().get("models", [])
                    available_models = [m.get("name") for m in models_data]
                    model_found = any(settings.OLLAMA_MODEL in m for m in available_models)
                    self.is_ready = True
                    return {
                        "status": "connected",
                        "host": settings.OLLAMA_HOST,
                        "configured_model": settings.OLLAMA_MODEL,
                        "model_present": model_found,
                        "available_models": available_models
                    }
        except Exception as e:
            self.is_ready = False
            return {
                "status": "unreachable",
                "host": settings.OLLAMA_HOST,
                "configured_model": settings.OLLAMA_MODEL,
                "error": str(e)
            }
        return {"status": "unknown"}

    def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Builds the structured, citation-grounded prompt for the LLM.
        """
        if not retrieved_chunks:
            return f"Context:\nNo relevant documents found.\n\nQuestion: {question}\nAnswer:"

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            doc = chunk.get("document", "Unknown")
            page = chunk.get("page", "?")
            content = chunk.get("content", "").strip()
            context_blocks.append(f"--- [Source {i} | Document: {doc} | Page: {page}] ---\n{content}")

        context_str = "\n\n".join(context_blocks)

        prompt = f"""Context:
{context_str}

Question:
{question}

Answer (grounded strictly in the context above, with citations):"""
        return prompt

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates a grounded answer from Ollama using the retrieved chunks.
        Falls back cleanly if Ollama daemon is unreachable.
        """
        if not retrieved_chunks:
            return "I could not find this information in the provided documents."

        prompt = self.build_prompt(question, retrieved_chunks)

        try:
            if not self.client:
                self.client = ollama.Client(host=settings.OLLAMA_HOST)

            response = self.client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                options={
                    "temperature": 0.1,  # Low temperature for strict factual grounding
                    "top_p": 0.9
                }
            )
            return response.get("response", "").strip()

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}", exc_info=True)
            # Informative fallback if local Ollama server is offline or model is pulling
            sources_summary = "\n".join(
                [f"• [{c.get('document')} - Page {c.get('page')}]: {c.get('content', '')[:150]}..."
                 for c in retrieved_chunks[:2]]
            )
            return (
                f"[Ollama Service Notice]: Unable to connect to local Ollama LLM at {settings.OLLAMA_HOST} "
                f"({type(e).__name__}).\n\n"
                f"Relevant context retrieved from vector store:\n{sources_summary}\n\n"
                f"To enable full neural synthesis, ensure Ollama is running (`ollama serve`) and model is pulled (`ollama pull {settings.OLLAMA_MODEL}`)."
            )
