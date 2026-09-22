import logging
import json
from typing import List, Dict, Any, Optional
import httpx

try:
    from backend.app.core.config import settings
except ImportError:
    from app.core.config import settings

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
    Multi-backend generation service.
    Priority order:
      1. Google Gemini API  (set GEMINI_API_KEY in backend/.env)
      2. llama-server / local llama.cpp  (OpenAI-compatible, running on LLAMA_SERVER_URL)
      3. Context-summary fallback (shows retrieved chunks, no LLM)
    """
    _instance = None

    def __init__(self):
        self._gemini_client = None
        self.backend: str = "fallback"
        self.is_ready: bool = False

    @classmethod
    def get_instance(cls) -> "GenerationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self):
        """Tries each backend in priority order."""

        # 1) Google Gemini
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self.backend = "gemini"
                self.is_ready = True
                logger.info(f"✅ Gemini backend ready (model={settings.GEMINI_MODEL})")
                return
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        # 2) llama-server (OpenAI-compatible, running locally)
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{settings.LLAMA_SERVER_URL}/health")
                if res.status_code in (200, 503):   # 503 = still loading, that's OK
                    logger.info(f"✅ llama-server backend ready at {settings.LLAMA_SERVER_URL}")
                    self.backend = "llama_server"
                    self.is_ready = True
                    return
        except Exception as e:
            logger.warning(f"llama-server not reachable: {e}")

        # 3) Fallback
        logger.warning("No LLM backend available — using context-summary fallback.")
        self.backend = "fallback"
        self.is_ready = True

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        if self.backend == "gemini":
            return {
                "status": "connected",
                "backend": "Google Gemini",
                "configured_model": settings.GEMINI_MODEL,
                "model_present": True,
                "available_models": [settings.GEMINI_MODEL],
                "host": "https://generativelanguage.googleapis.com",
            }
        if self.backend == "llama_server":
            try:
                with httpx.Client(timeout=3.0) as client:
                    res = client.get(f"{settings.LLAMA_SERVER_URL}/health")
                    healthy = res.status_code == 200
                return {
                    "status": "connected" if healthy else "loading",
                    "backend": "llama-server (local)",
                    "configured_model": "qwen2.5-1.5b-instruct",
                    "model_present": True,
                    "available_models": ["qwen2.5-1.5b-instruct"],
                    "host": settings.LLAMA_SERVER_URL,
                }
            except Exception as e:
                return {"status": "unreachable", "backend": "llama-server", "error": str(e)}
        return {
            "status": "fallback",
            "backend": "Context-summary (no LLM configured)",
            "configured_model": "none",
            "model_present": False,
            "available_models": [],
        }

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return f"Context:\nNo relevant documents found.\n\nQuestion: {question}\nAnswer:"

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            doc = chunk.get("document", "Unknown")
            page = chunk.get("page", "?")
            content = chunk.get("content", "").strip()
            context_blocks.append(
                f"--- [Source {i} | Document: {doc} | Page: {page}] ---\n{content}"
            )

        context_str = "\n\n".join(context_blocks)
        return (
            f"Context:\n{context_str}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer (grounded strictly in the context above, with citations):"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "I could not find this information in the provided documents."

        prompt = self.build_prompt(question, retrieved_chunks)

        # --- Google Gemini ---
        if self.backend == "gemini" and self._gemini_client:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                top_p=0.9,
                max_output_tokens=1024,
            )
            # Try configured model first, then auto-fallback to reliable backup models if 503/rate-limited
            candidate_models = list(dict.fromkeys([
                settings.GEMINI_MODEL,
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash"
            ]))
            for model_name in candidate_models:
                try:
                    response = self._gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    if response.text:
                        return response.text.strip()
                except Exception as e:
                    logger.warning(f"Gemini model '{model_name}' call failed: {e}. Attempting next model...")
            logger.error("All Gemini candidate models failed.")

        # --- llama-server (OpenAI-compatible /v1/chat/completions) ---
        if self.backend == "llama_server":
            try:
                payload = {
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                }
                with httpx.Client(timeout=120.0) as client:
                    res = client.post(
                        f"{settings.LLAMA_SERVER_URL}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    res.raise_for_status()
                    data = res.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.error(f"llama-server generation error: {e}", exc_info=True)

        # --- Context-summary fallback ---
        sources_summary = "\n".join(
            [
                f"• **[{c.get('document')} – Page {c.get('page')}]**: {c.get('content', '')[:250]}..."
                for c in retrieved_chunks[:3]
            ]
        )
        return (
            "⚠️ **No LLM backend is active.**\n\n"
            "**Relevant context retrieved from your documents:**\n\n"
            f"{sources_summary}\n\n"
            "---\n"
            "*To enable full AI answers, either:*\n"
            "- *Add a `GEMINI_API_KEY` to `backend/.env` (free at https://aistudio.google.com/apikey)*\n"
            "- *Or ensure `llama-server` is running on port 11434*"
        )
