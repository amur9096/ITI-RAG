import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings
from backend.app.schemas.query import SourceItem

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for semantic similarity search over document chunks stored in ChromaDB.
    """
    _instance: Optional["RetrievalService"] = None

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.collection = None
        self.is_ready: bool = False

    @classmethod
    def get_instance(cls) -> "RetrievalService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """
        Loads the SentenceTransformer embedding model and attaches to ChromaDB.
        Called once during FastAPI lifespan startup.
        """
        try:
            logger.info(f"Loading SentenceTransformer model: {settings.EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

            chroma_dir = settings.get_absolute_chroma_path()
            logger.info(f"Connecting to ChromaDB persistent storage at: {chroma_dir}")
            
            os.makedirs(chroma_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=chroma_dir)

            # Retrieve or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            count = self.collection.count()
            logger.info(f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' loaded with {count} chunks.")
            self.is_ready = True
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalService: {e}", exc_info=True)
            self.is_ready = False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Embeds the incoming user query and retrieves the most relevant document chunks.
        """
        if not self.is_ready or self.embedding_model is None or self.collection is None:
            logger.warning("RetrievalService is not initialized or vector store is unavailable.")
            return []

        k = top_k or settings.TOP_K
        total_chunks = self.collection.count()
        if total_chunks == 0:
            logger.warning("ChromaDB collection contains 0 documents.")
            return []

        fetch_k = min(k, total_chunks)

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

            # Perform similarity search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved_chunks = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc_text, meta, dist in zip(docs, metas, dists):
                    # For cosine distance: cosine similarity = 1 - distance
                    sim_score = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0
                    retrieved_chunks.append({
                        "content": doc_text,
                        "document": meta.get("document", "Unknown Document"),
                        "page": meta.get("page", 1),
                        "chunk_id": meta.get("chunk_id", ""),
                        "score": round(sim_score, 4)
                    })

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:40]}...'")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error during retrieval execution: {e}", exc_info=True)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics about vector store."""
        if not self.is_ready:
            return {"status": "uninitialized", "count": 0}
        try:
            return {
                "status": "ready",
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "total_chunks": self.collection.count() if self.collection else 0,
                "model_name": settings.EMBEDDING_MODEL_NAME,
                "storage_path": settings.get_absolute_chroma_path()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
