"""FAISS-backed Vehicle RAG (optional).

Lazy-loads FAISS and sentence-transformers; falls back gracefully if unavailable.
"""

from __future__ import annotations

from typing import List, Optional
import logging

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)


class FAISSVehicleRAG:
    def __init__(
        self,
        documents: Optional[List[str]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.documents = documents or [
            "Engine displacement: 2.0L turbocharged",
            "Max power: 280 HP @ 6000 RPM",
            "Transmission: 8-speed automatic",
            "Fuel tank capacity: 60L",
            "Cabin temperature range: 16-32°C",
        ]
        self.embedding_model_name = embedding_model
        self._st_model = None
        self._faiss = None
        self._index = None
        self._embeddings = None

        # Build only if dependencies present
        try:
            self._lazy_setup()
        except Exception as e:  # pragma: no cover
            logger.warning("FAISS RAG disabled: %s", e)

    def _lazy_setup(self) -> None:
        if np is None:
            raise RuntimeError("numpy not available")
        from sentence_transformers import SentenceTransformer  # type: ignore
        import faiss  # type: ignore

        self._st_model = SentenceTransformer(self.embedding_model_name)
        self._faiss = faiss
        embeddings = self._st_model.encode(self.documents)
        self._embeddings = np.array(embeddings).astype("float32")
        d = self._embeddings.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(self._embeddings)
        logger.info("FAISS index built for %d docs", len(self.documents))

    def available(self) -> bool:
        return self._index is not None

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.available():
            return []
        assert self._st_model is not None and self._embeddings is not None
        query_emb = self._st_model.encode([query])
        query_vec = np.array(query_emb).astype("float32")
        distances, indices = self._index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]

    def retrieve_context(self, query: str, command: str) -> dict:
        """Mirror VehicleRAG.retrieve_context contract."""
        docs = self.retrieve(query, top_k=3)
        return {
            "context": {"docs": docs, "procedure": "Context from FAISS index"},
            "success": bool(docs),
            "query": query,
            "command": command,
        }
