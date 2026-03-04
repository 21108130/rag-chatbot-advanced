

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings
from src.utils.logger import logger


class EmbeddingGenerator:

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None          # Lazy-loaded
        self._cache: Dict[str, np.ndarray] = {}



    @property
    def model(self):

        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(
                    f"Embedding model loaded. "
                    f"Dimension: {self._model.get_sentence_embedding_dimension()}"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


    def embed_text(self, text: str) -> np.ndarray:

        if not text or not text.strip():
            logger.warning("Attempted to embed empty text; returning zero vector.")
            return np.zeros(self.dimension, dtype=np.float32)

        cache_key = self._hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        self._cache[cache_key] = vector
        return vector

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> List[np.ndarray]:

        if not texts:
            return []

        results: Dict[int, np.ndarray] = {}
        uncached_indices: List[int] = []
        uncached_texts:   List[str] = []

        for i, text in enumerate(texts):
            ck = self._hash(text)
            if ck in self._cache:
                results[i] = self._cache[ck]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text or "")

        if uncached_texts:
            logger.info(
                f"Embedding {len(uncached_texts)} texts "
                f"({len(results)} served from cache) …"
            )
            vectors = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)

            for idx, vec in zip(uncached_indices, vectors):
                self._cache[self._hash(texts[idx])] = vec
                results[idx] = vec

        return [results[i] for i in range(len(texts))]

    def embed_query(self, query: str) -> np.ndarray:

        logger.debug(f"Embedding query: '{query[:80]}…'")
        return self.embed_text(query)

    def clear_cache(self) -> None:

        cleared = len(self._cache)
        self._cache.clear()
        logger.debug(f"Embedding cache cleared ({cleared} entries removed).")


    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()




@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingGenerator:
    """Return a cached singleton EmbeddingGenerator."""
    return EmbeddingGenerator()
