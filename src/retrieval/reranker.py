
from __future__ import annotations

import time
from functools import lru_cache
from typing import List, Optional

from src.utils.logger import logger
from src.utils.models import RetrievalResult, RetrievedChunk


class CrossEncoderReranker:


    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model     = None



    @property
    def model(self):
        if self._model is None:
            logger.info(f"[Reranker] Loading cross-encoder: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, max_length=512)
                logger.info("[Reranker] Cross-encoder ready.")
            except ImportError:
                raise ImportError("sentence-transformers not installed.")
        return self._model



    def rerank(
        self,
        query:    str,
        result:   RetrievalResult,
        top_k:    Optional[int] = None,
    ) -> RetrievalResult:

        if not result.chunks:
            return result

        start = time.perf_counter()


        pairs = [(query, chunk.content) for chunk in result.chunks]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.error(f"[Reranker] Prediction failed: {exc}. Returning original order.")
            return result

        scored_chunks = sorted(
            zip(result.chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        k = top_k or len(scored_chunks)
        reranked: List[RetrievedChunk] = []

        for chunk, ce_score in scored_chunks[:k]:
            reranked.append(
                RetrievedChunk(
                    chunk_id         = chunk.chunk_id,
                    doc_id           = chunk.doc_id,
                    content          = chunk.content,
                    similarity_score = float(ce_score),
                    metadata         = {
                        **chunk.metadata,
                        "cross_encoder_score": float(ce_score),
                        "original_score":      chunk.similarity_score,
                        "reranked":            True,
                    },
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[Reranker] Reranked {len(result.chunks)} → {len(reranked)} chunks "
            f"in {elapsed_ms:.1f}ms"
        )

        return RetrievalResult(
            query      = result.query,
            chunks     = reranked,
            latency_ms = result.latency_ms + elapsed_ms,
        )

    def is_available(self) -> bool:

        try:
            _ = self.model
            return True
        except Exception:
            return False



@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
