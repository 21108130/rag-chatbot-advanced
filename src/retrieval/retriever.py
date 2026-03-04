

from __future__ import annotations

import time
from typing import Optional

from config.settings import settings
from src.embeddings.embedder import EmbeddingGenerator, get_embedder
from src.utils.logger import logger
from src.utils.models import RetrievalResult
from src.vectordb.chroma_store import ChromaVectorStore


class Retriever:


    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore]  = None,
        embedder:     Optional[EmbeddingGenerator] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder     = embedder     or get_embedder()



    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            logger.warning("[Retriever] Empty query received.")
            return RetrievalResult(query=query, chunks=[], latency_ms=0.0)

        if self.vector_store.count() == 0:
            logger.warning("[Retriever] Vector store is empty — no documents indexed.")
            return RetrievalResult(query=query, chunks=[], latency_ms=0.0)

        start = time.perf_counter()


        query_vector = self.embedder.embed_query(query)


        k = top_k or settings.top_k_results
        chunks = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=k,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"[Retriever] Query: '{query[:60]}…' → "
            f"{len(chunks)} chunks in {elapsed_ms:.1f}ms"
        )

        return RetrievalResult(
            query      = query,
            chunks     = chunks,
            latency_ms = elapsed_ms,
        )

    def format_context(self, result: RetrievalResult, max_chars: int = 6000) -> str:
      
        if not result.chunks:
            return ""

        parts = []
        total = 0

        for i, chunk in enumerate(result.chunks, 1):
            source   = chunk.metadata.get("source", chunk.doc_id)
            section  = f"[Source {i}: {source} | similarity={chunk.similarity_score:.2f}]"
            block    = f"{section}\n{chunk.content}"
            block_len = len(block)

            if total + block_len > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(block[:remaining] + " …[truncated]")
                break

            parts.append(block)
            total += block_len

        return "\n\n".join(parts)
