
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from src.embeddings.embedder import EmbeddingGenerator, get_embedder
from src.utils.logger import logger
from src.utils.models import RetrievalResult, RetrievedChunk
from src.vectordb.chroma_store import ChromaVectorStore


class HybridRetriever:


    RRF_K = 60
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT   = 0.3

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore]  = None,
        embedder:     Optional[EmbeddingGenerator] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder     = embedder     or get_embedder()
        self._bm25_index  = None
        self._bm25_docs:  List[str]         = []
        self._bm25_chunks: List[RetrievedChunk] = []



    def _build_bm25_index(self, chunks: List[RetrievedChunk]) -> None:

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        tokenized = [doc.content.lower().split() for doc in chunks]
        self._bm25_index  = BM25Okapi(tokenized)
        self._bm25_docs   = [doc.content for doc in chunks]
        self._bm25_chunks = chunks

    def _bm25_search(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[Tuple[RetrievedChunk, float]]:

        if not candidates:
            return []

        self._build_bm25_index(candidates)

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)


        max_score = scores.max() if scores.max() > 0 else 1.0
        norm_scores = scores / max_score

        ranked = sorted(
            zip(candidates, norm_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


    def _reciprocal_rank_fusion(
        self,
        vector_results:  List[RetrievedChunk],
        bm25_results:    List[Tuple[RetrievedChunk, float]],
    ) -> List[RetrievedChunk]:

        scores: Dict[str, float] = {}
        chunks: Dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results, 1):
            cid = chunk.chunk_id
            scores[cid]  = scores.get(cid, 0) + self.VECTOR_WEIGHT / (self.RRF_K + rank)
            chunks[cid]  = chunk
        for rank, (chunk, _bm25_score) in enumerate(bm25_results, 1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.BM25_WEIGHT / (self.RRF_K + rank)
            chunks[cid] = chunk


        merged = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        max_score = scores[merged[0]] if merged else 1.0
        result = []
        for cid in merged:
            chunk = chunks[cid]
            normalized_score = scores[cid] / max_score
            result.append(
                RetrievedChunk(
                    chunk_id         = chunk.chunk_id,
                    doc_id           = chunk.doc_id,
                    content          = chunk.content,
                    similarity_score = round(normalized_score, 3),
                    metadata         = {
                        **chunk.metadata,
                        "retrieval_method": "hybrid_rrf",
                        "raw_rrf_score":    round(scores[cid], 4),
                        "original_similarity": round(chunk.similarity_score, 3),
                    },
                )
            )
        return result



    def retrieve(
        self,
        query:       str,
        top_k:       Optional[int]  = None,
        where:       Optional[dict] = None,
    ) -> RetrievalResult:

        if not query.strip():
            return RetrievalResult(query=query, chunks=[], latency_ms=0.0)

        if self.vector_store.count() == 0:
            logger.warning("[HybridRetriever] Vector store is empty.")
            return RetrievalResult(query=query, chunks=[], latency_ms=0.0)

        k = top_k or settings.top_k_results
        fetch_k = min(k * 4, 40)

        start = time.perf_counter()


        query_vector = self.embedder.embed_query(query)
        vector_chunks = self.vector_store.similarity_search(
            query_vector = query_vector,
            top_k        = fetch_k,
            where        = where,
        )

        if not vector_chunks:
            elapsed = (time.perf_counter() - start) * 1000
            return RetrievalResult(query=query, chunks=[], latency_ms=elapsed)


        try:
            bm25_ranked = self._bm25_search(query, vector_chunks, top_k=fetch_k)
        except Exception as exc:
            logger.warning(f"[HybridRetriever] BM25 failed, falling back to vector-only: {exc}")
            bm25_ranked = []


        if bm25_ranked:
            fused = self._reciprocal_rank_fusion(vector_chunks, bm25_ranked)
        else:
            fused = vector_chunks

        final = fused[:k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"[HybridRetriever] Query: '{query[:60]}' → "
            f"{len(final)} chunks in {elapsed_ms:.1f}ms "
            f"(vector={len(vector_chunks)}, bm25={len(bm25_ranked)})"
        )

        return RetrievalResult(query=query, chunks=final, latency_ms=elapsed_ms)

    def format_context(self, result: RetrievalResult, max_chars: int = 6000) -> str:

        if not result.chunks:
            return ""

        parts = []
        total = 0
        for i, chunk in enumerate(result.chunks, 1):
            source  = chunk.metadata.get("source", chunk.doc_id)
            method  = chunk.metadata.get("retrieval_method", "vector")
            section = f"[Source {i}: {source} | score={chunk.similarity_score:.3f} | via={method}]"
            block   = f"{section}\n{chunk.content}"

            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(block[:remaining] + " …[truncated]")
                break

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)