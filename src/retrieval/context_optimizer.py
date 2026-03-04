
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

from src.utils.logger import logger
from src.utils.models import RetrievalResult, RetrievedChunk


class ContextOptimizer:

    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        max_tokens:           int   = 3000,
        dedup_similarity:     float = 0.85,
        min_chunk_tokens:     int   = 20,
    ) -> None:
        self.max_tokens       = max_tokens
        self.dedup_similarity = dedup_similarity
        self.min_chunk_tokens = min_chunk_tokens
        self._tokenizer       = None



    @property
    def tokenizer(self):
        """Lazy-load tiktoken for accurate token counting."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._tokenizer = None
        return self._tokenizer

    def count_tokens(self, text: str) -> int:

        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // self.CHARS_PER_TOKEN



    def _jaccard_similarity(self, a: str, b: str) -> float:

        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union        = len(set_a | set_b)
        return intersection / union

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:

        if len(chunks) <= 1:
            return chunks

        kept    = [chunks[0]]
        for candidate in chunks[1:]:
            is_dup = False
            for existing in kept:
                sim = self._jaccard_similarity(candidate.content, existing.content)
                if sim >= self.dedup_similarity:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(candidate)

        removed = len(chunks) - len(kept)
        if removed:
            logger.debug(f"[ContextOptimizer] Removed {removed} duplicate chunks.")
        return kept



    def _truncate_to_sentence(self, text: str, max_chars: int) -> str:

        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]

        last_period = max(
            truncated.rfind(". "),
            truncated.rfind(".\n"),
            truncated.rfind("! "),
            truncated.rfind("? "),
        )
        if last_period > max_chars // 2:
            return truncated[:last_period + 1] + " …"
        return truncated + " …"



    def optimize(
        self,
        result: RetrievalResult,
        query:  Optional[str] = None,
    ) -> RetrievalResult:

        chunks = result.chunks
        if not chunks:
            return result


        chunks = [
            c for c in chunks
            if self.count_tokens(c.content) >= self.min_chunk_tokens
        ]


        chunks = self._deduplicate(chunks)


        budget_chars   = self.max_tokens * self.CHARS_PER_TOKEN
        kept_chunks: List[RetrievedChunk] = []
        total_tokens   = 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk.content)
            if total_tokens + chunk_tokens > self.max_tokens:

                remaining_tokens = self.max_tokens - total_tokens
                if remaining_tokens > self.min_chunk_tokens:
                    max_chars     = remaining_tokens * self.CHARS_PER_TOKEN
                    truncated_text = self._truncate_to_sentence(chunk.content, max_chars)
                    kept_chunks.append(
                        RetrievedChunk(
                            chunk_id         = chunk.chunk_id,
                            doc_id           = chunk.doc_id,
                            content          = truncated_text,
                            similarity_score = chunk.similarity_score,
                            metadata         = {**chunk.metadata, "truncated": True},
                        )
                    )
                break
            kept_chunks.append(chunk)
            total_tokens += chunk_tokens

        total_final = sum(self.count_tokens(c.content) for c in kept_chunks)
        logger.info(
            f"[ContextOptimizer] {len(result.chunks)} → {len(kept_chunks)} chunks, "
            f"~{total_final} tokens (budget={self.max_tokens})"
        )

        return RetrievalResult(
            query      = result.query,
            chunks     = kept_chunks,
            latency_ms = result.latency_ms,
        )

    def to_context_string(
        self,
        result: RetrievalResult,
    ) -> str:
        if not result.chunks:
            return ""

        parts = []
        for i, chunk in enumerate(result.chunks, 1):
            source   = chunk.metadata.get("source", chunk.doc_id)
            score    = chunk.similarity_score
            method   = chunk.metadata.get("retrieval_method", "")
            reranked = chunk.metadata.get("reranked", False)

            flags = []
            if method:    flags.append(f"via={method}")
            if reranked:  flags.append("reranked=✓")
            if chunk.metadata.get("truncated"): flags.append("truncated")

            flag_str = f" | {', '.join(flags)}" if flags else ""
            header   = f"[Source {i}: {source} | score={score:.3f}{flag_str}]"
            parts.append(f"{header}\n{chunk.content}")

        return "\n\n".join(parts)
