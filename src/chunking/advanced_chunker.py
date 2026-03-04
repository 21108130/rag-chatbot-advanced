
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from src.utils.logger import logger



class BaseChunker(ABC):


    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, doc_type: str = "general") -> List[str]:

        ...

    def _count_tokens_approx(self, text: str) -> int:

        return len(text) // 4

    def _is_too_small(self, text: str, min_tokens: int = 20) -> bool:
        return self._count_tokens_approx(text) < min_tokens


class SemanticChunker(BaseChunker):

    HEADING_PATTERNS = [
        r'^\s{0,3}#{1,4}\s+.+$',
        r'^[A-Z][A-Z\s]{5,50}$',
        r'^\d+\.\s+[A-Z]',
        r'^[IVX]+\.\s+[A-Z]',
    ]

    def __init__(
        self,
        chunk_size:    int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self._heading_re    = re.compile(
            "|".join(self.HEADING_PATTERNS), re.MULTILINE
        )

    def _is_heading(self, line: str) -> bool:
        return bool(self._heading_re.match(line.strip()))

    def _split_into_paragraphs(self, text: str) -> List[str]:

        text = text.replace('\r\n', '\n').replace('\r', '\n')
        raw_blocks = re.split(r'\n\s*\n', text)
        paragraphs = [b.strip() for b in raw_blocks if b.strip()]
        return paragraphs

    def chunk(self, text: str, doc_type: str = "general") -> List[str]:
        if not text.strip():
            return []

        paragraphs = self._split_into_paragraphs(text)
        chunks: List[str] = []
        current_chunk     = ""

        for para in paragraphs:

            if self._is_heading(para):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n"
                continue


            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:

                if current_chunk.strip() and not self._is_too_small(current_chunk):
                    chunks.append(current_chunk.strip())


                if len(para) <= self.chunk_size:
                    current_chunk = para
                else:

                    sub_chunks = self._split_large_paragraph(para)
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1]
                    else:
                        current_chunk = para

        if current_chunk.strip() and not self._is_too_small(current_chunk):
            chunks.append(current_chunk.strip())


        chunks = self._add_overlap(chunks)

        logger.debug(
            f"[SemanticChunker] {len(text)} chars → {len(chunks)} semantic chunks"
        )
        return chunks

    def _split_large_paragraph(self, text: str) -> List[str]:
        """Split oversized paragraph by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sub_chunks: List[str] = []
        current   = ""
        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    sub_chunks.append(current.strip())
                current = sent
        if current:
            sub_chunks.append(current.strip())
        return sub_chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped




class OverlapChunker(BaseChunker):


    def __init__(
        self,
        chunk_size:    int   = 512,
        overlap_ratio: float = 0.15,     # 15% overlap
    ) -> None:
        overlap = int(chunk_size * overlap_ratio)
        super().__init__(chunk_size, overlap)
        self.overlap_ratio = overlap_ratio

    def chunk(self, text: str, doc_type: str = "general") -> List[str]:
        if not text.strip():
            return []


        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: List[str] = []
        current           = ""

        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sent

        if current:
            chunks.append(current.strip())


        overlapped = [chunks[0]] if chunks else []
        for i in range(1, len(chunks)):
            tail = chunks[i-1][-self.chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())

        logger.debug(
            f"[OverlapChunker] {len(text)} chars → {len(overlapped)} chunks "
            f"(overlap={self.chunk_overlap})"
        )
        return overlapped




class DynamicChunker(BaseChunker):


    TYPE_CONFIGS = {
        "technical":  {"chunk_size": 512,  "overlap": 64,  "strategy": "semantic"},
        "legal":      {"chunk_size": 1024, "overlap": 150, "strategy": "overlap"},
        "narrative":  {"chunk_size": 768,  "overlap": 100, "strategy": "overlap"},
        "qa_pairs":   {"chunk_size": 256,  "overlap": 30,  "strategy": "semantic"},
        "general":    {"chunk_size": 512,  "overlap": 64,  "strategy": "semantic"},
        "scientific": {"chunk_size": 640,  "overlap": 80,  "strategy": "semantic"},
        "financial":  {"chunk_size": 512,  "overlap": 64,  "strategy": "overlap"},
    }

    def __init__(self, default_type: str = "general") -> None:
        cfg = self.TYPE_CONFIGS.get(default_type, self.TYPE_CONFIGS["general"])
        super().__init__(cfg["chunk_size"], cfg["overlap"])
        self.default_type = default_type
        self._chunkers: dict = {}

    def _detect_doc_type(self, text: str) -> str:
        """Heuristically detect document type from content."""
        text_lower = text.lower()
        # Legal
        if any(kw in text_lower for kw in ["whereas", "hereinafter", "indemnify", "notwithstanding"]):
            return "legal"
        # Scientific / Technical
        if any(kw in text_lower for kw in ["abstract", "methodology", "hypothesis", "algorithm", "implementation"]):
            return "technical"
        # Q&A
        qa_ratio = text.count("?") / max(len(text.split()), 1)
        if qa_ratio > 0.05:
            return "qa_pairs"
        # Financial
        if any(kw in text_lower for kw in ["revenue", "ebitda", "quarterly", "fiscal year", "earnings"]):
            return "financial"
        return "general"

    def _get_chunker(self, doc_type: str) -> BaseChunker:
        if doc_type not in self._chunkers:
            cfg = self.TYPE_CONFIGS.get(doc_type, self.TYPE_CONFIGS["general"])
            if cfg["strategy"] == "semantic":
                self._chunkers[doc_type] = SemanticChunker(
                    chunk_size=cfg["chunk_size"], chunk_overlap=cfg["overlap"]
                )
            else:
                self._chunkers[doc_type] = OverlapChunker(
                    chunk_size=cfg["chunk_size"],
                    overlap_ratio=cfg["overlap"] / cfg["chunk_size"],
                )
        return self._chunkers[doc_type]

    def chunk(self, text: str, doc_type: str = "auto") -> List[str]:
        if doc_type == "auto":
            doc_type = self._detect_doc_type(text)

        chunker = self._get_chunker(doc_type)
        chunks  = chunker.chunk(text, doc_type=doc_type)

        logger.info(
            f"[DynamicChunker] type={doc_type} → {len(chunks)} chunks "
            f"(size={chunker.chunk_size}, overlap={chunker.chunk_overlap})"
        )
        return chunks


class SentenceWindowChunker(BaseChunker):


    def __init__(
        self,
        window_size: int = 3,
    ) -> None:
        super().__init__(chunk_size=256, chunk_overlap=0)
        self.window_size = window_size

    def chunk(self, text: str, doc_type: str = "general") -> List[str]:

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks = []
        for i, sent in enumerate(sentences):
            start = max(0, i - self.window_size)
            end   = min(len(sentences), i + self.window_size + 1)
            window = " ".join(sentences[start:end])
            chunks.append(window)

        deduped = [chunks[0]] if chunks else []
        for c in chunks[1:]:
            if c != deduped[-1]:
                deduped.append(c)

        logger.debug(
            f"[SentenceWindowChunker] {len(sentences)} sentences → "
            f"{len(deduped)} window chunks (window={self.window_size})"
        )
        return deduped


def get_chunker(
    strategy:   str  = "semantic",
    chunk_size: Optional[int] = None,
    overlap:    Optional[int] = None,
) -> BaseChunker:
 
    strategies = {
        "semantic":        SemanticChunker,
        "overlap":         OverlapChunker,
        "dynamic":         DynamicChunker,
        "sentence_window": SentenceWindowChunker,
    }

    if strategy not in strategies:
        logger.warning(f"[Chunking] Unknown strategy '{strategy}', using 'semantic'")
        strategy = "semantic"

    cls = strategies[strategy]
    kwargs: dict = {}

    if strategy == "dynamic":
        chunker = DynamicChunker()
    elif strategy == "sentence_window":
        chunker = SentenceWindowChunker(window_size=overlap or 3)
    elif strategy == "overlap":
        cs = chunk_size or 512
        ratio = (overlap or 77) / cs
        chunker = OverlapChunker(chunk_size=cs, overlap_ratio=min(ratio, 0.25))
    else:
        chunker = SemanticChunker(
            chunk_size    = chunk_size or 800,
            chunk_overlap = overlap    or 100,
        )

    return chunker
