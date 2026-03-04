
from typing import List

from config.settings import settings
from src.utils.logger import logger


class RecursiveTextSplitter:

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> None:
        self.chunk_size    = chunk_size    or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap



    def split_text(self, text: str) -> List[str]:

        if not text or not text.strip():
            return []

        chunks = self._split_recursive(text.strip(), self.SEPARATORS)
        chunks = self._merge_small_chunks(chunks)
        chunks = self._add_overlap(chunks)

        logger.debug(
            f"Split text ({len(text)} chars) → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks


    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:

        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining = separators[1:] if separators else []

        if sep == "":

            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        parts = text.split(sep)
        chunks: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                chunks.append(part)
            else:

                chunks.extend(self._split_recursive(part, remaining))

        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:

        if not chunks:
            return []

        merged: List[str] = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            candidate = current + " " + next_chunk
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current.strip())
                current = next_chunk

        merged.append(current.strip())
        return [c for c in merged if c]

    def _add_overlap(self, chunks: List[str]) -> List[str]:

        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :]
            overlapped.append((tail + " " + chunks[i]).strip())

        return overlapped
