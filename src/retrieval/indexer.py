

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from config.settings import settings
from src.embeddings.embedder import EmbeddingGenerator, get_embedder
from src.utils.document_loader import DocumentLoader
from src.utils.logger import logger
from src.utils.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentUploadResult,
)
from src.utils.text_splitter import RecursiveTextSplitter
from src.vectordb.chroma_store import ChromaVectorStore


class DocumentIndexer:


    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore]   = None,
        embedder:     Optional[EmbeddingGenerator]  = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder     = embedder     or get_embedder()
        self.loader       = DocumentLoader()
        self.splitter     = RecursiveTextSplitter(
            chunk_size    = settings.chunk_size,
            chunk_overlap = settings.chunk_overlap,
        )



    def index_file(self, file_path: str | Path) -> DocumentUploadResult:

        path = Path(file_path)
        doc_id = str(uuid4())
        start = time.perf_counter()

        logger.info(f"[Indexer] Starting indexing: {path.name} (doc_id={doc_id})")

        try:

            text, file_metadata = self.loader.load(path)
            if not text.strip():
                raise ValueError(f"No text extracted from '{path.name}'.")


            raw_chunks = self.splitter.split_text(text)
            if not raw_chunks:
                raise ValueError(f"Text from '{path.name}' produced zero chunks.")


            chunks: List[DocumentChunk] = [
                DocumentChunk(
                    doc_id      = doc_id,
                    content     = chunk_text,
                    chunk_index = idx,
                    metadata    = {
                        **file_metadata,
                        "source": path.name,
                    },
                )
                for idx, chunk_text in enumerate(raw_chunks)
            ]


            logger.info(f"[Indexer] Generating embeddings for {len(chunks)} chunks …")
            embeddings = self.embedder.embed_batch(
                [c.content for c in chunks],
                show_progress=True,
            )


            self.vector_store.add_chunks(chunks, embeddings)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                f"[Indexer] Done: {path.name} → {len(chunks)} chunks "
                f"in {elapsed:.0f}ms"
            )

            return DocumentUploadResult(
                doc_id      = doc_id,
                filename    = path.name,
                status      = DocumentStatus.INDEXED,
                chunk_count = len(chunks),
                message     = f"Successfully indexed {len(chunks)} chunks.",
            )

        except Exception as exc:
            logger.error(f"[Indexer] Failed to index '{path.name}': {exc}")
            return DocumentUploadResult(
                doc_id      = doc_id,
                filename    = path.name,
                status      = DocumentStatus.FAILED,
                chunk_count = 0,
                message     = str(exc),
            )

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for *doc_id* from the vector store."""
        return self.vector_store.delete_document(doc_id)
