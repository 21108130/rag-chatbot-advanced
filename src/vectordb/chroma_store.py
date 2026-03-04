

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from config.settings import settings
from src.utils.logger import logger
from src.utils.models import DocumentChunk, RetrievedChunk


class ChromaVectorStore:


    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir     = persist_dir     or str(settings.chroma_persist_path)
        self._client         = None
        self._collection     = None



    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def _create_client(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        logger.info(f"Connecting to ChromaDB at: {self.persist_dir}")
        client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return client

    def _get_or_create_collection(self):
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},      # cosine similarity
        )
        logger.info(
            f"Collection '{self.collection_name}' ready "
            f"({collection.count()} documents stored)."
        )
        return collection



    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[np.ndarray],
    ) -> int:

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have equal length."
            )
        if not chunks:
            return 0

        ids        = [c.chunk_id for c in chunks]
        documents  = [c.content  for c in chunks]
        metadatas  = [
            {
                **c.metadata,
                "doc_id":      c.doc_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        vectors    = [e.tolist() for e in embeddings]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

        logger.info(
            f"Added {len(chunks)} chunks for doc_id={chunks[0].doc_id!r}"
        )
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:

        results = self.collection.get(where={"doc_id": doc_id})
        ids = results.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for doc_id={doc_id!r}")
        else:
            logger.warning(f"No chunks found for doc_id={doc_id!r}")

        return len(ids)



    def similarity_search(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:

        k = top_k or settings.top_k_results

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector.tolist()],
            "n_results":        min(k, max(1, self.collection.count())),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        retrieved: List[RetrievedChunk] = []

        ids       = results.get("ids",       [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):

            similarity = float(1.0 - dist
            )
            if similarity < settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id         = chunk_id,
                    doc_id           = meta.get("doc_id", ""),
                    content          = doc,
                    similarity_score = similarity,
                    metadata         = meta,
                )
            )

        logger.debug(
            f"Similarity search returned {len(retrieved)}/{k} results "
            f"(threshold={settings.similarity_threshold})"
        )
        return retrieved



    def count(self) -> int:

        return self.collection.count()

    def list_documents(self) -> List[str]:

        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return list({m.get("doc_id", "") for m in all_meta if m.get("doc_id")})

    def get_stats(self) -> Dict[str, Any]:

        doc_ids = self.list_documents()
        return {
            "collection":   self.collection_name,
            "total_chunks": self.count(),
            "total_docs":   len(doc_ids),
            "persist_dir":  self.persist_dir,
        }

    def reset(self) -> None:

        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Collection '{self.collection_name}' has been reset.")
