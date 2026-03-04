
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.models import DocumentChunk
from src.vectordb.chroma_store import ChromaVectorStore


@pytest.fixture
def mock_collection():
    col = MagicMock()
    col.count.return_value = 0
    col.query.return_value = {
        "ids":       [["chunk-1", "chunk-2"]],
        "documents": [["Context A", "Context B"]],
        "metadatas": [[
            {"doc_id": "doc-1", "source": "test.pdf", "chunk_index": 0},
            {"doc_id": "doc-1", "source": "test.pdf", "chunk_index": 1},
        ]],
        "distances": [[0.1, 0.3]],    
    }
    return col


@pytest.fixture
def store(mock_collection):
    s = ChromaVectorStore()
    s._collection = mock_collection
    s._client     = MagicMock()
    return s


class TestChromaVectorStore:

    def test_add_chunks(self, store, mock_collection):
        chunks = [
            DocumentChunk(doc_id="d1", content="hello", chunk_index=0),
            DocumentChunk(doc_id="d1", content="world", chunk_index=1),
        ]
        vecs = [np.zeros(384, dtype=np.float32)] * 2
        count = store.add_chunks(chunks, vecs)
        assert count == 2
        mock_collection.add.assert_called_once()

    def test_add_chunks_mismatched_lengths_raises(self, store):
        chunks = [DocumentChunk(doc_id="d1", content="x", chunk_index=0)]
        vecs   = [np.zeros(384)] * 2
        with pytest.raises(ValueError):
            store.add_chunks(chunks, vecs)

    def test_similarity_search_returns_chunks(self, store, mock_collection):
        mock_collection.count.return_value = 5
        query_vec = np.random.rand(384).astype(np.float32)
        results = store.similarity_search(query_vec, top_k=2)
        assert len(results) == 2
        assert results[0].similarity_score == pytest.approx(0.9, abs=0.01)

    def test_similarity_search_empty_store(self, store, mock_collection):
        mock_collection.count.return_value = 0
        results = store.similarity_search(np.zeros(384), top_k=5)
        assert results == []

    def test_delete_document(self, store, mock_collection):
        mock_collection.get.return_value = {"ids": ["c1", "c2"]}
        deleted = store.delete_document("doc-1")
        assert deleted == 2
        mock_collection.delete.assert_called_once_with(ids=["c1", "c2"])

    def test_count(self, store, mock_collection):
        mock_collection.count.return_value = 42
        assert store.count() == 42
