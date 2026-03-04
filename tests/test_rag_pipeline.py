

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.models import (
    ChatRequest,
    DocumentStatus,
    DocumentUploadResult,
    RetrievedChunk,
)


def _make_pipeline_with_mocks():
    pipeline = RAGPipeline()


    vs = MagicMock()
    vs.count.return_value = 10
    vs.get_stats.return_value = {"total_docs": 2, "total_chunks": 10,
                                 "collection": "test", "persist_dir": "/tmp"}
    pipeline._vector_store = vs


    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(384, dtype=np.float32)
    embedder.embed_batch.return_value = [np.zeros(384, dtype=np.float32)] * 3
    pipeline._embedder = embedder

    retriever = MagicMock()
    retriever.retrieve.return_value = MagicMock(
        query="test",
        chunks=[
            RetrievedChunk(
                chunk_id="c1", doc_id="d1",
                content="RAG stands for Retrieval Augmented Generation.",
                similarity_score=0.92,
                metadata={"source": "paper.pdf"},
            )
        ],
        latency_ms=15.0,
    )
    retriever.format_context.return_value = "RAG stands for Retrieval Augmented Generation."
    pipeline._retriever = retriever


    llm = MagicMock()
    llm.generate.return_value = {
        "answer": "RAG is a technique that enhances LLMs with retrieved context.",
        "tokens_used": 45,
        "model": "llama3-8b-8192",
    }
    pipeline._llm = llm

    return pipeline


class TestRAGPipeline:

    def test_chat_returns_response(self):
        pipeline = _make_pipeline_with_mocks()
        request  = ChatRequest(query="What is RAG?")
        response = pipeline.chat(request)

        assert response.answer
        assert len(response.sources) == 1
        assert response.conversation_id
        assert response.latency_ms >= 0

    def test_conversation_id_is_persisted(self):
        pipeline = _make_pipeline_with_mocks()
        r1 = pipeline.chat(ChatRequest(query="First question"))
        r2 = pipeline.chat(ChatRequest(
            query="Follow-up", conversation_id=r1.conversation_id
        ))
        assert r1.conversation_id == r2.conversation_id

    def test_clear_session(self):
        pipeline = _make_pipeline_with_mocks()
        r = pipeline.chat(ChatRequest(query="Hello"))
        pipeline.clear_session(r.conversation_id)
        assert pipeline.get_session(r.conversation_id) is None

    def test_kb_stats(self):
        pipeline = _make_pipeline_with_mocks()
        stats = pipeline.get_kb_stats()
        assert "total_chunks" in stats
        assert "total_docs" in stats

    def test_ingest_document_failure_handled(self, tmp_path):
        pipeline = _make_pipeline_with_mocks()
        fake_file = tmp_path / "empty.txt"
        fake_file.write_text("")

        pipeline._indexer = MagicMock()
        pipeline._indexer.index_file.return_value = DocumentUploadResult(
            doc_id="x", filename="empty.txt",
            status=DocumentStatus.FAILED,
            chunk_count=0,
            message="No text extracted",
        )
        result = pipeline.ingest_document(fake_file)
        assert result.status == DocumentStatus.FAILED
