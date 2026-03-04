

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from config.settings import settings
from src.embeddings.embedder import get_embedder
from src.llm.groq_client import GroqLLMClient
from src.retrieval.indexer import DocumentIndexer
from src.retrieval.retriever import Retriever
from src.utils.logger import logger
from src.utils.models import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    DocumentUploadResult,
    MessageRole,
)
from src.vectordb.chroma_store import ChromaVectorStore


class RAGPipeline:


    def __init__(self) -> None:
        self._vector_store: Optional[ChromaVectorStore] = None
        self._embedder     = None
        self._indexer:      Optional[DocumentIndexer]  = None
        self._retriever:    Optional[Retriever]        = None
        self._llm:          Optional[GroqLLMClient]    = None


        self._sessions: Dict[str, ConversationHistory] = {}


    @property
    def vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore()
        return self._vector_store

    @property
    def indexer(self) -> DocumentIndexer:
        if self._indexer is None:
            self._indexer = DocumentIndexer(
                vector_store = self.vector_store,
                embedder     = get_embedder(),
            )
        return self._indexer

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store = self.vector_store,
                embedder     = get_embedder(),
            )
        return self._retriever

    @property
    def llm(self) -> GroqLLMClient:
        if self._llm is None:
            self._llm = GroqLLMClient()
        return self._llm



    def ingest_document(self, file_path: str | Path) -> DocumentUploadResult:

        return self.indexer.index_file(file_path)


    def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        start = time.perf_counter()


        conv_id = request.conversation_id or str(uuid4())
        if conv_id not in self._sessions:
            self._sessions[conv_id] = ConversationHistory(
                conversation_id=conv_id
            )
        session = self._sessions[conv_id]

        logger.info(f"[RAG] Chat | conv={conv_id} | query='{request.query[:60]}'")

        retrieval = self.retriever.retrieve(
            query = request.query,
            top_k = request.top_k or settings.top_k_results,
        )
        context = self.retriever.format_context(retrieval)


        llm_result = self.llm.generate(
            query   = request.query,
            context = context,
            history = session,
        )

        answer = llm_result["answer"]

        session.add_message(MessageRole.USER,      request.query)
        session.add_message(MessageRole.ASSISTANT, answer)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"[RAG] Response generated in {elapsed_ms:.0f}ms | "
            f"sources={len(retrieval.chunks)}"
        )

        return ChatResponse(
            answer          = answer,
            sources         = retrieval.chunks,
            conversation_id = conv_id,
            latency_ms      = elapsed_ms,
            tokens_used     = llm_result.get("tokens_used"),
        )



    def get_session(self, conversation_id: str) -> Optional[ConversationHistory]:
        return self._sessions.get(conversation_id)

    def clear_session(self, conversation_id: str) -> None:
        self._sessions.pop(conversation_id, None)
        logger.debug(f"[RAG] Session cleared: {conversation_id}")

    def clear_all_sessions(self) -> None:
        self._sessions.clear()
        logger.debug("[RAG] All sessions cleared.")


    def get_kb_stats(self) -> Dict:
        return self.vector_store.get_stats()
