

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Generator, Optional
from uuid import uuid4

from config.settings import settings
from src.embeddings.embedder import get_embedder
from src.feedback.feedback_store import get_feedback_store
from src.llm.multi_provider_client import MultiProviderLLMClient, get_llm_client
from src.observability.metrics import get_metrics, timed_operation
from src.retrieval.context_optimizer import ContextOptimizer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.indexer import DocumentIndexer
from src.retrieval.reranker import get_reranker
from src.utils.logger import logger
from src.utils.models import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    DocumentUploadResult,
    MessageRole,
)
from src.vectordb.chroma_store import ChromaVectorStore


class AdvancedRAGPipeline:


    def __init__(
        self,
        enable_reranking:  bool = True,
        enable_streaming:  bool = True,
        max_context_tokens: int = 3000,
    ) -> None:
        self.enable_reranking   = enable_reranking
        self.enable_streaming   = enable_streaming
        self.max_context_tokens = max_context_tokens


        self._vector_stores:  Dict[str, ChromaVectorStore] = {}
        self._retrievers:     Dict[str, HybridRetriever]   = {}
        self._indexers:       Dict[str, DocumentIndexer]   = {}

        self._llm: Optional[MultiProviderLLMClient] = None  # lazy-loaded
        self._reranker       = get_reranker() if enable_reranking else None
        self._optimizer      = ContextOptimizer(max_tokens=max_context_tokens)
        self._metrics        = get_metrics()
        self._feedback       = get_feedback_store()
        self._embedder       = get_embedder()


        self._sessions: Dict[str, Dict[str, ConversationHistory]] = {}

        logger.info(
            f"[AdvancedRAG] Pipeline initialized "
            f"(reranking={enable_reranking}, streaming={enable_streaming})"
        )


    @property
    def llm(self) -> MultiProviderLLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def get_llm_for_user(self, user_id: Optional[str] = None) -> MultiProviderLLMClient:

        if user_id:
            return get_llm_client(user_id=user_id)
        return self.llm



    def _get_collection_name(self, user_id: str) -> str:

        safe_id = user_id.replace("-", "")[:16]
        return f"user_{safe_id}"

    def _get_vector_store(self, user_id: str) -> ChromaVectorStore:
        if user_id not in self._vector_stores:
            self._vector_stores[user_id] = ChromaVectorStore(
                collection_name = self._get_collection_name(user_id),
            )
        return self._vector_stores[user_id]

    def _get_retriever(self, user_id: str) -> HybridRetriever:
        if user_id not in self._retrievers:
            self._retrievers[user_id] = HybridRetriever(
                vector_store = self._get_vector_store(user_id),
                embedder     = self._embedder,
            )
        return self._retrievers[user_id]

    def _get_indexer(self, user_id: str) -> DocumentIndexer:
        if user_id not in self._indexers:
            self._indexers[user_id] = DocumentIndexer(
                vector_store = self._get_vector_store(user_id),
                embedder     = self._embedder,
            )
        return self._indexers[user_id]


    @property
    def retriever(self):
        """Default retriever (anonymous user) — for agent compatibility."""
        return self._get_retriever("anonymous")



    def ingest_document(
        self,
        file_path: str | Path,
        user_id:   str = "anonymous",
    ) -> DocumentUploadResult:

        indexer = self._get_indexer(user_id)
        result  = indexer.index_file(file_path)
        logger.info(
            f"[AdvancedRAG] Ingested '{Path(file_path).name}' "
            f"for user={user_id} → {result.chunk_count} chunks"
        )
        return result


    def chat(
        self,
        request: ChatRequest,
        user_id: str = "anonymous",
    ) -> ChatResponse:

        pipeline_start = time.perf_counter()


        conv_id = request.conversation_id or str(uuid4())
        if user_id not in self._sessions:
            self._sessions[user_id] = {}
        if conv_id not in self._sessions[user_id]:
            self._sessions[user_id][conv_id] = ConversationHistory(conversation_id=conv_id)
        session = self._sessions[user_id][conv_id]

        logger.info(f"[AdvancedRAG] Chat | user={user_id} | query='{request.query[:60]}'")

        retrieval_latency = 0.0
        llm_latency       = 0.0
        error_occurred    = False

        try:

            retrieval_start = time.perf_counter()
            retriever = self._get_retriever(user_id)
            retrieval = retriever.retrieve(
                query = request.query,
                top_k = request.top_k or settings.top_k_results,
            )
            retrieval_latency = (time.perf_counter() - retrieval_start) * 1000


            if self.enable_reranking and self._reranker and retrieval.chunks:
                try:
                    retrieval = self._reranker.rerank(
                        query  = request.query,
                        result = retrieval,
                        top_k  = request.top_k or settings.top_k_results,
                    )
                except Exception as exc:
                    logger.warning(f"[AdvancedRAG] Reranking skipped: {exc}")


            optimized = self._optimizer.optimize(retrieval, query=request.query)
            context   = self._optimizer.to_context_string(optimized)


            llm_start = time.perf_counter()
            llm_result = self.get_llm_for_user(user_id).generate(
                query   = request.query,
                context = context,
                history = session,
            )
            llm_latency = (time.perf_counter() - llm_start) * 1000
            answer      = llm_result["answer"]

        except Exception as exc:
            logger.error(f"[AdvancedRAG] Pipeline error: {exc}")
            answer          = f"I encountered an error processing your request: {exc}"
            optimized       = retrieval if 'retrieval' in dir() else type('R', (), {'chunks': []})()
            llm_result      = {"tokens_used": 0, "model": settings.llm_model}
            error_occurred  = True


        session.add_message(MessageRole.USER,      request.query)
        session.add_message(MessageRole.ASSISTANT, answer)

        total_latency = (time.perf_counter() - pipeline_start) * 1000


        self._metrics.record_query(
            latency_ms         = total_latency,
            retrieval_latency  = retrieval_latency,
            llm_latency        = llm_latency,
            tokens_used        = llm_result.get("tokens_used") or 0,
            chunks_retrieved   = len(optimized.chunks),
            user_id            = user_id,
            model              = llm_result.get("model", settings.llm_model),
            error              = error_occurred,
        )

        logger.info(
            f"[AdvancedRAG] Response in {total_latency:.0f}ms | "
            f"retrieval={retrieval_latency:.0f}ms | llm={llm_latency:.0f}ms | "
            f"sources={len(optimized.chunks)}"
        )

        return ChatResponse(
            answer          = answer,
            sources         = optimized.chunks,
            conversation_id = conv_id,
            latency_ms      = total_latency,
            tokens_used     = llm_result.get("tokens_used"),
        )



    def chat_stream(
        self,
        query:           str,
        user_id:         str            = "anonymous",
        conversation_id: Optional[str] = None,
        top_k:           Optional[int] = None,
    ) -> Generator[str, None, None]:

        retriever = self._get_retriever(user_id)
        retrieval = retriever.retrieve(query=query, top_k=top_k or settings.top_k_results)

        if self.enable_reranking and self._reranker and retrieval.chunks:
            try:
                retrieval = self._reranker.rerank(query=query, result=retrieval)
            except Exception:
                pass

        optimized = self._optimizer.optimize(retrieval, query=query)
        context   = self._optimizer.to_context_string(optimized)


        conv_id = conversation_id or str(uuid4())
        if user_id not in self._sessions:
            self._sessions[user_id] = {}
        if conv_id not in self._sessions[user_id]:
            self._sessions[user_id][conv_id] = ConversationHistory(conversation_id=conv_id)
        session = self._sessions[user_id][conv_id]

        full_answer = ""
        for token in self.get_llm_for_user(user_id).stream(query=query, context=context, history=session):
            full_answer += token
            yield token


        session.add_message(MessageRole.USER,      query)
        session.add_message(MessageRole.ASSISTANT, full_answer)



    def record_feedback(
        self,
        user_id:         str,
        query:           str,
        answer:          str,
        rating:          int,
        conversation_id: Optional[str] = None,
        comment:         Optional[str] = None,
        latency_ms:      Optional[float] = None,
    ) -> str:

        return self._feedback.record(
            user_id         = user_id,
            query           = query,
            rating          = rating,
            answer          = answer,
            conversation_id = conversation_id,
            comment         = comment,
            latency_ms      = latency_ms,
        )


    def get_session(
        self,
        user_id:         str,
        conversation_id: str,
    ) -> Optional[ConversationHistory]:
        return self._sessions.get(user_id, {}).get(conversation_id)

    def clear_session(self, user_id: str, conversation_id: str) -> None:
        if user_id in self._sessions:
            self._sessions[user_id].pop(conversation_id, None)

    def get_kb_stats(self, user_id: str = "anonymous") -> Dict:
        return self._get_vector_store(user_id).get_stats()

    def get_metrics_summary(self) -> Dict:
        return self._metrics.get_summary()

    def get_feedback_summary(self, user_id: Optional[str] = None) -> Dict:
        return self._feedback.get_summary(user_id=user_id)