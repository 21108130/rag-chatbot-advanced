
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Generator, List, Optional
from uuid import uuid4

from config.settings import settings
from src.retrieval.advanced_rag_pipeline import AdvancedRAGPipeline
from src.utils.logger import logger
from src.utils.models import ChatRequest, ChatResponse


class EnhancedRAGPipeline(AdvancedRAGPipeline):


    def __init__(
        self,
        enable_reranking:       bool = True,
        enable_streaming:       bool = True,
        max_context_tokens:     int  = 3000,

        enable_smart_routing:   bool = True,
        enable_self_correction: bool = True,
        enable_query_transform: bool = False,
        enable_quality_extract: bool = True,
        enable_rag_metrics:     bool = True,
        chunking_strategy:      str  = "semantic",
    ) -> None:
        super().__init__(
            enable_reranking=enable_reranking,
            enable_streaming=enable_streaming,
            max_context_tokens=max_context_tokens,
        )

        self.enable_smart_routing   = enable_smart_routing
        self.enable_self_correction = enable_self_correction
        self.enable_query_transform = enable_query_transform
        self.enable_quality_extract = enable_quality_extract
        self.enable_rag_metrics     = enable_rag_metrics
        self.chunking_strategy      = chunking_strategy


        self._smart_router     = None
        self._corrective_rag   = None
        self._query_transformer = None
        self._quality_extractor = None
        self._rag_metrics      = None
        self._structured_fb    = None
        self._tenant_manager   = None
        self._priority_fw      = None
        self._evaluator        = None

        logger.info(
            f"[EnhancedRAG] Initialized with enhancements: "
            f"smart_routing={enable_smart_routing} "
            f"self_correct={enable_self_correction} "
            f"query_transform={enable_query_transform}"
        )



    @property
    def smart_router(self):
        if self._smart_router is None and self.enable_smart_routing:
            from src.routing.smart_router import get_smart_router
            self._smart_router = get_smart_router()
        return self._smart_router

    @property
    def corrective_rag(self):
        if self._corrective_rag is None:
            from src.self_correct.reflective_rag import CorrectiveRAG
            self._corrective_rag = CorrectiveRAG(
                pipeline           = self,
                enable_web_search  = True,
            )
        return self._corrective_rag

    @property
    def query_transformer(self):
        if self._query_transformer is None:
            from src.query_transform.transformer import QueryTransformer
            self._query_transformer = QueryTransformer(llm_fn=None)  # rule-based
        return self._query_transformer

    @property
    def quality_extractor(self):
        if self._quality_extractor is None:
            from src.extraction.quality_extractor import QualityExtractor
            self._quality_extractor = QualityExtractor()
        return self._quality_extractor

    @property
    def rag_metrics(self):
        if self._rag_metrics is None and self.enable_rag_metrics:
            from src.observability.rag_metrics import get_rag_metrics
            self._rag_metrics = get_rag_metrics()
        return self._rag_metrics

    @property
    def structured_feedback(self):
        if self._structured_fb is None:
            from src.feedback.structured_feedback import get_structured_feedback
            self._structured_fb = get_structured_feedback()
        return self._structured_fb

    @property
    def tenant_manager(self):
        if self._tenant_manager is None:
            from src.tenant.isolation import get_tenant_manager
            self._tenant_manager = get_tenant_manager()
        return self._tenant_manager

    @property
    def priority_framework(self):
        if self._priority_fw is None:
            from src.prioritization.priority_framework import get_priority_framework
            self._priority_fw = get_priority_framework()
        return self._priority_fw

    @property
    def evaluator(self):
        if self._evaluator is None:
            from src.evaluation.eval_pipeline import EvaluationPipeline
            self._evaluator = EvaluationPipeline(self)
        return self._evaluator



    def ingest_document_with_quality(
        self,
        file_path: str | Path,
        user_id:   str = "anonymous",
        chunking_strategy: Optional[str] = None,
    ):

        path = Path(file_path)

        if self.enable_quality_extract:

            result = self.quality_extractor.extract(path)
            if not result.is_acceptable:
                logger.warning(
                    f"[EnhancedRAG] Document quality too low ({result.quality_score:.2f}): "
                    f"{path.name} — {result.issues}"
                )
            else:
                logger.info(
                    f"[EnhancedRAG] Quality OK ({result.quality_score:.2f}): {path.name}"
                )


        ingest_result = self.ingest_document(file_path, user_id=user_id)


        if ingest_result.chunk_count > 0:
            self.tenant_manager.increment_doc_count(user_id, by=1)

        return ingest_result


    def chat_with_query_transform(
        self,
        query:   str,
        user_id: str = "anonymous",
        strategy: str = "multi_query",
    ) -> ChatResponse:

        variants = self.query_transformer.transform(query, strategy=strategy, n=3)
        logger.info(f"[EnhancedRAG] Query variants: {variants}")


        retriever = self._get_retriever(user_id)
        all_chunks = []
        for variant in variants:
            result = retriever.retrieve(query=variant, top_k=5)
            all_chunks.extend(result.chunks)


        deduped = self.query_transformer.deduplicate_results(all_chunks)
        top_chunks = deduped[:settings.top_k_results]


        from src.utils.models import RetrievalResult
        merged_result = RetrievalResult(query=query, chunks=top_chunks)
        optimized = self._optimizer.optimize(merged_result, query=query)
        context   = self._optimizer.to_context_string(optimized)

        # Generate
        llm_result = self.llm.generate(query=query, context=context)

        return ChatResponse(
            answer          = llm_result["answer"],
            sources         = top_chunks,
            conversation_id = str(uuid4()),
            latency_ms      = 0.0,
            tokens_used     = llm_result.get("tokens_used"),
        )



    def chat_corrective(
        self,
        query:   str,
        user_id: str = "anonymous",
    ) -> Dict:
        """
        Self-corrective RAG chat with confidence scoring.

        Enhancement 8: Validates retrieval quality before generation.
        Falls back to expanded query / web search if confidence is low.
        """
        return self.corrective_rag.chat_with_correction(
            query=query, user_id=user_id
        )


    def record_structured_feedback(
        self,
        user_id:           str,
        query:             str,
        answer:            str,
        rating:            int,
        failure_type_str:  Optional[str] = None,
        expected_response: Optional[str] = None,
        sources:           Optional[list] = None,
        confidence_scores: Optional[Dict] = None,
        comment:           Optional[str]  = None,
        latency_ms:        Optional[float] = None,
    ) -> str:

        from src.feedback.structured_feedback import FailureType

        ft = None
        if failure_type_str:
            try:
                ft = FailureType(failure_type_str)
            except ValueError:
                ft = FailureType.OTHER

        doc_names = []
        ret_scores = []
        if sources:
            doc_names  = [c.metadata.get("source", c.doc_id) for c in sources]
            ret_scores = [c.similarity_score for c in sources]

        record_id = self.structured_feedback.record(
            user_id            = user_id,
            query              = query,
            answer             = answer,
            rating             = rating,
            failure_type       = ft,
            expected_response  = expected_response,
            retrieved_doc_names = doc_names,
            retrieval_scores   = ret_scores,
            confidence_scores  = confidence_scores or {},
            comment            = comment,
            latency_ms         = latency_ms,
        )


        if rating < 0 and ft:
            from src.prioritization.priority_framework import UserPersona
            self.priority_framework.record_query_failure(
                query        = query,
                failure_type = ft.value,
                user_id      = user_id,
            )

        return record_id



    def run_evaluation(
        self,
        category:   Optional[str] = None,
        priority:   Optional[str] = None,
        max_queries: Optional[int] = None,
        save_report: bool = True,
    ):

        from src.evaluation.eval_dataset import load_eval_dataset
        queries = load_eval_dataset(
            category=category, priority=priority, max_queries=max_queries
        )
        logger.info(f"[EnhancedRAG] Running evaluation with {len(queries)} queries")

        report = self.evaluator.run(
            queries,
            config_snapshot={
                "top_k":           settings.top_k_results,
                "chunk_size":      settings.chunk_size,
                "embedding_model": settings.embedding_model,
                "llm_model":       settings.llm_model,
                "chunking":        self.chunking_strategy,
            }
        )

        if save_report:
            self.evaluator.save_report(report)

        return report



    def get_routing_status(self) -> Dict:

        if self.smart_router:
            return self.smart_router.get_status()
        return {}



    def get_observability_dashboard(self) -> Dict:

        dashboard = {}
        if self.rag_metrics:
            dashboard["rag_metrics"] = self.rag_metrics.get_dashboard()
            dashboard["document_health"] = self.rag_metrics.get_document_health()

        dashboard["routing_status"] = self.get_routing_status()

        try:
            dashboard["feedback_analysis"] = self.structured_feedback.get_failure_analysis()
        except Exception:
            pass

        try:
            dashboard["action_plan"] = [
                {
                    "rank": a.rank,
                    "issue": a.issue,
                    "urgency": a.urgency,
                    "fix": a.suggested_fix,
                    "score": a.priority_score,
                }
                for a in self.priority_framework.get_action_plan(top_n=5)
            ]
        except Exception:
            pass

        return dashboard



    def register_tenant(
        self,
        tenant_id:       str,
        isolation_level: str = "medium",
        plan:            str = "standard",
    ) -> Dict:

        from src.tenant.isolation import IsolationLevel
        level = IsolationLevel(isolation_level)
        info  = self.tenant_manager.register_tenant(tenant_id, level, plan=plan)
        return self.tenant_manager.get_tenant_info(tenant_id) or {}

    def list_tenants(self) -> List[Dict]:
        """List all registered tenants."""
        return self.tenant_manager.list_tenants()



_enhanced_pipeline: Optional[EnhancedRAGPipeline] = None

def get_enhanced_pipeline() -> EnhancedRAGPipeline:
    global _enhanced_pipeline
    if _enhanced_pipeline is None:
        _enhanced_pipeline = EnhancedRAGPipeline()
    return _enhanced_pipeline
