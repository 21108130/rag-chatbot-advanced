
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Tuple

from src.utils.logger import logger



@dataclass
class RetrievalEvent:
    timestamp:        float
    query:            str
    query_id:         str
    retrieved_docs:   List[str]
    relevance_scores: List[float]
    retrieval_method: str
    latency_ms:       float
    top_k_requested:  int
    chunks_returned:  int
    user_id:          str


@dataclass
class GenerationEvent:
    timestamp:        float
    query_id:         str
    answer_length:    int
    context_length:   int
    context_tokens:   int
    has_citations:    bool
    citation_count:   int
    groundedness:     float
    latency_ms:       float
    model:            str
    provider:         str
    tokens_used:      int


@dataclass
class BusinessEvent:
    timestamp:        float
    user_id:          str
    query_id:         str
    cost_usd:         float
    satisfaction:     Optional[int]
    query_category:   str
    failure_type:     Optional[str]




class RAGMetricsTracker:


    def __init__(self, window_size: int = 500) -> None:

        self._retrieval_events:   Deque[RetrievalEvent]   = deque(maxlen=window_size)
        self._generation_events:  Deque[GenerationEvent]  = deque(maxlen=window_size)
        self._business_events:    Deque[BusinessEvent]    = deque(maxlen=window_size)


        self._doc_retrieval_counts: DefaultDict[str, int]    = defaultdict(int)
        self._doc_relevance_sums:   DefaultDict[str, float]  = defaultdict(float)
        self._query_categories:     DefaultDict[str, int]    = defaultdict(int)
        self._failure_types:        DefaultDict[str, int]    = defaultdict(int)
        self._hourly_queries:       DefaultDict[str, int]    = defaultdict(int)


        self._total_cost_usd: float = 0.0
        self._total_tokens:   int   = 0


        self._retrieval_latencies: Deque[float] = deque(maxlen=200)
        self._llm_latencies:       Deque[float] = deque(maxlen=200)

        logger.info("[RAGMetrics] RAG-specific metrics tracker initialized")



    @staticmethod
    def compute_groundedness(answer: str, context: str) -> float:

        import re
        if not answer or not context:
            return 0.0

        context_lower = context.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 20]
        if not sentences:
            return 0.5

        grounded = 0
        for sent in sentences:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
            if not words:
                continue
            matches = sum(1 for w in words if w in context_lower)
            if matches / len(words) >= 0.30:
                grounded += 1

        return round(grounded / len(sentences), 3)

    @staticmethod
    def count_citations(answer: str) -> int:
        """Count [Source N] style citations in an answer."""
        import re
        return len(re.findall(r'\[Source\s*\d+\]', answer, re.IGNORECASE))



    def record_retrieval_event(
        self,
        query:            str,
        query_id:         str,
        chunks:           list,
        retrieval_method: str  = "hybrid",
        latency_ms:       float = 0.0,
        top_k:            int   = 5,
        user_id:          str   = "anonymous",
    ) -> None:
        """Record a retrieval event with per-document tracking."""
        doc_names   = [c.metadata.get("source", c.doc_id) for c in chunks]
        scores      = [c.similarity_score for c in chunks]

        event = RetrievalEvent(
            timestamp        = time.time(),
            query            = query[:80],
            query_id         = query_id,
            retrieved_docs   = doc_names,
            relevance_scores = scores,
            retrieval_method = retrieval_method,
            latency_ms       = latency_ms,
            top_k_requested  = top_k,
            chunks_returned  = len(chunks),
            user_id          = user_id,
        )
        self._retrieval_events.append(event)


        for doc, score in zip(doc_names, scores):
            self._doc_retrieval_counts[doc] += 1
            self._doc_relevance_sums[doc]   += score

        self._retrieval_latencies.append(latency_ms)


        hour_key = datetime.utcnow().strftime("%Y-%m-%d %H:00")
        self._hourly_queries[hour_key] += 1

    def record_generation_event(
        self,
        query_id:        str,
        answer:          str,
        context:         str,
        context_tokens:  int  = 0,
        latency_ms:      float = 0.0,
        model:           str  = "unknown",
        provider:        str  = "unknown",
        tokens_used:     int  = 0,
    ) -> None:
        """Record a generation event with quality scoring."""
        groundedness  = self.compute_groundedness(answer, context)
        citation_count = self.count_citations(answer)

        event = GenerationEvent(
            timestamp      = time.time(),
            query_id       = query_id,
            answer_length  = len(answer),
            context_length = len(context),
            context_tokens = context_tokens,
            has_citations  = citation_count > 0,
            citation_count = citation_count,
            groundedness   = groundedness,
            latency_ms     = latency_ms,
            model          = model,
            provider       = provider,
            tokens_used    = tokens_used,
        )
        self._generation_events.append(event)
        self._llm_latencies.append(latency_ms)
        self._total_tokens += tokens_used

    def record_business_event(
        self,
        user_id:        str,
        query_id:       str,
        cost_usd:       float         = 0.0,
        satisfaction:   Optional[int] = None,
        query_category: str           = "general",
        failure_type:   Optional[str] = None,
    ) -> None:

        event = BusinessEvent(
            timestamp      = time.time(),
            user_id        = user_id,
            query_id       = query_id,
            cost_usd       = cost_usd,
            satisfaction   = satisfaction,
            query_category = query_category,
            failure_type   = failure_type,
        )
        self._business_events.append(event)
        self._total_cost_usd   += cost_usd
        self._query_categories[query_category] += 1

        if failure_type:
            self._failure_types[failure_type] += 1


    def get_dashboard(self) -> Dict[str, Any]:

        def percentile(data, p):
            if not data:
                return 0.0
            s = sorted(data)
            idx = max(0, int(len(s) * p) - 1)
            return round(s[idx], 1)

        def avg(data):
            return round(sum(data) / len(data), 3) if data else 0.0


        ret_events = list(self._retrieval_events)
        ret_latencies = list(self._retrieval_latencies)
        avg_scores = [
            avg(e.relevance_scores) for e in ret_events if e.relevance_scores
        ]
        avg_chunks = avg([e.chunks_returned for e in ret_events])


        gen_events = list(self._generation_events)
        groundedness_vals = [e.groundedness for e in gen_events]
        citation_rates    = [1.0 if e.has_citations else 0.0 for e in gen_events]

        biz_events = list(self._business_events)
        sat_events = [e for e in biz_events if e.satisfaction is not None]
        pos_sat    = sum(1 for e in sat_events if e.satisfaction and e.satisfaction > 0)


        doc_quality = {}
        for doc, count in self._doc_retrieval_counts.items():
            avg_rel = self._doc_relevance_sums[doc] / count if count else 0
            doc_quality[doc] = {
                "retrieval_count": count,
                "avg_relevance":   round(avg_rel, 3),
            }


        now  = datetime.utcnow()
        trend_24h = {}
        for i in range(24):
            hour_key = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:00")
            trend_24h[hour_key] = self._hourly_queries.get(hour_key, 0)

        return {

            "total_queries":       len(ret_events),
            "total_tokens":        self._total_tokens,
            "total_cost_usd":      round(self._total_cost_usd, 6),
            "cost_per_query":      round(self._total_cost_usd / max(len(ret_events), 1), 6),


            "retrieval": {
                "avg_chunks_returned":    round(avg_chunks, 1),
                "avg_relevance_score":    avg(avg_scores),
                "avg_latency_ms":         avg(ret_latencies),
                "p95_latency_ms":         percentile(ret_latencies, 0.95),
                "methods_used":           list({e.retrieval_method for e in ret_events}),
                "doc_retrieval_counts":   dict(sorted(
                    self._doc_retrieval_counts.items(), key=lambda x: -x[1]
                )[:10]),
                "doc_quality":            doc_quality,
            },


            "generation": {
                "avg_groundedness":    avg(groundedness_vals),
                "citation_rate":       round(avg(citation_rates) * 100, 1),
                "avg_answer_length":   avg([e.answer_length for e in gen_events]),
                "avg_context_tokens":  avg([e.context_tokens for e in gen_events]),
                "avg_latency_ms":      avg(list(self._llm_latencies)),
                "p95_latency_ms":      percentile(list(self._llm_latencies), 0.95),
                "models_used":         list({e.model for e in gen_events}),
                "providers_used":      list({e.provider for e in gen_events}),
            },


            "business": {
                "satisfaction_rate":   round(pos_sat / max(len(sat_events), 1) * 100, 1),
                "total_feedback":      len(sat_events),
                "query_categories":    dict(self._query_categories),
                "failure_types":       dict(sorted(
                    self._failure_types.items(), key=lambda x: -x[1]
                )),
                "hourly_trend_24h":    dict(sorted(trend_24h.items())),
            },
        }

    def get_document_health(self) -> List[Dict]:
        """Return per-document health stats for admin dashboard."""
        result = []
        for doc, count in sorted(
            self._doc_retrieval_counts.items(), key=lambda x: -x[1]
        ):
            avg_rel = self._doc_relevance_sums[doc] / count if count else 0.0
            result.append({
                "document":        doc,
                "times_retrieved": count,
                "avg_relevance":   round(avg_rel, 3),
                "quality_label":   (
                    "✅ High" if avg_rel >= 0.7 else
                    "🟡 Medium" if avg_rel >= 0.4 else
                    "🔴 Low"
                ),
            })
        return result




_tracker: Optional[RAGMetricsTracker] = None

def get_rag_metrics() -> RAGMetricsTracker:
    global _tracker
    if _tracker is None:
        _tracker = RAGMetricsTracker()
    return _tracker
