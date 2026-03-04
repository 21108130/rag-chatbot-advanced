
from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

from src.utils.logger import logger



class MetricsCollector:
    

    def __init__(self) -> None:
        self._prometheus_available = False
        self._in_memory            = {
            "total_queries":       0,
            "total_errors":        0,
            "total_tokens":        0,
            "total_latency_ms":    0.0,
            "retrieval_latency_ms": 0.0,
            "llm_latency_ms":      0.0,
            "active_users":        0,
            "chunks_retrieved":    0,
        }
        self._latency_samples: list = []

        self._init_prometheus()

    def _init_prometheus(self) -> None:
        try:
            from prometheus_client import (
                Counter, Histogram, Gauge, REGISTRY, start_http_server
            )
            self._query_counter    = Counter("rag_queries_total",       "Total queries processed", ["user_id"])
            self._error_counter    = Counter("rag_errors_total",        "Total errors",            ["error_type"])
            self._token_counter    = Counter("rag_tokens_total",        "Total tokens used",       ["model"])
            self._query_latency    = Histogram("rag_query_latency_ms",  "Query latency in ms",     buckets=[50,100,250,500,1000,2000,5000])
            self._retrieval_latency= Histogram("rag_retrieval_latency_ms", "Retrieval latency ms")
            self._llm_latency      = Histogram("rag_llm_latency_ms",   "LLM latency ms")
            self._active_users     = Gauge("rag_active_users",         "Active user sessions")
            self._chunks_gauge     = Gauge("rag_chunks_retrieved",     "Chunks retrieved per query")
            self._prometheus_available = True
            logger.info("[Metrics] Prometheus metrics initialized.")
        except ImportError:
            logger.warning("[Metrics] prometheus-client not installed. Using in-memory metrics only.")

    def start_prometheus_server(self, port: int = 8001) -> None:
        """Start the Prometheus HTTP metrics endpoint on /metrics."""
        if not self._prometheus_available:
            logger.warning("[Metrics] Prometheus not available — skipping server start.")
            return
        try:
            from prometheus_client import start_http_server
            start_http_server(port)
            logger.info(f"[Metrics] Prometheus metrics server started on port {port}")
        except Exception as exc:
            logger.error(f"[Metrics] Failed to start metrics server: {exc}")


    def record_query(
        self,
        latency_ms:         float,
        retrieval_latency:  float,
        llm_latency:        float,
        tokens_used:        int,
        chunks_retrieved:   int,
        user_id:            str  = "anonymous",
        model:              str  = "unknown",
        error:              bool = False,
    ) -> None:
        """Record metrics for a single query."""
        # In-memory
        self._in_memory["total_queries"]        += 1
        self._in_memory["total_tokens"]         += tokens_used
        self._in_memory["total_latency_ms"]     += latency_ms
        self._in_memory["retrieval_latency_ms"] += retrieval_latency
        self._in_memory["llm_latency_ms"]       += llm_latency
        self._in_memory["chunks_retrieved"]     += chunks_retrieved
        if error:
            self._in_memory["total_errors"]     += 1


        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 100:
            self._latency_samples.pop(0)


        if self._prometheus_available:
            try:
                self._query_counter.labels(user_id=user_id).inc()
                self._query_latency.observe(latency_ms)
                self._retrieval_latency.observe(retrieval_latency)
                self._llm_latency.observe(llm_latency)
                self._token_counter.labels(model=model).inc(tokens_used)
                self._chunks_gauge.set(chunks_retrieved)
                if error:
                    self._error_counter.labels(error_type="query_error").inc()
            except Exception as exc:
                logger.debug(f"[Metrics] Prometheus record error: {exc}")

        logger.info(
            f"[Metrics] query latency={latency_ms:.0f}ms "
            f"retrieval={retrieval_latency:.0f}ms llm={llm_latency:.0f}ms "
            f"tokens={tokens_used} chunks={chunks_retrieved}"
        )

    def record_error(self, error_type: str, user_id: str = "anonymous") -> None:
        self._in_memory["total_errors"] += 1
        if self._prometheus_available:
            try:
                self._error_counter.labels(error_type=error_type).inc()
            except Exception:
                pass

    def set_active_users(self, count: int) -> None:
        self._in_memory["active_users"] = count
        if self._prometheus_available:
            try:
                self._active_users.set(count)
            except Exception:
                pass



    def get_summary(self) -> dict:
        """Return summary metrics for dashboard display."""
        n = self._in_memory["total_queries"]
        return {
            "total_queries":       n,
            "total_errors":        self._in_memory["total_errors"],
            "error_rate":          (self._in_memory["total_errors"] / n * 100) if n else 0.0,
            "avg_latency_ms":      self._in_memory["total_latency_ms"] / n if n else 0.0,
            "avg_tokens":          self._in_memory["total_tokens"] / n if n else 0.0,
            "total_tokens":        self._in_memory["total_tokens"],
            "active_users":        self._in_memory["active_users"],
            "p95_latency_ms":      self._p95_latency(),
            "prometheus_enabled":  self._prometheus_available,
        }

    def _p95_latency(self) -> float:
        if len(self._latency_samples) < 5:
            return 0.0
        sorted_samples = sorted(self._latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]




@contextmanager
def timed_operation(name: str):

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"[Timing] {name}: {elapsed_ms:.1f}ms")




_metrics: Optional[MetricsCollector] = None

def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
