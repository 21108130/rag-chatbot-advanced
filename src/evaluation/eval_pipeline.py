
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import logger



@dataclass
class EvalQuery:
    """A single evaluation test case."""
    query_id:        str
    query:           str
    expected_answer: str
    relevant_doc_keywords: List[str]
    category:        str = "general"
    priority:        str = "medium"


@dataclass
class EvalResult:

    query_id:           str
    query:              str
    category:           str
    priority:           str

    # Pipeline outputs
    actual_answer:      str = ""
    retrieved_chunks:   int = 0
    latency_ms:         float = 0.0
    retrieval_latency:  float = 0.0
    llm_latency:        float = 0.0

    # Scores (0.0 – 1.0)
    retrieval_precision: float = 0.0
    answer_correctness:  float = 0.0
    groundedness:        float = 0.0
    overall_score:       float = 0.0

    # Flags
    error:              Optional[str] = None
    passed:             bool = False


@dataclass
class EvalReport:

    run_id:          str
    timestamp:       str
    total_queries:   int
    passed:          int
    failed:          int
    pass_rate:       float


    avg_retrieval_precision: float
    avg_answer_correctness:  float
    avg_groundedness:        float
    avg_overall_score:       float
    avg_latency_ms:          float
    p95_latency_ms:          float

    by_category:     Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_priority:     Dict[str, Dict[str, float]] = field(default_factory=dict)


    results:         List[EvalResult] = field(default_factory=list)
    failing_queries: List[Dict] = field(default_factory=list)


    config_snapshot: Dict[str, Any] = field(default_factory=dict)




class EvaluationPipeline:


    def __init__(
        self,
        pipeline,
        user_id:         str   = "eval_user",
        pass_threshold:  float = 0.6,
        retrieval_weight: float = 0.35,
        correctness_weight: float = 0.40,
        groundedness_weight: float = 0.25,
    ) -> None:
        self.pipeline            = pipeline
        self.user_id             = user_id
        self.pass_threshold      = pass_threshold
        self.retrieval_weight    = retrieval_weight
        self.correctness_weight  = correctness_weight
        self.groundedness_weight = groundedness_weight



    def _score_retrieval_precision(
        self,
        chunks: list,
        relevant_keywords: List[str],
    ) -> float:

        if not chunks or not relevant_keywords:
            return 0.0

        relevant_count = 0
        kws = [kw.lower() for kw in relevant_keywords]
        for chunk in chunks:
            text = chunk.content.lower()
            if any(kw in text for kw in kws):
                relevant_count += 1

        return round(relevant_count / len(chunks), 3)

    def _score_answer_correctness(
        self,
        actual: str,
        expected: str,
    ) -> float:

        if not actual or not expected:
            return 0.0


        def tokenize(text: str) -> set:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

            stops = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are',
                     'was', 'has', 'have', 'been', 'will', 'can', 'not', 'its'}
            return {w for w in words if w not in stops}

        expected_tokens = tokenize(expected)
        actual_tokens   = tokenize(actual)

        if not expected_tokens:
            return 0.0


        tp = len(expected_tokens & actual_tokens)
        precision = tp / len(actual_tokens) if actual_tokens else 0.0
        recall    = tp / len(expected_tokens)

        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 3)

    def _score_groundedness(
        self,
        answer: str,
        chunks: list,
    ) -> float:

        if not answer or not chunks:
            return 0.0

        context = " ".join(c.content.lower() for c in chunks)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return 0.5

        grounded_count = 0
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())

            matches = sum(1 for w in words if w in context)
            match_ratio = matches / len(words) if words else 0
            if match_ratio >= 0.35:
                grounded_count += 1

        return round(grounded_count / len(sentences), 3)

    def _evaluate_one(self, eq: EvalQuery) -> EvalResult:

        result = EvalResult(
            query_id  = eq.query_id,
            query     = eq.query,
            category  = eq.category,
            priority  = eq.priority,
        )

        try:
            from src.utils.models import ChatRequest
            start = time.perf_counter()

            chat_req = ChatRequest(query=eq.query)
            response = self.pipeline.chat(chat_req, user_id=self.user_id)

            result.latency_ms       = response.latency_ms
            result.retrieved_chunks = len(response.sources)
            result.actual_answer    = response.answer

            result.retrieval_precision = self._score_retrieval_precision(
                response.sources, eq.relevant_doc_keywords
            )


            result.answer_correctness = self._score_answer_correctness(
                response.answer, eq.expected_answer
            )


            result.groundedness = self._score_groundedness(
                response.answer, response.sources
            )

            result.overall_score = round(
                self.retrieval_weight    * result.retrieval_precision +
                self.correctness_weight  * result.answer_correctness +
                self.groundedness_weight * result.groundedness,
                3
            )

            result.passed = result.overall_score >= self.pass_threshold

        except Exception as exc:
            result.error  = str(exc)
            result.passed = False
            logger.error(f"[Eval] Error on query '{eq.query[:40]}': {exc}")

        logger.info(
            f"[Eval] {eq.query_id} | score={result.overall_score:.2f} "
            f"(ret={result.retrieval_precision:.2f} "
            f"corr={result.answer_correctness:.2f} "
            f"grnd={result.groundedness:.2f}) | "
            f"{'PASS' if result.passed else 'FAIL'}"
        )
        return result



    def run(
        self,
        queries: List[EvalQuery],
        config_snapshot: Optional[Dict] = None,
    ) -> EvalReport:

        logger.info(f"[Eval] Starting evaluation run with {len(queries)} queries")
        start_ts = datetime.utcnow().isoformat()
        run_id   = f"eval_{int(time.time())}"

        results: List[EvalResult] = []
        for i, eq in enumerate(queries):
            logger.info(f"[Eval] Query {i+1}/{len(queries)}: {eq.query_id}")
            r = self._evaluate_one(eq)
            results.append(r)


        n = len(results)
        valid = [r for r in results if r.error is None]
        passed = [r for r in valid if r.passed]

        def avg(vals): return round(sum(vals) / len(vals), 3) if vals else 0.0

        latencies = [r.latency_ms for r in valid]
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[p95_idx] if sorted_lat else 0.0


        categories = {}
        for r in valid:
            cat = r.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        by_category = {}
        for cat, cat_results in categories.items():
            by_category[cat] = {
                "count":       len(cat_results),
                "pass_rate":   avg([1.0 if r.passed else 0.0 for r in cat_results]),
                "avg_score":   avg([r.overall_score for r in cat_results]),
                "avg_latency": avg([r.latency_ms for r in cat_results]),
            }

        priorities = {}
        for r in valid:
            p = r.priority
            if p not in priorities:
                priorities[p] = []
            priorities[p].append(r)

        by_priority = {}
        for pri, pri_results in priorities.items():
            by_priority[pri] = {
                "count":     len(pri_results),
                "pass_rate": avg([1.0 if r.passed else 0.0 for r in pri_results]),
                "avg_score": avg([r.overall_score for r in pri_results]),
            }

        failing = [
            {
                "query_id":   r.query_id,
                "query":      r.query[:80],
                "score":      r.overall_score,
                "retrieval":  r.retrieval_precision,
                "correctness": r.answer_correctness,
                "groundedness": r.groundedness,
                "error":      r.error,
            }
            for r in valid if not r.passed
        ]

        report = EvalReport(
            run_id        = run_id,
            timestamp     = start_ts,
            total_queries = n,
            passed        = len(passed),
            failed        = n - len(passed),
            pass_rate     = round(len(passed) / n * 100, 1) if n else 0.0,
            avg_retrieval_precision = avg([r.retrieval_precision for r in valid]),
            avg_answer_correctness  = avg([r.answer_correctness  for r in valid]),
            avg_groundedness        = avg([r.groundedness        for r in valid]),
            avg_overall_score       = avg([r.overall_score       for r in valid]),
            avg_latency_ms          = avg(latencies),
            p95_latency_ms          = round(p95_lat, 1),
            by_category             = by_category,
            by_priority             = by_priority,
            results                 = results,
            failing_queries         = failing,
            config_snapshot         = config_snapshot or {},
        )

        logger.info(
            f"[Eval] Run complete: {len(passed)}/{n} passed "
            f"({report.pass_rate}%) | avg_score={report.avg_overall_score}"
        )
        return report

    def save_report(
        self,
        report: EvalReport,
        output_path: str = "./data/eval_reports/",
    ) -> str:
        """Save evaluation report as JSON."""
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"{report.run_id}.json"

        data = asdict(report)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"[Eval] Report saved: {file_path}")
        return str(file_path)

    def print_summary(self, report: EvalReport) -> None:
        """Print a human-readable summary to stdout."""
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT — {report.run_id}")
        print(sep)
        print(f"  Timestamp  : {report.timestamp}")
        print(f"  Total      : {report.total_queries} queries")
        print(f"  Pass Rate  : {report.pass_rate}% ({report.passed}/{report.total_queries})")
        print(f"\n  Avg Scores:")
        print(f"    Retrieval Precision : {report.avg_retrieval_precision:.3f}")
        print(f"    Answer Correctness  : {report.avg_answer_correctness:.3f}")
        print(f"    Groundedness        : {report.avg_groundedness:.3f}")
        print(f"    Overall             : {report.avg_overall_score:.3f}")
        print(f"\n  Latency:")
        print(f"    Average : {report.avg_latency_ms:.0f} ms")
        print(f"    P95     : {report.p95_latency_ms:.0f} ms")

        if report.by_category:
            print(f"\n  By Category:")
            for cat, stats in report.by_category.items():
                print(f"    {cat:20s} pass={stats['pass_rate']*100:.0f}% "
                      f"score={stats['avg_score']:.2f} "
                      f"lat={stats['avg_latency']:.0f}ms")

        if report.failing_queries:
            print(f"\n  Top Failures ({len(report.failing_queries)}):")
            for f in report.failing_queries[:5]:
                print(f"    [{f['query_id']}] score={f['score']:.2f} — {f['query'][:60]}")
        print(sep)
