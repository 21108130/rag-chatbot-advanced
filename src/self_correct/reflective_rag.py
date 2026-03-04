
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger
from src.utils.models import ChatRequest, ChatResponse, RetrievalResult, RetrievedChunk



@dataclass
class ConfidenceScores:
    
    retrieval_score:   float
    coverage_score:    float
    groundedness_score: float
    overall:           float

    def is_confident(self, threshold: float = 0.4) -> bool:
        return self.overall >= threshold




class RetrievalSelfEvaluator:


    def __init__(
        self,
        min_similarity:   float = 0.25,
        min_coverage:     float = 0.40,
        min_chunks:       int   = 2,
    ) -> None:
        self.min_similarity = min_similarity
        self.min_coverage   = min_coverage
        self.min_chunks     = min_chunks

    def score(
        self,
        query:  str,
        result: RetrievalResult,
    ) -> ConfidenceScores:

        chunks = result.chunks


        if not chunks:
            return ConfidenceScores(
                retrieval_score=0.0, coverage_score=0.0,
                groundedness_score=0.0, overall=0.0
            )

        sim_scores = [c.similarity_score for c in chunks]
        avg_sim    = sum(sim_scores) / len(sim_scores)
        retrieval_score = min(1.0, avg_sim / max(self.min_similarity, 0.01))


        query_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        stops = {'what', 'how', 'when', 'where', 'does', 'can', 'will', 'that', 'this'}
        query_terms -= stops

        if query_terms:
            context = " ".join(c.content.lower() for c in chunks)
            covered = sum(1 for t in query_terms if t in context)
            coverage_score = covered / len(query_terms)
        else:
            coverage_score = 1.0


        total_context = " ".join(c.content for c in chunks)
        groundedness = min(1.0, len(total_context) / 1000)

        overall = round(
            0.45 * retrieval_score +
            0.35 * coverage_score +
            0.20 * groundedness,
            3
        )

        scores = ConfidenceScores(
            retrieval_score    = round(retrieval_score, 3),
            coverage_score     = round(coverage_score, 3),
            groundedness_score = round(groundedness, 3),
            overall            = overall,
        )

        logger.info(
            f"[SelfEval] scores: retrieval={scores.retrieval_score:.2f} "
            f"coverage={scores.coverage_score:.2f} "
            f"grnd={scores.groundedness_score:.2f} "
            f"overall={scores.overall:.2f}"
        )
        return scores




class AnswerGrader:


    GRADE_PROMPT = """You are grading an AI answer for quality.

{context}

User question: {query}

AI answer: {answer}

Grade the answer on these dimensions (score 1-5):
1. Is the answer grounded in the context? (not hallucinated)
2. Does it fully answer the question?
3. Are sources cited?

Respond with ONLY: GRADE:<score>/5 ISSUE:<one sentence issue if score<4>
Example: GRADE:4/5 ISSUE:Missing citation for the main claim"""

    REFINE_PROMPT = """The following answer has quality issues: {issue}

Original question: {query}
Context: {context}
Original answer: {answer}

Write an improved answer that fixes the issue. Be specific, cite [Source N] references, and stay grounded in the context."""

    def __init__(self, llm_fn: Optional = None) -> None:
        self.llm_fn = llm_fn

    def grade(self, query: str, answer: str, context: str) -> Tuple[int, Optional[str]]:

        if not self.llm_fn:
            return 5, None

        try:
            prompt = self.GRADE_PROMPT.format(
                context=context[:2000],
                query=query,
                answer=answer[:1000],
            )
            response = self.llm_fn(prompt)


            grade_match = re.search(r'GRADE:(\d)/5', response)
            issue_match = re.search(r'ISSUE:(.+)', response)

            score = int(grade_match.group(1)) if grade_match else 3
            issue = issue_match.group(1).strip() if issue_match else None

            return score, issue

        except Exception as e:
            logger.warning(f"[AnswerGrader] Grading failed: {e}")
            return 5, None

    def refine(self, query: str, answer: str, context: str, issue: str) -> str:

        if not self.llm_fn:
            return answer

        try:
            prompt = self.REFINE_PROMPT.format(
                issue=issue, query=query,
                context=context[:2000], answer=answer[:1000],
            )
            refined = self.llm_fn(prompt)
            if len(refined) > 50:
                logger.info(f"[AnswerGrader] Refined answer (issue: {issue[:50]})")
                return refined.strip()
        except Exception as e:
            logger.warning(f"[AnswerGrader] Refinement failed: {e}")

        return answer



class CorrectiveRAG:


    HIGH_CONFIDENCE  = 0.55
    LOW_CONFIDENCE   = 0.30
    VERY_LOW         = 0.15

    def __init__(
        self,
        pipeline,
        llm_fn:            Optional = None,
        enable_grading:    bool = False,
        enable_web_search: bool = True,
    ) -> None:
        self.pipeline          = pipeline
        self.enable_grading    = enable_grading
        self.enable_web_search = enable_web_search
        self._evaluator        = RetrievalSelfEvaluator()
        self._grader           = AnswerGrader(llm_fn) if enable_grading else None

    def _expand_query(self, query: str) -> str:

        expansions = []
        if not query.lower().startswith("what"):
            expansions.append(f"what is {query}")
        if not query.lower().startswith("explain"):
            expansions.append(f"explain {query}")
        return f"{query} {' '.join(expansions[:2])}"

    def _web_search_fallback(self, query: str) -> List[RetrievedChunk]:

        try:
            from duckduckgo_search import DDGS
            from src.utils.models import RetrievedChunk
            import time as t
            t.sleep(0.5)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, region="wt-wt"))

            web_chunks = []
            for i, r in enumerate(results):
                body = r.get("body", "") or ""
                if len(body) > 50:
                    web_chunks.append(RetrievedChunk(
                        chunk_id         = f"web_{i}",
                        doc_id           = "web_search",
                        content          = f"{r.get('title', '')}: {body}",
                        similarity_score = 0.5,
                        metadata         = {
                            "source": "Web Search",
                            "url":    r.get("href", ""),
                            "retrieval_method": "web_fallback",
                        },
                    ))
            return web_chunks
        except Exception as e:
            logger.warning(f"[CorrectiveRAG] Web fallback failed: {e}")
            return []

    def chat_with_correction(
        self,
        query:   str,
        user_id: str = "anonymous",
        top_k:   int = 5,
    ) -> Dict:

        start = time.perf_counter()
        correction_applied = "none"


        retriever = self.pipeline._get_retriever(user_id)
        retrieval = retriever.retrieve(query=query, top_k=top_k)


        confidence = self._evaluator.score(query, retrieval)


        if confidence.overall < self.VERY_LOW:
            logger.info(
                f"[CorrectiveRAG] VERY LOW confidence ({confidence.overall:.2f}) — "
                f"expanding query + web search"
            )
            expanded_query = self._expand_query(query)
            retrieval2 = retriever.retrieve(query=expanded_query, top_k=top_k)


            seen = {c.chunk_id for c in retrieval.chunks}
            extra = [c for c in retrieval2.chunks if c.chunk_id not in seen]
            merged_chunks = retrieval.chunks + extra


            if self.enable_web_search:
                web_chunks = self._web_search_fallback(query)
                merged_chunks.extend(web_chunks)

            from src.utils.models import RetrievalResult
            retrieval = RetrievalResult(
                query      = query,
                chunks     = merged_chunks[:top_k * 2],
                latency_ms = retrieval.latency_ms,
            )
            correction_applied = "re-retrieved + web"

        elif confidence.overall < self.LOW_CONFIDENCE:
            logger.info(
                f"[CorrectiveRAG] LOW confidence ({confidence.overall:.2f}) — "
                f"re-retrieving with expanded query"
            )
            expanded_query = self._expand_query(query)
            retrieval2     = retriever.retrieve(query=expanded_query, top_k=top_k)


            seen  = {c.chunk_id for c in retrieval.chunks}
            extra = [c for c in retrieval2.chunks if c.chunk_id not in seen]
            from src.utils.models import RetrievalResult
            retrieval = RetrievalResult(
                query      = query,
                chunks     = retrieval.chunks + extra[:top_k],
                latency_ms = retrieval.latency_ms,
            )
            correction_applied = "re-retrieved"

        else:
            logger.info(f"[CorrectiveRAG] Confidence OK ({confidence.overall:.2f})")


        from src.utils.models import ChatRequest
        chat_req = ChatRequest(query=query)
        chat_req.__dict__['_override_retrieval'] = retrieval

        context = self.pipeline._optimizer.to_context_string(
            self.pipeline._optimizer.optimize(retrieval, query=query)
        )

        llm_result = self.pipeline.llm.generate(
            query=query, context=context
        )
        answer = llm_result["answer"]


        grade_info = {}
        if self.enable_grading and self._grader:
            score, issue = self._grader.grade(query, answer, context)
            if score < 3 and issue:
                logger.info(
                    f"[CorrectiveRAG] Graded {score}/5, refining: {issue[:50]}"
                )
                answer = self._grader.refine(query, answer, context, issue)
                grade_info = {"grade": score, "issue": issue, "refined": True}
            else:
                grade_info = {"grade": score, "refined": False}

        total_ms = (time.perf_counter() - start) * 1000

        return {
            "answer":              answer,
            "sources":             retrieval.chunks,
            "confidence":          confidence,
            "correction_applied":  correction_applied,
            "latency_ms":          round(total_ms, 1),
            "retrieval_score":     confidence.retrieval_score,
            "coverage_score":      confidence.coverage_score,
            "grade_info":          grade_info,
        }
