
from __future__ import annotations

import re
from typing import Callable, List, Optional

from src.utils.logger import logger


class MultiQueryGenerator:


    SYSTEM_PROMPT = """You are an expert at rephrasing search queries.
Given a user's question, generate {n} alternative phrasings that preserve the meaning
but use different vocabulary, perspectives, or specificity levels.
Output ONLY the alternative queries, one per line, no numbering or explanations."""

    def __init__(self, llm_fn: Optional[Callable] = None) -> None:
        """
        Args:
            llm_fn: Callable that takes a prompt str and returns str.
                    If None, uses rule-based expansion only.
        """
        self.llm_fn = llm_fn

    def generate(self, query: str, n: int = 4) -> List[str]:

        alternatives = []

        if self.llm_fn:
            try:
                prompt = (
                    f"Original query: {query}\n\n"
                    f"Generate {n} alternative phrasings. "
                    f"One per line, no numbering."
                )
                response = self.llm_fn(prompt)
                lines = [l.strip() for l in response.split('\n') if l.strip()]

                for line in lines[:n]:
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if len(line) > 10 and line.lower() != query.lower():
                        alternatives.append(line)
            except Exception as e:
                logger.warning(f"[MultiQuery] LLM generation failed: {e}. Using rule-based.")

        if len(alternatives) < 2:
            alternatives = self._rule_based_expand(query)

        result = [query] + alternatives[:n]
        logger.info(f"[MultiQuery] Generated {len(result)} query variants for: '{query[:50]}'")
        return result

    def _rule_based_expand(self, query: str) -> List[str]:
        """Rule-based query expansion when LLM is unavailable."""
        expansions = []
        q_lower = query.lower()


        if not q_lower.startswith(("what is", "explain", "describe")):
            expansions.append(f"explain {query}")


        words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', query) if w.lower() not in
                 {'what', 'how', 'when', 'where', 'which', 'does', 'can', 'will', 'should'}]
        if words:
            expansions.append(" ".join(words[:5]))

        if "?" in query:
            clean = query.replace("?", "").strip()
            expansions.append(f"definition of {clean}")


        expansions.append(f"overview of {query}")

        return expansions[:4]




class HyDE:


    HYDE_PROMPT = """Write a short passage (2-4 sentences) that would be
the ideal answer to the following question. Write as if this appears
in a document, not as a direct answer.

Question: {query}

Passage:"""

    def __init__(self, llm_fn: Optional[Callable] = None) -> None:
        self.llm_fn = llm_fn

    def generate_hypothesis(self, query: str) -> str:

        if not self.llm_fn:
            logger.debug("[HyDE] No LLM function, returning original query")
            return query

        try:
            prompt    = self.HYDE_PROMPT.format(query=query)
            hypothesis = self.llm_fn(prompt)
            hypothesis = hypothesis.strip()

            if len(hypothesis) < 20:
                return query

            logger.info(
                f"[HyDE] Generated hypothesis ({len(hypothesis)} chars) "
                f"for: '{query[:50]}'"
            )
            return hypothesis

        except Exception as e:
            logger.warning(f"[HyDE] Generation failed: {e}. Using original query.")
            return query




class QueryExpander:

    DOMAIN_SYNONYMS = {
        "rag":            ["retrieval augmented generation", "document qa", "knowledge retrieval"],
        "llm":            ["large language model", "language model", "gpt", "transformer"],
        "embedding":      ["vector", "dense representation", "semantic vector"],
        "chunk":          ["segment", "passage", "text fragment", "window"],
        "retrieval":      ["search", "lookup", "fetch", "query"],
        "hallucination":  ["fabrication", "confabulation", "made-up", "unsupported"],
        "accuracy":       ["precision", "correctness", "quality", "performance"],
        "agent":          ["agentic", "autonomous", "assistant", "bot"],
        "document":       ["file", "text", "corpus", "knowledge base"],
        "summary":        ["overview", "synopsis", "abstract", "condensed"],
        "cost":           ["price", "expense", "token usage", "billing"],
        "latency":        ["speed", "response time", "delay", "performance"],
    }

    def __init__(self, llm_fn: Optional[Callable] = None) -> None:
        self.llm_fn = llm_fn

    def expand(self, query: str, max_terms: int = 5) -> str:
        """
        Expand query with related terms.

        Returns:
            Expanded query string.
        """
        added_terms = set()

        q_lower = query.lower()
        for key, synonyms in self.DOMAIN_SYNONYMS.items():
            if key in q_lower:
                for syn in synonyms[:2]:
                    if syn not in q_lower:
                        added_terms.add(syn)
                if len(added_terms) >= max_terms:
                    break

        if not added_terms and self.llm_fn:

            try:
                prompt = (
                    f"List 3-5 related terms or synonyms for this search query: '{query}'\n"
                    f"Output only comma-separated terms, no explanation."
                )
                response = self.llm_fn(prompt)
                terms = [t.strip() for t in response.split(',')]
                added_terms.update(terms[:max_terms])
            except Exception as e:
                logger.debug(f"[QueryExpander] LLM expansion failed: {e}")

        if added_terms:
            expansion = " ".join(list(added_terms)[:max_terms])
            expanded = f"{query} {expansion}"
            logger.info(f"[QueryExpander] Expanded: '{query[:40]}' + {list(added_terms)[:3]}")
            return expanded

        return query



class QueryTransformer:


    def __init__(self, llm_fn: Optional[Callable] = None) -> None:
        self.llm_fn    = llm_fn
        self._mq       = MultiQueryGenerator(llm_fn)
        self._hyde     = HyDE(llm_fn)
        self._expander = QueryExpander(llm_fn)

    def transform(
        self,
        query:    str,
        strategy: str = "multi_query",
        n:        int = 3,
    ) -> List[str]:

        if strategy == "multi_query":
            return self._mq.generate(query, n=n)

        elif strategy == "hyde":
            hypothesis = self._hyde.generate_hypothesis(query)
            return [hypothesis] if hypothesis != query else [query]

        elif strategy == "expansion":
            expanded = self._expander.expand(query)
            return [expanded]

        elif strategy == "combined":
            variants = self._mq.generate(query, n=n)
            expanded = [self._expander.expand(v) for v in variants]

            seen = set()
            result = []
            for q in expanded:
                if q not in seen:
                    seen.add(q)
                    result.append(q)
            return result

        else:
            logger.warning(f"[QueryTransformer] Unknown strategy '{strategy}', using original")
            return [query]

    def deduplicate_results(self, all_chunks: list) -> list:

        seen: dict = {}
        for chunk in all_chunks:
            cid = chunk.chunk_id
            if cid not in seen or chunk.similarity_score > seen[cid].similarity_score:
                seen[cid] = chunk
      
        return sorted(seen.values(), key=lambda c: c.similarity_score, reverse=True)
