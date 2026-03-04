
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import logger



@dataclass
class Tool:
    name:        str
    description: str
    func:        Callable[..., str]
    parameters:  Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "description": self.description,
            "parameters":  self.parameters,
        }




def web_search(query: str, max_results: int = 3) -> str:
    """Search the web — tries DuckDuckGo with multiple fallback strategies."""

    try:
        from duckduckgo_search import DDGS
        import random, time
        time.sleep(random.uniform(0.5, 1.5))
        with DDGS(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}) as ddgs:
            results = list(ddgs.text(query, max_results=max_results * 2, region="wt-wt"))

        english = []
        for r in results:
            text = (r.get("title") or "") + (r.get("body") or "")
            ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
            if ascii_ratio > 0.7:
                english.append(r)
        final = (english or results)[:max_results]
        if final:
            lines = [f"[Web {i}] {r.get('title','')}\n{r.get('body','')}" for i, r in enumerate(final, 1)]
            return "\n\n".join(lines)
    except Exception:
        pass


    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "wt-wt"},
            headers=headers,
            timeout=10
        )
        import re
        snippets = re.findall(r'class="result__snippet">(.*?)</a>', resp.text)
        titles   = re.findall(r'class="result__title">.*?<a[^>]*>(.*?)</a>', resp.text)
        if snippets:
            lines = []
            for i, (t, s) in enumerate(zip(titles[:max_results], snippets[:max_results]), 1):
                clean_s = re.sub(r'<[^>]+>', '', s).strip()
                clean_t = re.sub(r'<[^>]+>', '', t).strip()
                lines.append(f"[Web {i}] {clean_t}\n{clean_s}")
            return "\n\n".join(lines)
    except Exception:
        pass

    return "Web search temporarily unavailable. Using knowledge base only."



class RAGAgent:


    MAX_ITERATIONS = 4

    AGENT_SYSTEM_PROMPT = """You are an expert research assistant. Your job is to answer questions using the knowledge base documents as your PRIMARY source.

Available Tools:
{tool_descriptions}

Knowledge Base Status: {kb_status}

To use a tool, respond with:
<tool_call>
{{"tool": "tool_name", "args": {{"param": "value"}}}}
</tool_call>

When you have enough information to answer, respond with:
<final_answer>
Your complete answer here.
</final_answer>

STRICT RULES — follow these exactly:
1. ALWAYS call kb_search first. The knowledge base is your most important source.
2. When kb_search returns results, you MUST base your answer primarily on those results.
3. Quote or paraphrase the KB content directly — do not replace it with vague general knowledge.
4. Use web_search ONLY to add extra real-world context AFTER using KB content.
5. In your final answer, cite KB results as [KB Source 1], [KB Source 2] etc.
6. Your final answer must be detailed and well-structured — use headings if helpful.
7. NEVER ignore KB results in favour of generic knowledge. KB content = ground truth.
8. If kb_search found something, your answer must include that specific content.
"""

    def __init__(
        self,
        pipeline,
        user_id:  str = "anonymous",
        model:    Optional[str] = None,
        has_docs: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.user_id  = user_id

        from config.settings import settings as _s
        self.model    = model or getattr(_s, "llm_agent_model", "llama-3.3-70b-versatile")
        self.has_docs = has_docs


        self.tools: Dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self) -> None:

        def kb_search(query: str, top_k: int = 5) -> str:

            retriever = self.pipeline._get_retriever(self.user_id)
            result = retriever.retrieve(query=query, top_k=int(top_k))
            if not result.chunks:
                return "No relevant documents found in the knowledge base."
            parts = []
            for i, chunk in enumerate(result.chunks, 1):
                source   = chunk.metadata.get("source", chunk.doc_id)
                orig_sim = chunk.metadata.get("original_similarity", chunk.similarity_score)
                parts.append(f"[KB Source {i}: {source} | relevance={orig_sim:.2f}]\n{chunk.content[:1200]}")
            return "\n\n".join(parts)

        self.tools["kb_search"] = Tool(
            name        = "kb_search",
            description = "Search the user's uploaded knowledge base documents for relevant information.",
            func        = kb_search,
            parameters  = {
                "query": "The search query",
                "top_k": "Number of results (default: 5)",
            },
        )


        self.tools["web_search"] = Tool(
            name        = "web_search",
            description = "Search the web for current information, news, and facts not in documents.",
            func        = web_search,
            parameters  = {
                "query":       "The search query",
                "max_results": "Number of results (default: 3)",
            },
        )


        def summarize_kb(topic: str) -> str:
            return kb_search(f"summarize {topic}", top_k=8)

        self.tools["summarize"] = Tool(
            name        = "summarize",
            description = "Get a comprehensive summary of a topic from the knowledge base.",
            func        = summarize_kb,
            parameters  = {"topic": "The topic to summarize"},
        )

    def _build_tool_descriptions(self) -> str:
        lines = []
        for tool in self.tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def _build_kb_status(self) -> str:
        try:
            stats = self.pipeline.get_kb_stats(user_id=self.user_id)
            total = stats.get("total_chunks", 0)
            docs  = stats.get("total_docs", 0)
            if total > 0:
                return f"AVAILABLE — {docs} document(s), {total} chunks indexed. Use kb_search to find information."
            return "EMPTY — No documents uploaded yet. Use web_search only."
        except Exception:
            return "UNKNOWN — Try kb_search anyway."


    def _call_groq(self, messages: List[Dict]) -> str:

        from src.llm.multi_provider_client import get_llm_client
        from config.settings import settings as cfg

        client = get_llm_client()


        last_user_msg = ""
        system_msg = ""
        history_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] == "user":
                last_user_msg = m["content"]
            else:
                history_msgs.append(m)


        providers_tried = []
        last_error = None


        agent_model = self.model or getattr(cfg, "llm_agent_model", cfg.llm_model)
        try:
            from groq import Groq
            import os
            api_key = os.environ.get("GROQ_API_KEY", "") or cfg.groq_api_key
            groq_client = Groq(api_key=api_key)
            resp = groq_client.chat.completions.create(
                model       = agent_model,
                messages    = messages,
                temperature = 0.1,
                max_tokens  = 1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "decommissioned" in err.lower():
                logger.warning(f"[Agent] Groq unavailable ({err[:80]}), trying fallback providers...")
            else:
                raise


        result = client.generate(
            query      = last_user_msg,
            context    = system_msg,
            max_tokens = 1500,
        )
        logger.info(f"[Agent] Used fallback provider: {result.get('provider', 'unknown')}")
        return result["answer"]

    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """Extract tool call from model response."""

        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass


        match2 = re.search(r'\{"tool"\s*:\s*"(\w+)".*?"args"\s*:\s*(\{.*?\})\s*\}', response, re.DOTALL)
        if match2:
            try:
                full = re.search(r'\{"tool".*?"args".*?\}.*?\}', response, re.DOTALL)
                if full:
                    return json.loads(full.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """Extract final answer from model response."""
        match = re.search(r"<final_answer>(.*?)</final_answer>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        if "<tool_call>" not in response:
            return response

        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if len(cleaned) > 100:
            return cleaned
        return None

    def run(self, query: str) -> Dict[str, Any]:

        logger.info(f"[Agent] Starting agent run: '{query[:60]}'")

        system_prompt = self.AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self._build_tool_descriptions(),
            kb_status=self._build_kb_status(),
        )

        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": f"Question: {query}\n\nPlease use the available tools to research this thoroughly. Search the knowledge base with multiple relevant queries to gather all available information. Then write a detailed, well-structured answer that covers the topic comprehensively."},
        ]

        steps      = []
        tools_used = []

        kb_search_count = 0

        for iteration in range(self.MAX_ITERATIONS):
            logger.debug(f"[Agent] Iteration {iteration + 1}")

            if kb_search_count >= 3:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have now searched the knowledge base multiple times and gathered enough information. "
                        "STOP searching and write your FINAL comprehensive answer now using ALL the KB content you found. "
                        "Structure it with clear headings. Do not call any more tools."
                    )
                })
                kb_search_count = -999

            response = self._call_groq(messages)
            messages.append({"role": "assistant", "content": response})


            has_substantial_text = len(re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL).strip()) > 200
            final = self._extract_final_answer(response)
            if final and ("<tool_call>" not in response or has_substantial_text):
                logger.info(f"[Agent] Final answer reached after {iteration + 1} iterations.")
                return {
                    "answer":     final,
                    "steps":      steps,
                    "tools_used": list(set(tools_used)),
                    "iterations": iteration + 1,
                }


            tool_call = self._parse_tool_call(response)
            if not tool_call:

                return {
                    "answer":     response,
                    "steps":      steps,
                    "tools_used": list(set(tools_used)),
                    "iterations": iteration + 1,
                }

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            if tool_name not in self.tools:
                observation = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    tool = self.tools[tool_name]
                    observation = tool.func(**tool_args)
                    tools_used.append(tool_name)
                    if tool_name == "kb_search":
                        kb_search_count += 1
                    logger.info(f"[Agent] Used tool '{tool_name}' → {len(observation)} chars")
                except Exception as exc:
                    observation = f"Tool '{tool_name}' failed: {exc}"

            step = {
                "iteration":   iteration + 1,
                "tool":        tool_name,
                "args":        tool_args,
                "observation": observation[:300] + "…" if len(observation) > 300 else observation,
            }
            steps.append(step)

            
            messages.append({
                "role":    "user",
                "content": f"<tool_result>\n{observation}\n</tool_result>\n\nContinue your reasoning.",
            })


        logger.warning("[Agent] Max iterations reached.")
        return {
            "answer":     "I reached the maximum reasoning steps. Here is what I found so far:\n\n" + (steps[-1]["observation"] if steps else "No results."),
            "steps":      steps,
            "tools_used": list(set(tools_used)),
            "iterations": self.MAX_ITERATIONS,
        }