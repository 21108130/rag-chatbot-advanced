
from __future__ import annotations

from typing import Dict, Generator, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from src.utils.logger import logger
from src.utils.models import ConversationHistory, MessageRole


SYSTEM_PROMPT = """You are an intelligent assistant that answers questions \
based strictly on the provided document context.

Guidelines:
- Answer ONLY using information from the context below.
- If the context does not contain enough information, say so clearly.
- Be concise, accurate, and helpful.
- Cite the source reference (e.g. "[Source 1]") when referencing specific facts.
- Do not fabricate information or reference external knowledge not in the context.
"""

RAG_CONTEXT_TEMPLATE = """{system_prompt}

─── Retrieved Context ────────────────────────────────────────
{context}
─────────────────────────────────────────────────────────────

Answer the user's question based on the context above.
If you cannot find the answer in the context, say: \
"I could not find relevant information in the uploaded documents."
"""

NO_CONTEXT_PROMPT = """You are a helpful AI assistant.
No documents have been uploaded yet. Let the user know they should upload
documents first so you can answer questions based on them.
"""


class GroqStreamingClient:


    def __init__(
        self,
        api_key:     Optional[str]   = None,
        model:       Optional[str]   = None,
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
    ) -> None:
        self.api_key     = api_key     or settings.groq_api_key
        self.model       = model       or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens  = max_tokens  or settings.llm_max_tokens
        self._client     = None

    @property
    def client(self):
        if self._client is None:
            import os

            api_key = (
                self.api_key
                or os.environ.get("GROQ_API_KEY", "")
                or settings.groq_api_key
            )
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. "
                    "Add GROQ_API_KEY=your_key to your .env file."
                )
            self.api_key = api_key  # cache it
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"[LLM] Groq streaming client ready (model={self.model})")
        return self._client



    def _build_messages(
        self,
        query:   str,
        context: str,
        history: Optional[ConversationHistory],
    ) -> List[Dict[str, str]]:
        system_content = (
            RAG_CONTEXT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, context=context)
            if context.strip()
            else NO_CONTEXT_PROMPT
        )
        messages = [{"role": "system", "content": system_content}]

        if history:
            for msg in history.to_llm_messages(last_n=8):
                if msg["role"] != MessageRole.SYSTEM.value:
                    messages.append(msg)

        messages.append({"role": "user", "content": query})
        return messages

    #

    def stream(
        self,
        query:   str,
        context: str = "",
        history: Optional[ConversationHistory] = None,
    ) -> Generator[str, None, None]:

        messages = self._build_messages(query, context, history)

        stream = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            stream      = True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content



    @retry(
        retry   = retry_if_exception_type(Exception),
        stop    = stop_after_attempt(3),
        wait    = wait_exponential(multiplier=1, min=1, max=10),
        reraise = True,
    )
    def generate(
        self,
        query:   str,
        context: str = "",
        history: Optional[ConversationHistory] = None,
    ) -> Dict:
     
        messages = self._build_messages(query, context, history)

        completion = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
        )

        answer      = completion.choices[0].message.content.strip()
        tokens_used = completion.usage.total_tokens if completion.usage else None

        return {"answer": answer, "tokens_used": tokens_used, "model": self.model}

    def health_check(self) -> bool:
        try:
            result = self.generate(query="Reply with the single word: OK", context="")
            return "ok" in result["answer"].lower()
        except Exception as exc:
            logger.warning(f"[LLM] Health check failed: {exc}")
            return False