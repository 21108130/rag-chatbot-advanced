
from __future__ import annotations

import os
import time
from typing import Dict, Generator, List, Optional

from config.settings import settings
from src.utils.logger import logger
from src.utils.models import ConversationHistory, MessageRole



SYSTEM_PROMPT = """You are an expert research assistant. Your job is to answer questions using the knowledge base documents as your PRIMARY source.

Guidelines:
- Base your answers on the provided document context.
- Be detailed, structured, and comprehensive.
- Cite sources as [Source 1], [Source 2] etc.
- If the context lacks information, say so clearly.
"""

RAG_CONTEXT_TEMPLATE = """{system_prompt}

─── Document Context ─────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────

Answer the user's question based on the context above.
"""

NO_CONTEXT_PROMPT = """You are a helpful AI assistant.
No documents have been uploaded yet. Ask the user to upload documents first.
"""


class MultiProviderLLMClient:


    def __init__(
        self,
        groq_api_key:       Optional[str] = None,
        gemini_api_key:     Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ) -> None:

        self._user_groq_key       = groq_api_key       or None
        self._user_gemini_key     = gemini_api_key     or None
        self._user_openrouter_key = openrouter_api_key or None

        self._groq_client       = None
        self._gemini_client     = None
        self._openrouter_client = None
        self._current_provider  = "groq"


    def _resolve_groq_key(self) -> str:
        """User-supplied key takes priority; falls back to .env."""
        return self._user_groq_key or os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key

    def _resolve_gemini_key(self) -> str:
        return self._user_gemini_key or os.environ.get("GEMINI_API_KEY", "")

    def _resolve_openrouter_key(self) -> str:
        return self._user_openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

    def using_user_keys(self) -> bool:

        return bool(self._user_groq_key or self._user_gemini_key or self._user_openrouter_key)

    def update_keys(
        self,
        groq_api_key:       Optional[str] = None,
        gemini_api_key:     Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ) -> None:

        changed = False
        if groq_api_key is not None and groq_api_key != self._user_groq_key:
            self._user_groq_key = groq_api_key or None
            self._groq_client   = None   # force reinit
            changed = True
        if gemini_api_key is not None and gemini_api_key != self._user_gemini_key:
            self._user_gemini_key = gemini_api_key or None
            self._gemini_client   = None
            changed = True
        if openrouter_api_key is not None and openrouter_api_key != self._user_openrouter_key:
            self._user_openrouter_key = openrouter_api_key or None
            self._openrouter_client   = None
            changed = True
        if changed:
            logger.info("[LLM] API keys updated — provider clients reset")



    def _get_groq(self):
        if self._groq_client is None:
            api_key = self._resolve_groq_key()
            if not api_key:
                return None
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                logger.info("[LLM] Groq client ready")
            except Exception as e:
                logger.warning(f"[LLM] Groq init failed: {e}")
                return None
        return self._groq_client

    def _get_gemini(self):
        if self._gemini_client is None:
            api_key = self._resolve_gemini_key()
            if not api_key:
                return None
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._gemini_client = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("[LLM] Gemini client ready")
            except Exception as e:
                logger.warning(f"[LLM] Gemini init failed: {e}")
                return None
        return self._gemini_client

    def _get_openrouter(self):
        if self._openrouter_client is None:
            api_key = self._resolve_openrouter_key()
            if not api_key:
                return None
            try:
                from openai import OpenAI
                self._openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                logger.info("[LLM] OpenRouter client ready")
            except Exception as e:
                logger.warning(f"[LLM] OpenRouter init failed: {e}")
                return None
        return self._openrouter_client


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
            for msg in history.to_llm_messages(last_n=6):
                if msg["role"] != MessageRole.SYSTEM.value:
                    messages.append(msg)
        messages.append({"role": "user", "content": query})
        return messages



    def _generate_groq(self, messages: List[Dict], model: str, max_tokens: int) -> Dict:
        client = self._get_groq()
        if not client:
            raise RuntimeError("Groq not available")
        completion = client.chat.completions.create(
            model       = model,
            messages    = messages,
            temperature = settings.llm_temperature,
            max_tokens  = max_tokens,
        )
        return {
            "answer":      completion.choices[0].message.content.strip(),
            "tokens_used": completion.usage.total_tokens if completion.usage else 0,
            "model":       model,
            "provider":    "groq",
        }

    def _stream_groq(self, messages: List[Dict], model: str, max_tokens: int) -> Generator[str, None, None]:
        client = self._get_groq()
        if not client:
            raise RuntimeError("Groq not available")
        stream = client.chat.completions.create(
            model       = model,
            messages    = messages,
            temperature = settings.llm_temperature,
            max_tokens  = max_tokens,
            stream      = True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content



    def _generate_gemini(self, messages: List[Dict], max_tokens: int) -> Dict:
        client = self._get_gemini()
        if not client:
            raise RuntimeError("Gemini not available")

        prompt = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        response = client.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.1}
        )
        return {
            "answer":      response.text.strip(),
            "tokens_used": 0,
            "model":       "gemini-2.0-flash",
            "provider":    "gemini",
        }



    def _generate_openrouter(self, messages: List[Dict], max_tokens: int) -> Dict:
        client = self._get_openrouter()
        if not client:
            raise RuntimeError("OpenRouter not available")
        completion = client.chat.completions.create(
            model      = "meta-llama/llama-3.1-8b-instruct:free",
            messages   = messages,
            max_tokens = max_tokens,
        )
        return {
            "answer":      completion.choices[0].message.content.strip(),
            "tokens_used": 0,
            "model":       "llama-3.1-8b-instruct:free",
            "provider":    "openrouter",
        }


    def generate(
        self,
        query:      str,
        context:    str = "",
        history:    Optional[ConversationHistory] = None,
        model:      Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:

        messages   = self._build_messages(query, context, history)
        llm_model  = model or getattr(settings, "llm_agent_model", settings.llm_model)
        max_tok    = max_tokens or settings.llm_max_tokens


        providers = [
            ("groq",       lambda: self._generate_groq(messages, llm_model, max_tok)),
            ("gemini",     lambda: self._generate_gemini(messages, max_tok)),
            ("openrouter", lambda: self._generate_openrouter(messages, max_tok)),
        ]

        last_error = None
        for provider_name, fn in providers:
            try:
                result = fn()
                if provider_name != self._current_provider:
                    logger.info(f"[LLM] Switched to provider: {provider_name}")
                    self._current_provider = provider_name
                return result
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower():
                    logger.warning(f"[LLM] {provider_name} rate limited — trying next provider")
                elif "401" in err_str or "invalid" in err_str.lower():
                    logger.warning(f"[LLM] {provider_name} auth failed — trying next provider")
                else:
                    logger.warning(f"[LLM] {provider_name} error: {e} — trying next provider")
                last_error = e
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}\n"
            "Please check your API keys in .env or wait for rate limits to reset."
        )


    def stream(
        self,
        query:      str,
        context:    str = "",
        history:    Optional[ConversationHistory] = None,
        model:      Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:

        messages  = self._build_messages(query, context, history)
        llm_model = model or settings.llm_model
        max_tok   = max_tokens or settings.llm_max_tokens


        try:
            yield from self._stream_groq(messages, llm_model, max_tok)
            return
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                logger.warning("[LLM] Groq rate limited for streaming — falling back")
            else:
                logger.warning(f"[LLM] Groq streaming failed: {e} — falling back")


        try:
            result = self.generate(query, context, history, model, max_tokens)

            words = result["answer"].split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.01)
        except Exception as e:
            yield f"\n\n⚠️ All providers failed: {e}"

    def get_active_provider(self) -> str:
        return self._current_provider



_default_client: Optional[MultiProviderLLMClient] = None


_user_clients: dict = {}


def get_llm_client(
    user_id:            Optional[str] = None,
    groq_api_key:       Optional[str] = None,
    gemini_api_key:     Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
) -> MultiProviderLLMClient:

    global _default_client

    has_user_keys = any([groq_api_key, gemini_api_key, openrouter_api_key])

    if user_id and has_user_keys:
        if user_id not in _user_clients:
            _user_clients[user_id] = MultiProviderLLMClient(
                groq_api_key=groq_api_key,
                gemini_api_key=gemini_api_key,
                openrouter_api_key=openrouter_api_key,
            )
            logger.info(f"[LLM] Created per-user client for user_id={user_id}")
        else:
            _user_clients[user_id].update_keys(
                groq_api_key=groq_api_key,
                gemini_api_key=gemini_api_key,
                openrouter_api_key=openrouter_api_key,
            )
        return _user_clients[user_id]

    if user_id and user_id in _user_clients:
        return _user_clients[user_id]


    if _default_client is None:
        _default_client = MultiProviderLLMClient()
    return _default_client


def clear_user_client(user_id: str) -> None:

    _user_clients.pop(user_id, None)