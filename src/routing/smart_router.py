
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from src.utils.logger import logger



class RoutingStrategy(str, Enum):
    PRIORITY   = "priority"
    COST_AWARE = "cost_aware"
    LOAD_BALANCE = "load_balance"
    FASTEST    = "fastest"


@dataclass
class ProviderConfig:

    name:             str
    priority:         int
    cost_per_1k_tok:  float
    max_tokens:       int
    models:           List[str]
    default_model:    str


    failure_count:    int   = 0
    success_count:    int   = 0
    last_failure_ts:  float = 0.0
    is_open:          bool  = False
    avg_latency_ms:   float = 500.0
    total_requests:   int   = 0
    total_errors:     int   = 0

    failure_threshold:  int   = 3
    recovery_timeout_s: float = 60.0


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        priority=1,
        cost_per_1k_tok=0.0,
        max_tokens=8192,
        models=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        default_model="llama-3.1-8b-instant",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        priority=2,
        cost_per_1k_tok=0.0,
        max_tokens=8192,
        models=["gemini-2.0-flash", "gemini-1.5-pro"],
        default_model="gemini-2.0-flash",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        priority=3,
        cost_per_1k_tok=0.0,
        max_tokens=4096,
        models=["meta-llama/llama-3.1-8b-instruct:free"],
        default_model="meta-llama/llama-3.1-8b-instruct:free",
    ),
}




class CircuitBreaker:


    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider
        self._lock    = threading.Lock()

    def is_available(self) -> bool:

        with self._lock:
            if not self.provider.is_open:
                return True

            elapsed = time.time() - self.provider.last_failure_ts
            if elapsed >= self.provider.recovery_timeout_s:
                logger.info(
                    f"[CircuitBreaker] {self.provider.name} entering HALF-OPEN "
                    f"after {elapsed:.0f}s"
                )
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.provider.failure_count = 0
            self.provider.is_open       = False
            self.provider.success_count += 1

    def record_failure(self) -> None:
        with self._lock:
            self.provider.failure_count  += 1
            self.provider.total_errors   += 1
            self.provider.last_failure_ts = time.time()
            if self.provider.failure_count >= self.provider.failure_threshold:
                if not self.provider.is_open:
                    logger.warning(
                        f"[CircuitBreaker] {self.provider.name} CIRCUIT OPEN "
                        f"after {self.provider.failure_count} failures"
                    )
                self.provider.is_open = True

    def update_latency(self, latency_ms: float) -> None:
        with self._lock:

            alpha = 0.2
            self.provider.avg_latency_ms = (
                alpha * latency_ms +
                (1 - alpha) * self.provider.avg_latency_ms
            )
            self.provider.total_requests += 1




class SmartLLMRouter:

    def __init__(self) -> None:
        self.providers: Dict[str, ProviderConfig]  = {
            k: v for k, v in PROVIDER_REGISTRY.items()
        }
        self.breakers: Dict[str, CircuitBreaker] = {
            k: CircuitBreaker(v) for k, v in self.providers.items()
        }
        self._rr_index = 0
        self._lock     = threading.Lock()



    def _get_provider_order(self, strategy: RoutingStrategy) -> List[ProviderConfig]:

        available = [
            p for p in self.providers.values()
            if self.breakers[p.name].is_available()
        ]
        if not available:

            available = list(self.providers.values())
            logger.warning("[Router] All circuits open — attempting recovery calls")

        if strategy == RoutingStrategy.PRIORITY:
            return sorted(available, key=lambda p: p.priority)

        elif strategy == RoutingStrategy.COST_AWARE:
            return sorted(available, key=lambda p: p.cost_per_1k_tok)

        elif strategy == RoutingStrategy.FASTEST:
            return sorted(available, key=lambda p: p.avg_latency_ms)

        elif strategy == RoutingStrategy.LOAD_BALANCE:
            with self._lock:
                if not available:
                    return []
                start = self._rr_index % len(available)
                self._rr_index += 1

            return available[start:] + available[:start]

        return sorted(available, key=lambda p: p.priority)



    def _call_groq(
        self,
        messages: list,
        model: str,
        max_tokens: int,
    ) -> Dict:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.1, max_tokens=max_tokens,
        )
        return {
            "answer":      resp.choices[0].message.content.strip(),
            "tokens_used": resp.usage.total_tokens if resp.usage else 0,
            "model":       model,
            "provider":    "groq",
        }

    def _call_gemini(self, messages: list, max_tokens: int) -> Dict:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.1}
        )
        return {
            "answer":      resp.text.strip(),
            "tokens_used": 0,
            "model":       "gemini-2.0-flash",
            "provider":    "gemini",
        }

    def _call_openrouter(self, messages: list, max_tokens: int) -> Dict:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        resp = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=messages,
            max_tokens=max_tokens,
        )
        return {
            "answer":      resp.choices[0].message.content.strip(),
            "tokens_used": 0,
            "model":       "llama-3.1-8b-instruct:free",
            "provider":    "openrouter",
        }

    def _dispatch(
        self,
        provider: ProviderConfig,
        messages: list,
        max_tokens: int,
        model_override: Optional[str] = None,
    ) -> Dict:
        """Dispatch call to a specific provider."""
        model = model_override or provider.default_model
        if provider.name == "groq":
            return self._call_groq(messages, model, max_tokens)
        elif provider.name == "gemini":
            return self._call_gemini(messages, max_tokens)
        elif provider.name == "openrouter":
            return self._call_openrouter(messages, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider.name}")



    def route(
        self,
        messages:       list,
        max_tokens:     int = 1024,
        strategy:       RoutingStrategy = RoutingStrategy.PRIORITY,
        model_override: Optional[str] = None,
    ) -> Dict:

        ordered = self._get_provider_order(strategy)
        if not ordered:
            raise RuntimeError("No LLM providers configured")

        last_error = None
        for provider in ordered:
            breaker = self.breakers[provider.name]
            start   = time.perf_counter()

            try:
                logger.info(
                    f"[Router] Trying {provider.name} "
                    f"(strategy={strategy}, cost={provider.cost_per_1k_tok}/1k, "
                    f"avg_lat={provider.avg_latency_ms:.0f}ms)"
                )
                result = self._dispatch(provider, messages, max_tokens, model_override)

                elapsed_ms = (time.perf_counter() - start) * 1000
                breaker.record_success()
                breaker.update_latency(elapsed_ms)

                result["routing_strategy"] = strategy.value
                result["latency_ms"]       = round(elapsed_ms, 1)
                result["cost_estimate"]    = (
                    result.get("tokens_used", 0) / 1000 * provider.cost_per_1k_tok
                )

                logger.info(
                    f"[Router] Success: {provider.name} in {elapsed_ms:.0f}ms "
                    f"tokens={result.get('tokens_used', 0)}"
                )
                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                err_str    = str(e)
                breaker.record_failure()

                if "429" in err_str or "rate_limit" in err_str.lower():
                    logger.warning(f"[Router] {provider.name} rate limited")
                elif "401" in err_str:
                    logger.warning(f"[Router] {provider.name} auth failed — skipping")
                else:
                    logger.warning(f"[Router] {provider.name} error: {e}")

                last_error = e
                continue

        raise RuntimeError(
            f"All providers exhausted (strategy={strategy}). "
            f"Last error: {last_error}"
        )



    def get_status(self) -> Dict:

        return {
            name: {
                "status":         "OPEN" if p.is_open else "CLOSED",
                "failure_count":  p.failure_count,
                "success_count":  p.success_count,
                "total_requests": p.total_requests,
                "total_errors":   p.total_errors,
                "error_rate":     round(p.total_errors / max(p.total_requests, 1) * 100, 1),
                "avg_latency_ms": round(p.avg_latency_ms, 1),
                "cost_per_1k":    p.cost_per_1k_tok,
                "priority":       p.priority,
            }
            for name, p in self.providers.items()
        }

    def reset_circuit(self, provider_name: str) -> None:
        """Manually reset a provider's circuit breaker (admin action)."""
        if provider_name in self.providers:
            p = self.providers[provider_name]
            p.is_open        = False
            p.failure_count  = 0
            p.last_failure_ts = 0.0
            logger.info(f"[Router] Circuit reset for {provider_name}")

    def reset_all_circuits(self) -> None:
        for name in self.providers:
            self.reset_circuit(name)




_router: Optional[SmartLLMRouter] = None

def get_smart_router() -> SmartLLMRouter:
    global _router
    if _router is None:
        _router = SmartLLMRouter()
    return _router
