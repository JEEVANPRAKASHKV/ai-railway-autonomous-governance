"""
core/llm_router.py
==================
Tri-mode LLM router — Golden Rule 2: Zero-Crash Guarantee.

Transparently routes LLM calls to:
  - PRIMARY:   Google Gemini via the REST API (when GEMINI_API_KEY is configured). A model chain is tried in
               order (configured model -> current Flash-Lite -> current Flash), so a retired or rate-limited
               model never takes the platform down.
  - SECONDARY: Azure OpenAI (optional, only when Azure credentials are configured)
  - FALLBACK:  Intelligent rule-based FallbackEngine (always available, 0 external deps)

Usage:
    from core.llm_router import llm_router
    response = llm_router.complete(prompt="...", system_prompt="...")
    print(response.content)   # Works in ALL modes
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.config import settings
from core.schemas import LLMResponse
from core.fallback_engine import FallbackEngine

logger = logging.getLogger(__name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Model families Google has retired — configuring one of these must not cost a failed call on every request.
GEMINI_RETIRED_PREFIXES = ("gemini-1.0", "gemini-1.5", "gemini-pro", "gemini-2.0")
GEMINI_DEFAULT_CHAIN = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash")
GEMINI_TIMEOUT_S = 12.0


class LLMRouter:
    """
    Tri-mode LLM router.

    Automatically selects Google Gemini, Azure OpenAI, or rule-based fallback
    based on environment configuration. All callers use the same interface
    regardless of which mode is active.
    """

    def __init__(self) -> None:
        self._mode = settings.llm_mode
        self._fallback = FallbackEngine()
        self._client: Optional[Any] = None
        self._gemini_client: Optional[Any] = None
        self._gemini_chain: List[str] = []
        self._cooldown: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "requests": 0, "gemini_ok": 0, "fallback_used": 0,
            "last_model": None, "last_error": None, "last_latency_ms": None,
        }

        if self._mode == "gemini":
            self._init_gemini_client()
        elif self._mode == "azure_openai":
            self._init_azure_client()
        else:
            logger.info(
                "[LLMRouter] External LLM credentials not found — "
                "activating intelligent rule-based fallback mode. "
                "All features remain fully functional."
            )

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_gemini_client(self) -> None:
        """Gemini is called over plain HTTPS (no SDK to break); this just builds the model chain."""
        chain: List[str] = []
        for m in [settings.gemini_model.strip(), *GEMINI_DEFAULT_CHAIN]:
            m = m.replace("models/", "")
            if m and m not in chain and not m.startswith(GEMINI_RETIRED_PREFIXES):
                chain.append(m)
        self._gemini_chain = chain
        self._gemini_client = True  # truthy flag: REST client is ready
        self._cooldown: Dict[str, float] = {}
        logger.info(f"[LLMRouter] Gemini ready. Model chain: {' -> '.join(chain)}")

    def _init_azure_client(self) -> None:
        try:
            from langchain_openai import AzureChatOpenAI
            self._client = AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_key,
                azure_deployment=settings.azure_openai_deployment_name,
                api_version=settings.azure_openai_api_version,
                temperature=0.3,
                request_timeout=15,
                max_retries=2,
            )
            logger.info("[LLMRouter] Azure OpenAI client initialized successfully.")
        except Exception as e:
            logger.warning(f"[LLMRouter] Azure OpenAI init failed: {e}. Switching to fallback.")
            self._mode = "fallback"
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_gemini(self) -> bool:
        return self._mode == "gemini"

    @property
    def is_azure(self) -> bool:
        return self._mode == "azure_openai"

    def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful Indian Railways AI assistant.",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Main completion method. Routes to Gemini, Azure OpenAI, or fallback.
        Never raises — always returns an LLMResponse.
        """
        if self._mode == "gemini" and self._gemini_client is not None:
            return self._call_gemini(prompt, system_prompt, temperature, max_tokens, **kwargs)
        if self._mode == "azure_openai" and self._client is not None:
            return self._call_azure(prompt, system_prompt, temperature, max_tokens, **kwargs)
        return self._call_fallback(prompt, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Multi-turn chat completion. Accepts messages list.
        """
        if self._mode == "gemini" and self._gemini_client is not None:
            return self._call_gemini_chat(messages, temperature, max_tokens)
        if self._mode == "azure_openai" and self._client is not None:
            return self._call_azure_chat(messages, temperature, max_tokens)
        # Fallback: combine messages into a single prompt
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self._call_fallback(prompt)

    # ── Google Gemini (REST) ──────────────────────────────────────────────────

    def _gemini_generate(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, str, int]:
        """Try each model in the chain; return (text, model, tokens). Raises RuntimeError if all fail."""
        errors: List[str] = []
        now = time.time()
        for model in self._gemini_chain:
            if self._cooldown.get(model, 0) > now:
                continue
            for with_thinking_off in ((True, False) if not model.endswith("lite") else (False,)):
                gen_cfg: Dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max(max_tokens, 512)}
                if with_thinking_off:
                    gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}  # keep latency low, keep the budget for the answer
                body: Dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
                if system_prompt:
                    body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                try:
                    r = requests.post(
                        GEMINI_ENDPOINT.format(model=model),
                        headers={"x-goog-api-key": settings.gemini_api_key.strip(), "content-type": "application/json"},
                        json=body,
                        timeout=GEMINI_TIMEOUT_S,
                    )
                except requests.RequestException as e:
                    errors.append(f"{model}: network {type(e).__name__}")
                    self._cooldown[model] = time.time() + 15
                    break
                if r.status_code == 200:
                    data = r.json()
                    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                    if text:
                        tokens = int((data.get("usageMetadata") or {}).get("totalTokenCount", 0) or 0)
                        return text, model, tokens
                    errors.append(f"{model}: empty response")
                    break
                msg = ""
                try:
                    msg = r.json().get("error", {}).get("message", "")[:160]
                except Exception:
                    pass
                if r.status_code == 400 and with_thinking_off:
                    continue  # this model rejects the thinking override — retry without it
                errors.append(f"{model}: HTTP {r.status_code} {msg}".strip())
                if r.status_code == 404:
                    self._cooldown[model] = time.time() + 3600      # retired / unavailable
                elif r.status_code == 429:
                    self._cooldown[model] = time.time() + 60        # rate limited (free tier) — rotate model
                elif r.status_code in (401, 403):
                    for m in self._gemini_chain:
                        self._cooldown[m] = time.time() + 300       # bad key: don't hammer
                else:
                    self._cooldown[model] = time.time() + 20
                break
        raise RuntimeError("; ".join(errors) or "all Gemini models cooling down")

    def _record(self, ok: bool, model: Optional[str] = None, error: Optional[str] = None, latency_ms: Optional[float] = None) -> None:
        with self._lock:
            self._stats["requests"] += 1
            if ok:
                self._stats["gemini_ok"] += 1
                self._stats["last_model"] = model
                self._stats["last_error"] = None
            else:
                self._stats["fallback_used"] += 1
                if not (error or "").startswith("all Gemini models cooling") or not self._stats["last_error"]:
                    self._stats["last_error"] = error
            if latency_ms is not None:
                self._stats["last_latency_ms"] = round(latency_ms, 1)

    def status(self) -> Dict[str, Any]:
        """Operational view of the LLM layer (used by /health and the UI)."""
        with self._lock:
            out = dict(self._stats)
        out.update({"mode": self._mode, "model_chain": list(self._gemini_chain),
                    "cooling_down": [m for m, t in self._cooldown.items() if t > time.time()]})
        return out

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            text, model, tokens = self._gemini_generate(
                [{"role": "user", "parts": [{"text": prompt}]}], system_prompt, temperature, max_tokens
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self._record(True, model, latency_ms=elapsed)
            return LLMResponse(content=text, mode="gemini", model=model, tokens_used=tokens, latency_ms=elapsed)
        except Exception as e:
            logger.warning(f"[LLMRouter] Gemini unavailable ({e}). Using rule-based fallback for this request.")
            self._record(False, error=str(e)[:300])
            return self._call_fallback(prompt, error=f"Gemini: {e}", **kwargs)

    def _call_gemini_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        system_prompt = None
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                system_prompt = content
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]})
        try:
            text, model, tokens = self._gemini_generate(contents, system_prompt, temperature, max_tokens)
            elapsed = (time.perf_counter() - t0) * 1000
            self._record(True, model, latency_ms=elapsed)
            return LLMResponse(content=text, mode="gemini", model=model, tokens_used=tokens, latency_ms=elapsed)
        except Exception as e:
            logger.warning(f"[LLMRouter] Gemini chat unavailable ({e}). Using fallback.")
            self._record(False, error=str(e)[:300])
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            return self._call_fallback(prompt, error=f"Gemini: {e}")

    # ── Azure OpenAI Call ─────────────────────────────────────────────────────

    def _call_azure(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
            response = self._client.invoke(messages)
            elapsed = (time.perf_counter() - t0) * 1000
            content = response.content if hasattr(response, "content") else str(response)
            tokens = getattr(response, "usage_metadata", {})
            total_tokens = tokens.get("total_tokens", 0) if isinstance(tokens, dict) else 0
            return LLMResponse(
                content=content,
                mode="azure_openai",
                model=settings.azure_openai_deployment_name,
                tokens_used=total_tokens,
                latency_ms=elapsed,
            )
        except Exception as e:
            logger.error(f"[LLMRouter] Azure OpenAI call failed: {e}. Switching to fallback for this request.")
            return self._call_fallback(prompt, error=str(e))

    def _call_azure_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            lc_messages = []
            for m in messages:
                role, content = m["role"], m["content"]
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
            response = self._client.invoke(lc_messages)
            elapsed = (time.perf_counter() - t0) * 1000
            content = response.content if hasattr(response, "content") else str(response)
            return LLMResponse(content=content, mode="azure_openai", latency_ms=elapsed)
        except Exception as e:
            logger.error(f"[LLMRouter] Azure chat failed: {e}")
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            return self._call_fallback(prompt, error=str(e))

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _call_fallback(self, prompt: str, error: Optional[str] = None, **kwargs: Any) -> LLMResponse:
        t0 = time.perf_counter()
        prompt_lower = prompt.lower()

        # Route to appropriate fallback handler based on prompt context & task hints
        task = kwargs.get("task", "")
        if task == "support":
            response = self._fallback.support_query(prompt)
            content = response.content
        elif task == "planner":
            result = self._fallback.plan_travel(prompt)
            content = self._format_itinerary(result)
        elif task == "rca":
            result = self._fallback.analyze_rca(kwargs.get("alert_type", "default"), prompt)
            content = self._format_rca(result)
        elif any(kw in prompt_lower for kw in ["refund", "cancel", "policy", "rules", "rule", "pnr", "tatkal", "berth", "chart", "how to", "what is", "can i", "help", "support", "faq"]):
            response = self._fallback.support_query(prompt)
            content = response.content
        elif any(kw in prompt_lower for kw in ["travel plan", "itinerary", "find train", "trains from", "going from", "route", "book train"]):
            result = self._fallback.plan_travel(prompt)
            content = self._format_itinerary(result)
        elif any(kw in prompt_lower for kw in ["root cause", "rca", "degraded", "alert", "failure"]):
            result = self._fallback.analyze_rca("default", prompt)
            content = self._format_rca(result)
        elif any(kw in prompt_lower for kw in ["summary", "governance", "status", "health"]):
            content = "**Governance Summary (Rule-Based Mode)**\nAll models are operating within normal parameters. Telemetry is being collected continuously."
        else:
            response = self._fallback.support_query(prompt)
            content = response.content

        if error:
            logger.info(f"[LLMRouter] fallback engaged: {error}")  # internal detail stays in logs, not in the passenger's answer

        elapsed = (time.perf_counter() - t0) * 1000
        return LLMResponse(content=content, mode="fallback", latency_ms=elapsed, error=error)

    def _format_itinerary(self, result: Dict[str, Any]) -> str:
        lines = [f"**Travel Plan (Rule-Based Mode)**\n\nExtracted Constraints: {result.get('extracted_constraints', {})}\n"]
        for i, itin in enumerate(result.get("itineraries", []), 1):
            lines.append(
                f"**Option {i}: {itin.get('train_name', 'Train')} ({itin.get('train_number', '')})**\n"
                f"- Route: {itin.get('source', '?')} → {itin.get('destination', '?')}\n"
                f"- Departure: {itin.get('departure', '?')} | Arrival: {itin.get('arrival', '?')}\n"
                f"- Sleeper Fare: ₹{itin.get('base_fare_sl', 'N/A')} | 3A Fare: ₹{itin.get('base_fare_3a', 'N/A')}\n"
                f"- Punctuality: {itin.get('punctuality_pct', '?')}% | WL Confirm Prob: {itin.get('wl_confirmation_prob', 0):.0%}\n"
            )
        return "\n".join(lines)

    def _format_rca(self, result: Dict[str, Any]) -> str:
        remediations = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(result.get("remediations", [])))
        return (
            f"**Root Cause Analysis — {result.get('module', 'Unknown')}**\n\n"
            f"**Identified Cause:** {result.get('root_cause_description', 'Unknown')}\n"
            f"**Confidence:** {result.get('confidence', 0):.0%}\n\n"
            f"**Recommended Remediations:**\n{remediations}"
        )

    # ── Convenience methods for specific use-cases ────────────────────────────

    def plan_travel(self, query: str) -> Dict[str, Any]:
        """Direct travel planning call."""
        return self._fallback.plan_travel(query)

    def support_query(self, query: str, language: str = "en") -> LLMResponse:
        """Direct support query call."""
        if self.is_gemini or self.is_azure:
            system = (
                "You are RailSeva AI, the official intelligent customer support assistant for Indian Railways. "
                "Answer questions about PNR status, ticket booking, refunds, cancellations, "
                "Tatkal booking, waitlist/RAC, and railway policies. "
                "Be concise, accurate, empathetic, and helpful. If uncertain, say so."
            )
            return self.complete(query, system_prompt=system, task="support")
        return self._fallback.support_query(query, language)

    def analyze_rca(self, module: str, alert: str) -> Dict[str, Any]:
        """Direct RCA call."""
        if self.is_gemini or self.is_azure:
            prompt = f"Perform root cause analysis for module '{module}' with alert: {alert}. Explain likely root cause and actionable remediations."
            response = self.complete(prompt, task="rca")
            return {"module": module, "analysis": response.content, "mode": self._mode}
        return self._fallback.analyze_rca(module, alert)

    def supervisor_tick(self, metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Always use rule-based supervisor (deterministic + fast)."""
        return self._fallback.supervisor_tick(metrics)


# Module-level singleton
llm_router = LLMRouter()
