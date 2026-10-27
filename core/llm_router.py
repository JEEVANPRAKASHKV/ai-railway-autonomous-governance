"""
core/llm_router.py
==================
Multi-provider LLM router — Golden Rule 2: Zero-Crash Guarantee.

Providers are tried in failover order (LLM_PROVIDER_ORDER, default "sarvam,gemini,azure_openai"); only the
ones with credentials take part, and the rule-based FallbackEngine is always the last resort:

  - Sarvam AI    (SARVAM_API_KEY)  — Indian LLM, OpenAI-compatible chat completions; default model sarvam-105b
                                     with reasoning switched off for fast, cheap support answers.
  - Google Gemini (GEMINI_API_KEY) — REST, with its own model chain (configured -> Flash-Lite -> Flash).
  - Azure OpenAI (optional)        — via langchain-openai.
  - Rule-based FallbackEngine      — always available, no external dependency.

Governance features of the router itself:
  - per-provider health: calls, failures, latency, tokens; a failing provider cools down instead of being hammered
  - spend tracking in ₹ from each provider's published price list, and a daily budget cap (LLM_DAILY_BUDGET_INR):
    when the cap is reached, paid providers are skipped until the next day and answers come from the next provider
  - `probe()` checks every configured provider with a tiny request (Control Tower "Test LLM engines")

Usage:
    from core.llm_router import llm_router
    response = llm_router.complete(prompt="...", system_prompt="...")
    print(response.content)   # Works in ALL modes
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.config import settings
from core.schemas import LLMResponse
from core.fallback_engine import FallbackEngine

logger = logging.getLogger(__name__)

# ── Sarvam AI ─────────────────────────────────────────────────────────────────
SARVAM_V1 = "/v1/chat/completions"      # Sarvam's own models (sarvam-105b, sarvam-105b-conversations)
SARVAM_V2 = "/v2/chat/completions"      # open-weight models hosted by Sarvam (gemma4, glm5.3, ...)
SARVAM_TIMEOUT_S = 20.0
# ₹ per 1M tokens (input, output) — Sarvam price list, Oct 2026. Unknown models are costed at the sarvam-105b rate.
SARVAM_PRICES_INR = {
    "sarvam-105b": (15.0, 60.0), "sarvam-105b-conversations": (15.0, 60.0), "gemma4": (35.0, 90.0),
    "deepseekv4.1-flash": (25.0, 105.0), "glm5.3": (126.0, 390.0), "mimo-v2.6-flash": (12.5, 25.0),
}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)

# ── Google Gemini ─────────────────────────────────────────────────────────────
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Model families Google has retired — configuring one of these must not cost a failed call on every request.
GEMINI_RETIRED_PREFIXES = ("gemini-1.0", "gemini-1.5", "gemini-pro", "gemini-2.0")
GEMINI_DEFAULT_CHAIN = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash")
GEMINI_TIMEOUT_S = 12.0

PROVIDER_LABELS = {"sarvam": "Sarvam AI", "gemini": "Gemini", "azure_openai": "Azure OpenAI", "fallback": "Built-in engine"}
PAID_PROVIDERS = ("sarvam", "azure_openai")


class ProviderError(RuntimeError):
    """One provider could not answer; the router moves on to the next one."""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class LLMRouter:
    """Routes every LLM call through the configured providers in order, then the rule-based engine."""

    def __init__(self) -> None:
        self._providers: List[str] = list(settings.llm_providers)
        self._mode = self._providers[0] if self._providers else "fallback"
        self._fallback = FallbackEngine()
        self._client: Optional[Any] = None          # Azure (langchain) client
        self._gemini_client: Optional[Any] = None
        self._gemini_chain: List[str] = []
        self._cooldown: Dict[str, float] = {}       # key: "sarvam" | Gemini model name | "azure_openai"
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "requests": 0, "llm_ok": 0, "fallback_used": 0,
            "last_provider": None, "last_model": None, "last_error": None, "last_latency_ms": None,
        }
        self._per: Dict[str, Dict[str, Any]] = {
            p: {"calls": 0, "ok": 0, "failed": 0, "tokens_in": 0, "tokens_out": 0, "cost_inr": 0.0,
                "latency_ms_total": 0.0, "last_error": None}
            for p in self._providers
        }
        self._spend_day, self._spend_today = _today(), 0.0

        if "gemini" in self._providers:
            self._init_gemini_client()
        if "azure_openai" in self._providers:
            self._init_azure_client()
        if self._providers:
            logger.info(f"[LLMRouter] Provider order: {' -> '.join(self._providers)} -> rule-based engine "
                        f"(daily paid-LLM budget ₹{settings.llm_daily_budget_inr:.0f})")
        else:
            logger.info("[LLMRouter] No external LLM credentials — rule-based engine answers. All features still work.")

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
            logger.warning(f"[LLMRouter] Azure OpenAI init failed: {e}. Skipping Azure.")
            self._client = None
            self._providers = [p for p in self._providers if p != "azure_openai"]
            self._mode = self._providers[0] if self._providers else "fallback"

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Primary provider ('sarvam', 'gemini', 'azure_openai') or 'fallback'."""
        return self._mode

    @property
    def providers(self) -> List[str]:
        return list(self._providers)

    @property
    def is_gemini(self) -> bool:
        return self._mode == "gemini"

    @property
    def is_azure(self) -> bool:
        return self._mode == "azure_openai"

    @property
    def is_sarvam(self) -> bool:
        return self._mode == "sarvam"

    @property
    def has_llm(self) -> bool:
        return bool(self._providers)

    def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful Indian Railways AI assistant.",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Main completion method. Tries each provider in order, then the rule-based engine. Never raises."""
        return self._run([{"role": "user", "content": prompt}], system_prompt, temperature, max_tokens,
                         fallback_prompt=prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        """Multi-turn chat completion (OpenAI-style messages; a 'system' message is honoured)."""
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        turns = [m for m in messages if m["role"] != "system"]
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self._run(turns, system, temperature, max_tokens, fallback_prompt=prompt)

    # ── Routing core ──────────────────────────────────────────────────────────

    def _run(self, messages: List[Dict[str, str]], system: Optional[str], temperature: float, max_tokens: int,
             fallback_prompt: str, **kwargs: Any) -> LLMResponse:
        t0 = time.perf_counter()
        errors: List[str] = []
        for provider in self._providers:
            skip = self._skip_reason(provider)
            if skip:
                errors.append(f"{provider}: {skip}")
                continue
            p0 = time.perf_counter()
            try:
                text, model, tin, tout = self._call(provider, messages, system, temperature, max_tokens)
            except ProviderError as e:
                self._record_provider(provider, ok=False, error=str(e), latency_ms=(time.perf_counter() - p0) * 1000)
                errors.append(f"{provider}: {e}")
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            self._record_provider(provider, ok=True, model=model, tokens=(tin, tout),
                                  latency_ms=(time.perf_counter() - p0) * 1000)
            self._record(True, provider, model, latency_ms=elapsed)
            return LLMResponse(content=text, mode=provider, model=model, tokens_used=tin + tout, latency_ms=elapsed)

        err = "; ".join(errors) or None
        if self._providers:
            logger.warning(f"[LLMRouter] No LLM provider answered ({err}). Using rule-based engine for this request.")
            self._record(False, error=(err or "")[:300])
        return self._call_fallback(fallback_prompt, error=err, **kwargs)

    def _skip_reason(self, provider: str) -> Optional[str]:
        if provider in PAID_PROVIDERS and self.spend_today() >= settings.llm_daily_budget_inr > 0:
            return f"daily budget ₹{settings.llm_daily_budget_inr:.0f} reached"
        now = time.time()
        if provider == "gemini":
            if self._gemini_chain and all(self._cooldown.get(m, 0) > now for m in self._gemini_chain):
                return "all models cooling down after errors"
        elif self._cooldown.get(provider, 0) > now:
            return "cooling down after an error"
        return None

    def _call(self, provider: str, messages: List[Dict[str, str]], system: Optional[str], temperature: float,
              max_tokens: int) -> Tuple[str, str, int, int]:
        if provider == "sarvam":
            return self._sarvam_generate(messages, system, temperature, max_tokens)
        if provider == "gemini":
            contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                        for m in messages]
            try:
                text, model, tokens = self._gemini_generate(contents, system, temperature, max_tokens)
            except RuntimeError as e:
                raise ProviderError(str(e)) from None
            return text, model, tokens, 0
        if provider == "azure_openai":
            return self._azure_generate(messages, system)
        raise ProviderError(f"unknown provider {provider}")

    # ── Sarvam AI ─────────────────────────────────────────────────────────────

    def _sarvam_generate(self, messages: List[Dict[str, str]], system: Optional[str], temperature: float,
                         max_tokens: int) -> Tuple[str, str, int, int]:
        model = settings.sarvam_model.strip() or "sarvam-105b"
        base = (settings.sarvam_base_url or "https://api.sarvam.ai").rstrip("/")
        url = base + (SARVAM_V1 if model.startswith("sarvam-") else SARVAM_V2)
        body: Dict[str, Any] = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "temperature": temperature,
            "max_tokens": max(int(max_tokens), 256),
        }
        if model.startswith("sarvam-105b"):
            body["reasoning_effort"] = None        # hybrid model: thinking off = faster, cheaper, full budget for the answer
        elif model in ("glm5.3", "deepseekv4.1-flash"):
            body["reasoning_effort"] = "low"
        headers = {"api-subscription-key": settings.sarvam_api_key.strip(), "Content-Type": "application/json"}

        for attempt in range(2):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=SARVAM_TIMEOUT_S)
            except requests.RequestException as e:
                self._cooldown["sarvam"] = time.time() + 15
                raise ProviderError(f"network {type(e).__name__}") from None
            if r.status_code == 200:
                break
            msg = ""
            try:
                err = r.json().get("error") or {}
                msg = (err.get("message") if isinstance(err, dict) else str(err)) or ""
            except Exception:
                pass
            if r.status_code == 400 and attempt == 0 and "reasoning_effort" in body:
                body.pop("reasoning_effort")       # model/endpoint rejected the override — retry once without it
                continue
            if r.status_code in (401, 403):
                self._cooldown["sarvam"] = time.time() + 300     # bad key: don't hammer
            elif r.status_code in (429, 502, 503):
                self._cooldown["sarvam"] = time.time() + 30      # rate limited / overloaded
            else:
                self._cooldown["sarvam"] = time.time() + 20
            raise ProviderError(f"HTTP {r.status_code} {msg[:160]}".strip())

        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = _THINK_RE.sub("", str((choice.get("message") or {}).get("content") or "")).strip()
        if not text:
            raise ProviderError(f"empty answer (finish_reason={choice.get('finish_reason')})")
        usage = data.get("usage") or {}
        return text, str(data.get("model") or model), int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

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

    # ── Azure OpenAI ──────────────────────────────────────────────────────────

    def _azure_generate(self, messages: List[Dict[str, str]], system: Optional[str]) -> Tuple[str, str, int, int]:
        if self._client is None:
            raise ProviderError("client not initialised")
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            lc = [SystemMessage(content=system)] if system else []
            for m in messages:
                lc.append(AIMessage(content=m["content"]) if m["role"] == "assistant" else HumanMessage(content=m["content"]))
            response = self._client.invoke(lc)
        except Exception as e:
            self._cooldown["azure_openai"] = time.time() + 20
            raise ProviderError(str(e)[:200]) from None
        content = response.content if hasattr(response, "content") else str(response)
        usage = getattr(response, "usage_metadata", {}) or {}
        return (content, settings.azure_openai_deployment_name,
                int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0))

    # ── Accounting ────────────────────────────────────────────────────────────

    @staticmethod
    def cost_inr(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
        """Estimated ₹ cost of one call (Sarvam's published per-token prices; Gemini free tier = 0)."""
        if provider != "sarvam":
            return 0.0
        pin, pout = SARVAM_PRICES_INR.get(model, SARVAM_PRICES_INR["sarvam-105b"])
        return (tokens_in * pin + tokens_out * pout) / 1_000_000

    def spend_today(self) -> float:
        with self._lock:
            if self._spend_day != _today():
                self._spend_day, self._spend_today = _today(), 0.0
            return self._spend_today

    def _record_provider(self, provider: str, ok: bool, model: str = "", tokens: Tuple[int, int] = (0, 0),
                         latency_ms: float = 0.0, error: Optional[str] = None) -> None:
        cost = self.cost_inr(provider, model, *tokens) if ok else 0.0
        self.spend_today()                         # rolls the day over if needed
        with self._lock:
            s = self._per.setdefault(provider, {"calls": 0, "ok": 0, "failed": 0, "tokens_in": 0, "tokens_out": 0,
                                                "cost_inr": 0.0, "latency_ms_total": 0.0, "last_error": None})
            s["calls"] += 1
            s["ok" if ok else "failed"] += 1
            s["tokens_in"] += tokens[0]
            s["tokens_out"] += tokens[1]
            s["cost_inr"] += cost
            if ok:
                s["latency_ms_total"] += latency_ms
                s["last_model"] = model
            else:
                s["last_error"] = (error or "")[:200]
            self._spend_today += cost

    def _record(self, ok: bool, provider: Optional[str] = None, model: Optional[str] = None,
                error: Optional[str] = None, latency_ms: Optional[float] = None) -> None:
        with self._lock:
            self._stats["requests"] += 1
            if ok:
                self._stats["llm_ok"] += 1
                self._stats["last_provider"] = provider
                self._stats["last_model"] = model
                self._stats["last_error"] = None
            else:
                self._stats["fallback_used"] += 1
                self._stats["last_error"] = error
            if latency_ms is not None:
                self._stats["last_latency_ms"] = round(latency_ms, 1)

    def status(self) -> Dict[str, Any]:
        """Operational view of the LLM layer (used by /health, the sidebar and the Control Tower)."""
        spend = self.spend_today()
        with self._lock:
            out = dict(self._stats)
            per = {}
            for p, s in self._per.items():
                per[p] = {
                    "label": PROVIDER_LABELS.get(p, p), "calls": s["calls"], "ok": s["ok"], "failed": s["failed"],
                    "success_pct": round(100.0 * s["ok"] / s["calls"], 1) if s["calls"] else None,
                    "avg_latency_ms": round(s["latency_ms_total"] / s["ok"], 1) if s["ok"] else None,
                    "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"], "cost_inr": round(s["cost_inr"], 4),
                    "model": s.get("last_model") or (settings.sarvam_model if p == "sarvam" else
                                                     (self._gemini_chain[0] if p == "gemini" and self._gemini_chain else "")),
                    "last_error": s["last_error"],
                    "cooling_down": self._cooldown.get(p, 0) > time.time() if p != "gemini" else
                    all(self._cooldown.get(m, 0) > time.time() for m in self._gemini_chain) if self._gemini_chain else False,
                }
        out.update({
            "mode": self._mode, "providers": list(self._providers), "per_provider": per,
            "model_chain": list(self._gemini_chain),
            "cooling_down": [m for m, t in self._cooldown.items() if t > time.time()],
            "spend_today_inr": round(spend, 4), "daily_budget_inr": settings.llm_daily_budget_inr,
            "budget_reached": settings.llm_daily_budget_inr > 0 and spend >= settings.llm_daily_budget_inr,
            "gemini_ok": out.get("llm_ok", 0),   # backwards-compatible key
        })
        return out

    def probe(self) -> List[Dict[str, Any]]:
        """Send a tiny request to every configured provider and report what happened (Control Tower test button).
        Cooldowns are ignored so an operator can check a key right after fixing it."""
        results: List[Dict[str, Any]] = []
        msgs = [{"role": "user", "content": "Reply with exactly: OK"}]
        for p in self._providers:
            if p == "sarvam":
                self._cooldown.pop("sarvam", None)
            elif p == "gemini":
                for m in self._gemini_chain:
                    self._cooldown.pop(m, None)
            t0 = time.perf_counter()
            try:
                text, model, tin, tout = self._call(p, msgs, "You are a connectivity check.", 0.0, 32)
                ms = (time.perf_counter() - t0) * 1000
                self._record_provider(p, ok=True, model=model, tokens=(tin, tout), latency_ms=ms)
                results.append({"provider": p, "label": PROVIDER_LABELS.get(p, p), "ok": True, "model": model,
                                "latency_ms": round(ms, 1), "reply": text[:60],
                                "cost_inr": round(self.cost_inr(p, model, tin, tout), 5)})
            except ProviderError as e:
                ms = (time.perf_counter() - t0) * 1000
                self._record_provider(p, ok=False, error=str(e), latency_ms=ms)
                results.append({"provider": p, "label": PROVIDER_LABELS.get(p, p), "ok": False,
                                "latency_ms": round(ms, 1), "error": str(e)[:200]})
        return results

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
        if self.has_llm:
            system = (
                "You are RailSeva AI, the official intelligent customer support assistant for Indian Railways. "
                "Answer questions about PNR status, ticket booking, refunds, cancellations, "
                "Tatkal booking, waitlist/RAC, and railway policies. "
                "Be concise, accurate, empathetic, and helpful. Always reply in English. If uncertain, say so."
            )
            return self.complete(query, system_prompt=system, task="support")
        return self._fallback.support_query(query, language)

    def analyze_rca(self, module: str, alert: str) -> Dict[str, Any]:
        """Direct RCA call."""
        if self.has_llm:
            prompt = f"Perform root cause analysis for module '{module}' with alert: {alert}. Explain likely root cause and actionable remediations."
            response = self.complete(prompt, task="rca")
            return {"module": module, "analysis": response.content, "mode": response.mode}
        return self._fallback.analyze_rca(module, alert)

    def supervisor_tick(self, metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Always use rule-based supervisor (deterministic + fast)."""
        return self._fallback.supervisor_tick(metrics)


# Module-level singleton
llm_router = LLMRouter()
