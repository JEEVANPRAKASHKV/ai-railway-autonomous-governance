"""
core/llm_router.py
==================
Tri-mode LLM router — Golden Rule 2: Zero-Crash Guarantee.

Transparently routes LLM calls to:
  - PRIMARY:   Google Gemini 1.5 Flash (when GEMINI_API_KEY is configured)
  - SECONDARY: Azure OpenAI GPT-4o-mini (when Azure credentials are configured)
  - FALLBACK:  Intelligent rule-based FallbackEngine (always available, 0 external deps)

Usage:
    from core.llm_router import llm_router
    response = llm_router.complete(prompt="...", system_prompt="...")
    print(response.content)   # Works in ALL modes
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.config import settings
from core.schemas import LLMResponse
from core.fallback_engine import FallbackEngine

logger = logging.getLogger(__name__)


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
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._gemini_client = genai
            logger.info(f"[LLMRouter] Google Gemini initialized successfully with model '{settings.gemini_model}'.")
        except Exception as e:
            logger.warning(f"[LLMRouter] Gemini init failed: {e}. Checking Azure / falling back.")
            if settings.azure_available:
                self._mode = "azure_openai"
                self._init_azure_client()
            else:
                self._mode = "fallback"
                self._gemini_client = None

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

    # ── Google Gemini Call ────────────────────────────────────────────────────

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
            genai = self._gemini_client
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=system_prompt if system_prompt else None,
                generation_config=generation_config,
            )
            response = model.generate_content(prompt)
            elapsed = (time.perf_counter() - t0) * 1000

            content = response.text if hasattr(response, "text") and response.text else ""
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

            return LLMResponse(
                content=content,
                mode="gemini",
                model=settings.gemini_model,
                tokens_used=tokens_used,
                latency_ms=elapsed,
            )
        except Exception as e:
            logger.error(f"[LLMRouter] Gemini call failed: {e}. Switching to fallback for this request.")
            return self._call_fallback(prompt, error=f"Gemini: {e}", **kwargs)

    def _call_gemini_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            genai = self._gemini_client
            system_prompt = None
            history = []
            last_user_message = ""

            for m in messages:
                role, content = m["role"], m["content"]
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    history.append({"role": "user", "parts": [content]})
                    last_user_message = content
                elif role == "assistant":
                    history.append({"role": "model", "parts": [content]})

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=system_prompt if system_prompt else None,
                generation_config=generation_config,
            )

            if history and history[-1]["role"] == "user":
                chat_session = model.start_chat(history=history[:-1])
                response = chat_session.send_message(last_user_message)
            else:
                combined_prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
                response = model.generate_content(combined_prompt)

            elapsed = (time.perf_counter() - t0) * 1000
            content = response.text if hasattr(response, "text") and response.text else ""
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

            return LLMResponse(
                content=content,
                mode="gemini",
                model=settings.gemini_model,
                tokens_used=tokens_used,
                latency_ms=elapsed,
            )
        except Exception as e:
            logger.error(f"[LLMRouter] Gemini chat failed: {e}. Switching to fallback.")
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
            content += f"\n\n*[Note: Azure OpenAI unavailable ({error}). Using rule-based fallback.]*"

        elapsed = (time.perf_counter() - t0) * 1000
        return LLMResponse(content=content, mode="fallback", latency_ms=elapsed)

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
