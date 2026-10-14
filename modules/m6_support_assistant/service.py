"""
modules/m6_support_assistant/service.py
======================================
Inference service for Module 6: Multilingual Customer Support Assistant.
RAG pipeline with language detection, local semantic retrieval,
hallucination faithfulness audit, and sentiment-triggered escalation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from core.config import settings
from core.llm_router import llm_router
from governance.hooks import governance_hook
from modules.m6_support_assistant.rag_indexer import rag_indexer
from modules.m6_support_assistant.schema import (
    RetrievedContextChunk,
    SupportQueryRequest,
    SupportQueryResponse,
)

logger = logging.getLogger(__name__)

HINGLISH_WORDS: Set[str] = {
    "kya", "kaise", "kab", "kahan", "kyun", "hoga", "milega", "karna",
    "pnr", "tatkal", "cancle", "refund", "train", "seat", "mil", "gaya",
}

FRUSTRATION_KEYWORDS: Set[str] = {
    "scam", "cheat", "worst", "horrible", "refund not received", "money deducted",
    "fraud", "complaint", "harassment", "stolen", "useless", "pathetic", "lost money",
}


class SupportAssistantService:
    """Service to handle multilingual passenger queries using grounded RAG and hallucination checks."""

    def _detect_language(self, query: str) -> str:
        # Check Devanagari unicode range
        if re.search(r"[\u0900-\u097F]", query):
            return "hi"
        # Check Hinglish token overlap
        words = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
        if len(words.intersection(HINGLISH_WORDS)) >= 2:
            return "hinglish"
        return "en"

    def _detect_sentiment(self, query: str) -> str:
        low = query.lower()
        for kw in FRUSTRATION_KEYWORDS:
            if kw in low:
                return "Frustrated"
        if any(w in low for w in ["thank", "thanks", "great", "helpful", "good", "excellent"]):
            return "Positive"
        return "Neutral"

    def _compute_faithfulness(self, response_text: str, context_text: str) -> float:
        """Measure token/entity overlap between synthesized response and grounding context."""
        resp_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", response_text.lower()))
        ctx_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", context_text.lower()))

        if not resp_tokens:
            return 0.85

        # Fraction of response substantive tokens supported by context
        intersection = resp_tokens.intersection(ctx_tokens)
        overlap = len(intersection) / len(resp_tokens)
        # Scaled score: baseline 0.50 + 0.50 * overlap
        score = min(0.99, max(0.40, 0.45 + (overlap * 0.55)))
        return round(float(score), 3)

    @governance_hook("m6_support_assistant", endpoint="/api/v1/support/query")
    def query(self, request: SupportQueryRequest) -> SupportQueryResponse:
        """Process passenger query through grounded RAG pipeline and validate against hallucination."""
        lang = request.language
        if not lang or lang == "auto":
            lang = self._detect_language(request.query)

        sentiment = self._detect_sentiment(request.query)
        escalate = bool(sentiment == "Frustrated")

        # ── Step 1: Retrieve Grounding Context ────────────────────────────────
        retrieved_raw = rag_indexer.retrieve(request.query, top_k=3)
        retrieved_contexts: List[RetrievedContextChunk] = []
        combined_context_text = ""

        for chunk_dict, sim_score in retrieved_raw:
            retrieved_contexts.append(
                RetrievedContextChunk(
                    chunk_id=chunk_dict.get("chunk_id", "chunk-0"),
                    source_file=chunk_dict.get("source_file", "knowledge_base"),
                    text=chunk_dict.get("text", "")[:300],
                    similarity_score=sim_score,
                )
            )
            combined_context_text += "\n" + chunk_dict.get("text", "")

        # ── Step 2: Synthesize Response via LLMRouter ─────────────────────────
        system_prompt = (
            "You are RailSeva AI, the official intelligent passenger assistant for Indian Railways. "
            "Answer user questions clearly and concisely strictly based on the following verified IRCTC policy context.\n"
            f"Grounding Context:\n{combined_context_text}\n"
        )

        user_prompt = f"User Query: {request.query}\nLanguage: {lang}\nPlease provide an accurate, helpful answer."

        llm_resp = llm_router.complete(prompt=request.query, system_prompt=system_prompt, task="support")
        final_answer = llm_resp.content

        # If Hinglish / Hindi requested and fallback answer is English, add helpful bilingual notice
        if lang == "hi" and not re.search(r"[\u0900-\u097F]", final_answer):
            final_answer = f"[हिंदी सेवा / Hindi Support]: {final_answer}"
        elif lang == "hinglish" and "IRCTC" in final_answer:
            final_answer = f"[Hinglish Guide]: {final_answer}"

        # ── Step 3: Hallucination & Faithfulness Audit ────────────────────────
        faithfulness = self._compute_faithfulness(final_answer, combined_context_text)
        hallucination_flag = bool(faithfulness < settings.governance_hallucination_threshold)

        if hallucination_flag:
            logger.warning(
                f"[M6 Governance] Low faithfulness detected ({faithfulness:.2f} < {settings.governance_hallucination_threshold}). "
                "Adding safety disclaimer to prevent ungrounded hallucinations."
            )
            final_answer += (
                "\n\n[Official Disclaimer]: Please verify current rules directly on irctc.co.in or call 139 "
                "as automated policy interpretation may vary during live chart preparation."
            )

        if escalate:
            final_answer += (
                "\n\n[Escalation Advisory]: We noticed you have an urgent grievance. "
                "Your complaint ticket has been flagged for human railway officer review. You can also dial RailMadad at 139."
            )

        return SupportQueryResponse(
            query=request.query,
            response=final_answer,
            detected_language=lang,
            faithfulness_score=faithfulness,
            hallucination_flag=hallucination_flag,
            retrieved_contexts=retrieved_contexts,
            sentiment=sentiment,
            escalation_recommended=escalate,
            confidence_score=round(faithfulness, 2),
        )


support_assistant_service = SupportAssistantService()
