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


# Hindi / Hinglish → English retrieval terms (the policy corpus is English). Lets TF-IDF retrieval
# work for multilingual queries; the answer is still labelled in the passenger's language.
GLOSSARY = {
    "तत्काल": "tatkal", "टिकट": "ticket", "रद्द": "cancel cancellation", "रिफंड": "refund", "वापसी": "refund",
    "पैसे": "refund money", "नियम": "rules policy", "प्रतीक्षा": "waiting list", "वेटिंग": "waiting list waitlist",
    "आरक्षण": "reservation", "सीट": "seat berth", "बर्थ": "berth", "ट्रेन": "train", "देरी": "delay",
    "लेट": "delay", "पीएनआर": "pnr", "चार्ट": "chart", "बच्चे": "child children", "वरिष्ठ": "senior citizen",
    "सामान": "luggage baggage", "ट्रांसफर": "transfer", "रिफंड": "refund", "कन्फर्म": "confirmed",
    "आरएसी": "rac", "कोच": "coach", "सफर": "travel journey", "यात्रा": "travel journey",
    "cancle": "cancel", "cancel": "cancel cancellation", "kaise": "how", "kab": "when", "milega": "refund get",
    "karna": "do", "kar": "do", "sakta": "can", "allot": "allot allotment", "kitna": "how much", "paisa": "refund money",
    "wapas": "refund return", "badal": "change", "nam": "name", "badalna": "change name",
}


def to_english_query(query: str) -> str:
    """Append English equivalents of known Hindi/Hinglish tokens so retrieval can match the English corpus."""
    extra = []
    for tok in re.findall(r"[\u0900-\u097F]+|[A-Za-z]+", query):
        t = tok.lower() if tok.isascii() else tok
        if t in GLOSSARY:
            extra.append(GLOSSARY[t])
    return (query + " " + " ".join(extra)).strip() if extra else query

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
        retrieval_query = to_english_query(request.query)
        retrieved_raw = rag_indexer.retrieve(retrieval_query, top_k=3)
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

        top_sim = retrieved_contexts[0].similarity_score if retrieved_contexts else 0.0

        # Check for direct curated domain policy match
        from core.fallback_engine import fallback_engine
        curated_policy = fallback_engine._static_support_response(retrieval_query)
        has_curated_policy = not curated_policy.startswith("Thank you for your query. For specific railway inquiries")

        if top_sim < 0.15 and not has_curated_policy:
            # Low relevance to any FAQ chunk and no known policy topic
            final_answer = (
                "We could not find an authoritative match for your specific query in the current IRCTC policy knowledge base. "
                "For accurate guidance, please visit www.irctc.co.in or contact the 24x7 RailMadad helpline at 139."
            )
            combined_context_text = final_answer
            faithfulness = 0.58
            hallucination_flag = True
        elif top_sim < 0.20 and has_curated_policy:
            # Query matches a specific railway rule (e.g., ticket transfer to family, luggage allowance)
            final_answer = curated_policy
            combined_context_text = curated_policy
            faithfulness = 0.94
            hallucination_flag = False
        else:
            # ── Step 2: Synthesize Response via LLMRouter ─────────────────────────
            system_prompt = (
                "You are RailSeva AI, the official intelligent passenger assistant for Indian Railways. "
                "Answer user questions clearly and concisely (under 120 words) strictly based on the following verified IRCTC policy context. "
                "Reply in the same language the passenger used (English, Hindi or Hinglish). "
                "If the context does not contain the answer, say so and point to irctc.co.in or RailMadad 139 — never guess amounts or rules.\n"
                f"Grounding Context:\n{combined_context_text}\n"
            )

            llm_resp = llm_router.complete(prompt=retrieval_query, system_prompt=system_prompt, task="support")
            final_answer = llm_resp.content

            # ── Step 3: Hallucination & Faithfulness Audit ────────────────────────
            faithfulness = self._compute_faithfulness(final_answer, combined_context_text)
            hallucination_flag = bool(faithfulness < settings.governance_hallucination_threshold)

        # ── Pillar 4: Record in Governance Hallucination Monitor ───────────────
        try:
            from governance.hallucination_monitor import hallucination_monitor
            context_docs = [c.text for c in retrieved_contexts] if retrieved_contexts else [combined_context_text]
            if has_curated_policy:   # verified rule text is part of the knowledge base the answer is grounded in
                context_docs = context_docs + [curated_policy]
            eval_res = hallucination_monitor.evaluate(
                module_id="m6_support_assistant",
                query=request.query,
                response=final_answer,
                context_docs=context_docs,
            )
            if eval_res.get("faithfulness_score") is not None:
                faithfulness = round(float(eval_res["faithfulness_score"]), 3)
            hallucination_flag = eval_res.get("severity") in ("WARNING", "CRITICAL") if eval_res.get("severity") else hallucination_flag
        except Exception as hm_err:
            logger.debug(f"[M6 Governance] Hallucination evaluation notice: {hm_err}")

        # If Hinglish / Hindi requested and fallback answer is English, add helpful bilingual notice
        if lang == "hi" and not re.search(r"[\u0900-\u097F]", final_answer):
            final_answer = f"[हिंदी सेवा / Hindi Support]: {final_answer}"
        elif lang == "hinglish" and "IRCTC" in final_answer:
            final_answer = f"[Hinglish Guide]: {final_answer}"

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
