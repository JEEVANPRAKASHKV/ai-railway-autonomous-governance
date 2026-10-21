"""
modules/m6_support_assistant/service.py
======================================
Inference service for Module 6: Customer Support Assistant (English).
Tools first (PNR status, refunds, train suggestions), then RAG over IRCTC policy text,
hallucination faithfulness audit, and sentiment-triggered escalation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from core.config import settings
from core.llm_router import llm_router
from governance.audit_store import audit_store
from governance.context import report_inference
from governance.hooks import governance_hook
from modules.m6_support_assistant.rag_indexer import rag_indexer
from modules.m6_support_assistant.tools import support_tools
from modules.m6_support_assistant.schema import (
    RetrievedContextChunk,
    SupportQueryRequest,
    SupportQueryResponse,
)

logger = logging.getLogger(__name__)

NON_LATIN = re.compile(r"[^\x00-\x7F\u00A0-\u024F\u2000-\u206F\u20B9]")   # anything outside English text (₹ allowed)
ENGLISH_ONLY_REPLY = ("I can help in English only. Please type your question in English — for example "
                      "\"What is the refund if my waitlist ticket is not confirmed?\" or \"What is the status of PNR 0000001?\"")


def is_english(query: str) -> bool:
    """True when the query is written in English (Latin script). Railway Buddy supports English only."""
    letters = re.findall(r"[^\W\d_]", query)
    if not letters:
        return True
    foreign = len(NON_LATIN.findall(query))
    return foreign / max(1, len(letters)) < 0.2


FRUSTRATION_KEYWORDS: Set[str] = {
    "scam", "cheat", "worst", "horrible", "refund not received", "money deducted",
    "fraud", "complaint", "harassment", "stolen", "useless", "pathetic", "lost money",
}


class SupportAssistantService:
    """Service to handle passenger queries (English) using grounded RAG and hallucination checks."""

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
        lang = "en"
        if not is_english(request.query):
            report_inference(prediction="english_only_notice", version="language guard")
            return SupportQueryResponse(
                answered_by="language guard", query=request.query, response=ENGLISH_ONLY_REPLY,
                detected_language="unsupported", faithfulness_score=1.0, hallucination_flag=False,
                retrieved_contexts=[], sentiment="Neutral", escalation_recommended=False, confidence_score=1.0)

        sentiment = self._detect_sentiment(request.query)
        escalate = bool(sentiment == "Frustrated")

        # ── Step 1: Retrieve Grounding Context ────────────────────────────────
        retrieval_query = request.query
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
        answered_by = "rule-based policy engine"
        tool_meta: Dict[str, Any] = {}

        # ── Step 1b: Tools first — PNR status / refund tracking, train suggestions ──
        tool_out = support_tools.pnr_record(request.query)
        if tool_out is None:
            rt = support_tools.route(retrieval_query)
            if rt:
                tool_out = support_tools.train_suggestions(*rt)
        if tool_out is not None:
            answer_text, record_text, tool_meta = tool_out
            retrieved_contexts = [RetrievedContextChunk(chunk_id=tool_meta.get("tool", "tool"),
                                                        source_file=f"tool:{tool_meta.get('tool')}",
                                                        text=record_text[:300], similarity_score=1.0)]
            combined_context_text = record_text
            top_sim = 1.0

        # Check for direct curated domain policy match
        from core.fallback_engine import fallback_engine
        curated_policy = fallback_engine._static_support_response(retrieval_query)
        has_curated_policy = not curated_policy.startswith("Thank you for your query. For specific railway inquiries")

        if tool_out is not None:
            final_answer = answer_text
            if support_tools.wants_planning(request.query):
                final_answer += "\nTo plan the whole journey and book after your approval, use the AI Travel Planner."
            faithfulness, hallucination_flag = 1.0, False
            answered_by = f"tool: {tool_meta.get('tool')}"
            has_curated_policy = False
        elif top_sim < 0.15 and not has_curated_policy:
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
                "Always reply in clear, simple English. "
                "If the context does not contain the answer, say so and point to irctc.co.in or RailMadad 139 — never guess amounts or rules.\n"
                f"Grounding Context:\n{combined_context_text}\n"
            )

            llm_resp = llm_router.complete(prompt=retrieval_query, system_prompt=system_prompt, task="support")
            final_answer = llm_resp.content
            answered_by = f"Gemini ({llm_resp.model})" if llm_resp.mode == "gemini" else (
                "Azure OpenAI" if llm_resp.mode == "azure_openai" else "rule-based policy engine")

            # ── Step 3: Hallucination & Faithfulness Audit ────────────────────────
            faithfulness = self._compute_faithfulness(final_answer, combined_context_text)
            hallucination_flag = bool(faithfulness < settings.governance_hallucination_threshold)

        # ── Pillar 4: Record in Governance Hallucination Monitor ───────────────
        try:
            from governance.hallucination_monitor import hallucination_monitor
            context_docs = ([combined_context_text] if tool_out is not None else
                            [c.text for c in retrieved_contexts] if retrieved_contexts else [combined_context_text])
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

        # ── Conversation audit (compliance) + register for satisfaction feedback ──
        report_inference(prediction="answered", version=answered_by)
        try:
            audit_store.append("conversation", {
                "session_id": request.session_id, "user_id": request.user_id, "language": lang,
                "query": request.query[:500], "answer": final_answer[:1200], "answered_by": answered_by,
                "sources": [c.source_file for c in retrieved_contexts], "faithfulness": faithfulness,
                "hallucination_flag": hallucination_flag, "sentiment": sentiment, "escalated": escalate,
                "tool": tool_meta or None,
            }, module="m6_support_assistant", ref=request.session_id or "", actor="passenger")
        except Exception:
            pass

        return SupportQueryResponse(
            answered_by=answered_by,
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
