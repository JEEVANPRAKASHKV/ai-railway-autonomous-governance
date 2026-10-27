"""
modules/m6_support_assistant/service.py
======================================
Module 6: Customer Support Assistant (English).

Every question takes exactly one route, and the reply says which (`route`, `answered_by`):

  language   → non-English text gets an English-only notice
  tool       → PNR status / refund tracking from booking records, train suggestions from the timetable
  smalltalk  → greetings and thanks get a short reply and what the assistant can do
  offtopic   → questions with nothing to do with railways or this site get a polite scope reply
  knowledge  → the verified knowledge base (data/knowledge/support_kb.json) answers:
                 · with an LLM configured (Sarvam AI → Gemini), the model writes the reply ONLY from the matched
                   entries; if the hallucination monitor flags the draft, or no model answers, the verified entry
                   text is returned word for word instead
                 · without an LLM, the verified entry text is returned word for word
  no_match   → nothing in the knowledge base matches well: say so, suggest the closest topics and point to 139

Nothing here invents a rule, an amount or an action that did not happen.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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

NON_LATIN = re.compile(r"[^\x00-\x7F -ɏ -⁯₹]")   # anything outside English text (₹ allowed)
ENGLISH_ONLY_REPLY = ("I can help in English only. Please type your question in English — for example "
                      "\"What is the refund if my waitlist ticket is not confirmed?\" or \"What is the status of PNR 0000001?\"")

# Retrieval confidence (best-phrasing score, 0–1), calibrated on tests/data/support_eval.json
CONFIDENT = 0.40     # answer directly
POSSIBLE = 0.24      # answer, but say which topic we matched and offer alternatives

SMALLTALK = re.compile(r"^\s*(hi+|hello+|hey+|hiya|namaste|good\s+(morning|afternoon|evening|night)|thanks?( you)?|"
                       r"thank\s*u|thx|ty|ok(ay)?|cool|great|nice|bye|goodbye|see you)\b[\s!.,?]*(there|so much|a lot|"
                       r"that was (helpful|useful|great)|bot|buddy)?[\s!.,?]*$", re.I)
RAIL_WORDS = re.compile(
    r"\b(trains?|tickets?|pnr|rail\w*|irctc|berths?|coach(es)?|stations?|tatkal|refunds?|cancel\w*|book\w*|journey|"
    r"travel\w*|fares?|seats?|chart|rac|wait\w*|wl|gnwl|rlwl|pqwl|tqwl|rswl|tte|platforms?|luggage|baggage|food|meals?|"
    r"tdr|boarding|aadhaar|senior|ladies|women|quota|sleeper|[123]a|[23]ac|ac|class(es)?|upgrade\w*|passengers?|"
    r"concession|discount|elderly|pregnant|helpline|complain\w*|139|id|identity|child(ren)?|kids?|infant|lost|fine|penalty|delay\w*|late|"
    r"reserv\w*|catering|pantry|e-?ticket|counter|agent|premium|status|madad|chair car|rajdhani|shatabdi|duronto|"
    r"vande|express|route|planner)\b", re.I)
SITE_WORDS = re.compile(r"\b(what (can|are|do) you|who are you|how (can|do|will) you help|help me|^help\b|"
                        r"this (site|website|app|page|chat|bot|tool)|chat ?bot|assistant|railway buddy|features?|"
                        r"services?|how does (it|this) work|is this (real|official|live))", re.I)

FRUSTRATION_KEYWORDS: Set[str] = {
    "scam", "cheat", "worst", "horrible", "refund not received", "money deducted", "fraud", "harassment", "stolen",
    "useless", "pathetic", "lost money", "disgusting", "terrible", "angry", "fed up",
}
SMALLTALK_REPLY = ("Hello! I can answer questions about Indian Railways tickets and travel: refunds and cancellation charges, "
                   "Tatkal, waitlist and RAC, chart timing, boarding changes, ID proof, luggage and more. You can also check "
                   "a demo PNR (type PNR 0000001) or ask for trains between two cities, e.g. \"trains from Delhi to Mumbai\".")
THANKS_REPLY = "You're welcome! Ask me anything else about your journey, tickets or refunds."
OFFTOPIC_REPLY = ("I can only help with Indian Railways travel and this app: tickets, refunds, cancellations, Tatkal, waitlist "
                  "and RAC, PNR status, train options and rules. Please ask me something about your train journey.")


def is_english(query: str) -> bool:
    """True when the query is written in English (Latin script). Railway Buddy supports English only."""
    letters = re.findall(r"[^\W\d_]", query)
    if not letters:
        return True
    foreign = len(NON_LATIN.findall(query))
    return foreign / max(1, len(letters)) < 0.2


def classify(query: str) -> str:
    """Route before retrieval: 'smalltalk', 'offtopic' or 'question'."""
    q = query.strip()
    if SMALLTALK.match(q):
        return "smalltalk"
    if not RAIL_WORDS.search(q) and not SITE_WORDS.search(q):
        return "offtopic"
    return "question"


class SupportAssistantService:
    """Grounded support assistant over the verified knowledge base, with booking-record tools."""

    def _detect_sentiment(self, query: str) -> str:
        low = query.lower()
        if any(kw in low for kw in FRUSTRATION_KEYWORDS):
            return "Frustrated"
        if any(w in low for w in ["thank", "thanks", "great", "helpful", "excellent"]):
            return "Positive"
        return "Neutral"

    # ── LLM-written answer from the matched entries only ──────────────────────
    def _llm_answer(self, query: str, hits: List[Tuple[Dict[str, Any], float]]) -> Tuple[Optional[str], str, str]:
        """Returns (answer or None, answered_by, context_text)."""
        context = "\n\n".join(f"[{i + 1}] {h['title']}\n{h['text']}" for i, (h, _) in enumerate(hits))
        system = (
            "You are the Railway Buddy support assistant for Indian Railways passengers. Answer the passenger's question "
            "using ONLY the verified knowledge below. Do not add rules, amounts, times or dates that are not in it. "
            "If the knowledge does not answer the question, say you don't have that information and suggest calling 139 "
            "or checking irctc.co.in. Reply in clear, simple English, in at most 120 words, without headings, "
            "and do not mention 'knowledge', 'context' or entry numbers.\n\n"
            f"VERIFIED KNOWLEDGE:\n{context}"
        )
        resp = llm_router.complete(prompt=query, system_prompt=system, temperature=0.2, max_tokens=400, task="support")
        if resp.mode == "fallback" or not resp.content.strip():
            return None, "", context
        label = {"sarvam": f"Sarvam AI ({resp.model})", "gemini": f"Gemini ({resp.model})",
                 "azure_openai": "Azure OpenAI"}.get(resp.mode, resp.mode)
        return resp.content.strip(), label, context

    def _verify(self, query: str, answer: str, docs: List[str]) -> Tuple[float, bool]:
        """Hallucination monitor on an LLM draft → (faithfulness, flagged)."""
        try:
            from governance.hallucination_monitor import hallucination_monitor
            ev = hallucination_monitor.evaluate(module_id="m6_support_assistant", query=query, response=answer, context_docs=docs)
            faith = round(float(ev.get("faithfulness_score") or 0.0), 3)
            return faith, ev.get("severity") in ("WARNING", "CRITICAL")
        except Exception as e:
            logger.debug(f"[M6] hallucination check skipped: {e}")
            return 0.5, False

    @governance_hook("m6_support_assistant", endpoint="/api/v1/support/query")
    def query(self, request: SupportQueryRequest) -> SupportQueryResponse:
        q = request.query.strip()
        sentiment = self._detect_sentiment(q)
        escalate = sentiment == "Frustrated"
        contexts: List[RetrievedContextChunk] = []
        faithfulness, flagged = 1.0, False
        tool_meta: Dict[str, Any] = {}
        match_score: Optional[float] = None

        # 1 ── language
        if not is_english(q):
            route, answer, answered_by = "language", ENGLISH_ONLY_REPLY, "language guard"
            detected = "unsupported"
        else:
            detected = "en"
            tool_out = support_tools.pnr_record(q)
            if tool_out is None:
                rt = support_tools.route(q)
                if rt:
                    tool_out = support_tools.train_suggestions(*rt)
            kind = classify(q)

            # 2 ── tools: booking records and timetable
            if tool_out is not None:
                answer, record, tool_meta = tool_out
                if support_tools.wants_planning(q):
                    answer += "\nTo plan the whole journey and book after your approval, use the AI Travel Planner."
                route, answered_by = "tool", f"tool: {tool_meta.get('tool')}"
                contexts = [RetrievedContextChunk(chunk_id=tool_meta.get("tool", "tool"), source_file=f"Booking records ({tool_meta.get('tool')})",
                                                  text=record[:300], similarity_score=1.0)]
            # 3 ── greetings / thanks
            elif kind == "smalltalk":
                route, answered_by = "smalltalk", "assistant"
                answer = THANKS_REPLY if re.search(r"\b(thank|thx|ty)\b", q, re.I) else SMALLTALK_REPLY
            # 4 ── off-topic
            elif kind == "offtopic":
                route, answered_by, answer = "offtopic", "assistant", OFFTOPIC_REPLY
            else:
                # 5 ── verified knowledge base
                hits = rag_indexer.retrieve(q, top_k=3)
                best, score = (hits[0] if hits else (None, 0.0))
                match_score = score
                contexts = [RetrievedContextChunk(chunk_id=h["chunk_id"], source_file=h["title"], text=h["text"][:300],
                                                  similarity_score=s, sources=h.get("sources", []))
                            for h, s in hits if s >= POSSIBLE] or []
                if best is None or score < POSSIBLE:
                    route, answered_by = "no_match", "knowledge base"
                    near = [h["title"] for h, s in hits[:3] if s >= 0.12]
                    answer = ("I don't have a verified answer for that yet, so I won't guess. "
                              + (f"Related topics I can help with: {', '.join(near)}. " if near else "")
                              + "For anything else, call 139 (Rail Madad) or check irctc.co.in.")
                    contexts = []
                else:
                    route = "knowledge"
                    use = [(h, s) for h, s in hits if s >= POSSIBLE] or [(best, score)]
                    draft, label, ctx = (None, "", "")
                    if llm_router.has_llm:
                        draft, label, ctx = self._llm_answer(q, use if score < CONFIDENT else use[:2])
                    if draft:
                        faithfulness, flagged = self._verify(q, draft, [h["text"] for h, _ in use])
                    if draft and not flagged:
                        answer, answered_by = draft, label
                    else:
                        # verbatim verified answer: no model, the model was unavailable, or its draft was flagged
                        answer = best["text"]
                        answered_by = ("verified answer (model draft rejected)" if draft else "verified knowledge base")
                        faithfulness, flagged = 1.0, False
                    if score < CONFIDENT:
                        alt = [h["title"] for h, s in hits[1:3] if s >= POSSIBLE]
                        answer = (f"I think you're asking about {best['title'].lower()}.\n\n{answer}"
                                  + (f"\n\nIf not, I can also help with: {', '.join(alt)}." if alt else ""))

        if escalate:
            answer += ("\n\nI'm sorry about the trouble. To raise a formal complaint, call 139 or use Rail Madad "
                       "(railmadad.indianrailways.gov.in) with your PNR — I can't file complaints myself.")

        report_inference(prediction=route, version=answered_by)
        try:
            audit_store.append("conversation", {
                "session_id": request.session_id, "user_id": request.user_id, "route": route,
                "query": q[:500], "answer": answer[:1200], "answered_by": answered_by,
                "sources": [c.chunk_id for c in contexts], "match_score": match_score, "faithfulness": faithfulness,
                "hallucination_flag": flagged, "sentiment": sentiment, "escalated": escalate,
                "kb_version": rag_indexer.version, "tool": tool_meta or None,
            }, module="m6_support_assistant", ref=request.session_id or "", actor="passenger")
        except Exception:
            pass

        return SupportQueryResponse(
            answered_by=answered_by, query=request.query, response=answer, detected_language=detected,
            faithfulness_score=faithfulness, hallucination_flag=flagged, retrieved_contexts=contexts,
            sentiment=sentiment, escalation_recommended=escalate, confidence_score=round(faithfulness, 2),
            route=route, match_score=match_score, kb_version=rag_indexer.version,
        )


support_assistant_service = SupportAssistantService()
