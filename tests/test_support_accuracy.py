"""
tests/test_support_accuracy.py
==============================
Accuracy and honesty of the Support Assistant:
- the verified knowledge base is well-formed and free of known-outdated rules;
- held-out passenger questions route to the right answer (two independent sets);
- LLM drafts are used only when they stay faithful to the verified entries — a draft that invents an amount
  is replaced by the verified text;
- greetings / off-topic / unknown questions get honest replies, not hallucination flags or false claims.
"""

import json
import re
from pathlib import Path

import pytest

from core.schemas import LLMResponse
from modules.m6_support_assistant.rag_indexer import rag_indexer
from modules.m6_support_assistant.schema import SupportQueryRequest
import modules.m6_support_assistant.service as m6

ROOT = Path(__file__).resolve().parents[1]
EVAL_SETS = [ROOT / "tests" / "data" / "support_eval.json", ROOT / "tests" / "data" / "support_eval_fresh.json"]


def ask(q: str):
    return m6.support_assistant_service.query(SupportQueryRequest(query=q))


# ── Knowledge base integrity ──────────────────────────────────────────────────

def test_knowledge_base_is_well_formed():
    kb = json.loads((ROOT / "data" / "knowledge" / "support_kb.json").read_text(encoding="utf-8"))
    assert re.match(r"\d{4}-\d{2}-\d{2}$", kb["version"])
    ids = [e["id"] for e in kb["entries"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 35
    for e in kb["entries"]:
        assert e["title"] and e["answer"] and len(e["questions"]) >= 3, e["id"]
        assert e["sources"], f'{e["id"]} has no source'


def test_knowledge_base_has_current_rules():
    by_id = {e["id"]: e["answer"] for e in rag_indexer.entries}
    assert "60 days" in by_id["advance_booking"]
    assert "8 hours" in by_id["cancel_confirmed"] and "72 hours" in by_id["cancel_confirmed"]
    assert "not been restored" in by_id["senior_citizen"]
    assert "30 minutes" in by_id["boarding_change"]
    assert "8 PM" in by_id["chart_time"]


# ── Routing accuracy on held-out questions (no LLM) ───────────────────────────

@pytest.mark.parametrize("path", EVAL_SETS, ids=lambda p: p.stem)
def test_support_routes_held_out_questions_correctly(path):
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    wrong = []
    for c in cases:
        r = ask(c["q"])
        exp = c["expect"]
        if exp in ("smalltalk", "offtopic", "tool"):
            got = r.route
        else:
            got = r.retrieved_contexts[0].chunk_id if r.route == "knowledge" and r.retrieved_contexts else r.route
        if got != exp:
            wrong.append((c["q"], exp, got))
    accuracy = 1 - len(wrong) / len(cases)
    assert accuracy >= 0.95, f"accuracy {accuracy:.0%}; misses: {wrong}"


def test_without_llm_the_verified_text_is_returned_verbatim(monkeypatch):
    monkeypatch.setattr(type(m6.llm_router), "has_llm", property(lambda self: False))
    r = ask("how far ahead can I reserve a train ticket")
    assert r.route == "knowledge" and r.answered_by == "verified knowledge base"
    assert rag_indexer.get("advance_booking")["answer"] in r.response
    assert r.retrieved_contexts[0].sources and r.kb_version


def test_greeting_offtopic_and_unknown_are_honest():
    for q, route in [("hi", "smalltalk"), ("what can i get from this site", "knowledge"),
                     ("who won yesterday's cricket match", "offtopic"),
                     ("what is the rule for carrying a pet dog in a train coupe", "no_match")]:
        r = ask(q)
        assert r.route == route, (q, r.route, r.response)
        assert r.hallucination_flag is False
        assert "Disclaimer" not in r.response


def test_frustrated_passenger_gets_no_false_promise():
    r = ask("my refund has not come, this is pathetic")
    assert r.escalation_recommended
    assert "flagged for human" not in r.response.lower()
    assert "139" in r.response


# ── LLM path: drafts are checked against the verified entries ─────────────────

class _FakeRouter:
    has_llm = True

    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, prompt, system_prompt="", **kw):
        self.calls.append(system_prompt)
        return LLMResponse(content=self.text, mode="sarvam", model="sarvam-105b")


def test_faithful_llm_draft_is_used(monkeypatch):
    fake = _FakeRouter("You can book reserved train tickets up to 60 days in advance, not counting the day of the journey.")
    monkeypatch.setattr(m6, "llm_router", fake)
    r = ask("how many days in advance can I book a ticket")
    assert r.answered_by == "Sarvam AI (sarvam-105b)" and "60 days" in r.response
    assert "VERIFIED KNOWLEDGE" in fake.calls[0] and "60 days" in fake.calls[0]      # grounded prompt


def test_unfaithful_llm_draft_is_replaced_by_verified_text(monkeypatch):
    fake = _FakeRouter("You can book up to 120 days in advance and pay a ₹500 fee for early booking.")
    monkeypatch.setattr(m6, "llm_router", fake)
    r = ask("how many days in advance can I book a ticket")
    assert r.answered_by == "verified answer (model draft rejected)"
    assert "120 days in advance and pay" not in r.response and "60 days" in r.response


def test_rule_engine_fallback_uses_the_knowledge_base():
    from core.fallback_engine import fallback_engine
    out = fallback_engine.support_query("how many days before the journey can i book").content
    assert "60 days" in out and "120 days in advance excluding" not in out
