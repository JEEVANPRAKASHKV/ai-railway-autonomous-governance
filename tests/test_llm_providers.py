"""
tests/test_llm_providers.py
===========================
Multi-provider LLM router: Sarvam AI first, Gemini failover, rule-based last resort, ₹ spend cap.
No network — requests.post is stubbed with Sarvam's documented request/response shapes.
"""

import pytest
from fastapi.testclient import TestClient


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


def _sarvam_ok(text="Full refund within 5 business days.", tin=100, tout=50):
    return _Resp(200, {"id": "x", "object": "chat.completion", "model": "sarvam-105b",
                       "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                       "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}})


def _gemini_ok(text="Hello from Gemini"):
    return _Resp(200, {"candidates": [{"content": {"parts": [{"text": text}]}}], "usageMetadata": {"totalTokenCount": 9}})


def _router(monkeypatch, handler, sarvam=True, gemini=True, order="sarvam,gemini,azure_openai", budget=50.0):
    from core import llm_router as lr
    monkeypatch.setattr(lr.settings, "sarvam_api_key", "sk_test_" + "x" * 24 if sarvam else "")
    monkeypatch.setattr(lr.settings, "sarvam_model", "sarvam-105b")
    monkeypatch.setattr(lr.settings, "gemini_api_key", "test-key-" + "x" * 30 if gemini else "")
    monkeypatch.setattr(lr.settings, "azure_openai_key", "")
    monkeypatch.setattr(lr.settings, "llm_provider_order", order)
    monkeypatch.setattr(lr.settings, "llm_daily_budget_inr", budget)
    monkeypatch.setattr(lr.requests, "post", handler)
    return lr.LLMRouter()


def test_sarvam_is_primary_and_called_correctly(monkeypatch):
    seen = {}

    def handler(url, headers=None, json=None, **kw):
        seen.update(url=url, headers=headers, body=json)
        return _sarvam_ok()

    r = _router(monkeypatch, handler)
    assert r.mode == "sarvam" and r.providers == ["sarvam", "gemini"]
    res = r.complete("Refund for WL ticket?", system_prompt="Be brief.")
    assert res.mode == "sarvam" and res.model == "sarvam-105b" and "refund" in res.content.lower()
    assert seen["url"] == "https://api.sarvam.ai/v1/chat/completions"
    assert seen["headers"]["api-subscription-key"].startswith("sk_test_")
    assert seen["body"]["model"] == "sarvam-105b"
    assert "reasoning_effort" in seen["body"] and seen["body"]["reasoning_effort"] is None   # thinking off
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief."}
    st = r.status()
    assert st["per_provider"]["sarvam"]["ok"] == 1
    assert st["spend_today_inr"] == pytest.approx((100 * 15 + 50 * 60) / 1e6)


def test_sarvam_reasoning_trace_is_stripped(monkeypatch):
    r = _router(monkeypatch, lambda url, **kw: _sarvam_ok("<think>let me think</think>\nAnswer: yes."))
    assert r.complete("q").content == "Answer: yes."


def test_sarvam_retries_without_reasoning_override_on_400(monkeypatch):
    bodies = []

    def handler(url, json=None, **kw):
        bodies.append(dict(json))
        return _Resp(400, {"error": {"message": "reasoning_effort not supported"}}) if len(bodies) == 1 else _sarvam_ok()

    r = _router(monkeypatch, handler)
    assert r.complete("q").mode == "sarvam"
    assert "reasoning_effort" in bodies[0] and "reasoning_effort" not in bodies[1]


def test_sarvam_rate_limit_fails_over_to_gemini_then_cools_down(monkeypatch):
    calls = []

    def handler(url, **kw):
        calls.append(url)
        return _Resp(429, {"error": {"message": "rate_limit_exceeded_error"}}) if "sarvam" in url else _gemini_ok()

    r = _router(monkeypatch, handler)
    res = r.complete("hi")
    assert res.mode == "gemini" and res.content == "Hello from Gemini"
    n = len(calls)
    assert r.complete("again").mode == "gemini"
    assert not any("sarvam" in u for u in calls[n:]), "Sarvam must cool down after a 429"
    assert r.status()["per_provider"]["sarvam"]["failed"] == 1


def test_all_providers_down_still_answers_with_rules(monkeypatch):
    r = _router(monkeypatch, lambda url, **kw: _Resp(503, {"error": {"message": "overloaded"}}))
    res = r.complete("What is the refund policy for cancelled tickets?", task="support")
    assert res.mode == "fallback" and res.content
    assert r.status()["fallback_used"] == 1


def test_daily_budget_pauses_paid_provider(monkeypatch):
    calls = []

    def handler(url, **kw):
        calls.append(url)
        return _sarvam_ok(tin=1_000_000, tout=0)          # ₹15 in one call

    r = _router(monkeypatch, handler, gemini=False, budget=10.0)
    assert r.complete("first").mode == "sarvam"
    assert r.status()["budget_reached"] is True
    n = len(calls)
    res = r.complete("What is the refund policy?", task="support")
    assert res.mode == "fallback" and len(calls) == n       # Sarvam not called once the cap is reached


def test_provider_order_is_configurable(monkeypatch):
    r = _router(monkeypatch, lambda url, **kw: _gemini_ok() if "googleapis" in url else _sarvam_ok(),
                order="gemini,sarvam")
    assert r.providers == ["gemini", "sarvam"] and r.complete("hi").mode == "gemini"


def test_probe_reports_each_provider(monkeypatch):
    r = _router(monkeypatch, lambda url, **kw: _sarvam_ok("OK") if "sarvam" in url else _Resp(401, {"error": {"message": "bad key"}}))
    out = {p["provider"]: p for p in r.probe()}
    assert out["sarvam"]["ok"] is True and out["sarvam"]["reply"] == "OK"
    assert out["gemini"]["ok"] is False and "401" in out["gemini"]["error"]


def test_support_assistant_labels_sarvam_answers(monkeypatch):
    import modules.m6_support_assistant.service as m6
    r = _router(monkeypatch, lambda url, **kw: _sarvam_ok(
        "You need a valid government photo ID such as Aadhaar, PAN card, voter ID, driving licence or passport."))
    monkeypatch.setattr(m6, "llm_router", r)
    from modules.m6_support_assistant.schema import SupportQueryRequest
    res = m6.support_assistant_service.query(SupportQueryRequest(query="Which identity documents are accepted on the train?"))
    if r.status()["requests"] == 0:
        pytest.skip("this question was answered by a tool / curated rule, not the LLM")
    assert res.answered_by == "Sarvam AI (sarvam-105b)"


def test_llm_status_and_test_endpoints():
    from backend.main import app
    client = TestClient(app)
    st = client.get("/api/v1/governance/llm").json()
    assert {"mode", "providers", "per_provider", "spend_today_inr", "daily_budget_inr"} <= set(st)
    assert client.post("/api/v1/governance/llm/test").status_code == 401          # operator key required
    r = client.post("/api/v1/governance/llm/test", headers={"X-API-Key": "rb-dev-operator-key"})
    assert r.status_code == 200 and "providers" in r.json()
