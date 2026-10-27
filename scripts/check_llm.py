"""
scripts/check_llm.py
====================
Checks the configured LLM engines with your real keys from .env — without printing any key.

    python scripts/check_llm.py

1. Shows which providers are configured and their failover order.
2. Sends a tiny test request to each provider (costs a fraction of a paisa).
3. Asks the Support Assistant one real question end-to-end (retrieval → LLM → hallucination check)
   and shows which engine answered, the grounding score, latency, tokens and ₹ cost.

Exit code 0 = the primary provider works; 1 = it does not (the app still answers via failover / rules).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # settings read .env from the working directory
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # ₹ and ✅ on Windows terminals
    except Exception:
        pass
    try:
        from core.config import settings
        from core.llm_router import llm_router
    except ImportError as e:
        print(f"Missing dependency ({e}). Run:  pip install -r requirements.txt")
        return 1

    print("LLM engine check")
    print("─" * 60)
    key = settings.sarvam_api_key.strip()
    print(f"SARVAM_API_KEY : {'set (' + str(len(key)) + ' chars)' if key else 'NOT SET'}   model: {settings.sarvam_model}")
    print(f"GEMINI_API_KEY : {'set' if settings.gemini_api_key.strip() else 'not set'}")
    print(f"Failover order : {' → '.join(llm_router.providers + ['rule-based engine'])}")
    print(f"Daily budget   : ₹{settings.llm_daily_budget_inr:.0f}")
    if not llm_router.providers:
        print("\nNo LLM provider is configured — only the rule-based engine will answer.")
        return 1

    print("\n1) Connectivity test (tiny request to each provider)")
    primary_ok = False
    for r in llm_router.probe():
        if r["ok"]:
            print(f"   ✅ {r['label']:<12} {r.get('model', ''):<22} {r['latency_ms']:>7.0f} ms   reply: {r.get('reply', '')!r}")
        else:
            print(f"   ❌ {r['label']:<12} {r.get('error', '')}")
            hint = r.get("error", "")
            if "401" in hint or "403" in hint:
                print("      → the key was rejected: check SARVAM_API_KEY in .env (no quotes or spaces) and your credit balance.")
            elif "network" in hint:
                print("      → could not connect: check your internet / firewall / proxy.")
        if r["provider"] == llm_router.providers[0]:
            primary_ok = bool(r["ok"])

    print("\n2) Support Assistant, end to end")
    from modules.m6_support_assistant.schema import SupportQueryRequest
    from modules.m6_support_assistant.service import support_assistant_service
    q = "If my waitlisted ticket is not confirmed when the chart is prepared, what happens to my money?"
    t0 = time.perf_counter()
    res = support_assistant_service.query(SupportQueryRequest(query=q, user_id="llm_check"))
    ms = (time.perf_counter() - t0) * 1000
    print(f"   Q: {q}")
    print(f"   A: {res.response[:600]}")
    print(f"   answered by: {res.answered_by} · grounded {res.faithfulness_score:.0%}"
          f"{' (flagged)' if res.hallucination_flag else ''} · {ms:.0f} ms")

    st = llm_router.status()
    print("\n3) Usage this run")
    for p, s in st["per_provider"].items():
        print(f"   {s['label']:<12} calls {s['calls']}  ok {s['ok']}  failed {s['failed']}  "
              f"tokens {s['tokens_in']}/{s['tokens_out']}  cost ₹{s['cost_inr']:.4f}")
    print(f"   Spent today: ₹{st['spend_today_inr']:.4f} of ₹{st['daily_budget_inr']:.0f}")

    print("\nResult:", "✅ primary engine works" if primary_ok else "❌ primary engine failed — see above")
    return 0 if primary_ok else 1


if __name__ == "__main__":
    sys.exit(main())
