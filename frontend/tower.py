"""
frontend/tower.py
─────────────────
AI Governance Control Tower — ONE scrollable page, ordered by importance:

   Command centre → Needs attention → Pillar status → 1 Incidents & RCA → 2 Model health & SLA →
   3 Drift & data quality → 4 Bias & fairness → 5 Hallucination → 6 Model lifecycle →
   7 Supervisor agent → 8 Agents → 9 Security & fraud → 10 Compliance & audit trail → 11 Chaos lab
"""

from __future__ import annotations

import json
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.ui import (
    API_BASE, api_get, api_post, badge, callout, engine_label, esc, gauge, hero, inject_theme,
    kpi, kpi_row, rows_card, section, style_fig, tone_for,
)

MODULE_NAMES = {
    "m1_ticket_predictor": "Ticket Predictor",
    "m2_train_recommender": "Train Recommender",
    "m3_delay_predictor": "Delay Predictor",
    "m4_travel_planner": "Travel Planner",
    "m5_fare_predictor": "Fare Forecaster",
    "m6_support_assistant": "Support Assistant",
    "m7_fraud_detector": "Fraud Detector",
}
SEV_ORDER = {"bad": 0, "warn": 1, "info": 2}


def mname(mid: str) -> str:
    return MODULE_NAMES.get(mid, mid)


def _df(rows: List[dict], **kw) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, **kw)


def _ts(s: Any) -> str:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).strftime("%d %b %H:%M:%S")
    except Exception:
        return str(s)[:19]


# ──────────────────────────────────────────────────────────────────────────────
# Live traffic generator (exercises the REAL models through the governance hooks)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_traffic(rounds: int, drifted: bool, quiet: bool = False) -> int:
    """Replay held-out historical requests (normal) or deliberately shifted ones through the governed services.
    Runs server-side, so it is fast and every request goes through the real governance hooks."""
    r = api_post("/governance/traffic/simulate", {"rounds": int(rounds), "shifted": bool(drifted)}, timeout=180)
    return int((r or {}).get("requests_ok", 0))


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

ANCHORS = [
    ("attention", "Needs attention"), ("pillars", "Pillars"), ("incidents", "1 Incidents"), ("quality", "2 Live quality"),
    ("health", "3 Health & resources"), ("drift", "4 Drift & data"), ("bias", "5 Bias"), ("halluc", "6 Hallucination"),
    ("lifecycle", "7 Lifecycle"), ("agent", "8 Agents"), ("security", "9 Security & fraud"),
    ("reports", "10 Compliance & audit"), ("chaos", "11 Chaos lab"),
]


def tower(pages: Dict[str, Any] = None) -> None:
    inject_theme("tower")
    hero(
        "🏛️ AI Governance Control Tower",
        "AI agents watching AI. Every model behind Railway Buddy is monitored for health, drift, bias and hallucination — "
        "problems are diagnosed, explained and fixed automatically, with humans approving the high-risk moves.",
        ["Autonomous supervisor", "Root-cause analysis", "Auto-retrain & rollback", "Full audit trail"],
    )

    top1, top2, top3 = st.columns([1.2, 1.2, 3])
    live = top1.toggle("Live auto-refresh", value=False, help="Refresh every 10 seconds")
    if top2.button("🔄 Refresh now", width="stretch"):
        st.rerun()
    top3.caption(f"Engine: **{engine_label()}** · API: `{API_BASE}` · Last updated {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    body = st.fragment(run_every=10 if live else None)(_tower_body)
    body(pages or {})


def _tower_body(pages: Dict[str, Any]) -> None:
    dash = api_get("/governance/supervisor/dashboard")
    if not dash:
        callout("<b>The governance backend isn't reachable.</b> It starts automatically with the app — wait a few seconds and press "
                "<b>Refresh now</b>. Locally, run <code>python run_platform.py</code>.", "bad", "⚠️")
        return

    health = dash.get("platform_health") or {}
    modules = health.get("modules") or {}
    score = float(dash.get("governance_score", 0))
    rca_sum = dash.get("rca_summary") or {}
    lifecycle = dash.get("lifecycle") or {}
    hall = dash.get("hallucination_stats") or {}
    incidents = api_get("/governance/incidents/open") or []
    drift_raw = api_get("/governance/drift/all")
    bias_raw = api_get("/governance/bias/audit/all")
    drift, bias = drift_raw or [], bias_raw or []
    sla = api_get("/governance/health/sla") or []
    hitl = dash.get("hitl_pending") or api_get("/governance/supervisor/hitl/pending") or []
    perf = dash.get("performance") or []
    degraded = [p for p in perf if p.get("degraded")]
    retrain_q = lifecycle.get("retraining_queue", [])
    events = api_get("/governance/telemetry", {"limit": 300}) or []
    sec_alerts = api_get("/security/alerts", {"n": 60}) or []
    blocked_n = sum(1 for a in sec_alerts if a.get("action") == "BLOCK_AND_REPORT")

    drifting = [d for d in drift if d.get("overall_drift_detected")]
    biased = [b for b in bias if not b.get("overall_fair", True)]
    sla_bad = [s for s in sla if not s.get("compliant", True)]
    open_cb = health.get("open_circuits", [])
    status = health.get("platform_status", "UNKNOWN")
    n_open = rca_sum.get("open_incidents", len(incidents))

    # ── Command centre ───────────────────────────────────────────────────────
    g, k = st.columns([1, 2.2], gap="large")
    with g:
        color = "#059669" if score >= 80 else "#D97706" if score >= 60 else "#DC2626"
        st.plotly_chart(gauge(score, "Governance score", "", color=color,
                              steps=[(0, 60, "#FEE2E2"), (60, 80, "#FEF3C7"), (80, 100, "#D1FAE5")], height=220),
                        width="stretch")
    with k:
        kpi_row([
            ("Platform", esc(status.title()), f"{len(modules)} of 7 modules reporting", tone_for(status)),
            ("Inferences", f'{health.get("total_requests", 0):,}', f'{health.get("overall_error_rate_pct", 0):.2f}% errors',
             "ok" if health.get("overall_error_rate_pct", 0) < 5 else "bad"),
            ("Open incidents", n_open, f'{rca_sum.get("critical_incidents", 0)} critical', "ok" if n_open == 0 else "bad" if rca_sum.get("critical_incidents") else "warn"),
        ])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        kpi_row([
            ("Drifting models", len(drifting), f"of {len(drift)} monitored", "ok" if not drifting else "bad"),
            ("Retraining queue", len(retrain_q), "awaiting retrain", "ok" if not retrain_q else "warn"),
            ("Supervisor", "Active" if dash.get("supervisor_running") else "Idle", f'{dash.get("hitl_pending_count", 0)} approvals pending',
             "ok" if dash.get("supervisor_running") else "warn"),
        ])

    st.markdown('<div class="rb-anchors">' + "".join(f'<a href="#{a}" target="_self">{t}</a>' for a, t in ANCHORS) + "</div>",
                unsafe_allow_html=True)

    # ── Needs attention ──────────────────────────────────────────────────────
    section("Needs your attention", "A single prioritised queue across every pillar.", "attention")
    items: List[tuple] = []
    for inc in incidents:
        p = inc.get("priority", "MEDIUM")
        items.append(("bad" if p == "CRITICAL" else "warn" if p == "HIGH" else "info", f"Incident {inc['incident_id']}",
                      f'{mname(inc.get("module_id", ""))}: {inc.get("primary_event_type", "").replace("_", " ").title()}'
                      f'{" ×" + str(inc.get("occurrences")) if inc.get("occurrences", 1) > 1 else ""}', "→ see Incidents & RCA"))
    for c in open_cb:
        items.append(("bad", "Circuit breaker OPEN", f"{mname(c)} is failing — traffic is being shed", "→ see Model health"))
    for s in sla_bad:
        items.append(("warn", "SLA breach", f'{mname(s["module_id"])}: ' + "; ".join(s.get("breaches", [])), "→ see Model health"))
    for d in drifting:
        items.append(("bad", "Drift detected", f'{mname(d["model_id"])} inputs no longer match training data', "→ see Drift"))
    for b in biased:
        items.append(("warn", "Fairness flag", f'{mname(b["module_id"])}: ' + "; ".join(b.get("adverse_impact_flags", [])[:1]), "→ see Bias"))
    if hall.get("total_quarantined"):
        items.append(("warn", "Responses quarantined", f'{hall["total_quarantined"]} support answers were blocked for weak grounding', "→ see Hallucination"))
    for p_ in degraded:
        items.append(("bad", "Quality dropped", f'{mname(p_["module_id"])}: {p_.get("status_reason", "")}', "→ see Live quality"))
    for m in retrain_q:
        items.append(("warn", "Retrain queued", f"{mname(m)} is waiting to be retrained", "→ see Lifecycle"))
    if blocked_n:
        items.append(("info", "Security", f"{blocked_n} bot / scalper sessions were blocked and reported recently", "→ see Security & fraud"))
    items.sort(key=lambda x: SEV_ORDER[x[0]])

    if hitl:
        for req in hitl:
            pl = req.get("payload") or {}
            ev = (pl.get("evidence") or {}).get("canary_analysis") or {}
            evtxt = ""
            if ev:
                evtxt = (f' · live {esc(ev.get("metric"))}: champion {esc(ev.get("champion", {}).get(ev.get("metric")))} '
                         f'vs challenger {esc(ev.get("challenger", {}).get(ev.get("metric")))}')
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f'<div class="rb-card" style="border-left:4px solid #F59E0B;margin:0">{badge("Approval needed", "warn")} '
                        f'<b>Promote {esc(mname(pl.get("model_id", "")))} {esc(pl.get("version", ""))} to 100% traffic</b> '
                        f'<span class="rb-sub">(from {esc(pl.get("from_version", "—"))}) · requested by {esc(req.get("requested_by", ""))} · '
                        f'{esc(req.get("request_id"))}{evtxt}</span></div>', unsafe_allow_html=True)
            if c2.button("✅ Approve", key=f'ap_{req["request_id"]}', width="stretch"):
                r = api_post("/governance/supervisor/hitl/decide", {"request_id": req["request_id"], "approved": True})
                if r and r.get("error"):
                    st.error(r["error"])
                st.rerun()
            if c3.button("❌ Reject", key=f'rj_{req["request_id"]}', width="stretch"):
                api_post("/governance/supervisor/hitl/decide", {"request_id": req["request_id"], "approved": False})
                st.rerun()
    if not items and not hitl:
        callout("<b>All clear.</b> No incidents, drift, bias flags or SLA breaches. The supervisor keeps watching.", "ok", "✅")
    for tone, title, text, link in items[:8]:
        st.markdown(f'<div class="rb-card" style="padding:10px 16px;margin-bottom:6px;border-left:4px solid {"#EF4444" if tone == "bad" else "#F59E0B" if tone == "warn" else "#60A5FA"}">'
                    f'{badge(title, tone)} &nbsp;{esc(text)} <span class="rb-sub" style="float:right">{link}</span></div>', unsafe_allow_html=True)
    if len(items) > 8:
        st.caption(f"+ {len(items) - 8} more — see the sections below.")

    # ── Pillar status ────────────────────────────────────────────────────────
    section("Eight governance pillars", "Each pillar is a working service — status is computed live from telemetry.", "pillars")
    valid = [e for e in events if "schema_valid" in e]
    dq_rate = (sum(1 for e in valid if e.get("schema_valid")) / len(valid) * 100) if valid else 100.0
    hq = hall.get("total_evaluations", 0)
    actions = dash.get("action_log_recent") or []
    pillars = [
        ("📊", "Model health", status, f'{health.get("total_requests", 0):,} requests · {len(open_cb)} open circuits'),
        ("🧪", "Data quality", "HEALTHY" if dq_rate >= 98 else "WARNING", f"{dq_rate:.1f}% of inputs passed validation"),
        ("📉", "Drift & performance", "UNAVAILABLE" if drift_raw is None else "DRIFT" if (drifting or degraded) else "STABLE",
         "Drift service not responding" if drift_raw is None else f"{len(drifting)} drifting · {len(degraded)} with lower live quality"),
        ("⚖️", "Bias & fairness", "BIAS_DETECTED" if biased else "FAIR", f"{len(bias)} modules audited"),
        ("🧠", "Hallucination guard", "WARNING" if hall.get("total_quarantined") else "STABLE", f'{hall.get("total_quarantined", 0)} of {hq} answers quarantined'),
        ("🚨", "Root-cause analysis", "INCIDENT" if n_open else "HEALTHY", f"{n_open} open incidents"),
        ("🔄", "Model lifecycle", "RETRAINING" if retrain_q else "HEALTHY", f'{lifecycle.get("total_model_versions", 0)} versions · {len(retrain_q)} queued'),
        ("🤖", "Supervisor agent", "ACTIVE" if dash.get("supervisor_running") else "IDLE", f"{len(actions)} recent autonomous actions"),
    ]
    for row in (pillars[:4], pillars[4:]):
        cols = st.columns(4)
        for col, (ico, t, st_, m) in zip(cols, row):
            col.markdown(f'<div class="rb-pillar"><div class="h"><span>{ico} {t}</span>{badge(st_.replace("_", " ").title())}</div>'
                         f'<div class="m">{esc(m)}</div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── 1 Incidents & RCA ───────────────────────────────────────────────────
    section("Incidents & root-cause analysis", "Faults are correlated, explained with 5-Why reasoning and paired with a remediation playbook.", "incidents", "1")
    if not incidents:
        callout("No open incidents. Use the <b>Chaos lab</b> at the bottom to see the loop in action.", "ok", "✅")
    for i_, inc in enumerate(incidents[:10]):
        p = inc.get("priority", "MEDIUM")
        tone = "bad" if p == "CRITICAL" else "warn" if p == "HIGH" else "info"
        with st.expander(f'{p} · {mname(inc.get("module_id", ""))} · {inc.get("primary_event_type", "").replace("_", " ").title()} · {inc["incident_id"]}',
                         expanded=(i_ == 0 and p == "CRITICAL")):
            a, b = st.columns(2)
            with a:
                st.markdown(f'{badge(p, tone)} &nbsp; opened {_ts(inc.get("timestamp"))} · seen **{inc.get("occurrences", 1)}×** · '
                            f'{inc.get("correlated_events_count", 0)} correlated events', unsafe_allow_html=True)
                if inc.get("evidence"):
                    st.markdown("**Evidence**")
                    for e_ in inc["evidence"]:
                        st.markdown(f"- {e_}")
                if inc.get("root_cause_module") and inc.get("root_cause_module") != inc.get("module_id"):
                    callout(f"<b>Traced upstream:</b> the likely root cause is <b>{esc(mname(inc['root_cause_module']))}</b>, "
                            f"which {esc(mname(inc.get('module_id', '')))} depends on.", "warn", "🧭")
                if inc.get("downstream_impact"):
                    st.caption("Downstream services affected: " + ", ".join(mname(m) for m in inc["downstream_impact"]))
                st.markdown("**Likely upstream causes**")
                for c in inc.get("inferred_causes", []):
                    st.markdown(f"- {c.replace('_', ' ').title()}")
                st.markdown("**5-Why analysis**")
                for i, w in enumerate(inc.get("five_whys", []), 1):
                    st.markdown(f"{i}. {w}")
            with b:
                callout(f'<b>Remediation playbook</b><br>{esc(inc.get("remediation_playbook", ""))}', "info", "💡")
                if st.button("✅ Mark resolved", key=f'res_{inc["incident_id"]}'):
                    api_post(f'/governance/incidents/{inc["incident_id"]}/resolve')
                    st.rerun()
    if len(incidents) > 10:
        st.caption(f"Showing 10 of {len(incidents)} open incidents.")
    if len(incidents) > 1 and st.button("Resolve all open incidents"):
        for inc in incidents:
            api_post(f'/governance/incidents/{inc["incident_id"]}/resolve')
        st.rerun()
    hist = [h for h in (api_get("/governance/incidents/history/all", {"n": 40}) or []) if h.get("resolved")]
    if hist:
        with st.expander(f"✔️ Recently resolved incidents ({len(hist)})"):
            _df([{"Resolved": _ts(h.get("resolved_at")), "Module": mname(h.get("module_id", "")),
                  "Incident": h.get("primary_event_type", "").replace("_", " ").title(), "How": h.get("resolution", "")}
                 for h in reversed(hist[-12:])])

    # ── 2 Live model quality (ground truth) ─────────────────────────────────
    section("Live model quality", "Measured on real outcomes (final PNR status, actual delay, fare paid, analyst verdicts, "
            "chosen trains, satisfaction ratings) — this is how concept drift is caught.", "quality", "2")
    if perf:
        rows = []
        for p_ in perf:
            m = p_.get("metrics") or {}
            key = p_.get("primary_metric")
            rows.append({"Module": mname(p_["module_id"]), "Outcome measured": p_.get("outcome"),
                         "Outcomes": p_.get("n_outcomes", 0), "Live": ", ".join(f"{k}={v}" for k, v in m.items() if v is not None) or "—",
                         "Baseline": str(p_.get("baseline")) if p_.get("baseline") is not None else (
                             "FPR ≤ 12% · FNR ≤ 15%" if p_["module_id"] == "m7_fraud_detector" else "—"), "Status": "🔴 degraded" if p_.get("degraded") else ("⏳ waiting" if p_.get("n_outcomes", 0) < 30 else "✅ ok"),
                         "Why": p_.get("status_reason", ""), "Sources": ", ".join(p_.get("sources", []))})
        _df(rows)
        canaries = [(p_, v) for p_ in perf for v in (p_.get("by_version") or {}) if len(p_.get("by_version") or {}) > 1]
        if canaries:
            st.markdown("**Champion vs canary on live outcomes**")
            _df([{"Module": mname(p_["module_id"]), "Version": v, **{k: x for k, x in p_["by_version"][v].items()}} for p_, v in canaries])
        st.caption("Outcomes arrive from passengers (the feedback boxes on each service page), operators and analysts. "
                   "In this demo you can also generate simulated outcomes in the Chaos lab — they are labelled 'simulated'.")
    else:
        callout("No outcomes yet.", "info", "ℹ️")

    # ── 2 Model health & SLA ────────────────────────────────────────────────
    section("Model health & SLA", "Latency percentiles, error rates, circuit breakers and resource use per AI module (SLA: P95 < 3000 ms, errors < 5%).", "health", "3")
    res_ = health.get("resources") or {}
    if res_:
        kpi_row([("Process CPU", f'{res_.get("process_cpu_pct", 0):.0f}%', f'{res_.get("host_cpus", "?")} CPUs on host', ""),
                 ("Process memory", f'{res_.get("process_memory_mb", 0):,.0f} MB', f'host memory {res_.get("host_memory_pct", "—")}%', ""),
                 ("Threads", res_.get("process_threads", "—"), "API process", ""),
                 ("Open circuits", len(open_cb), "requests shed fast (503) while open", "ok" if not open_cb else "bad")])
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if modules:
        rows = []
        for mid, s in modules.items():
            rows.append({"Module": mname(mid), "Requests": s.get("total_requests", 0), "Error %": s.get("error_rate_pct", 0),
                         "Avg ms": s.get("avg_latency_ms", 0), "P95 ms": s.get("p95_ms", 0), "P99 ms": s.get("p99_ms", 0),
                         "CPU ms/req": s.get("avg_cpu_ms", 0), "Uptime %": s.get("uptime_pct", 100),
                         "Circuit": s.get("circuit_state", "—"), "Times opened": s.get("circuit_opened_count", 0),
                         "SLA": "✅" if not any(x["module_id"] == mid for x in sla_bad) else "🔴"})
        l, r = st.columns([1.5, 1])
        with l:
            _df(rows)
        with r:
            names = [mname(m) for m in modules]
            p95 = [modules[m].get("p95_ms", 0) for m in modules]
            fig = go.Figure(go.Bar(x=p95, y=names, orientation="h", marker_color="#3B82F6", text=[f"{v:.0f} ms" for v in p95], textposition="auto"))
            fig.add_vline(x=3000, line_dash="dash", line_color="#DC2626", annotation_text="SLA 3000 ms")
            fig.update_layout(title="<b>P95 latency vs SLA</b>")
            style_fig(fig, 300)
            st.plotly_chart(fig, width="stretch")
        # resource utilisation proxy from telemetry
        if events:
            ev = pd.DataFrame(events)
            if "latency_ms" in ev and "timestamp" in ev:
                ev["t"] = pd.to_datetime(ev["timestamp"], errors="coerce")
                ev = ev.dropna(subset=["t"]).sort_values("t")
                if len(ev) > 5:
                    fig = go.Figure()
                    for mod, grp in ev.groupby("module"):
                        fig.add_trace(go.Scatter(x=grp["t"], y=grp["latency_ms"], mode="lines+markers", name=mname(mod),
                                                 line=dict(width=1.4), marker=dict(size=5)))
                    fig.update_layout(title="<b>Live latency stream</b> (all modules)", yaxis_title="ms")
                    style_fig(fig, 260)
                    st.plotly_chart(fig, width="stretch")
    else:
        callout("No traffic yet — use <b>Generate live traffic</b> in the Chaos lab (or open any Passenger service).", "info", "ℹ️")

    # ── 3 Drift & data quality ──────────────────────────────────────────────
    section("Drift & data quality", "Inputs are validated before inference; live feature distributions are compared with the training data (KS test with p-value + small-sample-corrected PSI).", "drift", "4")
    q1, q2 = st.columns([1, 2.4])
    with q1:
        st.markdown(kpi("Input validation pass rate", f"{dq_rate:.1f}%", f"{len(valid)} recent inferences checked", "ok" if dq_rate >= 98 else "warn"), unsafe_allow_html=True)
    with q2:
        dqv = [e for e in events if e.get("event_type") == "DATA_QUALITY_VIOLATION"]
        blocked = sum(1 for e in dqv if (e.get("payload") or {}).get("blocking"))
        callout("<b>Data-quality gate</b> checks required fields, ranges, corrupted values (NaN / infinity, malformed train numbers, "
                f"impossible dates) and outliers <i>before</i> the model runs. Recent: <b>{len(dqv)}</b> flagged, "
                f"<b>{blocked}</b> blocked with HTTP 422 so no unreliable prediction was served.", "info", "🧪")
    if drift_raw is None:
        callout("The drift service did not respond. Press <b>Refresh now</b>; if it persists check the API logs.", "bad", "⚠️")
    for d in drift:
        feats = [f for f in d.get("feature_drift", []) if "feature" in f]
        waiting = [f for f in d.get("feature_drift", []) if f.get("status") == "INSUFFICIENT_DATA"]
        is_d = bool(d.get("overall_drift_detected"))
        label = f'{mname(d["model_id"])} — {"DRIFT DETECTED" if is_d else "stable"}'
        with st.expander(label, expanded=is_d):
            if feats:
                rows = [{"Feature": f["feature"].replace("_", " ").title(), "KS statistic": f.get("ks_statistic"), "PSI": f.get("psi"),
                         "PSI level": f.get("psi_severity"), "Mean shift %": f.get("mean_shift_pct"),
                         "Drift": "🔴 yes" if f.get("drift_detected") else "✅ no"} for f in feats]
                _df(rows)
                fig = go.Figure(go.Bar(x=[r["Feature"] for r in rows], y=[r["PSI"] or 0 for r in rows],
                                       marker_color=["#DC2626" if r["Drift"].startswith("🔴") else "#60A5FA" for r in rows]))
                fig.add_hline(y=0.2, line_dash="dash", line_color="#DC2626", annotation_text="PSI 0.20 = significant")
                fig.update_layout(title="<b>Population Stability Index per feature</b>")
                style_fig(fig, 240)
                st.plotly_chart(fig, width="stretch")
            elif waiting:
                n = max(f.get("n_samples", 0) for f in waiting)
                st.progress(min(1.0, n / 30), text=f"Collecting samples… {n}/30 needed for a statistical test")
            else:
                st.caption("No baseline features are registered for this model yet.")
    if not drift:
        callout("Drift windows are empty — generate traffic below.", "info", "ℹ️")
    beh = api_get("/governance/behaviour") or []
    if beh:
        st.markdown("**User-behaviour change** — has the mix of classes, quotas and traveller types people ask about shifted from history?")
        rows = []
        for b_ in beh:
            for f in b_.get("fields", []):
                if f.get("status") == "INSUFFICIENT_DATA":
                    continue
                ch = "; ".join(f'{c["category"]}: {c["baseline_pct"]}% → {c["live_pct"]}%' for c in f.get("largest_changes", []))
                rows.append({"Service": mname(b_["model_id"]), "Field": f["field"], "PSI": f.get("psi"),
                             "Shift": "🔴 yes" if f.get("shift_detected") else "✅ no", "Largest changes": ch})
        if rows:
            _df(rows)
        else:
            st.caption("Collecting requests — each field needs enough samples before its mix can be compared with history.")

    # ── 4 Bias & fairness ───────────────────────────────────────────────────
    section("Bias & fairness", "Disparate Impact Ratio (≥ 0.80), Demographic Parity gap (≤ 0.10) and Equal Opportunity gap (needs outcomes), for protected groups: age, disability, gender. Groups need ≥ 20 observations.", "bias", "5")
    if bias:
        for b in bias:
            fair = b.get("overall_fair", True)
            with st.expander(f'{mname(b["module_id"])} — {"fair" if fair else "BIAS FLAGGED"} · fairness score {b.get("fairness_score", "—")}/100', expanded=not fair):
                for f in b.get("adverse_impact_flags", []):
                    callout(esc(f), "warn", "⚠️")
                rows = []
                for a in b.get("attributes_audited", []):
                    rows.append({"Protected attribute": a["attribute"].replace("_", " ").title(), "DIR": a.get("dir"), "DPD": a.get("dpd"),
                                 "EOD": a.get("eod"), "Groups": ", ".join(f'{g} (n={m.get("n_samples", "?")})' for g, m in a.get("groups", {}).items()),
                                 "Result": "✅ fair" if not a.get("adverse_impact") else "⚠️ review"})
                _df(rows)
    else:
        callout("No fairness observations yet. Passenger requests carry protected attributes (senior citizen, divyangjan, gender); "
                "they appear here as soon as traffic flows — use <b>Generate live traffic</b> in the Chaos lab.", "info", "ℹ️")

    # ── 5 Hallucination ─────────────────────────────────────────────────────
    section("Hallucination & response quality", "Every Support Assistant answer is scored for content-word and numeric-fact support against the retrieved policy text, and checked for contradictions (e.g. 'refundable' vs 'non-refundable'); weak answers are flagged.", "halluc", "6")
    tot, quar = hall.get("total_evaluations", 0), hall.get("total_quarantined", 0)
    kpi_row([("Answers evaluated", tot, "", ""), ("Quarantined", quar, "weak grounding", "ok" if not quar else "warn"),
             ("Quarantine rate", f"{(quar / tot * 100):.1f}%" if tot else "0%", "", "ok" if not tot or quar / tot < 0.2 else "warn"),
             ("Avg faithfulness", f'{(sum(m.get("avg_faithfulness", 0) for m in (hall.get("modules") or {}).values()) / max(1, len(hall.get("modules") or {}))):.0%}',
              "target ≥ 65%", "")])
    recent = api_get("/governance/hallucination/recent", {"n": 15}) or []
    if recent:
        rows = [{"Time": _ts(e.get("timestamp", "")), "Query": (e.get("query") or "")[:70], "Faithfulness": round(e.get("faithfulness_score", 0), 2),
                 "Quality score": round(e.get("composite_score", 0), 2), "Contradictions": len(e.get("contradictions_detected") or []),
                 "Blocked": "🔴 yes" if e.get("quarantine") else "✅ no", "Severity": e.get("severity", "—")} for e in recent]
        _df(rows)
    else:
        st.caption("Ask the Support Assistant a question to populate this table.")

    # ── 6 Lifecycle ─────────────────────────────────────────────────────────
    section("Model lifecycle", "Retraining on the fixed hold-out set (+ recent outcomes), minimum-gain rule, 10% live canary, human-approved promotion, one-click rollback.", "lifecycle", "7")
    mids = lifecycle.get("models_registered", [])
    prod, chal = lifecycle.get("production_versions", {}), lifecycle.get("challenger_versions", {})
    rows = []
    for mid in mids:
        st_ = api_get(f"/governance/lifecycle/model/{mid}") or {}
        vers = {v["version"]: v for v in st_.get("versions", [])}
        pv = vers.get(prod.get(mid, ""), {})
        met = pv.get("metrics", {})
        can = (lifecycle.get("canary") or {}).get(mid)
        rows.append({"Model": mname(mid), "Production": prod.get(mid, "—"),
                     "Canary": f'{can["version"]} @ {can["traffic_pct"]}%' if can else "—",
                     "Versions": st_.get("total_versions", 0), "Key metrics": ", ".join(f"{k}: {v}" for k, v in list(met.items())[:3]) or "—",
                     "Retrain queued": "⚠️ yes" if mid in retrain_q else "no"})
    _df(rows)
    l1, l2 = st.columns(2)
    with l1:
        with st.container(border=True):
            st.markdown("**🔁 Retrain a model**")
            rm = st.selectbox("Model", mids, format_func=mname, key="rt_model") if mids else None
            if st.button("Retrain now", disabled=not mids, key="rt_btn", width="stretch"):
                with st.spinner("Retraining on fresh data and evaluating against the champion…"):
                    r = api_post("/governance/lifecycle/retrain", {"model_id": rm})
                if r and r.get("success") and r.get("outcome") == "not_applicable":
                    st.info(r.get("detail", "Not applicable."))
                elif r and r.get("success"):
                    (st.success if r.get("outcome") == "challenger_better" else st.info)(
                        f'{mname(rm)}: {r.get("outcome", "").replace("_", " ")} → {r.get("new_version", "")}. {r.get("decision", "")}')
                    for k_, v_ in (r.get("comparison") or {}).items():
                        st.caption(f"{k_}: {v_}")
                    st.caption(r.get("next_step", ""))
                else:
                    st.error((r or {}).get("error", "Retraining failed."))
            if rm and chal.get(rm):
                st.caption(f'Challenger **{chal[rm]}** serves 10% of live traffic as a canary.')
                p1, p2 = st.columns(2)
                if p1.button("🚀 Request promotion", key="pr_btn", width="stretch",
                             help="Promotion to 100% traffic is risk-increasing, so it waits for an admin approval (see Needs attention)."):
                    r = api_post("/governance/lifecycle/promote", {"model_id": rm})
                    if r and r.get("success"):
                        st.success(f'Approval requested ({r.get("request_id")}). Approve it under "Needs your attention".')
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error((r or {}).get("error", "Request failed."))
                if p2.button("⏹ Stop canary", key="sc_btn", width="stretch"):
                    r = api_post("/governance/lifecycle/canary/stop", {"model_id": rm})
                    (st.success if r and r.get("success") else st.error)(r.get("stopped_version", r.get("error", "")) if r else "failed")
                    st.rerun()
    with l2:
        with st.container(border=True):
            st.markdown("**⏪ Roll back a model**")
            rb = st.selectbox("Model", mids, format_func=mname, key="rb_model") if mids else None
            reason = st.text_input("Reason", "operator_request", key="rb_reason")
            if st.button("Rollback to previous version", disabled=not mids, key="rb_btn", width="stretch"):
                r = api_post("/governance/supervisor/rollback", {"model_id": rb, "reason": reason})
                if r and r.get("success"):
                    st.success(f'Rolled back {mname(rb)} to {r.get("restored_version")}')
                else:
                    st.error((r or {}).get("error", "Rollback failed."))
    rh = api_get("/governance/lifecycle/retrain-history", {"n": 12}) or []
    if rh:
        st.markdown("**Retraining history** — every run is kept, accepted or not, with the reason")
        _df([{"When": _ts(h.get("at")), "Model": mname(h.get("model_id", "")), "Version": h.get("version", "—"),
              "Trigger": h.get("reason", ""), "Decision": "✅ canary" if h.get("accepted") else ("❌ failed" if h.get("failed") else "kept champion"),
              "Why": h.get("why", ""), "Production rows used": h.get("production_rows_used", 0),
              "Seconds": h.get("train_seconds")} for h in reversed(rh)])

    # ── 7 Supervisor agent ──────────────────────────────────────────────────
    section("Agents monitoring agents", "The supervisor polls every pillar each 30 s, retrains on concept drift, stops bad canaries, "
            "requests approval for promotions and closes incidents once their signal clears. The planner agent is itself monitored.", "agent", "8")
    am = api_get("/agent/metrics") or {}
    kpi_row([("Supervisor cycles", dash.get("cycles_completed", 0), f'last {_ts(dash.get("last_cycle_at"))}' if dash.get("last_cycle_at") else "", "ok"),
             ("Planner plans", am.get("plans", 0), f'{am.get("blocked_injections", 0)} injections blocked', ""),
             ("Planner approval rate", f'{am["approval_rate"]:.0%}' if am.get("approval_rate") is not None else "—", "passenger decisions", ""),
             ("Agent anomalies", len(am.get("recent_anomalies") or []), "slow plans, tool overuse, rejection streaks",
              "warn" if am.get("recent_anomalies") else "ok")])
    if am.get("recent_anomalies"):
        _df([{"Time": _ts(a.get("at")), "Anomaly": a.get("kind", "").replace("_", " "),
              "Detail": ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("kind", "at"))} for a in reversed(am["recent_anomalies"])])
    if actions:
        _df([{"Time": _ts(a.get("timestamp", "")), "Action": str(a.get("action_type", "")).replace("_", " ").title(), "Module": mname(a.get("module_id", "")),
              "Trigger": str(a.get("trigger", "")).replace("_", " ").title(), "Executed by": a.get("executor", ""), "Outcome": a.get("outcome", "")} for a in reversed(actions[-12:])])
    else:
        callout("No autonomous actions yet. They appear when the supervisor detects drift, SLA breaches or incidents — try the Chaos lab.", "info", "🤖")

    # ── 9 Security & fraud operations ───────────────────────────────────────
    section("Security & fraud operations", "Module 7 scores every booking session for bots, scalpers and account misuse; "
            "analyst verdicts measure its live false-positive and false-negative rates.", "security", "9")
    m7 = modules.get("m7_fraud_detector") or {}
    m7p = next((p_ for p_ in perf if p_.get("module_id") == "m7_fraud_detector"), {})
    m7m = m7p.get("metrics") or {}
    fpr, fnr = m7m.get("fpr"), m7m.get("fnr")
    kpi_row([("Sessions scored", f'{m7.get("total_requests", 0):,}', "through the governed model", ""),
             ("Alerts raised", len(sec_alerts), f"{blocked_n} blocked & reported", "warn" if sec_alerts else "ok"),
             ("False-positive rate", f"{fpr:.1%}" if fpr is not None else "—",
              f'{m7p.get("n_outcomes", 0)} analyst verdicts · limit 12%', "bad" if fpr is not None and fpr > 0.12 else "ok"),
             ("False-negative rate", f"{fnr:.1%}" if fnr is not None else "—", "missed threats · limit 15%",
              "bad" if fnr is not None and fnr > 0.15 else "ok")])
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if sec_alerts:
        _df([{"Time": _ts(a.get("at")), "Session": a.get("session_id"), "Category": a.get("category"),
              "Action": str(a.get("action", "")).replace("_", " ").title(), "Risk": a.get("risk_score"),
              "Evidence": ", ".join(a.get("why", [])[:4])} for a in sec_alerts[:12]])
    else:
        callout("No security alerts yet. Score a session in the Fraud & Bot Shield, or send <b>Shifted traffic</b> from the Chaos lab "
                "(it includes bot-like sessions).", "info", "🛡️")
    if pages.get("fraud"):
        st.page_link(pages["fraud"], label="Open the Fraud & Bot Shield to score and investigate a session", icon="🛡️")

    # ── 10 Reports & audit ──────────────────────────────────────────────────
    section("Compliance, reports & audit trail", "Live ethical-AI checks with evidence, a tamper-evident audit trail that survives restarts, "
            "and downloadable governance reports.", "reports", "10")
    comp = api_get("/governance/compliance") or {}
    ver = api_get("/governance/audit/verify") or {}
    if comp:
        sm = comp.get("summary", {})
        kpi_row([("Compliance checks", f'{sm.get("PASS", 0)} pass', f'{sm.get("WARN", 0)} warn · {sm.get("FAIL", 0)} fail', tone_for(comp.get("overall", ""))),
                 ("Audit chain", "Intact" if ver.get("intact") else "BROKEN", f'{ver.get("entries_checked", 0):,} entries verified', "ok" if ver.get("intact") else "bad"),
                 ("Audit storage", "Durable" if ver.get("durable") else "Memory", "SQLite, hash-chained", "ok" if ver.get("durable") else "warn")])
        _df([{"Check": c["check"], "Principle": c["principle"], "Status": {"PASS": "✅ pass", "WARN": "⚠️ warn", "FAIL": "🔴 fail"}[c["status"]],
              "Evidence": c["evidence"]} for c in comp.get("checks", [])])
    rc1, rc2 = st.columns([1, 3])
    if rc1.button("📋 Generate report", width="stretch", type="primary"):
        with st.spinner("Auditing all pillars…"):
            rep = api_post("/governance/supervisor/report")
        if rep and "report_id" in rep:
            st.session_state["report"] = rep
    rep = st.session_state.get("report")
    if rep:
        tone = tone_for("CRITICAL" if rep.get("overall_risk_level") == "HIGH" else "WARNING" if rep.get("overall_risk_level") == "MEDIUM" else "HEALTHY")
        st.markdown(f'<div class="rb-card"><h4>{esc(rep["report_id"])}</h4>Overall risk {badge(rep.get("overall_risk_level", "—"), tone)} · '
                    f'platform {badge(rep.get("platform_status", "—"))} · generated {_ts(rep.get("generated_at"))}</div>', unsafe_allow_html=True)
        for f in rep.get("critical_flags", []):
            callout(esc(f), "bad", "🚨")
        for r_ in rep.get("recommendations", [])[:6]:
            st.markdown(f'**P{r_["priority"]}** · {badge(r_["urgency"])} {esc(r_["action"])}', unsafe_allow_html=True)
        md = f'# Governance report {rep["report_id"]}\n\n- Risk: {rep.get("overall_risk_level")}\n- Platform: {rep.get("platform_status")}\n- Generated: {rep.get("generated_at")}\n\n## Critical flags\n' + \
             ("\n".join(f"- {f}" for f in rep.get("critical_flags", [])) or "None") + "\n\n## Recommendations\n" + \
             "\n".join(f'{r_["priority"]}. [{r_["urgency"]}] {r_["action"]}' for r_ in rep.get("recommendations", []))
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇ Markdown", md, f'{rep["report_id"]}.md', width="stretch")
        d2.download_button("⬇ JSON", json.dumps(rep, indent=2, default=str), f'{rep["report_id"]}.json', width="stretch")
    with st.expander("🗄️ Durable audit log (personal data masked)"):
        kind = st.selectbox("Entry type", ["governance_action", "hitl_request", "hitl_decision", "booking", "conversation",
                                            "feedback", "inference", "plan_rejected"], key="aud_kind")
        al = api_get("/governance/audit/log", {"kind": kind, "limit": 25}) or {}
        st.caption("Entries by type: " + ", ".join(f"{k}: {v:,}" for k, v in (al.get("counts") or {}).items()))
        _df([{"Time": _ts(e["ts"]), "Module": mname(e.get("module") or ""), "Ref": e.get("ref"), "Actor": e.get("actor"),
              "Payload": json.dumps(e.get("payload"), default=str)[:160], "Hash": e.get("hash", "")[:12]} for e in al.get("entries", [])])
    with st.expander("📜 Live telemetry stream (most recent governed events)"):
        if events:
            ev = pd.DataFrame(events).tail(25).iloc[::-1]
            cols = [c for c in ["timestamp", "module", "event_type", "severity", "latency_ms", "confidence_score", "schema_valid", "event_id"] if c in ev]
            ev = ev[cols].copy()
            if "module" in ev:
                ev["module"] = ev["module"].map(mname)
            st.dataframe(ev, width="stretch", hide_index=True)
        else:
            st.caption("No events yet.")

    # ── 9 Chaos lab ─────────────────────────────────────────────────────────
    section("Chaos lab — watch the loop work", "Create traffic, report outcomes or inject faults, then scroll up to see detection → root cause → action.", "chaos", "11")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**🚦 Generate live traffic**")
            st.caption("Normal = replays held-out historical requests (should NOT drift). Shifted = very long waitlists, storms, "
                       "last-minute bookings and bot sessions (should drift).")
            n = st.slider("Passenger sessions", 20, 120, 60, step=10, key="sim_n")
            a, b = st.columns(2)
            if a.button("Normal traffic", width="stretch", key="sim_ok"):
                done = simulate_traffic(n, drifted=False)
                st.success(f"{done} governed inferences recorded.")
                time.sleep(0.6)
                st.rerun()
            if b.button("⚠️ Shifted traffic", width="stretch", key="sim_drift",
                        help="Longer waitlists, bad weather, bot-like sessions → should trigger drift detection"):
                done = simulate_traffic(n, drifted=True)
                st.warning(f"{done} inferences recorded with shifted input distributions.")
                time.sleep(0.6)
                st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("**💥 Inject a fault**")
            mod = st.selectbox("Target module", list(MODULE_NAMES), format_func=mname, key="ch_mod")
            kind = st.selectbox("Fault", [("latency_spike", "Latency spike (5 s responses)"), ("error_surge", "Error surge (8 failures)"),
                                          ("drift_simulation", "Data drift event")], format_func=lambda x: x[1], key="ch_kind")
            a, b = st.columns(2)
            if a.button("Inject fault", type="primary", width="stretch", key="ch_go"):
                r = api_post("/governance/supervisor/chaos", {"module_id": mod, "chaos_type": kind[0]})
                if r and r.get("success"):
                    st.success(r.get("message", "Injected."))
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(str(r)[:200])
            if b.button("Reset monitors", width="stretch", key="ch_reset"):
                api_post("/governance/supervisor/reset")
                st.rerun()
    o1, o2 = st.columns(2, gap="large")
    with o1:
        with st.container(border=True):
            st.markdown("**📬 Report outcomes (simulated ground truth)**")
            st.caption("Gives every recent prediction its 'real' outcome. Normal = the world behaves as the models learned. "
                       "Concept drift = same inputs, different outcomes (e.g. delays now much longer) → quality drops → "
                       "the supervisor retrains with these outcomes and starts a canary.")
            a, b = st.columns(2)
            if a.button("Normal outcomes", width="stretch", key="out_ok"):
                r = api_post("/governance/outcomes/simulate", {"mode": "normal"})
                st.success(f'{(r or {}).get("total", 0)} outcomes recorded.')
                time.sleep(0.5)
                st.rerun()
            if b.button("⚠️ Concept drift", width="stretch", key="out_drift"):
                r = api_post("/governance/outcomes/simulate", {"mode": "concept_drift"})
                st.warning(f'{(r or {}).get("total", 0)} outcomes recorded under a changed world.')
                time.sleep(0.5)
                st.rerun()
    with o2:
        with st.container(border=True):
            st.markdown("**🤖 Run the supervisor now**")
            st.caption("The supervisor runs every 30 seconds on its own; this triggers one cycle immediately.")
            if st.button("Run governance cycle", width="stretch", key="cyc", type="primary"):
                with st.spinner("Supervisor is checking every pillar…"):
                    r = api_post("/governance/supervisor/cycle", {})
                st.success(f'Cycle complete — governance score {(r or {}).get("governance_score", "—")}.')
                time.sleep(0.6)
                st.rerun()
