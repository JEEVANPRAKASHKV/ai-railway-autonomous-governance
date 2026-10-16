"""
frontend/tower.py
─────────────────
AI Governance Control Tower — ONE scrollable page, ordered by importance:

   Command centre → Needs attention → Pillar status → 1 Incidents & RCA → 2 Model health & SLA →
   3 Drift & data quality → 4 Bias & fairness → 5 Hallucination → 6 Model lifecycle →
   7 Supervisor agent → 8 Reports & audit trail → 9 Chaos lab
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, **kw)


def _ts(s: Any) -> str:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).strftime("%d %b %H:%M:%S")
    except Exception:
        return str(s)[:19]


# ──────────────────────────────────────────────────────────────────────────────
# Live traffic generator (exercises the REAL models through the governance hooks)
# ──────────────────────────────────────────────────────────────────────────────

SUPPORT_QS = [
    "How do I check my PNR status?", "What is the refund if my waitlist ticket is not confirmed?",
    "Can I transfer my ticket to a family member?", "What are the Tatkal booking rules?",
    "Kya RAC passenger 3A coach me travel kar sakta hai?", "तत्काल टिकट रद्द करने पर रिफंड मिलता है?",
]


def simulate_traffic(rounds: int, drifted: bool, quiet: bool = False) -> int:
    """Send realistic passenger traffic through the live API so every governance pillar sees real data."""
    from frontend.ui import load_reference
    trains = list(load_reference()["trains"].values())[:40]
    stations = ["BPL", "NDLS", "CSTM", "HWH", "SBC", "MAS", "LKO"]
    personas = ["budget", "premium", "student", "senior", "business", "family"]
    classes = ["3A", "2A", "1A", "SL", "CC"]
    rnd = random.Random(time.time_ns())
    ok = 0
    bar = None if quiet else st.progress(0.0, text="Sending live passenger traffic through the AI modules…")
    for i in range(rounds):
        t = rnd.choice(trains)
        jd = date.today() + timedelta(days=rnd.randint(1, 60))
        calls = [
            ("/predict/ticket", {
                "train_number": t, "journey_class": rnd.choice(classes), "quota": rnd.choice(["GN", "GN", "TQ", "LADIES", "SENIOR"]),
                "journey_date": jd.isoformat(), "booking_date": date.today().isoformat(),
                "current_wl_number": rnd.randint(150, 320) if drifted else max(1, int(rnd.lognormvariate(2.6, 0.8))),
                "seniority_flag": rnd.random() < 0.25, "disability_flag": rnd.random() < 0.06}),
            ("/predict/delay", {
                "train_number": t, "station_code": rnd.choice(stations), "journey_date": jd.isoformat(),
                "weather_index": round(rnd.uniform(0.75, 1.0) if drifted else rnd.uniform(0.0, 0.45), 2),
                "congestion_index": round(rnd.uniform(0.7, 1.0) if drifted else rnd.uniform(0.1, 0.6), 2),
                "distance_from_origin_km": round(rnd.uniform(300, 2200), 0)}),
            ("/predict/fare", {"train_number": t, "journey_class": rnd.choice(classes), "journey_date": jd.isoformat(),
                               "tatkal_flag": rnd.random() < 0.2}),
            ("/security/score", {
                "session_id": f"SIM-{i}-{rnd.randint(1000, 9999)}", "ip_hash": "sim", "booking_time_hour": rnd.randint(6, 23),
                "requests_per_minute": round(rnd.uniform(70, 140) if drifted else abs(rnd.gauss(8, 5)) + 1, 1),
                "unique_pnr_attempts": rnd.randint(8, 30) if drifted else rnd.randint(1, 4),
                "account_age_days": rnd.randint(0, 20) if drifted else rnd.randint(30, 1500),
                "payment_attempts": rnd.randint(3, 8) if drifted else 1,
                "failed_captcha_count": rnd.randint(2, 7) if drifted else 0,
                "tatkal_flag": drifted, "is_bulk_booking": drifted and rnd.random() < 0.5, "ip_risk_score": round(rnd.random(), 2)}),
            ("/recommend/trains", {"source_station": "NDLS", "destination_station": "CSTM", "journey_date": jd.isoformat(),
                                   "persona": rnd.choice(personas), "preferred_class": rnd.choice(classes)}),
        ]
        if i % 3 == 0:
            calls.append(("/support/query", {"query": rnd.choice(SUPPORT_QS), "language": "auto"}))
        for path, body in calls:
            r = api_post(path, body)
            ok += 1 if r and "error" not in r else 0
        if bar:
            bar.progress((i + 1) / rounds, text=f"Sent {(i + 1)} of {rounds} passenger sessions…")
    if bar:
        bar.empty()
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

ANCHORS = [
    ("attention", "Needs attention"), ("pillars", "Pillars"), ("incidents", "1 Incidents"), ("health", "2 Model health"),
    ("drift", "3 Drift & data"), ("bias", "4 Bias"), ("halluc", "5 Hallucination"), ("lifecycle", "6 Lifecycle"),
    ("agent", "7 Supervisor"), ("reports", "8 Reports & audit"), ("chaos", "9 Chaos lab"),
]


def tower() -> None:
    inject_theme("tower")
    hero(
        "🏛️ AI Governance Control Tower",
        "AI agents watching AI. Every model behind Railway Buddy is monitored for health, drift, bias and hallucination — "
        "problems are diagnosed, explained and fixed automatically, with humans approving the high-risk moves.",
        ["Autonomous supervisor", "Root-cause analysis", "Auto-retrain & rollback", "Full audit trail"],
    )

    top1, top2, top3 = st.columns([1.2, 1.2, 3])
    live = top1.toggle("Live auto-refresh", value=False, help="Refresh every 10 seconds")
    if top2.button("🔄 Refresh now", use_container_width=True):
        st.rerun()
    top3.caption(f"Engine: **{engine_label()}** · API: `{API_BASE}` · Last updated {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    body = st.fragment(run_every=10 if live else None)(_tower_body)
    body()


def _tower_body() -> None:
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
    hitl = api_get("/governance/supervisor/hitl/pending") or []
    retrain_q = lifecycle.get("retraining_queue", [])
    events = api_get("/governance/telemetry", {"limit": 300}) or []

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
                        use_container_width=True)
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
    for m in retrain_q:
        items.append(("warn", "Retrain queued", f"{mname(m)} is waiting to be retrained", "→ see Lifecycle"))
    items.sort(key=lambda x: SEV_ORDER[x[0]])

    if hitl:
        for req in hitl:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f'<div class="rb-card" style="border-left:4px solid #F59E0B;margin:0">{badge("Approval needed", "warn")} '
                        f'<b>{esc(req.get("action_type"))}</b> <span class="rb-sub">· {esc(req.get("module_id", ""))} · {esc(req.get("request_id"))}</span></div>',
                        unsafe_allow_html=True)
            if c2.button("✅ Approve", key=f'ap_{req["request_id"]}', use_container_width=True):
                api_post("/governance/supervisor/hitl/decide", {"request_id": req["request_id"], "approved": True})
                st.rerun()
            if c3.button("❌ Reject", key=f'rj_{req["request_id"]}', use_container_width=True):
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
        ("📉", "Drift & performance", "UNAVAILABLE" if drift_raw is None else "DRIFT" if drifting else "STABLE", "Drift service not responding" if drift_raw is None else f"{len(drifting)} of {len(drift)} models drifting"),
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
    for inc in incidents[:10]:
        p = inc.get("priority", "MEDIUM")
        tone = "bad" if p == "CRITICAL" else "warn" if p == "HIGH" else "info"
        with st.expander(f'{p} · {mname(inc.get("module_id", ""))} · {inc.get("primary_event_type", "").replace("_", " ").title()} · {inc["incident_id"]}',
                         expanded=(p == "CRITICAL")):
            a, b = st.columns(2)
            with a:
                st.markdown(f'{badge(p, tone)} &nbsp; opened {_ts(inc.get("timestamp"))} · seen **{inc.get("occurrences", 1)}×** · '
                            f'{inc.get("correlated_events_count", 0)} correlated events', unsafe_allow_html=True)
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

    # ── 2 Model health & SLA ────────────────────────────────────────────────
    section("Model health & SLA", "Latency percentiles, error rates and circuit breakers per AI module (SLA: P95 < 3000 ms, errors < 5%).", "health", "2")
    if modules:
        rows = []
        for mid, s in modules.items():
            rows.append({"Module": mname(mid), "Requests": s.get("total_requests", 0), "Error %": s.get("error_rate_pct", 0),
                         "Avg ms": s.get("avg_latency_ms", 0), "P95 ms": s.get("p95_ms", 0), "P99 ms": s.get("p99_ms", 0),
                         "Uptime %": s.get("uptime_pct", 100), "Circuit": s.get("circuit_state", "—"),
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
            st.plotly_chart(fig, use_container_width=True)
        # resource utilisation proxy from telemetry
        if events:
            ev = pd.DataFrame(events)
            if "latency_ms" in ev and "timestamp" in ev:
                ev["t"] = pd.to_datetime(ev["timestamp"], errors="coerce")
                ev = ev.dropna(subset=["t"]).sort_values("t")
                if len(ev) > 5:
                    fig = go.Figure()
                    for mod, grp in ev.groupby("module"):
                        fig.add_trace(go.Scatter(x=grp["t"], y=grp["latency_ms"], mode="lines", name=mname(mod), line=dict(width=1.6)))
                    fig.update_layout(title="<b>Live latency stream</b> (all modules)", yaxis_title="ms")
                    style_fig(fig, 260)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        callout("No traffic yet — use <b>Generate live traffic</b> in the Chaos lab (or open any Passenger service).", "info", "ℹ️")

    # ── 3 Drift & data quality ──────────────────────────────────────────────
    section("Drift & data quality", "Inputs are validated before inference; live feature distributions are compared with training data (KS test + PSI).", "drift", "3")
    q1, q2 = st.columns([1, 2.4])
    with q1:
        st.markdown(kpi("Input validation pass rate", f"{dq_rate:.1f}%", f"{len(valid)} recent inferences checked", "ok" if dq_rate >= 98 else "warn"), unsafe_allow_html=True)
    with q2:
        callout("<b>Data-quality gate</b> checks schema, numeric ranges and z-score outliers on every request <i>before</i> the model runs. "
                "Bad inputs are flagged on the response and never silently trusted.", "info", "🧪")
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
                st.plotly_chart(fig, use_container_width=True)
            elif waiting:
                n = max(f.get("n_samples", 0) for f in waiting)
                st.progress(min(1.0, n / 30), text=f"Collecting samples… {n}/30 needed for a statistical test")
            else:
                st.caption("No baseline features are registered for this model yet.")
    if not drift:
        callout("Drift windows are empty — generate traffic below.", "info", "ℹ️")

    # ── 4 Bias & fairness ───────────────────────────────────────────────────
    section("Bias & fairness", "Disparate Impact Ratio (≥ 0.80), Demographic Parity gap (≤ 0.10) and Equal Opportunity gap, per protected group.", "bias", "4")
    if bias:
        for b in bias:
            fair = b.get("overall_fair", True)
            with st.expander(f'{mname(b["module_id"])} — {"fair" if fair else "BIAS FLAGGED"} · fairness score {b.get("fairness_score", "—")}/100', expanded=not fair):
                for f in b.get("adverse_impact_flags", []):
                    callout(esc(f), "warn", "⚠️")
                rows = []
                for a in b.get("attributes_audited", []):
                    rows.append({"Protected attribute": a["attribute"].replace("_", " ").title(), "DIR": a.get("dir"), "DPD": a.get("dpd"),
                                 "EOD": a.get("eod"), "Groups": ", ".join(f'{g} (n={m.get("n", "?")})' for g, m in a.get("groups", {}).items()),
                                 "Result": "✅ fair" if not a.get("adverse_impact") else "⚠️ review"})
                _df(rows)
    else:
        callout("No fairness observations yet. Passenger requests carry protected attributes (age group, quota, class, persona); "
                "they appear here as soon as traffic flows — use <b>Generate live traffic</b> in the Chaos lab.", "info", "ℹ️")

    # ── 5 Hallucination ─────────────────────────────────────────────────────
    section("Hallucination & response quality", "Every Support Assistant answer is scored for faithfulness to retrieved policy text; weak answers are quarantined.", "halluc", "5")
    tot, quar = hall.get("total_evaluations", 0), hall.get("total_quarantined", 0)
    kpi_row([("Answers evaluated", tot, "", ""), ("Quarantined", quar, "weak grounding", "ok" if not quar else "warn"),
             ("Quarantine rate", f"{(quar / tot * 100):.1f}%" if tot else "0%", "", "ok" if not tot or quar / tot < 0.2 else "warn"),
             ("Avg faithfulness", f'{(sum(m.get("avg_faithfulness", 0) for m in (hall.get("modules") or {}).values()) / max(1, len(hall.get("modules") or {}))):.0%}',
              "target ≥ 65%", "")])
    recent = api_get("/governance/hallucination/recent", {"n": 15}) or []
    if recent:
        rows = [{"Time": _ts(e.get("timestamp", "")), "Query": (e.get("query") or "")[:70], "Faithfulness": round(e.get("faithfulness_score", 0), 2),
                 "Quality score": round(e.get("composite_score", 0), 2), "Blocked": "🔴 yes" if e.get("quarantine") else "✅ no",
                 "Severity": e.get("severity", "—")} for e in recent]
        _df(rows)
    else:
        st.caption("Ask the Support Assistant a question to populate this table.")

    # ── 6 Lifecycle ─────────────────────────────────────────────────────────
    section("Model lifecycle", "Version registry, champion/challenger canaries, automated retraining and one-click rollback.", "lifecycle", "6")
    mids = lifecycle.get("models_registered", [])
    prod, chal = lifecycle.get("production_versions", {}), lifecycle.get("challenger_versions", {})
    rows = []
    for mid in mids:
        st_ = api_get(f"/governance/lifecycle/model/{mid}") or {}
        vers = {v["version"]: v for v in st_.get("versions", [])}
        pv = vers.get(prod.get(mid, ""), {})
        met = pv.get("metrics", {})
        rows.append({"Model": mname(mid), "Production": prod.get(mid, "—"), "Challenger": chal.get(mid, "—"),
                     "Versions": st_.get("total_versions", 0), "Key metrics": ", ".join(f"{k}: {v}" for k, v in list(met.items())[:3]) or "—",
                     "Retrain queued": "⚠️ yes" if mid in retrain_q else "no"})
    _df(rows)
    l1, l2 = st.columns(2)
    with l1:
        with st.container(border=True):
            st.markdown("**🔁 Retrain a model**")
            rm = st.selectbox("Model", mids, format_func=mname, key="rt_model") if mids else None
            if st.button("Retrain now", disabled=not mids, key="rt_btn", use_container_width=True):
                with st.spinner("Retraining on fresh data and evaluating against the champion…"):
                    r = api_post("/governance/lifecycle/retrain", {"model_id": rm})
                if r and r.get("success"):
                    st.success(f'{mname(rm)}: {r.get("outcome", "retrained")} → {r.get("new_version", "")}')
                    for k_, v_ in (r.get("comparison") or {}).items():
                        st.caption(f"{k_}: {v_}")
                else:
                    st.error(str(r)[:300])
            if rm and chal.get(rm):
                st.caption(f'Challenger **{chal[rm]}** is running as a 10% canary.')
                if st.button("🚀 Promote challenger to production", key="pr_btn", use_container_width=True):
                    r = api_post("/governance/lifecycle/promote", {"model_id": rm})
                    if r and r.get("success"):
                        st.success(f'{mname(rm)} now serving {r.get("promoted_version")} (hot-swapped, no restart).')
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(str(r)[:300])
    with l2:
        with st.container(border=True):
            st.markdown("**⏪ Roll back a model**")
            rb = st.selectbox("Model", mids, format_func=mname, key="rb_model") if mids else None
            reason = st.text_input("Reason", "operator_request", key="rb_reason")
            if st.button("Rollback to previous version", disabled=not mids, key="rb_btn", use_container_width=True):
                r = api_post("/governance/supervisor/rollback", {"model_id": rb, "reason": reason})
                if r and r.get("success"):
                    st.success(f'Rolled back {mname(rb)} to {r.get("restored_version")}')
                else:
                    st.error(str(r)[:300])

    # ── 7 Supervisor agent ──────────────────────────────────────────────────
    section("Supervisor agent", "The autonomous meta-agent polls every pillar, opens incidents, queues retraining and rolls back — logging every action.", "agent", "7")
    if actions:
        _df([{"Time": _ts(a.get("timestamp", "")), "Action": str(a.get("action_type", "")).replace("_", " ").title(), "Module": mname(a.get("module_id", "")),
              "Trigger": str(a.get("trigger", "")).replace("_", " ").title(), "Executed by": a.get("executor", ""), "Outcome": a.get("outcome", "")} for a in reversed(actions[-12:])])
    else:
        callout("No autonomous actions yet. They appear when the supervisor detects drift, SLA breaches or incidents — try the Chaos lab.", "info", "🤖")

    # ── 8 Reports & audit ───────────────────────────────────────────────────
    section("Governance reports & audit trail", "Regulator-ready reports (JSON + Markdown) and the immutable stream of governed inferences.", "reports", "8")
    rc1, rc2 = st.columns([1, 3])
    if rc1.button("📋 Generate report", use_container_width=True, type="primary"):
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
        d1.download_button("⬇ Markdown", md, f'{rep["report_id"]}.md', use_container_width=True)
        d2.download_button("⬇ JSON", json.dumps(rep, indent=2, default=str), f'{rep["report_id"]}.json', use_container_width=True)
    with st.expander("📜 Live audit trail (most recent governed inferences)"):
        if events:
            ev = pd.DataFrame(events).tail(25).iloc[::-1]
            cols = [c for c in ["timestamp", "module", "event_type", "severity", "latency_ms", "confidence_score", "schema_valid", "event_id"] if c in ev]
            ev = ev[cols].copy()
            if "module" in ev:
                ev["module"] = ev["module"].map(mname)
            st.dataframe(ev, use_container_width=True, hide_index=True)
        else:
            st.caption("No events yet.")

    # ── 9 Chaos lab ─────────────────────────────────────────────────────────
    section("Chaos lab — watch the loop work", "Create real traffic or inject faults, then scroll up to see detection → root cause → action.", "chaos", "9")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**🚦 Generate live traffic**")
            st.caption("Sends real requests to the actual models so drift, bias, SLA and data-quality pillars have live data.")
            n = st.slider("Passenger sessions", 20, 80, 45, step=5, key="sim_n")
            a, b = st.columns(2)
            if a.button("Normal traffic", use_container_width=True, key="sim_ok"):
                done = simulate_traffic(n, drifted=False)
                st.success(f"{done} governed inferences recorded.")
                time.sleep(0.6)
                st.rerun()
            if b.button("⚠️ Shifted traffic", use_container_width=True, key="sim_drift",
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
            if a.button("Inject fault", type="primary", use_container_width=True, key="ch_go"):
                r = api_post("/governance/supervisor/chaos", {"module_id": mod, "chaos_type": kind[0]})
                if r and r.get("success"):
                    st.success(r.get("message", "Injected."))
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(str(r)[:200])
            if b.button("Reset to healthy", use_container_width=True, key="ch_reset"):
                api_post("/governance/supervisor/reset")
                st.rerun()
