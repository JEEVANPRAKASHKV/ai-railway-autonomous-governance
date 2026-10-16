"""
frontend/passenger.py
─────────────────────
Railway Buddy — the passenger-facing screens (Modules 1–7).

Every screen follows the same pattern: hero banner → quick scenarios → inputs on the left,
a plain-language result on the right → "why" (SHAP) → governance receipt.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components.shap_chart import render_shap_chart
from frontend.components.telemetry_badge import render_telemetry_badge
from frontend.ui import (
    api_get, api_post, badge, callout, esc, gauge, hero, inject_theme, kpi, kpi_row,
    label_for, load_reference, rows_card, section, style_fig,
)

CLASSES = ["3A", "2A", "1A", "SL", "CC"]
PERSONAS = ["budget", "business", "family", "senior", "student", "premium"]


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setter(values: Dict[str, Any], run_flag: str):
    def _cb():
        for k, v in values.items():
            st.session_state[k] = v
        st.session_state[run_flag] = True
    return _cb


def presets(items: List[tuple], run_flag: str, key: str) -> None:
    """Row of one-click scenario buttons. items = [(label, {state_key: value})]."""
    st.caption("⚡ Try a ready-made scenario")
    cols = st.columns(len(items))
    for i, (col, (label, vals)) in enumerate(zip(cols, items)):
        col.button(label, key=f"{key}_p{i}", use_container_width=True, on_click=_setter(vals, run_flag))


def _err(result: Any, what: str) -> None:
    detail = ""
    if isinstance(result, dict):
        detail = result.get("error") or result.get("detail") or ""
    callout(f"We couldn't complete <b>{what}</b> right now. {esc(str(detail)[:200])}", "bad", "⚠️")


def _empty(msg: str) -> None:
    st.markdown(
        f'<div class="rb-card" style="text-align:center;padding:46px 20px;color:#5B6B82">'
        f'<div style="font-size:2.2rem">🧭</div><div style="font-weight:700;color:#0A2463;margin-top:6px">{msg}</div>'
        f'<div class="rb-sub">Fill in the details and press the button to see your result.</div></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Home
# ──────────────────────────────────────────────────────────────────────────────

SERVICES = [
    ("🎫", "Ticket Confirmation", "Will my waitlist ticket confirm? Probability, berth and the reasons why.", "ticket"),
    ("🚆", "Train Recommender", "Best train for your budget, time and travel style — fairness-audited.", "recommend"),
    ("⏱️", "Delay & Disruption", "Delay forecast, platform congestion and smarter alternatives.", "delay"),
    ("🧭", "AI Travel Planner", "Tell it your trip in plain English. It plans — you approve.", "planner"),
    ("💰", "Fare Forecast", "Where fares are heading and the cheapest time to book.", "fare"),
    ("💬", "Support Assistant", "Ask about PNR, refunds and rules in English, हिन्दी or Hinglish.", "support"),
    ("🛡️", "Fraud & Bot Shield", "Real-time bot and fraud risk scoring for every session.", "fraud"),
]


def home(pages: Dict[str, Any]) -> None:
    inject_theme("home")
    hero(
        "Hi, I'm Railway Buddy 👋",
        "Your AI companion for Indian Railways — predicts your ticket, plans your journey, watches delays and fares, "
        "and keeps you safe. Every answer is monitored by an autonomous AI governance layer.",
        ["7 AI services", "Explainable predictions", "Human approves every booking", "Governed 24×7"],
    )

    dash = api_get("/governance/supervisor/dashboard") or {}
    score = dash.get("governance_score")
    status = (dash.get("platform_health") or {}).get("platform_status", "—")
    kpi_row([
        ("AI services online", "7 / 7", "Passenger modules", "ok"),
        ("Platform status", esc(status), "Live from Control Tower", "ok" if status == "HEALTHY" else "warn"),
        ("Governance score", f"{score:.0f}" if score is not None else "—", "out of 100", "ok" if (score or 0) >= 80 else "warn"),
        ("Languages", "EN · हिं · Hinglish", "Auto-detected", ""),
    ])

    section("What would you like to do?", "Pick a service — each one shows its reasoning, not just an answer.")
    for row in (SERVICES[:4], SERVICES[4:]):
        cols = st.columns(4)
        for col, (icon, title, desc, key) in zip(cols, row):
            with col:
                st.markdown(
                    f'<div class="rb-card" style="height:196px"><div style="font-size:1.6rem">{icon}</div>'
                    f'<h4 style="margin-top:4px">{title}</h4><div class="rb-sub">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open →", key=f"home_{key}", use_container_width=True):
                    st.switch_page(pages[key])
        if row is SERVICES[4:]:
            with cols[3]:
                st.markdown(
                    '<div class="rb-card" style="height:196px;background:linear-gradient(135deg,#0A2463,#1D4ED8);border:0">'
                    '<div style="font-size:1.6rem">🏛️</div><h4 style="margin-top:4px;color:#fff">AI Governance Tower</h4>'
                    '<div class="rb-sub" style="color:#DBEAFE">See how AI agents monitor, explain and fix the models behind these services.</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open Control Tower →", key="home_tower", use_container_width=True, type="primary"):
                    st.switch_page(pages["tower"])

    section("How Railway Buddy stays trustworthy", "Every prediction passes through the same governance loop.")
    cols = st.columns(4)
    for col, (i, t, d) in zip(cols, [
        ("1", "Validate", "Inputs are checked for quality before any model runs."),
        ("2", "Predict & explain", "Models return a score, a confidence and the reasons (SHAP)."),
        ("3", "Monitor", "Latency, drift, bias and hallucination are tracked live."),
        ("4", "Self-heal", "Agents open incidents, retrain and roll back — humans approve the big moves."),
    ]):
        col.markdown(f'<div class="rb-card"><span class="rb-badge" style="color:#1D4ED8;background:#EFF6FF;border-color:#BFDBFE">Step {i}</span>'
                     f'<h4 style="margin-top:8px">{t}</h4><div class="rb-sub">{d}</div></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# M1 — Ticket confirmation
# ──────────────────────────────────────────────────────────────────────────────

def ticket() -> None:
    inject_theme("ticket")
    hero("🎫 Ticket Confirmation Predictor",
         "Waitlisted or RAC? See the chance your ticket confirms before you travel, the likely berth, and exactly why.",
         ["XGBoost", "SHAP explanations", "Confidence score", "Auto-retrained when accuracy drops"])
    ref = load_reference()
    trains = ref["trains"]

    presets([
        ("🟢 Almost confirmed", {"t_train": label_for(trains, "12951"), "t_cls": "3A", "t_quota": "GN", "t_wl": 2, "t_days": 14}),
        ("🟠 Tatkal, high WL", {"t_train": label_for(trains, "12951"), "t_cls": "2A", "t_quota": "TQ", "t_wl": 48, "t_days": 2}),
        ("🔵 Festival rush", {"t_train": label_for(trains, "12301"), "t_cls": "SL", "t_quota": "GN", "t_wl": 180, "t_days": 45}),
    ], "t_run", "tk")

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("ticket_form"):
            train = st.selectbox("Train", list(trains.keys()), key="t_train")
            c1, c2 = st.columns(2)
            cls = c1.selectbox("Class", CLASSES, key="t_cls")
            quota = c2.selectbox("Quota", ["GN", "TQ", "LADIES", "DEFENCE", "SENIOR"], key="t_quota")
            wl = c1.number_input("Waitlist / RAC no.", 1, 500, key="t_wl", value=st.session_state.get("t_wl", 15))
            days = c2.slider("Days to departure", 1, 120, key="t_days", value=st.session_state.get("t_days", 7))
            with st.expander("More options"):
                senior = st.checkbox("Senior citizen (60+)")
                disab = st.checkbox("Divyangjan quota")
            go_btn = st.form_submit_button("🔮 Predict confirmation", use_container_width=True)
        st.caption(f"Journey date: **{(date.today() + timedelta(days=days)).strftime('%d %b %Y')}**")

    with right:
        if go_btn or st.session_state.pop("t_run", False):
            with st.spinner("Analysing booking & cancellation patterns…"):
                res = api_post("/predict/ticket", {
                    "train_number": trains[train], "journey_class": cls, "quota": quota,
                    "journey_date": (date.today() + timedelta(days=days)).isoformat(),
                    "booking_date": date.today().isoformat(), "current_wl_number": int(wl),
                    "seniority_flag": senior, "disability_flag": disab,
                })
            st.session_state["t_res"] = res
        res = st.session_state.get("t_res")
        if not res:
            _empty("Your prediction will appear here")
        elif "confirmation_probability" not in res:
            _err(res, "the prediction")
        else:
            p = res["confirmation_probability"]
            tone = "ok" if p >= 0.75 else "warn" if p >= 0.4 else "bad"
            g1, g2 = st.columns([1.1, 1])
            g1.plotly_chart(gauge(p * 100, "Chance of confirmation", "%",
                                  color={"ok": "#059669", "warn": "#D97706", "bad": "#DC2626"}[tone]),
                            use_container_width=True)
            with g2:
                st.markdown(kpi("Verdict", esc(res.get("pnr_status_prediction", "—")), "", tone), unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(kpi("Likely berth", esc(res.get("estimated_berth_type", "—")), "Indicative estimate"), unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(kpi("Model confidence", f'{res.get("confidence_score", 0):.0%}', "Certainty of this prediction"), unsafe_allow_html=True)
            if res.get("booking_advisory"):
                callout(f"<b>Advice:</b> {esc(res['booking_advisory'])}", tone, "📢")

    if st.session_state.get("t_res") and "confirmation_probability" in st.session_state["t_res"]:
        res = st.session_state["t_res"]
        section("Why this prediction?", "The factors that moved the probability up or down.")
        a, b = st.columns([1.6, 1])
        with a:
            render_shap_chart(res.get("shap_explanation"), title="Top drivers of this prediction")
        with b:
            bb = res.get("berth_probability_breakdown") or {}
            if bb:
                fig = go.Figure(go.Bar(x=list(bb.values()), y=list(bb.keys()), orientation="h", marker_color="#60A5FA",
                                       text=[f"{v:.0%}" for v in bb.values()], textposition="auto"))
                fig.update_layout(title="<b>Berth likelihood</b>")
                style_fig(fig, 300)
                st.plotly_chart(fig, use_container_width=True)
        render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M2 — Train recommender
# ──────────────────────────────────────────────────────────────────────────────

def recommend() -> None:
    inject_theme("recommend")
    hero("🚆 Smart Train Recommender",
         "The best train for you — balancing fare, speed, punctuality, availability and comfort for your travel style.",
         ["Persona-aware scoring", "Fairness audit (DIR)", "Plain-language reasons"])
    stations = load_reference()["stations"]
    names = list(stations.keys())

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("rec_form"):
            o = st.selectbox("From", names, index=names.index(label_for(stations, "NDLS")))
            d = st.selectbox("To", names, index=names.index(label_for(stations, "CSTM")))
            dt = st.date_input("Travel date", value=date.today() + timedelta(days=5))
            c1, c2 = st.columns(2)
            persona = c1.selectbox("I travel as", PERSONAS)
            cls = c2.selectbox("Preferred class", CLASSES)
            budget = st.number_input("Max budget ₹ (0 = no limit)", 0, 20000, 0, step=500)
            go_btn = st.form_submit_button("🔍 Find best trains", use_container_width=True)
    with right:
        if go_btn:
            with st.spinner("Scoring trains on your corridor…"):
                res = api_post("/recommend/trains", {
                    "source_station": stations[o], "destination_station": stations[d], "journey_date": str(dt),
                    "persona": persona, "preferred_class": cls, "max_budget": float(budget) if budget > 0 else None})
            st.session_state["r_res"] = (res, stations[o], stations[d])
        stored = st.session_state.get("r_res")
        if not stored:
            _empty("Your recommended trains will appear here")
        else:
            res, oc, dc = stored
            if "recommendations" not in res:
                _err(res, "the recommendation")
            elif not res["recommendations"]:
                callout(f"No direct trains found on <b>{oc} → {dc}</b> in our dataset. Try a nearby station or a major corridor like NDLS → CSTM.", "warn", "🔎")
            else:
                dirv = float(res.get("disparate_impact_ratio", 1.0))
                for r in res["recommendations"]:
                    top = r.get("rank") == 1
                    tags = "".join(badge(t, "info") + " " for t in r.get("tags", []))
                    st.markdown(
                        f'<div class="rb-card" style="{"border:2px solid var(--accent)" if top else ""}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px">'
                        f'<div><span class="rb-badge" style="color:#fff;background:var(--accent);border-color:var(--accent)">#{r.get("rank")}{" · Best match" if top else ""}</span>'
                        f' <b style="font-size:1.05rem;color:var(--deep)">{esc(r.get("train_name"))}</b> <span class="rb-sub">{esc(r.get("train_number"))}</span></div>'
                        f'<div style="font-size:1.4rem;font-weight:800;color:var(--deep)">₹{r.get("fare_estimate_inr", 0):,.0f}</div></div>'
                        f'<div style="margin:10px 0;font-size:.95rem"><b>{esc(r.get("departure_time"))}</b> {esc(oc)} &nbsp;→&nbsp; '
                        f'<b>{esc(r.get("arrival_time"))}</b> {esc(dc)} &nbsp;·&nbsp; {r.get("duration_hours", 0):.1f} h &nbsp;·&nbsp; '
                        f'Punctuality <b>{r.get("punctuality_score", 0):.0%}</b> &nbsp;·&nbsp; Comfort <b>{r.get("comfort_score", 0):.0%}</b> &nbsp;·&nbsp; '
                        f'Match score <b>{r.get("composite_score", 0):.2f}</b></div>'
                        f'<div class="rb-sub">💡 {esc(r.get("recommendation_reason"))}</div><div style="margin-top:8px">{tags}</div></div>',
                        unsafe_allow_html=True)
                if dirv >= 0.8:
                    callout(f"<b>Fairness audit passed</b> — Disparate Impact Ratio <b>{dirv:.2f}</b> (≥ 0.80). Rankings give fair exposure across fare tiers.", "ok", "⚖️")
                else:
                    callout(f"<b>Fairness review</b> — Disparate Impact Ratio <b>{dirv:.2f}</b> is below 0.80; an equity adjustment was applied.", "warn", "⚖️")
                w = res.get("weights_used") or {}
                if w:
                    with st.expander("How the score was built (persona weights)"):
                        fig = go.Figure(go.Bar(x=list(w.keys()), y=list(w.values()), marker_color="#38BDF8",
                                               text=[f"{v:.0%}" for v in w.values()], textposition="auto"))
                        style_fig(fig, 240)
                        st.plotly_chart(fig, use_container_width=True)
                render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M3 — Delay & disruption
# ──────────────────────────────────────────────────────────────────────────────

def delay() -> None:
    inject_theme("delay")
    hero("⏱️ Delay & Disruption Predictor",
         "Know before you leave: expected delay, station congestion, and better trains if things look bad.",
         ["LightGBM", "Confidence interval", "Proactive alerts", "Drift-monitored"])
    ref = load_reference()
    trains, stations = ref["trains"], ref["stations"]

    presets([
        ("🟢 Clear day", {"d_train": label_for(trains, "12002"), "d_station": label_for(stations, "BPL"), "d_weather": 0.05, "d_cong": 0.15, "d_dist": 450.0}),
        ("🟠 Monsoon tracks", {"d_train": label_for(trains, "12951"), "d_station": label_for(stations, "CSTM"), "d_weather": 0.70, "d_cong": 0.65, "d_dist": 1380.0}),
        ("🔴 Dense fog + congestion", {"d_train": label_for(trains, "12301"), "d_station": label_for(stations, "NDLS"), "d_weather": 0.95, "d_cong": 0.90, "d_dist": 1450.0}),
    ], "d_run", "dl")

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("delay_form"):
            train = st.selectbox("Train", list(trains.keys()), key="d_train")
            stn = st.selectbox("Station", list(stations.keys()), key="d_station")
            jd = st.date_input("Journey date", value=date.today() + timedelta(days=1))
            wx = st.slider("Weather severity", 0.0, 1.0, st.session_state.get("d_weather", 0.25), 0.05, key="d_weather",
                           help="0 = clear skies · 1 = heavy fog / flooding")
            cg = st.slider("Station congestion", 0.0, 1.0, st.session_state.get("d_cong", 0.45), 0.05, key="d_cong")
            dist = st.number_input("Distance from origin (km)", 10.0, 4000.0, float(st.session_state.get("d_dist", 500.0)), 50.0, key="d_dist")
            go_btn = st.form_submit_button("⏱️ Forecast delay", use_container_width=True)
    with right:
        if go_btn or st.session_state.pop("d_run", False):
            with st.spinner("Forecasting arrival delay…"):
                res = api_post("/predict/delay", {"train_number": trains[train], "station_code": stations[stn],
                                                  "journey_date": str(jd), "weather_index": wx,
                                                  "congestion_index": cg, "distance_from_origin_km": dist})
            st.session_state["d_res"] = res
        res = st.session_state.get("d_res")
        if not res:
            _empty("Your delay forecast will appear here")
        elif "predicted_delay_minutes" not in res:
            _err(res, "the delay forecast")
        else:
            mins = res["predicted_delay_minutes"]
            tone = "ok" if mins <= 15 else "warn" if mins <= 60 else "bad"
            ci = res.get("confidence_interval") or {}
            lvl = res.get("congestion_level", "—")
            c1, c2, c3 = st.columns(3)
            c1.markdown(kpi("Expected delay", f"{mins:.0f} min", "At the selected station", tone), unsafe_allow_html=True)
            c2.markdown(kpi("Likely range", f'{ci.get("lower_bound", 0):.0f}–{ci.get("upper_bound", 0):.0f} min', "90% interval"), unsafe_allow_html=True)
            c3.markdown(kpi("Congestion", esc(lvl), "Platform pressure", {"Low": "ok", "Moderate": "warn"}.get(lvl, "bad")), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            callout(f"<b>Travel alert:</b> {esc(res.get('proactive_alert', ''))}", tone, "📢")
            drivers = res.get("delay_drivers") or []
            if isinstance(drivers, list) and drivers:
                fig = go.Figure(go.Bar(
                    x=[d.get("impact_minutes", 0) for d in drivers][::-1], y=[d.get("factor", "") for d in drivers][::-1],
                    orientation="h", marker_color="#22D3EE",
                    text=[f'{d.get("impact_minutes", 0):.1f} min' for d in drivers][::-1], textposition="auto"))
                fig.update_layout(title="<b>What's causing the delay</b>")
                style_fig(fig, 60 + 56 * len(drivers))
                st.plotly_chart(fig, use_container_width=True)
            for a in res.get("alternative_options", []):
                st.markdown(
                    f'<div class="rb-card"><div style="display:flex;justify-content:space-between;gap:8px"><div>'
                    f'<b style="color:var(--deep)">{esc(a.get("train_name"))}</b> <span class="rb-sub">{esc(a.get("train_number"))}</span><br>'
                    f'<span class="rb-sub">Departs {esc(a.get("departure_time"))} · Arrives {esc(a.get("arrival_time"))}</span><br>'
                    f'<span class="rb-sub">💡 {esc(a.get("reason"))}</span></div>'
                    f'{badge("Saves ~%d min" % a.get("delay_advantage_minutes", 0), "ok")}</div></div>', unsafe_allow_html=True)
            render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M4 — Travel planner agent
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.",
    "NDLS to CSTM next Friday, budget ₹2000, I'd like 3A with a confirmed berth.",
    "Cheapest way from Bhopal to Delhi this weekend.",
]


def planner() -> None:
    inject_theme("planner")
    hero("🧭 AI Travel Planner",
         "Describe your trip in your own words. A team of AI agents searches trains, checks confirmation odds, delays and fares — "
         "then proposes a plan. <b>Nothing is booked until you approve.</b>",
         ["Agentic AI", "Human-in-the-loop booking", "Full audit trail", "Prompt-injection guard"])

    st.caption("💬 Try an example")
    cols = st.columns(len(EXAMPLES))
    for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
        c.button(ex if len(ex) < 52 else ex[:50] + "…", key=f"ex{i}", use_container_width=True,
                 on_click=lambda e=ex: st.session_state.update(p_query=e))

    with st.form("plan_form"):
        q = st.text_area("Where do you want to go?", key="p_query", height=88,
                         value=st.session_state.get("p_query", EXAMPLES[0]))
        c1, c2, c3 = st.columns([1, 1, 1.2])
        persona = c1.selectbox("Traveller type", PERSONAS)
        budget = c2.number_input("Budget cap ₹ (0 = from message)", 0, 50000, 0, step=500)
        go_btn = c3.form_submit_button("🤖 Plan my journey", use_container_width=True)

    if go_btn:
        with st.spinner("Agents are searching routes, predicting delays and forecasting fares…"):
            res = api_post("/agent/plan", {"query": q, "user_id": "railway_buddy_user", "persona": persona,
                                           "max_budget": float(budget) if budget > 0 else None})
        if res and "recommended_itinerary" in res:
            st.session_state["plan"] = res
            st.session_state.pop("booking", None)
        else:
            _err(res, "the journey plan")

    plan = st.session_state.get("plan")
    if not plan:
        return
    rec = plan["recommended_itinerary"]
    status = plan.get("status", "PENDING_APPROVAL")
    booking = st.session_state.get("booking")

    section("Your proposed itinerary", f'Request <code>{esc(plan.get("request_id"))}</code>')
    st.markdown(
        f'<div class="rb-card" style="border:2px solid var(--accent)">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
        f'<div style="font-size:1.15rem;font-weight:800;color:var(--deep)">🚆 {esc(rec.get("train_name"))} '
        f'<span class="rb-sub">{esc(rec.get("train_number"))}</span></div>'
        f'{badge("Approved" if status == "APPROVED" else "Awaiting your approval" if status == "PENDING_APPROVAL" else status, "ok" if status == "APPROVED" else "warn")}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;margin:14px 0;flex-wrap:wrap">'
        f'<div><div style="font-size:1.5rem;font-weight:800">{esc(rec.get("departure_time"))}</div><div class="rb-sub">{esc(rec.get("source"))}</div></div>'
        f'<div style="flex:1;min-width:120px;text-align:center"><div class="rb-sub">{rec.get("duration_hours", 0):.1f} h · Class {esc(rec.get("travel_class"))}</div>'
        f'<div style="height:3px;background:linear-gradient(90deg,var(--accent),var(--soft));border-radius:3px;margin:4px 0"></div></div>'
        f'<div><div style="font-size:1.5rem;font-weight:800">{esc(rec.get("arrival_time"))}</div><div class="rb-sub">{esc(rec.get("destination"))}</div></div></div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap">'
        f'{badge("₹%s fare" % format(rec.get("estimated_fare_inr", 0), ",.0f"), "info")} '
        f'{badge("%.0f%% confirmation odds" % (rec.get("confirmation_probability", 0) * 100), "ok" if rec.get("confirmation_probability", 0) >= .7 else "warn")} '
        f'{badge("~%.0f min delay" % rec.get("predicted_delay_minutes", 0), "ok" if rec.get("predicted_delay_minutes", 0) < 20 else "warn")}</div>'
        f'</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="rb-card"><h4>👍 Why it fits</h4>' + "".join(f'<div style="padding:4px 0">✅ {esc(p)}</div>' for p in rec.get("pros", [])) + "</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="rb-card"><h4>⚠️ Keep in mind</h4>' + ("".join(f'<div style="padding:4px 0">• {esc(c)}</div>' for c in rec.get("cons", [])) or '<div style="padding:4px 0">No major concerns</div>') + "</div>", unsafe_allow_html=True)
    callout(f"<b>Agent reasoning:</b> {esc(plan.get('agent_reasoning_summary', ''))}", "info", "🧠")

    for alt in plan.get("alternative_itineraries", []):
        st.markdown(f'<div class="rb-card"><b>{esc(alt.get("train_name"))}</b> · {esc(alt.get("departure_time"))} → {esc(alt.get("arrival_time"))} · '
                    f'₹{alt.get("estimated_fare_inr", 0):,.0f}</div>', unsafe_allow_html=True)

    # Human-in-the-loop gate
    if booking:
        st.markdown(
            f'<div class="rb-card" style="border:2px solid #10B981;background:#ECFDF5"><h4 style="color:#047857">🎉 Booking prepared</h4>'
            f'<div class="rb-row"><span class="k">Booking ID</span><span class="v">{esc(booking.get("booking_id"))}</span></div>'
            f'<div class="rb-row"><span class="k">PNR (demo)</span><span class="v">{esc(booking.get("pnr_number"))}</span></div>'
            f'<div class="rb-row"><span class="k">Berth</span><span class="v">{esc(booking.get("allocated_berth"))}</span></div>'
            f'<div class="rb-row"><span class="k">Passenger</span><span class="v">{esc(booking.get("passenger_name"))}</span></div>'
            f'<div class="rb-sub" style="margin-top:6px">{esc(booking.get("message", ""))}</div></div>', unsafe_allow_html=True)
    elif status == "PENDING_APPROVAL":
        section("Your decision", "The agent stops here. It will only continue if you approve.")
        with st.form("hitl"):
            c1, c2, c3, c4 = st.columns([1.4, .7, .7, 1])
            nm = c1.text_input("Passenger name", "Demo Passenger")
            ag = c2.number_input("Age", 1, 110, 30)
            gd = c3.selectbox("Gender", ["M", "F", "O"])
            bp = c4.selectbox("Berth", ["Lower", "Middle", "Upper", "Side Lower", "Side Upper"])
            ok, no = st.columns(2)
            yes = ok.form_submit_button("✅ Approve & prepare booking", type="primary", use_container_width=True)
            rej = no.form_submit_button("↩️ Reject & re-plan", use_container_width=True)
        if yes:
            r = api_post("/agent/confirm", {"request_id": plan["request_id"], "selected_option_id": rec.get("option_id", "OPT-1"),
                                            "passenger_name": nm, "passenger_age": int(ag), "passenger_gender": gd, "berth_preference": bp})
            if r and r.get("booking_id"):
                st.session_state["booking"] = r
                plan["status"] = "APPROVED"
                st.rerun()
            else:
                _err(r, "the booking")
        if rej:
            api_post("/agent/reject", {"request_id": plan["request_id"], "rejection_reason": "Passenger requested a different option"})
            plan["status"] = "REPLANNING"
            st.rerun()
    elif status == "REPLANNING":
        callout("Plan rejected. Edit your request above and plan again.", "warn", "↩️")

    with st.expander("🔍 Agent execution trace (audit trail)"):
        for i, step in enumerate(plan.get("execution_trace", []), 1):
            detail = ", ".join(f"{k}: {v}" for k, v in step.items() if k != "step")
            st.markdown(f'**{i}. {esc(step.get("step"))}** — <span class="rb-sub">{esc(detail)}</span>', unsafe_allow_html=True)
    render_telemetry_badge(plan.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M5 — Fare forecast
# ──────────────────────────────────────────────────────────────────────────────

def fare() -> None:
    inject_theme("fare")
    hero("💰 Fare Forecast & Booking Window",
         "See where the fare is heading, when demand surges, and the smartest day to book.",
         ["XGBoost forecast", "Demand surge detection", "Cost-saving alternatives", "Anomaly-monitored"])
    trains = load_reference()["trains"]
    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("fare_form"):
            train = st.selectbox("Train", list(trains.keys()))
            cls = st.selectbox("Class", CLASSES)
            jd = st.date_input("Journey date", value=date.today() + timedelta(days=20))
            tk = st.checkbox("Tatkal quota")
            go_btn = st.form_submit_button("💰 Forecast fare", use_container_width=True)
    with right:
        if go_btn:
            with st.spinner("Forecasting fare trajectory…"):
                res = api_post("/predict/fare", {"train_number": trains[train], "journey_class": cls,
                                                 "journey_date": str(jd), "tatkal_flag": tk})
            st.session_state["f_res"] = res
        res = st.session_state.get("f_res")
        if not res:
            _empty("Your fare forecast will appear here")
        elif "current_predicted_fare_inr" not in res:
            _err(res, "the fare forecast")
        else:
            dem = res.get("demand_level", "—")
            c1, c2, c3 = st.columns(3)
            c1.markdown(kpi("Predicted fare", f'₹{res["current_predicted_fare_inr"]:,.0f}', f'Base ₹{res.get("base_fare_inr", 0):,.0f}'), unsafe_allow_html=True)
            c2.markdown(kpi("Surge", f'{res.get("surge_multiplier", 1):.2f}×', "vs. base fare",
                            "ok" if res.get("surge_multiplier", 1) < 1.1 else "warn"), unsafe_allow_html=True)
            c3.markdown(kpi("Demand", esc(dem), "Booking pressure", {"Low": "ok", "Moderate": "ok", "High": "warn"}.get(dem, "bad")), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            callout(f"<b>Best time to book:</b> {esc(res.get('optimal_booking_window', ''))}", "info", "💡")
            curve = res.get("forecast_curve") or []
            if curve:
                df = pd.DataFrame(curve)
                fig = go.Figure(go.Scatter(x=df["days_to_departure"], y=df["predicted_fare_inr"], mode="lines+markers",
                                           line=dict(color="#0EA5E9", width=3), fill="tozeroy", fillcolor="rgba(14,165,233,.10)",
                                           name="Predicted fare"))
                fig.add_hline(y=res.get("base_fare_inr", 0), line_dash="dash", line_color="#94A3B8",
                              annotation_text="Base fare", annotation_position="bottom right")
                fig.update_layout(title="<b>Fare vs days before departure</b>", xaxis_title="Days before departure",
                                  yaxis_title="₹", xaxis_autorange="reversed")
                style_fig(fig, 300)
                st.plotly_chart(fig, use_container_width=True)
    res = st.session_state.get("f_res")
    if res and "current_predicted_fare_inr" in res:
        alts = res.get("cost_saving_alternatives") or []
        if alts:
            section("Ways to save", "Small changes that lower your fare.")
            cols = st.columns(min(3, len(alts)))
            for c, a in zip(cols, alts):
                c.markdown(f'<div class="rb-card"><div class="rb-sub">{esc(a.get("alternative_type"))}</div>'
                           f'<div style="font-size:1.4rem;font-weight:800;color:#047857">Save ₹{a.get("potential_savings_inr", 0):,.0f}</div>'
                           f'<div class="rb-sub">{esc(a.get("description"))}</div></div>', unsafe_allow_html=True)
        if res.get("fare_drivers"):
            section("Why this fare?", "What's pushing the price up or down.")
            render_shap_chart(res["fare_drivers"], title="Fare drivers", target_metric="₹ impact")
        render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M6 — Support assistant
# ──────────────────────────────────────────────────────────────────────────────

QUICK_Q = [
    ("🚆 Refund on waitlist ticket", "What happens if my waitlist ticket is not confirmed before chart preparation? Will I get a full refund?"),
    ("👨‍👩‍👧 Transfer a ticket", "Can I transfer my confirmed train ticket to a family member like father, mother, or spouse?"),
    ("🇮🇳 तत्काल रिफंड (Hindi)", "तत्काल टिकट रद्द करने पर क्या कोई रिफंड मिलता है? IRCTC का क्या नियम है?"),
    ("⚡ RAC rules (Hinglish)", "Kya RAC passenger 3A coach me travel kar sakta hai aur berth kaise allot hoti hai?"),
]


def _ask(q: str, lang: str) -> None:
    hist = st.session_state.setdefault("chat", [])
    hist.append({"role": "user", "text": q})
    res = api_post("/support/query", {"query": q, "language": lang, "user_id": "railway_buddy_user",
                                      "session_id": st.session_state.get("chat_sid", "web")})
    hist.append({"role": "assistant", "res": res})


def support() -> None:
    inject_theme("support")
    hero("💬 Support Assistant",
         "Ask about PNR, refunds, cancellations, Tatkal or RAC rules — in English, हिन्दी or Hinglish. "
         "Every answer is grounded in official policy text and checked for hallucination.",
         ["24×7", "Multilingual", "RAG over IRCTC policy", "Faithfulness-audited"])
    c1, c2 = st.columns([1, 3])
    lang = c1.selectbox("Language", ["auto", "en", "hi", "hinglish"], help="'auto' detects it for you")
    c2.caption("💬 Quick questions")
    cols = st.columns(len(QUICK_Q))
    for i, (c, (lbl, q)) in enumerate(zip(cols, QUICK_Q)):
        c.button(lbl, key=f"qq{i}", use_container_width=True, on_click=lambda qq=q: st.session_state.update(chat_pending=qq))

    pending = st.session_state.pop("chat_pending", None)
    typed = st.chat_input("Ask Railway Buddy anything about your journey…")
    q = typed or pending
    if q:
        with st.spinner("Searching verified IRCTC policy…"):
            _ask(q, lang)

    hist = st.session_state.get("chat", [])
    if not hist:
        st.markdown('<div class="rb-card" style="text-align:center;padding:36px"><div style="font-size:2rem">👋</div>'
                    '<b style="color:#0A2463">Hello! How can I help with your journey today?</b>'
                    '<div class="rb-sub">Pick a quick question above or type your own below.</div></div>', unsafe_allow_html=True)
    for m in hist:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.write(m["text"])
        else:
            res = m["res"]
            with st.chat_message("assistant", avatar="🚆"):
                if not res or "response" not in res:
                    _err(res, "your question")
                    continue
                st.write(res["response"])
                faith = res.get("faithfulness_score", 0)
                flag = res.get("hallucination_flag")
                st.markdown(
                    badge(f'Language: {str(res.get("detected_language", "")).upper()}', "info") + " " +
                    badge(f'Grounded {faith:.0%}' if not flag else f'Low grounding {faith:.0%}', "ok" if not flag else "warn") + " " +
                    badge("Escalate to human" if res.get("escalation_recommended") else "Self-resolved", "warn" if res.get("escalation_recommended") else "ok") + " " +
                    badge(f'Sentiment: {res.get("sentiment", "Neutral")}', "mute"), unsafe_allow_html=True)
                if flag:
                    st.caption("⚠️ The governance layer flagged this answer as weakly supported by policy text — please verify on irctc.co.in or call 139.")
                chunks = res.get("retrieved_contexts") or []
                if chunks:
                    with st.expander(f"📚 Sources used ({len(chunks)})"):
                        for ch in chunks:
                            st.markdown(f'**{esc(ch.get("source_file"))}** · similarity {ch.get("similarity_score", 0):.2f}')
                            st.caption(ch.get("text", "")[:420])
    if hist:
        st.button("🧹 Clear conversation", on_click=lambda: st.session_state.update(chat=[]))


# ──────────────────────────────────────────────────────────────────────────────
# M7 — Fraud & bot shield
# ──────────────────────────────────────────────────────────────────────────────

def fraud() -> None:
    inject_theme("fraud")
    hero("🛡️ Fraud & Bot Shield",
         "Scores every booking session for bot, scalper and account-misuse risk — and explains the evidence behind each alert.",
         ["XGBoost classifier", "Tatkal abuse protection", "Explainable investigations"])
    presets([
        ("🟢 Genuine passenger", {"f_rpm": 4.0, "f_pnr": 1, "f_tatkal": False, "f_cap": 0, "f_age": 365, "f_pay": 1, "f_ip": 0.10, "f_bulk": False, "f_sess": "SESS-GENUINE-PASSENGER"}),
        ("🟠 Fast Tatkal user", {"f_rpm": 35.0, "f_pnr": 4, "f_tatkal": True, "f_cap": 1, "f_age": 45, "f_pay": 2, "f_ip": 0.40, "f_bulk": False, "f_sess": "SESS-TATKAL-FREQUENT"}),
        ("🔴 Scalper bot", {"f_rpm": 120.0, "f_pnr": 25, "f_tatkal": True, "f_cap": 6, "f_age": 1, "f_pay": 6, "f_ip": 0.85, "f_bulk": True, "f_sess": "SESS-SCALPER-BOTNET-99"}),
    ], "f_run", "fr")

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("fraud_form"):
            rpm = st.slider("Requests per minute", 1.0, 150.0, float(st.session_state.get("f_rpm", 20.0)), 1.0, key="f_rpm")
            c1, c2 = st.columns(2)
            pnr = c1.number_input("PNR lookups", 1, 50, int(st.session_state.get("f_pnr", 2)), key="f_pnr")
            age = c2.number_input("Account age (days)", 0, 2000, int(st.session_state.get("f_age", 180)), key="f_age")
            pay = c1.number_input("Payment attempts", 1, 20, int(st.session_state.get("f_pay", 1)), key="f_pay")
            cap = c2.slider("Failed captchas", 0, 10, int(st.session_state.get("f_cap", 0)), key="f_cap")
            tk = st.checkbox("During Tatkal window (10:00–10:15)", value=st.session_state.get("f_tatkal", False), key="f_tatkal")
            bulk = st.checkbox("Bulk booking (6+ passengers)", value=st.session_state.get("f_bulk", False), key="f_bulk")
            with st.expander("Session details"):
                sess = st.text_input("Session ID", st.session_state.get("f_sess", "SESS-DEMO-001"), key="f_sess")
                ip = st.slider("Network risk score", 0.0, 1.0, float(st.session_state.get("f_ip", 0.1)), 0.05, key="f_ip")
            go_btn = st.form_submit_button("🛡️ Score this session", use_container_width=True)
    with right:
        if go_btn or st.session_state.pop("f_run", False):
            with st.spinner("Classifying session behaviour…"):
                res = api_post("/security/score", {
                    "session_id": sess, "ip_hash": "a9b8c7d6e5f4", "requests_per_minute": rpm, "unique_pnr_attempts": int(pnr),
                    "tatkal_flag": tk, "account_age_days": int(age), "payment_attempts": int(pay),
                    "failed_captcha_count": int(cap), "booking_time_hour": 10, "is_bulk_booking": bulk, "ip_risk_score": ip})
            st.session_state["fr_res"] = res
        res = st.session_state.get("fr_res")
        if not res:
            _empty("The risk assessment will appear here")
        elif "fraud_risk_score" not in res:
            _err(res, "the risk score")
        else:
            sc, cat, act = res["fraud_risk_score"], res.get("risk_category", "Safe"), res.get("recommended_action", "ALLOW")
            tone = "bad" if cat in ("Bot Detected", "High Risk") else "warn" if cat == "Suspicious" else "ok"
            g1, g2 = st.columns([1.1, 1])
            g1.plotly_chart(gauge(sc, "Fraud risk score", " / 100", color={"ok": "#059669", "warn": "#D97706", "bad": "#DC2626"}[tone]),
                            use_container_width=True)
            with g2:
                st.markdown(kpi("Threat level", esc(cat), "", tone), unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                msg = {"BLOCK_AND_REPORT": "Block & report", "RATE_LIMIT": "Rate-limit", "STEP_UP_CHALLENGE": "Ask for extra verification", "ALLOW": "Allow"}.get(act, act)
                st.markdown(kpi("Action taken", msg, "Automated security policy", tone), unsafe_allow_html=True)
    res = st.session_state.get("fr_res")
    if res and "fraud_risk_score" in res:
        inv = res.get("investigation_features") or []
        if inv:
            section("Investigation evidence", "Signals the model examined, flagged by severity.")
            for it in inv:
                tone = {"High Anomaly": "bad", "Suspicious": "warn"}.get(it.get("severity"), "ok")
                st.markdown(f'<div class="rb-row" style="background:#fff;padding:9px 14px;border:1px solid #D9E4F5;border-radius:10px;margin-bottom:6px">'
                            f'<span class="k">{esc(it.get("description"))}</span><span class="v">{badge(it.get("severity", ""), tone)}</span></div>',
                            unsafe_allow_html=True)
        if res.get("shap_risk_drivers"):
            render_shap_chart(res["shap_risk_drivers"], title="What drove the risk score", target_metric="Risk impact")
        render_telemetry_badge(res.get("governance_telemetry"))
