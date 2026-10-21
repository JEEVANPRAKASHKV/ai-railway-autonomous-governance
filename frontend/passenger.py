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
    label_for, load_reference, rows_card, section, send_outcome, style_fig,
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


def outcome_box(key: str, res: Dict[str, Any], label: str, kind: str, **kw: Any) -> None:
    """'What really happened?' — closes the governance feedback loop for this exact prediction."""
    audit = (res.get("governance_telemetry") or {}).get("audit_id")
    if not audit:
        return
    with st.expander(f"📬 {label}"):
        st.caption("Real outcomes let the governance layer measure live accuracy, spot concept drift and retrain "
                   "the model. They are stored in the audit trail against this prediction's audit id.")
        c1, c2 = st.columns([2, 1])
        if kind == "choice":
            val = c1.selectbox("What happened?", kw["options"], key=f"{key}_v", label_visibility="collapsed")
        else:
            val = c1.number_input(kw.get("input_label", "Value"), min_value=0.0, value=float(kw.get("value", 0.0)),
                                  step=float(kw.get("step", 1.0)), key=f"{key}_v", label_visibility="collapsed")
        if c2.button("Send outcome", key=f"{key}_b", use_container_width=True):
            r = send_outcome(audit, val)
            if r and r.get("accepted"):
                st.success("Thanks — recorded. It now counts towards this model's live accuracy.")
            else:
                st.warning(f"Not recorded: {(r or {}).get('error', 'unknown error')}")


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
         ["XGBoost", "SHAP explanations", "Booking & cancellation trends", "Retrained when live accuracy drops"])
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
                st.markdown(kpi("Likely berth", esc(res.get("estimated_berth_type", "—")), esc(res.get("berth_estimate_basis", "Indicative estimate"))[:60]), unsafe_allow_html=True)
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
        hi = res.get("historical_insights") or {}
        if hi:
            section("Booking & cancellation trends", f'From {hi.get("comparable_bookings", 0):,} comparable past bookings ({esc(hi.get("scope", ""))}).')
            k = st.columns(4)
            k[0].markdown(kpi("Confirmed historically", f'{hi.get("historical_confirmed_pct", 0):.0f}%', "of comparable bookings"), unsafe_allow_html=True)
            k[1].markdown(kpi("Ended in RAC", f'{hi.get("historical_rac_pct", 0):.0f}%', "seat shared, travel allowed"), unsafe_allow_html=True)
            k[2].markdown(kpi("Stayed waitlisted", f'{hi.get("historical_wl_pct", 0):.0f}%', "auto-refunded"), unsafe_allow_html=True)
            k[3].markdown(kpi("Cancellation rate", f'{hi.get("cancellation_rate_pct", 0):.0f}%', "passengers who cancelled"), unsafe_allow_html=True)
            by_m = hi.get("confirmed_pct_by_month") or {}
            if by_m:
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                xs = sorted(int(m) for m in by_m)
                fig = go.Figure(go.Bar(x=[months[m - 1] for m in xs], y=[by_m.get(m, by_m.get(str(m), 0)) * 100 for m in xs],
                                       marker_color="#60A5FA"))
                fig.update_layout(title="<b>Confirmation rate by travel month</b>", yaxis_title="% confirmed")
                style_fig(fig, 250)
                st.plotly_chart(fig, use_container_width=True)
        outcome_box("t_out", res, "Report the final status after chart preparation", "choice",
                    options=["Confirmed", "RAC", "WL"])
        render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M2 — Train recommender
# ──────────────────────────────────────────────────────────────────────────────

def recommend() -> None:
    inject_theme("recommend")
    hero("🚆 Smart Train Recommender",
         "The best train for you — balancing fare, speed, punctuality, availability and comfort for your travel style.",
         ["Persona-aware scoring", "Personalised from history", "Learns from your choices", "Fairness audit"])
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
            budget = c1.number_input("Max budget ₹ (0 = no limit)", 0, 20000, 0, step=500)
            arrive = c2.text_input("Arrive by (HH:MM, optional)", "")
            uid = st.text_input("Passenger ID (optional, e.g. USR00001)", "",
                                help="Personalises using that passenger's travel history and usual class / budget.")
            go_btn = st.form_submit_button("🔍 Find best trains", use_container_width=True)
    with right:
        if go_btn:
            with st.spinner("Scoring trains on your corridor…"):
                res = api_post("/recommend/trains", {
                    "source_station": stations[o], "destination_station": stations[d], "journey_date": str(dt),
                    "persona": persona, "preferred_class": cls, "max_budget": float(budget) if budget > 0 else None,
                    "arrival_deadline": arrive.strip() or None, "user_id": uid.strip() or None})
            st.session_state["r_res"] = (res, stations[o], stations[d])
        stored = st.session_state.get("r_res")
        if not stored:
            _empty("Your recommended trains will appear here")
        else:
            res, oc, dc = stored
            if "recommendations" not in res:
                _err(res, "the recommendation")
            elif not res["recommendations"]:
                callout(esc(res.get("message") or f"No direct trains found on {oc} → {dc}."), "warn", "🔎")
            else:
                dirv = float(res.get("disparate_impact_ratio", 1.0))
                if res.get("personalisation"):
                    pz = res["personalisation"]
                    callout(f"<b>Personalised for {esc(pz.get('user_id'))}</b> — travels as <b>{esc(pz.get('persona_from_history'))}</b>, "
                            f"usually in <b>{esc(pz.get('usual_class'))}</b>, budget up to ₹{float(pz.get('usual_budget_max', 0)):,.0f}.", "info", "👤")
                if res.get("message"):
                    st.caption(esc(res["message"]))
                audit = (res.get("governance_telemetry") or {}).get("audit_id")
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
                    fc = r.get("factor_contributions") or {}
                    b1, b2 = st.columns([3, 1])
                    if fc:
                        with b1.expander("Score breakdown"):
                            fs = r.get("factor_scores") or {}
                            for f, v in sorted(fc.items(), key=lambda kv: -kv[1]):
                                st.markdown(f'<div class="rb-row"><span class="k">{esc(f.title())} — score {fs.get(f, 0):.0%}</span>'
                                            f'<span class="v">+{v:.3f}</span></div>', unsafe_allow_html=True)
                    if audit and b2.button("I'd pick this", key=f"pick_{r.get('train_number')}", use_container_width=True):
                        out = send_outcome(audit, r.get("train_number"))
                        if out and out.get("accepted"):
                            st.success("Thanks! Your choice is used to measure relevance and tune the ranking.")
                        else:
                            st.info((out or {}).get("error", "Choice already recorded."))
                if dirv >= 0.8:
                    callout(f"<b>Fairness audit passed</b> — Disparate Impact Ratio <b>{dirv:.2f}</b> (≥ 0.80). Rankings give fair exposure across fare tiers.", "ok", "⚖️")
                else:
                    callout(f"<b>Fairness review</b> — Disparate Impact Ratio <b>{dirv:.2f}</b> is below 0.80; an equity adjustment was applied.", "warn", "⚖️")
                w = res.get("weights_used") or {}
                if w:
                    with st.expander(f"How the score was built (persona weights, learned version {res.get('weights_version', 0)})"):
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
         ["LightGBM", "Arrival & departure delay", "Station congestion history", "SHAP", "Proactive alerts"])
    ref = load_reference()
    trains, stations = ref["trains"], ref["stations"]

    presets([
        ("🟢 Clear day", {"d_train": label_for(trains, "12002"), "d_station": label_for(stations, "BPL"), "d_weather": 0.05, "d_cong": 0.15, "d_auto": False}),
        ("🟠 Monsoon tracks", {"d_train": label_for(trains, "12951"), "d_station": label_for(stations, "CSTM"), "d_weather": 0.70, "d_cong": 0.65, "d_auto": False}),
        ("🔴 Dense fog + congestion", {"d_train": label_for(trains, "12301"), "d_station": label_for(stations, "NDLS"), "d_weather": 0.95, "d_cong": 0.90, "d_auto": False}),
    ], "d_run", "dl")

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("delay_form"):
            train = st.selectbox("Train", list(trains.keys()), key="d_train")
            stn = st.selectbox("Station", list(stations.keys()), key="d_station")
            jd = st.date_input("Journey date", value=date.today() + timedelta(days=1))
            auto = st.checkbox("Estimate weather & congestion automatically", value=st.session_state.get("d_auto", True), key="d_auto",
                               help="Uses the month's weather pattern and this station's own congestion history.")
            wx = st.slider("Weather severity", 0.0, 1.0, st.session_state.get("d_weather", 0.25), 0.05, key="d_weather",
                           help="0 = clear skies · 1 = heavy fog / flooding (ignored when automatic)")
            cg = st.slider("Station congestion", 0.0, 1.0, st.session_state.get("d_cong", 0.45), 0.05, key="d_cong",
                           help="Ignored when automatic")
            go_btn = st.form_submit_button("⏱️ Forecast delay", use_container_width=True)
    with right:
        if go_btn or st.session_state.pop("d_run", False):
            with st.spinner("Forecasting arrival delay…"):
                body = {"train_number": trains[train], "station_code": stations[stn], "journey_date": str(jd)}
                if not auto:
                    body.update({"weather_index": wx, "congestion_index": cg})
                res = api_post("/predict/delay", body)
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
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(kpi("Arrival delay", f"{mins:.0f} min", "At the selected station", tone), unsafe_allow_html=True)
            dep = res.get("departure_delay_minutes")
            c2.markdown(kpi("Departure delay", f"{dep:.0f} min" if dep is not None else "—", "From the origin"), unsafe_allow_html=True)
            c3.markdown(kpi("Likely range", f'{ci.get("lower_bound", 0):.0f}–{ci.get("upper_bound", 0):.0f} min',
                            f'{ci.get("coverage", 0.8):.0%} prediction interval'), unsafe_allow_html=True)
            sci = res.get("station_congestion_index")
            c4.markdown(kpi("Station congestion", esc(lvl), f'{sci:.0%} · {esc(res.get("congestion_source", ""))}' if sci is not None else "",
                            {"Low": "ok", "Moderate": "warn"}.get(lvl, "bad")), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            callout(f"<b>Travel alert:</b> {esc(res.get('proactive_alert', ''))}"
                    + (f" &nbsp;·&nbsp; <b>{res['disruption_probability']:.0%}</b> chance of a delay over 60 min."
                       if res.get("disruption_probability") is not None else ""), tone, "📢")
            for al in res.get("alerts", []):
                callout(f"<b>{esc(al.get('title'))}</b> — {esc(al.get('message'))} <i>{esc(al.get('action', ''))}</i>",
                        {"CRITICAL": "bad", "WARNING": "warn"}.get(al.get("severity"), "info"), "🔔")
            drivers = [d for d in (res.get("delay_drivers") or []) if d.get("feature") != "base_value"]
            if drivers:
                base = next((d for d in res["delay_drivers"] if d.get("feature") == "base_value"), None)
                fig = go.Figure(go.Bar(
                    x=[d.get("impact_minutes", 0) for d in drivers][::-1],
                    y=[f'{d.get("factor", "")} ({d.get("value")})' for d in drivers][::-1],
                    orientation="h", marker_color=["#0891B2" if d.get("impact_minutes", 0) >= 0 else "#F97316" for d in drivers][::-1],
                    text=[f'{d.get("impact_minutes", 0):+.1f} min' for d in drivers][::-1], textposition="auto"))
                fig.update_layout(title=f"<b>Why this delay (SHAP)</b> — typical delay {base['impact_minutes']:.0f} min, each bar adds or removes minutes" if base else "<b>Why this delay (SHAP)</b>")
                style_fig(fig, 60 + 50 * len(drivers))
                st.plotly_chart(fig, use_container_width=True)
            for a in res.get("alternative_options", []):
                st.markdown(
                    f'<div class="rb-card"><div style="display:flex;justify-content:space-between;gap:8px"><div>'
                    f'<b style="color:var(--deep)">{esc(a.get("train_name"))}</b> <span class="rb-sub">{esc(a.get("train_number"))}</span><br>'
                    f'<span class="rb-sub">Departs {esc(a.get("departure_time"))} · Arrives {esc(a.get("arrival_time"))}</span><br>'
                    f'<span class="rb-sub">💡 {esc(a.get("reason"))}</span></div>'
                    f'{badge("Saves ~%d min" % a.get("delay_advantage_minutes", 0), "ok")}</div></div>', unsafe_allow_html=True)
            outcome_box("d_out", res, "Report the actual delay", "number", input_label="Actual delay (min)", value=float(mins), step=5.0)
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
         ["Multi-agent", "One-change connections", "Validator agent", "Human-in-the-loop booking", "Injection guard"])

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
        with st.spinner("Agents are searching routes and connections, predicting delays and forecasting fares…"):
            res = api_post("/agent/plan", {"query": q, "user_id": "railway_buddy_user", "persona": persona,
                                           "max_budget": float(budget) if budget > 0 else None})
        if res and "status" in res and "error" not in res:
            st.session_state["plan"] = res
            st.session_state.pop("booking", None)
            st.session_state.pop("plan_choice", None)
        else:
            _err(res, "the journey plan")

    plan = st.session_state.get("plan")
    if not plan:
        return
    status = plan.get("status", "PENDING_APPROVAL")
    cons = plan.get("constraints") or {}

    if cons:
        chips = [f"From <b>{esc(cons.get('source'))}</b>", f"To <b>{esc(cons.get('destination'))}</b>",
                 f"On <b>{esc(cons.get('travel_date'))}</b> ({esc(cons.get('date_phrase', ''))})"]
        if cons.get("budget"):
            chips.append(f"Budget <b>₹{float(cons['budget']):,.0f}</b>")
        if cons.get("arrival_before"):
            chips.append(f"Arrive before <b>{esc(cons['arrival_before'])}</b>")
        if cons.get("preferred_class"):
            chips.append(f"Class <b>{esc(cons['preferred_class'])}</b>")
        st.markdown('<div class="rb-sub" style="margin:6px 0 2px">🧠 What the agent understood '
                    f'({esc(cons.get("method", "rules"))}):</div>' + " &nbsp;·&nbsp; ".join(chips), unsafe_allow_html=True)

    if status == "BLOCKED":
        callout(f"<b>Request blocked.</b> {esc(plan.get('agent_reasoning_summary', ''))}", "bad", "🛡️")
        render_telemetry_badge(plan.get("governance_telemetry"))
        return
    if status == "NEEDS_CLARIFICATION":
        callout(f"<b>The agent needs a detail from you.</b> {esc(plan.get('clarification_question') or plan.get('agent_reasoning_summary', ''))}", "warn", "❓")
        render_telemetry_badge(plan.get("governance_telemetry"))
        return

    options = [plan["recommended_itinerary"]] + list(plan.get("alternative_itineraries", []))
    choice_id = st.session_state.get("plan_choice", options[0].get("option_id"))
    rec = next((o for o in options if o.get("option_id") == choice_id), options[0])
    booking = st.session_state.get("booking")

    def leg_html(l):
        return (f'<div class="rb-row"><span class="k">🚆 <b>{esc(l.get("train_name"))}</b> {esc(l.get("train_number"))} · '
                f'{esc(l.get("source"))} {esc(l.get("departure", "")[11:16])} → {esc(l.get("destination"))} {esc(l.get("arrival", "")[11:16])}'
                f' <span class="rb-sub">({esc(l.get("arrival", "")[:10])})</span></span>'
                f'<span class="v">₹{l.get("fare_inr", 0):,.0f} · {l.get("confirmation_probability", 0):.0%} odds · ~{l.get("predicted_delay_minutes", 0):.0f} min</span></div>')

    section("Your proposed itinerary", f'Request <code>{esc(plan.get("request_id"))}</code>'
            + (f' · expires {esc(plan.get("expires_at", "")[11:16])} UTC' if plan.get("expires_at") else ""))
    exp = rec.get("expected_arrival_with_delay") or ""
    st.markdown(
        f'<div class="rb-card" style="border:2px solid var(--accent)">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
        f'<div style="font-size:1.15rem;font-weight:800;color:var(--deep)">{esc(rec.get("option_id"))} · {esc(rec.get("train_name"))}</div>'
        f'{badge("Approved" if status == "APPROVED" else "Awaiting your approval" if status == "PENDING_APPROVAL" else status, "ok" if status == "APPROVED" else "warn")}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;margin:14px 0;flex-wrap:wrap">'
        f'<div><div style="font-size:1.5rem;font-weight:800">{esc(rec.get("departure_time"))}</div><div class="rb-sub">{esc(rec.get("source"))} · {esc((rec.get("departure_datetime") or "")[:10])}</div></div>'
        f'<div style="flex:1;min-width:120px;text-align:center"><div class="rb-sub">{rec.get("duration_hours", 0):.1f} h · Class {esc(rec.get("travel_class"))}'
        f'{" · 1 change" if rec.get("changes") else " · direct"}</div>'
        f'<div style="height:3px;background:linear-gradient(90deg,var(--accent),var(--soft));border-radius:3px;margin:4px 0"></div></div>'
        f'<div><div style="font-size:1.5rem;font-weight:800">{esc(rec.get("arrival_time"))}</div><div class="rb-sub">{esc(rec.get("destination"))} · {esc((rec.get("arrival_datetime") or "")[:10])}</div></div></div>'
        + "".join(leg_html(l) for l in rec.get("legs", [])) +
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">'
        f'{badge("₹%s total" % format(rec.get("estimated_fare_inr", 0), ",.0f"), "info")} '
        f'{badge("%.0f%% confirmation odds" % (rec.get("confirmation_probability", 0) * 100), "ok" if rec.get("confirmation_probability", 0) >= .6 else "warn")} '
        f'{badge("Expected arrival " + exp[11:16] + " incl. delay", "info") if exp else ""} '
        f'{badge("Meets all your constraints", "ok") if rec.get("meets_all_constraints") else badge("Closest match — misses a constraint", "warn")}</div>'
        f'</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="rb-card"><h4>👍 Why it fits</h4>' + ("".join(f'<div style="padding:4px 0">✅ {esc(p)}</div>' for p in rec.get("pros", [])) or "<div class=rb-sub>—</div>") + "</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="rb-card"><h4>⚠️ Keep in mind</h4>' + ("".join(f'<div style="padding:4px 0">• {esc(c)}</div>' for c in rec.get("cons", [])) or '<div style="padding:4px 0">No major concerns</div>') + "</div>", unsafe_allow_html=True)
    callout(f"<b>Agent reasoning:</b> {esc(plan.get('agent_reasoning_summary', ''))}", "info", "🧠")
    val = plan.get("validation") or {}
    if val:
        checks = val.get("checks") or {}
        st.markdown('<div class="rb-sub" style="margin:4px 0">🔎 Validator agent: '
                    + " ".join(badge(k.replace("_", " "), "ok" if v else "warn") for k, v in checks.items()) + "</div>",
                    unsafe_allow_html=True)

    if len(options) > 1 and status == "PENDING_APPROVAL" and not booking:
        st.caption("Other options the agents compared — pick one to review it:")
        for o in options:
            if o.get("option_id") == rec.get("option_id"):
                continue
            c1, c2 = st.columns([5, 1])
            c1.markdown(f'<div class="rb-card" style="padding:10px 14px"><b>{esc(o.get("option_id"))} · {esc(o.get("train_name"))}</b> · '
                        f'{esc(o.get("departure_time"))} → {esc(o.get("arrival_time"))} · ₹{o.get("estimated_fare_inr", 0):,.0f} · '
                        f'{o.get("confirmation_probability", 0):.0%} odds{" · 1 change" if o.get("changes") else ""}'
                        f'{"" if o.get("meets_all_constraints") else " · misses a constraint"}</div>', unsafe_allow_html=True)
            if c2.button("Review", key=f"opt_{o.get('option_id')}", use_container_width=True):
                st.session_state["plan_choice"] = o.get("option_id")
                st.rerun()

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
        section("Your decision", f"The agents stop here. Approving books {esc(rec.get('option_id'))} only.")
        with st.form("hitl"):
            c1, c2, c3, c4 = st.columns([1.4, .7, .7, 1])
            nm = c1.text_input("Passenger name", "Demo Passenger")
            ag = c2.number_input("Age", 1, 110, 30)
            gd = c3.selectbox("Gender", ["M", "F", "O"])
            bp = c4.selectbox("Berth", ["Lower", "Middle", "Upper", "Side Lower", "Side Upper"])
            ok, no = st.columns(2)
            yes = ok.form_submit_button(f"✅ Approve {rec.get('option_id')} & prepare booking", type="primary", use_container_width=True)
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
            r = api_post("/agent/reject", {"request_id": plan["request_id"], "rejection_reason": "Passenger requested a different option"})
            if r and "error" in r:
                _err(r, "the rejection")
            else:
                plan["status"] = "REPLANNING"
                st.rerun()
    elif status == "REPLANNING":
        callout("Plan rejected. Edit your request above and plan again.", "warn", "↩️")

    with st.expander("🔍 Agent execution trace (audit trail)"):
        for i, step in enumerate(plan.get("execution_trace", []), 1):
            detail = ", ".join(f"{k}: {v}" for k, v in step.items() if k not in ("step", "agent", "status") and v not in (None, [], ""))
            st.markdown(f'**{i}. {esc(step.get("agent", ""))} — {esc(step.get("step"))}** {badge(step.get("status", ""), None)} '
                        f'<span class="rb-sub">{esc(detail)}</span>', unsafe_allow_html=True)
        m = plan.get("agent_metrics") or {}
        if m:
            st.caption(f"Tool calls: {m.get('tool_calls')} · planning time {m.get('plan_ms')} ms")
    render_telemetry_badge(plan.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M5 — Fare forecast
# ──────────────────────────────────────────────────────────────────────────────

def fare() -> None:
    inject_theme("fare")
    hero("💰 Fare Forecast & Booking Window",
         "See where the fare is heading, when demand surges, and the smartest day to book.",
         ["XGBoost forecast", "Demand surge detection", "SHAP fare drivers", "Pricing-anomaly guard"])
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
            callout(f"<b>Best time to book:</b> {esc(res.get('optimal_booking_window', ''))}"
                    + (f" &nbsp;·&nbsp; Demand surge expected from <b>{res['demand_surge_from_days']} days</b> before departure."
                       if res.get("demand_surge_from_days") is not None else ""), "info", "💡")
            if res.get("pricing_anomaly"):
                callout(f"<b>Pricing check:</b> {esc(res['pricing_anomaly'])} The governance layer logged this for review.", "warn", "🧪")
            if res.get("policy_note"):
                st.caption("ℹ️ " + esc(res["policy_note"]))
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
            section("Why this fare?", "SHAP: how much each factor adds to or removes from the fare, in ₹.")
            render_shap_chart(res["fare_drivers"], title="Fare drivers (SHAP)", target_metric="₹ impact")
        outcome_box("f_out", res, "Report the fare you actually paid", "number", input_label="Fare paid (₹)",
                    value=float(res.get("current_predicted_fare_inr", 0)), step=10.0)
        render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M6 — Support assistant
# ──────────────────────────────────────────────────────────────────────────────

QUICK_Q = [
    ("🔎 PNR status", "What is the status of PNR 0000001 and my refund?"),
    ("🚆 Refund on waitlist ticket", "What happens if my waitlist ticket is not confirmed before chart preparation? Will I get a full refund?"),
    ("👨‍👩‍👧 Transfer a ticket", "Can I transfer my confirmed train ticket to a family member like father, mother, or spouse?"),
    ("🇮🇳 तत्काल रिफंड (Hindi)", "तत्काल टिकट रद्द करने पर क्या कोई रिफंड मिलता है? IRCTC का क्या नियम है?"),
    ("⚡ RAC rules (Hinglish)", "Kya RAC passenger 3A coach me travel kar sakta hai aur berth kaise allot hoti hai?"),
    ("🚉 Trains Delhi → Mumbai", "Which trains from Delhi to Mumbai?"),
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
         ["24×7", "Multilingual", "PNR & refund lookup", "RAG over IRCTC policy", "Faithfulness-audited"])
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
    for idx, m in enumerate(hist):
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
                    badge(f'Answered by: {res.get("answered_by") or "policy engine"}', "mute") + " " +
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
                audit = (res.get("governance_telemetry") or {}).get("audit_id")
                if audit:
                    rated = st.session_state.setdefault("rated", {})
                    if audit in rated:
                        st.caption(f"Thanks for rating this answer {rated[audit]}/5.")
                    else:
                        stars = st.feedback("stars", key=f"fb_{idx}_{audit}")
                        if stars is not None:
                            r = send_outcome(audit, int(stars) + 1)
                            if r and r.get("accepted"):
                                rated[audit] = int(stars) + 1
                                st.caption("Thanks — your rating feeds the satisfaction monitor.")
    if hist:
        st.button("🧹 Clear conversation", on_click=lambda: st.session_state.update(chat=[]))


# ──────────────────────────────────────────────────────────────────────────────
# M7 — Fraud & bot shield
# ──────────────────────────────────────────────────────────────────────────────

def fraud() -> None:
    inject_theme("fraud")
    hero("🛡️ Fraud & Bot Shield",
         "Scores every booking session for bot, scalper and account-misuse risk — and explains the evidence behind each alert.",
         ["XGBoost classifier", "Anomaly detection", "Account-misuse checks", "SHAP investigations"])
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
                    "session_id": sess, "ip_hash": "a9b8c7d6e5f4", "user_id": "demo-account",
                    "requests_per_minute": rpm, "unique_pnr_attempts": int(pnr),
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
            pred = (res.get("incident_dossier") or {}).get("predicted_label", "")
            render_shap_chart(res["shap_risk_drivers"], title=f"Why the model says '{pred}' (SHAP)", target_metric="Impact on that class")
        audit = (res.get("governance_telemetry") or {}).get("audit_id")
        if audit:
            with st.expander("🕵️ Security analyst verdict (feeds false-positive / false-negative monitoring)"):
                c1, c2 = st.columns([2, 1])
                verdict = c1.selectbox("Confirmed label after investigation", ["legitimate", "suspicious", "bot"], key="fr_verdict")
                if c2.button("Record verdict", use_container_width=True):
                    r = api_post("/governance/outcomes", {"audit_id": audit, "actual": verdict, "source": "analyst"})
                    if r and r.get("accepted"):
                        st.success("Verdict recorded — it now counts towards the live false-positive / false-negative rates.")
                    else:
                        st.warning(f"Not recorded: {(r or {}).get('error', 'unknown error')}")
        render_telemetry_badge(res.get("governance_telemetry"))

    alerts = api_get("/security/alerts", {"n": 8}) or []
    if alerts:
        section("Recent security alerts", "Every alert carries its evidence so an analyst can investigate it.")
        for a in alerts:
            st.markdown(f'<div class="rb-row" style="background:#fff;padding:9px 14px;border:1px solid #D9E4F5;border-radius:10px;margin-bottom:6px">'
                        f'<span class="k"><b>{esc(a.get("session_id"))}</b> · {esc(", ".join(a.get("why", [])[:4]))}</span>'
                        f'<span class="v">{badge(a.get("category", ""), None)} {esc(a.get("risk_score"))}</span></div>',
                        unsafe_allow_html=True)
