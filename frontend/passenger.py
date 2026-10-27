"""
frontend/passenger.py
─────────────────────
Railway Buddy — the passenger-facing screens (Modules 1–5) and the Fraud & Bot Shield (Module 7, under AI Governance).
The Support Assistant (Module 6) is the floating chat button in frontend/chat_widget.py.

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
    NEG, POS, TONE_BAR, api_get, api_post, badge, callout, esc, hero, inject_theme, kpi, kpi_row,
    halt_labels, label_for, load_reference, meter, mono, provenance, rows_card, section, send_outcome, style_fig,
)

CLASSES = ["3A", "2A", "1A", "SL", "CC", "EC", "2S"]
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
    st.caption("Try a ready-made scenario")
    cols = st.columns(len(items))
    for i, (col, (label, vals)) in enumerate(zip(cols, items)):
        col.button(label, key=f"{key}_p{i}", width="stretch", on_click=_setter(vals, run_flag))


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
    with st.expander(f"{label}"):
        st.caption("Real outcomes let the governance layer measure live accuracy, spot concept drift and retrain "
                   "the model. They are stored in the audit trail against this prediction's audit id.")
        c1, c2 = st.columns([2, 1])
        if kind == "choice":
            val = c1.selectbox("What happened?", kw["options"], key=f"{key}_v", label_visibility="collapsed")
        else:
            val = c1.number_input(kw.get("input_label", "Value"), min_value=0.0, value=float(kw.get("value", 0.0)),
                                  step=float(kw.get("step", 1.0)), key=f"{key}_v", label_visibility="collapsed")
        if c2.button("Send outcome", key=f"{key}_b", width="stretch"):
            r = send_outcome(audit, val)
            if r and r.get("accepted"):
                st.success("Thanks — recorded. It now counts towards this model's live accuracy.")
            else:
                st.warning(f"Not recorded: {(r or {}).get('error', 'unknown error')}")


def _empty(msg: str) -> None:
    st.markdown(
        f'<div class="rb-card" style="padding:28px 20px;border-style:dashed">'
        f'<div style="font-weight:700">{msg}</div>'
        f'<div class="rb-sub">Fill in the details on the left and run it — or pick a ready-made scenario above.</div></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Home — a departure board of services
# ──────────────────────────────────────────────────────────────────────────────

# (code, page key, module id, name, what it answers)
SERVICES = [
    ("M1", "ticket", "m1_ticket_predictor", "Ticket confirmation", "Will my waitlisted ticket confirm, and which berth?"),
    ("M2", "recommend", "m2_train_recommender", "Train recommender", "Which train fits my budget, time and travel style?"),
    ("M3", "delay", "m3_delay_predictor", "Delay & disruption", "How late will it run, and is there a better train?"),
    ("M4", "planner", "m4_travel_planner", "AI travel planner", "Plan my trip from one sentence — I approve the booking."),
    ("M5", "fare", "m5_fare_predictor", "Fare forecast", "Where is the fare heading, and when should I book?"),
    ("M6", None, "m6_support_assistant", "Support assistant", "PNR status, refunds, Tatkal and RAC rules — chat button, bottom-right."),
]
GOVERNANCE = [
    ("GOV", "tower", None, "Governance control tower", "Health, drift, live quality, bias, incidents, retraining and audit."),
    ("M7", "fraud", "m7_fraud_detector", "Fraud & bot shield", "Bot, scalper and account-misuse scoring with evidence."),
]
_BOARD_COLS = [0.55, 3.2, 0.9, 0.9, 1.0, 0.9]


def _board(rows: List[tuple], pages: Dict[str, Any], modules: Dict[str, Any], key: str) -> None:
    cols = " ".join(f"{w}fr" for w in _BOARD_COLS)
    st.markdown(f'<div class="rb-thead" style="--cols:{cols}">' + "".join(
        f"<span>{h}</span>" for h in ["Code", "Service", "Requests", "P95 ms", "Status", ""]) + "</div>", unsafe_allow_html=True)
    for code, page, mid, name, desc in rows:
        m = modules.get(mid or "", {}) if mid else {}
        req = m.get("total_requests")
        p95 = m.get("p95_ms")
        err = float(m.get("error_rate_pct") or 0)
        status = ("—" if not m else "Delayed" if (p95 or 0) > 3000 or err >= 5 else "On time")
        with st.container(key=f"svcrow_{key}_{code}"):
            c = st.columns(_BOARD_COLS)
            c[0].markdown(f'<span class="rb-mono" style="font-weight:700">{code}</span>', unsafe_allow_html=True)
            c[1].markdown(f"**{name}**<br><span class='rb-sub'>{esc(desc)}</span>", unsafe_allow_html=True)
            c[2].markdown(mono(f"{req:,}" if req is not None else "—"), unsafe_allow_html=True)
            c[3].markdown(mono(f"{p95:.0f}" if p95 is not None else "—"), unsafe_allow_html=True)
            c[4].markdown(badge(status, "ok" if status == "On time" else "warn" if status == "Delayed" else "mute"),
                          unsafe_allow_html=True)
            if page and pages.get(page):
                c[5].page_link(pages[page], label="Open")
            else:
                c[5].markdown('<span class="rb-sub">Chat button ↘</span>', unsafe_allow_html=True)


def home(pages: Dict[str, Any]) -> None:
    inject_theme("home")
    hero(
        "Railway Buddy",
        "Predicts your ticket, plans your journey, watches delays and fares, and answers your questions. "
        "Every model is monitored live by an autonomous governance layer, and nothing is booked without your approval.",
        ["7 AI services", "Explanations on every prediction", "Human approval for bookings and model releases"],
    )

    dash = api_get("/governance/supervisor/dashboard") or {}
    score = dash.get("governance_score")
    health = dash.get("platform_health") or {}
    status = health.get("platform_status", "—")
    modules = health.get("modules") or {}
    kpi_row([
        ("Platform", esc(status.title() if status != "—" else status), "live from the control tower", "ok" if status == "HEALTHY" else "warn"),
        ("Governance score", f"{score:.0f}<small style='font-size:.9rem;color:#555'>/100</small>" if score is not None else "—",
         "health, drift, quality, bias", "ok" if (score or 0) >= 80 else "warn"),
        ("Requests served", f'{health.get("total_requests", 0):,}', f'{health.get("overall_error_rate_pct", 0):.2f}% errors', ""),
        ("Open incidents", (dash.get("rca_summary") or {}).get("open_incidents", 0), "handled by the supervisor agent", ""),
    ])

    section("Travel services", "Live request counts and latency from the governance layer.")
    _board(SERVICES, pages, modules, "svc")

    section("Operations", "Where the models are watched, secured and fixed.")
    _board(GOVERNANCE, pages, modules, "gov")

    section("How every prediction is governed", "The same four steps run on every request.")
    steps = [("1", "Validate", "Inputs are checked before any model runs; bad inputs are refused."),
             ("2", "Predict & explain", "A score, a confidence and the reasons behind it (SHAP)."),
             ("3", "Monitor", "Latency, drift, live accuracy, bias and hallucination, all tracked."),
             ("4", "Self-heal", "Agents open incidents, retrain and roll back; a human approves releases.")]
    st.markdown('<div class="rb-grid" style="--n:4">' + "".join(
        f'<div class="rb-kpi"><div class="l"><span class="rb-mono" style="font-weight:700;color:#000">{i}</span> &nbsp;{t}</div>'
        f'<div class="s" style="margin-top:4px;font-size:.84rem;color:#1F1F1F">{d}</div></div>' for i, t, d in steps) + "</div>",
        unsafe_allow_html=True)

    section("Where the data comes from", "Real where open data exists; simulated, and labelled, where it does not.")
    rows = [
        ("Trains, halts, timings", "REAL", "Indian Railways timetable via datameet/railways (CC0)", "2016 snapshot"),
        ("Stations & coordinates", "REAL", "datameet/railways — 2,719 stations served by these trains", "2016"),
        ("Fares", "REAL", "Railway Board Tariff (TAG 2025) fare tables + standard charges", "2025"),
        ("Weather", "LIVE", "Open-Meteo forecast at the station's coordinates", "up to 15 days ahead"),
        ("Booking outcomes, delays, demand, crowding", "SIM", "Simulated on the real network (no public data exists)", "seeded"),
        ("Passenger profiles, fraud sessions", "SIM", "Simulated", "seeded"),
    ]
    st.markdown('<table class="rb-table"><thead><tr><th>Data</th><th>Type</th><th>Source</th><th>As of</th></tr></thead><tbody>'
                + "".join(f'<tr><td>{esc(a)}</td><td>{badge(b, "ok" if b in ("REAL", "LIVE") else "warn")}</td>'
                          f'<td>{esc(c)}</td><td class="rb-mono">{esc(d)}</td></tr>' for a, b, c, d in rows)
                + "</tbody></table>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# M1 — Ticket confirmation
# ──────────────────────────────────────────────────────────────────────────────

def ticket() -> None:
    inject_theme("ticket")
    hero("Ticket Confirmation Predictor",
         "Waitlisted or RAC? See the chance your ticket confirms before you travel, the likely berth, and exactly why.",
         ["XGBoost", "SHAP explanations", "Booking & cancellation trends", "Retrained when live accuracy drops"])
    provenance("timetable", "outcomes")
    ref = load_reference()
    trains = ref["trains"]

    presets([
        ("Almost confirmed", {"t_train": label_for(trains, "12951"), "t_cls": "3A", "t_quota": "GN", "t_wl": 2, "t_days": 14}),
        ("Tatkal, high WL", {"t_train": label_for(trains, "12951"), "t_cls": "2A", "t_quota": "TQ", "t_wl": 48, "t_days": 2}),
        ("Festival rush", {"t_train": label_for(trains, "12301"), "t_cls": "SL", "t_quota": "GN", "t_wl": 180, "t_days": 45}),
    ], "t_run", "tk")

    st.session_state.setdefault("t_wl", 15)
    st.session_state.setdefault("t_days", 7)
    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("ticket_form"):
            train = st.selectbox("Train", list(trains.keys()), key="t_train")
            c1, c2 = st.columns(2)
            cls = c1.selectbox("Class", CLASSES, key="t_cls")
            quota = c2.selectbox("Quota", ["GN", "TQ", "LADIES", "DEFENCE", "SENIOR"], key="t_quota")
            wl = c1.number_input("Waitlist / RAC no.", 1, 500, key="t_wl")
            days = c2.slider("Days to departure", 1, 60, key="t_days", help="Reservations open 60 days before the journey.")
            with st.expander("More options"):
                senior = st.checkbox("Senior citizen (60+)")
                disab = st.checkbox("Divyangjan quota")
            go_btn = st.form_submit_button("Predict confirmation", type="primary", width="stretch")
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
            st.markdown(meter(p * 100, "Chance of confirmation", "%", tone=tone), unsafe_allow_html=True)
            kpi_row([("Verdict", esc(res.get("pnr_status_prediction", "—")), "at chart preparation", tone),
                     ("Likely berth", esc(res.get("estimated_berth_type", "—")), esc(res.get("berth_estimate_basis", "Indicative estimate"))[:60], ""),
                     ("Model confidence", f'{res.get("confidence_score", 0):.0%}', "certainty of this prediction", "")])
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
                fig = go.Figure(go.Bar(x=list(bb.values()), y=list(bb.keys()), orientation="h", marker_color=POS,
                                       text=[f"{v:.0%}" for v in bb.values()], textposition="auto"))
                fig.update_layout(title="<b>Berth likelihood</b>")
                style_fig(fig, 300)
                st.plotly_chart(fig, width="stretch")
        hi = res.get("historical_insights") or {}
        if hi:
            section("Booking & cancellation trends", f'From {hi.get("comparable_bookings", 0):,} comparable past bookings ({esc(hi.get("scope", ""))}).')
            kpi_row([("Confirmed historically", f'{hi.get("historical_confirmed_pct", 0):.0f}%', "of comparable bookings", ""),
                     ("Ended in RAC", f'{hi.get("historical_rac_pct", 0):.0f}%', "seat shared, travel allowed", ""),
                     ("Stayed waitlisted", f'{hi.get("historical_wl_pct", 0):.0f}%', "auto-refunded", ""),
                     ("Cancellation rate", f'{hi.get("cancellation_rate_pct", 0):.0f}%', "passengers who cancelled", "")])
            by_m = hi.get("confirmed_pct_by_month") or {}
            if by_m:
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                xs = sorted(int(m) for m in by_m)
                fig = go.Figure(go.Bar(x=[months[m - 1] for m in xs], y=[by_m.get(m, by_m.get(str(m), 0)) * 100 for m in xs],
                                       marker_color=POS))
                fig.update_layout(title="<b>Confirmation rate by travel month</b>", yaxis_title="% confirmed")
                style_fig(fig, 250)
                st.plotly_chart(fig, width="stretch")
        outcome_box("t_out", res, "Report the final status after chart preparation", "choice",
                    options=["Confirmed", "RAC", "WL"])
        render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M2 — Train recommender
# ──────────────────────────────────────────────────────────────────────────────

ARRIVE_BY = ["Any time"] + [f"{h:02d}:{m:02d}" for h in range(5, 24) for m in (0, 30)]


def _connections(res: Dict[str, Any]) -> None:
    conns = res.get("connections") or []
    if not conns:
        return
    st.markdown('<div class="rb-sub" style="margin:10px 0 6px"><b>One-change connections</b> — useful when few direct '
                'trains run on this corridor. Plan and book one in the AI Travel Planner.</div>', unsafe_allow_html=True)
    for c in conns:
        legs = " &nbsp;→&nbsp; ".join(
            f'<b>{esc(l["train_name"])}</b> {esc(l["train_number"])} ({esc(l["source"])} {esc(l["departure_time"])} → '
            f'{esc(l["destination"])} {esc(l["arrival_time"])})' for l in c.get("legs", []))
        st.markdown(
            f'<div class="rb-card" style="padding:12px 16px"><div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">'
            f'<div>Change at <b>{esc(c.get("via"))}</b> · {c.get("total_hours", 0):.1f} h door to door · '
            f'{c.get("layover_minutes", 0) // 60} h {c.get("layover_minutes", 0) % 60:02d} min layover</div>'
            f'<div style="font-weight:800;color:var(--deep)">₹{c.get("total_fare_inr", 0):,.0f}'
            f'{"" if c.get("within_budget", True) else " " + badge("over budget", "warn")}</div></div>'
            f'<div class="rb-sub" style="margin-top:6px">{legs}</div></div>', unsafe_allow_html=True)


def recommend() -> None:
    inject_theme("recommend")
    hero("Smart Train Recommender",
         "The best train for you — balancing fare, speed, punctuality, availability and comfort for your travel style.",
         ["Persona-aware scoring", "Personalised from history", "Learns from your choices", "Fairness audit"])
    provenance("timetable", "fares", "outcomes")
    stations = load_reference()["stations"]
    names = list(stations.keys())

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("rec_form"):
            o = st.selectbox("From", names, index=names.index(label_for(stations, "NDLS")))
            d = st.selectbox("To", names, index=names.index(label_for(stations, "CSTM")))
            dt = st.date_input("Travel date", value=date.today() + timedelta(days=5), min_value=date.today(),
                               max_value=date.today() + timedelta(days=60))
            st.caption("Finds every train that stops at both stations — including ones that start or end elsewhere — "
                       "and other stations in the same city.")
            c1, c2 = st.columns(2)
            persona = c1.selectbox("I travel as", PERSONAS)
            cls = c2.selectbox("Preferred class", CLASSES)
            budget = c1.number_input("Max fare ₹ (0 = any)", 0, 20000, 0, step=500)
            arrive = c2.selectbox("Arrive by", ARRIVE_BY, help="Trains arriving later are ranked lower")
            uid = st.text_input("Passenger ID (optional, e.g. USR00001)", "",
                                help="Personalises using that passenger's travel history and usual class / budget.")
            go_btn = st.form_submit_button("Find best trains", type="primary", width="stretch")
    with right:
        if go_btn and stations[o] == stations[d]:
            callout("Origin and destination are the same — pick two different stations.", "warn", "🔁")
            go_btn = False
        if go_btn:
            with st.spinner("Scoring trains on your corridor…"):
                res = api_post("/recommend/trains", {
                    "source_station": stations[o], "destination_station": stations[d], "journey_date": str(dt),
                    "persona": persona, "preferred_class": cls, "max_budget": float(budget) if budget > 0 else None,
                    "arrival_deadline": None if arrive == ARRIVE_BY[0] else arrive, "user_id": uid.strip() or None})
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
                _connections(res)
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
                        f'<div class="rb-card" style="padding:0;{"border:2px solid #000" if top else ""}">'
                        f'<div class="rb-tkt">'
                        f'<div class="rk{" top" if top else ""}">{r.get("rank")}</div>'
                        f'<div><b>{esc(r.get("train_name"))}</b><br><span class="rb-mono rb-sub">{esc(r.get("train_number"))}</span>'
                        f'{" &nbsp;" + badge("Best match", "ok") if top else ""}</div>'
                        f'<div class="rb-mono"><b>{esc(r.get("departure_time"))}</b> {esc(r.get("source") or oc)}<br><b>{esc(r.get("arrival_time"))}</b> {esc(r.get("destination") or dc)}</div>'
                        f'<div><span class="rb-sub">Duration</span><br><span class="rb-mono">{r.get("duration_hours", 0):.1f} h</span></div>'
                        f'<div><span class="rb-sub">On time</span><br><span class="rb-mono">{r.get("punctuality_score", 0):.0%}</span></div>'
                        f'<div><span class="rb-sub">Match</span><br><span class="rb-mono">{r.get("composite_score", 0):.2f}</span></div>'
                        f'<div class="fare rb-mono">₹{r.get("fare_estimate_inr", 0):,.0f}'
                        f'<br><span class="rb-sub" style="font-size:.75rem">{esc(r.get("fare_class", ""))}</span></div></div>'
                        f'<div style="padding:8px 14px;border-top:1px solid #ddd"><span class="rb-sub">{esc(r.get("recommendation_reason"))}</span>'
                        f'<div style="margin-top:6px">{tags}</div></div></div>',
                        unsafe_allow_html=True)
                    fc = r.get("factor_contributions") or {}
                    b1, b2 = st.columns([3, 1])
                    if fc:
                        with b1.expander("Score breakdown"):
                            fs = r.get("factor_scores") or {}
                            for f, v in sorted(fc.items(), key=lambda kv: -kv[1]):
                                st.markdown(f'<div class="rb-row"><span class="k">{esc(f.title())} — score {fs.get(f, 0):.0%}</span>'
                                            f'<span class="v">+{v:.3f}</span></div>', unsafe_allow_html=True)
                    if audit and b2.button("I'd pick this", key=f"pick_{r.get('train_number')}", width="stretch"):
                        out = send_outcome(audit, r.get("train_number"))
                        if out and out.get("accepted"):
                            st.success("Thanks! Your choice is used to measure relevance and tune the ranking.")
                        else:
                            st.info((out or {}).get("error", "Choice already recorded."))
                _connections(res)
                if dirv >= 0.8:
                    callout(f"<b>Fairness audit passed</b> — Disparate Impact Ratio <b>{dirv:.2f}</b> (≥ 0.80). Rankings give fair exposure across fare tiers.", "ok", "⚖️")
                else:
                    callout(f"<b>Fairness review</b> — Disparate Impact Ratio <b>{dirv:.2f}</b> is below 0.80; an equity adjustment was applied.", "warn", "⚖️")
                w = res.get("weights_used") or {}
                if w:
                    with st.expander(f"How the score was built (persona weights, learned version {res.get('weights_version', 0)})"):
                        fig = go.Figure(go.Bar(x=list(w.keys()), y=list(w.values()), marker_color=POS,
                                               text=[f"{v:.0%}" for v in w.values()], textposition="auto"))
                        style_fig(fig, 240)
                        st.plotly_chart(fig, width="stretch")
                render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M3 — Delay & disruption
# ──────────────────────────────────────────────────────────────────────────────

def delay() -> None:
    inject_theme("delay")
    hero("Delay & Disruption Predictor",
         "Know before you leave: expected delay, station congestion, and better trains if things look bad.",
         ["LightGBM", "Arrival & departure delay", "Live weather", "Station congestion history", "SHAP", "Proactive alerts"])
    provenance("timetable", "weather", "outcomes")
    ref = load_reference()
    trains, stations = ref["trains"], ref["stations"]

    presets([
        ("Clear day", {"d_train": label_for(trains, "12002"), "d_station": label_for(stations, "BPL"), "d_weather": 0.05, "d_cong": 0.15, "d_auto": False}),
        ("Monsoon tracks", {"d_train": label_for(trains, "12952"), "d_station": label_for(stations, "BCT"), "d_weather": 0.70, "d_cong": 0.65, "d_auto": False}),
        ("Dense fog + congestion", {"d_train": label_for(trains, "12301"), "d_station": label_for(stations, "NDLS"), "d_weather": 0.95, "d_cong": 0.90, "d_auto": False}),
    ], "d_run", "dl")

    left, right = st.columns([1, 1.65], gap="large")
    with left:
        st.session_state.setdefault("d_auto", True)
        st.session_state.setdefault("d_weather", 0.25)
        st.session_state.setdefault("d_cong", 0.45)
        train = st.selectbox("Train", list(trains.keys()), key="d_train")
        halts = halt_labels(ref, trains[train])
        if st.session_state.get("d_station") not in halts:
            st.session_state["d_station"] = halts[-1]
        stn = st.selectbox("Station (this train's halts, in order)", halts, key="d_station")
        stop = next((h for h in ref["stops"].get(trains[train], []) if h[0] == stations.get(stn, stn)), None)
        if stop:
            st.caption(f"Scheduled {'arrival ' + stop[1] if stop[1] else 'departure ' + stop[2]} · "
                       f"{stop[3]:,.0f} km from {ref['stops'][trains[train]][0][0]} (2016 timetable)")
        auto = st.toggle("Estimate weather & congestion automatically", key="d_auto",
                         help="Weather: live Open-Meteo forecast at this station (up to 15 days ahead), otherwise the "
                              "station's usual weather for that time of year. Congestion: this station's own history.")
        with st.form("delay_form"):
            jd = st.date_input("Journey date", value=date.today() + timedelta(days=1), min_value=date.today(),
                               max_value=date.today() + timedelta(days=60))
            if auto:
                st.caption("Weather comes from the live forecast for the station and day; congestion from the station's "
                           "history. Switch the toggle off to set them yourself.")
                wx, cg = None, None
            else:
                wx = st.slider("Weather severity", 0.0, 1.0, step=0.05, key="d_weather",
                               help="0 = clear skies · 1 = heavy fog / flooding")
                cg = st.slider("Station congestion", 0.0, 1.0, step=0.05, key="d_cong",
                               help="0 = empty platforms · 1 = extremely crowded")
            go_btn = st.form_submit_button("Forecast delay", type="primary", width="stretch")
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
            dep = res.get("departure_delay_minutes")
            sci = res.get("station_congestion_index")
            kpi_row([("Arrival delay", f"+{mins:.0f}<small style='font-size:.9rem'> min</small>", "at the selected station", tone),
                     ("Departure delay", f"+{dep:.0f}<small style='font-size:.9rem'> min</small>" if dep is not None else "—", "from the origin", ""),
                     ("Likely range", f'{ci.get("lower_bound", 0):.0f}–{ci.get("upper_bound", 0):.0f}<small style="font-size:.9rem"> min</small>',
                      f'{ci.get("coverage", 0.8):.0%} prediction interval', ""),
                     ("Station congestion", esc(lvl), f'{sci:.0%} · {esc(res.get("congestion_source", ""))}' if sci is not None else "",
                      {"Low": "ok", "Moderate": "warn"}.get(lvl, "bad"))])
            wc = res.get("weather_conditions") or {}
            wline = (f"rain {wc.get('precip_mm', 0):.0f} mm · visibility {wc.get('min_visibility_m', 0) / 1000:.1f} km · "
                     f"wind {wc.get('wind_kmh', 0):.0f} km/h{' · thunderstorm' if wc.get('storm') else ''}") if wc else ""
            rows_card("Inputs used", [
                ("Weather", f"{esc(res.get('weather_source', ''))}" + (f" — {esc(wline)}" if wline else "")
                 + (f" · index {res['weather_index_used']:.1f}/10" if res.get("weather_index_used") is not None else "")),
                ("Distance from origin", mono("%s km" % format(res.get("distance_from_origin_km") or 0, ",.0f"))
                 + " · " + esc(res.get("distance_source", ""))),
                ("Scheduled time", mono(res.get("scheduled_time") or "—")),
            ])
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
                    y=[f'{d.get("factor", "")} ({d.get("display", d.get("value"))})' for d in drivers][::-1],
                    orientation="h", marker_color=[POS if d.get("impact_minutes", 0) >= 0 else NEG for d in drivers][::-1],
                    text=[f'{d.get("impact_minutes", 0):+.1f} min' for d in drivers][::-1], textposition="auto"))
                fig.update_layout(title=f"<b>Why this delay (SHAP)</b> — typical delay {base['impact_minutes']:.0f} min, each bar adds or removes minutes" if base else "<b>Why this delay (SHAP)</b>")
                style_fig(fig, 60 + 50 * len(drivers))
                st.plotly_chart(fig, width="stretch")
            for a in res.get("alternative_options", []):
                st.markdown(
                    f'<div class="rb-card"><div style="display:flex;justify-content:space-between;gap:8px"><div>'
                    f'<b style="color:var(--deep)">{esc(a.get("train_name"))}</b> <span class="rb-sub">{esc(a.get("train_number"))}</span><br>'
                    f'<span class="rb-sub">Departs {esc(a.get("departure_time"))} · Arrives {esc(a.get("arrival_time"))}</span><br>'
                    f'<span class="rb-sub">{esc(a.get("reason"))}</span></div>'
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


def _expiry(ts: Any) -> str:
    """'· approve within 27 min' — the approval gate expires plans after 30 minutes."""
    try:
        from datetime import datetime, timezone
        exp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        mins = int((exp - datetime.now(timezone.utc)).total_seconds() // 60)
        return f" · approve within {mins} min" if mins > 0 else " · <b>expired — plan again</b>"
    except Exception:
        return ""


def planner() -> None:
    inject_theme("planner")
    hero("AI Travel Planner",
         "Describe your trip in your own words. A team of AI agents searches trains, checks confirmation odds, delays and fares — "
         "then proposes a plan. <b>Nothing is booked until you approve.</b>",
         ["Multi-agent", "One-change connections", "Validator agent", "Human-in-the-loop booking", "Injection guard"])
    provenance("timetable", "fares", "weather", "outcomes")

    st.caption("Try an example")
    cols = st.columns(len(EXAMPLES))
    for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
        c.button(ex if len(ex) < 52 else ex[:50] + "…", key=f"ex{i}", width="stretch",
                 on_click=lambda e=ex: st.session_state.update(p_query=e))

    with st.form("plan_form"):
        q = st.text_area("Where do you want to go?", key="p_query", height=88,
                         value=st.session_state.get("p_query", EXAMPLES[0]))
        c1, c2, c3 = st.columns([1, 1, 1.2])
        persona = c1.selectbox("Traveller type", PERSONAS)
        budget = c2.number_input("Budget cap ₹ (0 = from message)", 0, 50000, 0, step=500)
        go_btn = c3.form_submit_button("Plan my journey", type="primary", width="stretch")

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
        st.markdown('<div class="rb-sub" style="margin:6px 0 2px">What the agent understood '
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
        return (f'<div class="rb-row"><span class="k"><b>{esc(l.get("train_name"))}</b> <span class="rb-mono">{esc(l.get("train_number"))}</span> · '
                f'<span class="rb-mono">{esc(l.get("source"))} {esc(l.get("departure", "")[11:16])} → {esc(l.get("destination"))} {esc(l.get("arrival", "")[11:16])}</span>'
                f' <span class="rb-sub rb-mono">({esc(l.get("arrival", "")[:10])})</span></span>'
                f'<span class="v">₹{l.get("fare_inr", 0):,.0f} · {l.get("confirmation_probability", 0):.0%} odds · +{l.get("predicted_delay_minutes", 0):.0f} min</span></div>')

    section("Your proposed itinerary", f'Request <code>{esc(plan.get("request_id"))}</code>' + _expiry(plan.get("expires_at")))
    exp = rec.get("expected_arrival_with_delay") or ""
    st.markdown(
        f'<div class="rb-card" style="border:2px solid var(--accent)">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
        f'<div style="font-size:1.15rem;font-weight:800;color:var(--deep)">{esc(rec.get("option_id"))} · {esc(rec.get("train_name"))}</div>'
        f'{badge("Approved" if status == "APPROVED" else "Awaiting your approval" if status == "PENDING_APPROVAL" else status, "ok" if status == "APPROVED" else "warn")}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;margin:14px 0;flex-wrap:wrap">'
        f'<div><div class="rb-mono" style="font-size:1.6rem;font-weight:600">{esc(rec.get("departure_time"))}</div><div class="rb-sub">{esc(rec.get("source"))} · {esc((rec.get("departure_datetime") or "")[:10])}</div></div>'
        f'<div style="flex:1;min-width:120px;text-align:center"><div class="rb-sub">{rec.get("duration_hours", 0):.1f} h · Class {esc(rec.get("travel_class"))}'
        f'{" · 1 change" if rec.get("changes") else " · direct"}</div>'
        f'<div style="border-top:2px dashed #000;margin:6px 0"></div></div>'
        f'<div><div class="rb-mono" style="font-size:1.6rem;font-weight:600">{esc(rec.get("arrival_time"))}</div><div class="rb-sub">{esc(rec.get("destination"))} · {esc((rec.get("arrival_datetime") or "")[:10])}</div></div></div>'
        + "".join(leg_html(l) for l in rec.get("legs", [])) +
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">'
        f'{badge("₹%s total" % format(rec.get("estimated_fare_inr", 0), ",.0f"), "info")} '
        f'{badge("%.0f%% confirmation odds" % (rec.get("confirmation_probability", 0) * 100), "ok" if rec.get("confirmation_probability", 0) >= .6 else "warn")} '
        f'{badge("Expected arrival " + exp[11:16] + " incl. delay", "info") if exp else ""} '
        f'{badge("Meets all your constraints", "ok") if rec.get("meets_all_constraints") else badge("Closest match — misses a constraint", "warn")}</div>'
        f'</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="rb-card"><h4>Why it fits</h4>' + ("".join(f'<div style="padding:4px 0">{esc(p)}</div>' for p in rec.get("pros", [])) or "<div class=rb-sub>—</div>") + "</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="rb-card"><h4>Keep in mind</h4>' + ("".join(f'<div style="padding:4px 0">• {esc(c)}</div>' for c in rec.get("cons", [])) or '<div style="padding:4px 0">No major concerns</div>') + "</div>", unsafe_allow_html=True)
    callout(f"<b>Agent reasoning:</b> {esc(plan.get('agent_reasoning_summary', ''))}", "info", "🧠")
    val = plan.get("validation") or {}
    if val:
        checks = val.get("checks") or {}
        st.markdown('<div class="rb-sub" style="margin:4px 0">Validator agent: '
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
            if c2.button("Review", key=f"opt_{o.get('option_id')}", width="stretch"):
                st.session_state["plan_choice"] = o.get("option_id")
                st.rerun()

    # Human-in-the-loop gate
    if booking:
        st.markdown(
            f'<div class="rb-card" style="border-left:4px solid #13703A"><h4>Booking prepared</h4>'
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
            yes = ok.form_submit_button(f"Approve {rec.get('option_id')} & prepare booking", type="primary", width="stretch")
            rej = no.form_submit_button("Reject & re-plan", width="stretch")
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

    with st.expander("Agent execution trace (audit trail)"):
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
    hero("Fare Forecast & Booking Window",
         "The official fare for your exact journey, how it moves as seats sell on flexi-fare trains, and when to book.",
         ["Official TAG 2025 fares", "Flexi-fare forecast", "Tatkal rules", "SHAP fare drivers", "Pricing-anomaly guard"])
    provenance("timetable", "fares", "outcomes")
    ref = load_reference()
    trains = ref["trains"]
    left, right = st.columns([1, 1.65], gap="large")
    with left:
        train = st.selectbox("Train", list(trains.keys()), key="f_train")
        num = trains[train]
        halts = halt_labels(ref, num)
        if st.session_state.get("f_from") not in halts[:-1]:
            st.session_state["f_from"] = halts[0]
        if st.session_state.get("f_to") not in halts[1:]:
            st.session_state["f_to"] = halts[-1]
        c1, c2 = st.columns(2)
        frm = c1.selectbox("From", halts[:-1], key="f_from")
        to = c2.selectbox("To", halts[1:], key="f_to")
        offered = [c for c in CLASSES if c in ref["classes"].get(num, CLASSES)] or CLASSES
        with st.form("fare_form"):
            cls = st.selectbox("Class", offered)
            jd = st.date_input("Journey date", value=date.today() + timedelta(days=20), min_value=date.today(),
                               max_value=date.today() + timedelta(days=60))
            tk = st.checkbox("Tatkal quota", help="Tatkal opens one day before the journey and adds a fixed charge.")
            go_btn = st.form_submit_button("Forecast fare", type="primary", width="stretch")
    with right:
        bad_pair = halts.index(frm) >= halts.index(to) if frm in halts and to in halts else False
        if go_btn and bad_pair:
            callout("The train reaches the 'To' station before the 'From' station — pick a later 'To' station.", "warn")
            go_btn = False
        if go_btn:
            with st.spinner("Pricing the journey…"):
                res = api_post("/predict/fare", {"train_number": num, "journey_class": cls, "journey_date": str(jd),
                                                 "tatkal_flag": tk, "source_station": ref["stations"].get(frm),
                                                 "destination_station": ref["stations"].get(to)})
            st.session_state["f_res"] = res
        res = st.session_state.get("f_res")
        if not res:
            _empty("Your fare forecast will appear here")
        elif "current_predicted_fare_inr" not in res:
            _err(res, "the fare forecast")
        else:
            dem = res.get("demand_level", "—")
            rule = res.get("pricing_rule", "")
            kpi_row([("Fare", f'₹{res["current_predicted_fare_inr"]:,.0f}', f'standard ₹{res.get("base_fare_inr", 0):,.0f}', ""),
                     ("Pricing", {"flexi": "Flexi", "fixed": "Fixed", "tatkal": "Tatkal"}.get(rule, "—"),
                      {"flexi": "rises as seats sell", "fixed": "same on any booking day", "tatkal": "Tatkal charge added"}.get(rule, ""),
                      "ok" if rule == "fixed" else "warn"),
                     ("Surge", f'{res.get("surge_multiplier", 1):.2f}×', "vs. standard fare",
                      "ok" if res.get("surge_multiplier", 1) < 1.1 else "warn"),
                     ("Demand", esc(dem), "booking pressure", {"Low": "ok", "Moderate": "ok", "High": "warn"}.get(dem, "bad"))
                     if rule != "fixed" else
                     ("Distance", f'{float((res.get("fare_breakdown") or {}).get("distance_km", 0)):,.0f} km', "for this journey", "")])
            callout(f"<b>Best time to book:</b> {esc(res.get('optimal_booking_window', ''))}"
                    + (f" &nbsp;·&nbsp; Demand surge expected from <b>{res['demand_surge_from_days']} days</b> before departure."
                       if res.get("demand_surge_from_days") is not None and rule != "fixed" else ""), "info")
            if res.get("pricing_anomaly"):
                callout(f"<b>Pricing check:</b> {esc(res['pricing_anomaly'])} The governance layer logged this for review.", "warn")
            if res.get("policy_note"):
                st.caption(esc(res["policy_note"]))
            fb = res.get("fare_breakdown") or {}
            if fb:
                def inr(v: float) -> str:
                    return mono("₹" + format(float(v or 0), ",.0f"))
                rows_card(f"Standard fare · {esc(res.get('source_station') or '')} → {esc(res.get('destination_station') or '')} · "
                          f"{float(fb.get('distance_km', 0)):,.0f} km",
                          [("Base fare (" + str(fb.get("tariff_table", "")) + " table)", inr(fb.get("base_fare"))),
                           ("Reservation charge", inr(fb.get("reservation_charge"))),
                           ("Superfast charge", inr(fb.get("superfast_charge"))),
                           ("Tatkal charge", inr(fb.get("tatkal_charge"))),
                           ("GST (5%, AC classes)", inr(fb.get("gst"))),
                           ("Total payable", "<b>" + inr(fb.get("total")) + "</b>")],
                          note="Base fare from the Railway Board TAG 2025 fare tables for this distance. Catering on "
                               "Rajdhani/Duronto/Shatabdi and concessions are not included — confirm on IRCTC.")
            curve = res.get("forecast_curve") or []
            if curve:
                df = pd.DataFrame(curve)
                fig = go.Figure(go.Scatter(x=df["days_to_departure"], y=df["predicted_fare_inr"], mode="lines+markers",
                                           line=dict(color=POS, width=2), marker=dict(size=5),
                                           name="Predicted fare"))
                fig.add_hline(y=res.get("base_fare_inr", 0), line_dash="dash", line_color="#7A7A7A",
                              annotation_text="Standard fare", annotation_position="bottom right")
                fig.update_layout(title="<b>Fare vs days before departure</b>", xaxis_title="Days before departure",
                                  yaxis_title="₹", xaxis_autorange="reversed", yaxis_tickformat=",.0f")
                if rule == "fixed":
                    fig.update_yaxes(range=[0, float(res["current_predicted_fare_inr"]) * 1.2])
                style_fig(fig, 300)
                st.plotly_chart(fig, width="stretch")
    res = st.session_state.get("f_res")
    if res and "current_predicted_fare_inr" in res:
        alts = res.get("cost_saving_alternatives") or []
        if alts:
            section("Ways to save", "Small changes that lower your fare.")
            cols = st.columns(min(3, len(alts)))
            for c, a in zip(cols, alts):
                c.markdown(f'<div class="rb-card"><div class="rb-sub">{esc(a.get("alternative_type"))}</div>'
                           f'<div class="rb-mono" style="font-size:1.35rem;font-weight:600;color:#13703A">Save ₹{a.get("potential_savings_inr", 0):,.0f}</div>'
                           f'<div class="rb-sub">{esc(a.get("description"))}</div></div>', unsafe_allow_html=True)
        if res.get("fare_drivers"):
            section("Why this fare?", "SHAP: how much each factor adds to or removes from the fare, in ₹.")
            render_shap_chart(res["fare_drivers"], title="Fare drivers (SHAP)", target_metric="₹ impact")
        outcome_box("f_out", res, "Report the fare you actually paid", "number", input_label="Fare paid (₹)",
                    value=float(res.get("current_predicted_fare_inr", 0)), step=10.0)
        render_telemetry_badge(res.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# M7 — Fraud & bot shield
# ──────────────────────────────────────────────────────────────────────────────

def fraud() -> None:
    inject_theme("tower")
    hero("Fraud & Bot Shield",
         "Security operations for the booking platform: every session is scored for bot, scalper and account-misuse risk, "
         "each alert carries its evidence, and analyst verdicts feed live false-positive / false-negative monitoring.",
         ["AI Governance · security", "XGBoost classifier", "Anomaly detection", "Account-misuse checks", "SHAP investigations"])
    provenance("fraud")
    presets([
        ("Genuine passenger", {"f_rpm": 4.0, "f_pnr": 1, "f_tatkal": False, "f_cap": 0, "f_age": 365, "f_pay": 1, "f_ip": 0.10, "f_bulk": False, "f_sess": "SESS-GENUINE-PASSENGER"}),
        ("Fast Tatkal user", {"f_rpm": 35.0, "f_pnr": 4, "f_tatkal": True, "f_cap": 1, "f_age": 45, "f_pay": 2, "f_ip": 0.40, "f_bulk": False, "f_sess": "SESS-TATKAL-FREQUENT"}),
        ("Scalper bot", {"f_rpm": 120.0, "f_pnr": 25, "f_tatkal": True, "f_cap": 6, "f_age": 1, "f_pay": 6, "f_ip": 0.85, "f_bulk": True, "f_sess": "SESS-SCALPER-BOTNET-99"}),
    ], "f_run", "fr")

    for k_, v_ in {"f_rpm": 20.0, "f_pnr": 2, "f_age": 180, "f_pay": 1, "f_cap": 0, "f_tatkal": False,
                   "f_bulk": False, "f_sess": "SESS-DEMO-001", "f_ip": 0.1}.items():
        st.session_state.setdefault(k_, v_)
    left, right = st.columns([1, 1.65], gap="large")
    with left:
        with st.form("fraud_form"):
            rpm = st.slider("Requests per minute", 1.0, 150.0, step=1.0, key="f_rpm", format="%.0f")
            c1, c2 = st.columns(2)
            pnr = c1.number_input("PNR lookups", 1, 50, key="f_pnr")
            age = c2.number_input("Account age (days)", 0, 2000, key="f_age")
            pay = c1.number_input("Payment attempts", 1, 20, key="f_pay")
            cap = c2.slider("Failed captchas", 0, 10, key="f_cap")
            tk = st.checkbox("During Tatkal window (10:00–10:15)", key="f_tatkal")
            bulk = st.checkbox("Bulk booking (6+ passengers)", key="f_bulk")
            with st.expander("Session details"):
                sess = st.text_input("Session ID", key="f_sess")
                ip = st.slider("Network risk score", 0.0, 1.0, step=0.05, key="f_ip")
            go_btn = st.form_submit_button("Score this session", type="primary", width="stretch")
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
            st.markdown(meter(sc, "Fraud risk score", "/ 100", tone=tone, decimals=0), unsafe_allow_html=True)
            msg = {"BLOCK_AND_REPORT": "Block & report", "RATE_LIMIT": "Rate-limit", "STEP_UP_CHALLENGE": "Extra verification", "ALLOW": "Allow"}.get(act, act)
            kpi_row([("Threat level", esc(cat), "model verdict", tone), ("Action taken", msg, "automated security policy", tone)])
    res = st.session_state.get("fr_res")
    if res and "fraud_risk_score" in res:
        inv = res.get("investigation_features") or []
        if inv:
            section("Investigation evidence", "Signals the model examined, flagged by severity.")
            st.markdown('<div class="rb-list">' + "".join(
                f'<div class="rb-row"><span class="k">{esc(it.get("description"))}</span>'
                f'<span class="v">{badge(it.get("severity", ""), {"High Anomaly": "bad", "Suspicious": "warn"}.get(it.get("severity"), "ok"))}</span></div>'
                for it in inv) + "</div>", unsafe_allow_html=True)
        if res.get("shap_risk_drivers"):
            pred = (res.get("incident_dossier") or {}).get("predicted_label", "")
            render_shap_chart(res["shap_risk_drivers"], title=f"Why the model says '{pred}' (SHAP)", target_metric="Impact on that class")
        audit = (res.get("governance_telemetry") or {}).get("audit_id")
        if audit:
            with st.expander("Security analyst verdict (feeds false-positive / false-negative monitoring)"):
                c1, c2 = st.columns([2, 1])
                verdict = c1.selectbox("Confirmed label after investigation", ["legitimate", "suspicious", "bot"], key="fr_verdict")
                if c2.button("Record verdict", width="stretch"):
                    r = api_post("/governance/outcomes", {"audit_id": audit, "actual": verdict, "source": "analyst"})
                    if r and r.get("accepted"):
                        st.success("Verdict recorded — it now counts towards the live false-positive / false-negative rates.")
                    else:
                        st.warning(f"Not recorded: {(r or {}).get('error', 'unknown error')}")
        render_telemetry_badge(res.get("governance_telemetry"))

    alerts = api_get("/security/alerts", {"n": 60}) or []
    perf = api_get("/governance/performance") or []
    m7 = next((p_ for p_ in perf if p_.get("module_id") == "m7_fraud_detector"), {}) if isinstance(perf, list) else {}
    mm = m7.get("metrics") or {}
    fpr, fnr = mm.get("fpr"), mm.get("fnr")
    section("Security operations", "Live from the governance layer — every alert carries its evidence so an analyst can investigate it.")
    kpi_row([
        ("Alerts raised", len(alerts), "recent sessions flagged", "warn" if alerts else "ok"),
        ("Blocked & reported", sum(1 for a in alerts if a.get("action") == "BLOCK_AND_REPORT"), "bot / scalper sessions", ""),
        ("False-positive rate", f"{fpr:.1%}" if fpr is not None else "—",
         f'{m7.get("n_outcomes", 0)} analyst verdicts · limit 12%', "bad" if fpr is not None and fpr > 0.12 else "ok"),
        ("False-negative rate", f"{fnr:.1%}" if fnr is not None else "—", "missed threats · limit 15%",
         "bad" if fnr is not None and fnr > 0.15 else "ok"),
    ])
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if not alerts:
        st.caption("No alerts yet — try the Scalper bot scenario above.")
    if alerts:
        body = "".join(
            f'<tr><td class="m">{esc(str(a.get("at", ""))[11:19])}</td><td class="m">{esc(a.get("session_id"))}</td>'
            f'<td>{esc(", ".join(a.get("why", [])[:4]))}</td><td>{badge(a.get("category", ""), None)}</td>'
            f'<td>{esc(str(a.get("action", "")).replace("_", " ").capitalize())}</td><td class="m r">{esc(a.get("risk_score"))}</td></tr>'
            for a in alerts[:10])
        st.markdown('<table class="rb-table"><thead><tr><th>Time (UTC)</th><th>Session</th><th>Evidence</th><th>Category</th>'
                    f'<th>Action</th><th class="r">Risk</th></tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)
