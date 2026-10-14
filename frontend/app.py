"""
frontend/app.py
────────────────
AI Railway Autonomous Governance — Streamlit Control Tower Dashboard.

Multi-page Streamlit application providing:
  🏠 Home / Executive Overview
  📊 Platform Health & SLA Monitor
  📉 Statistical Drift Detector
  ⚖️  Fairness & Bias Auditor
  🧠 LLM Hallucination Monitor
  🚨 Incidents & RCA Engine
  🔄 Model Lifecycle Manager
  📋 Governance Reports
  🧪 Chaos Engineering Lab
  🤖 AI Travel Assistant (Module 4 Planner)
  🎫 Ticket Confirmation Predictor
  🚄 Train Recommender
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
PAGE_REFRESH_INTERVAL = 30  # seconds

st.set_page_config(
    page_title="AI Railway Governance Control Tower",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium dark theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1530 0%, #111827 100%);
        border-right: 1px solid #1e2d4d;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a2340 0%, #1e2d52 100%);
        border: 1px solid #2d4170;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .kpi-delta-good {
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .kpi-delta-bad {
        color: #f87171;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Status badges */
    .badge-healthy {
        background: linear-gradient(135deg, #065f46, #064e3b);
        color: #6ee7b7;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #059669;
    }
    .badge-warning {
        background: linear-gradient(135deg, #78350f, #92400e);
        color: #fde68a;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #d97706;
    }
    .badge-critical {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fca5a5;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #ef4444;
    }
    .badge-info {
        background: linear-gradient(135deg, #1e3a5f, #1e3a8a);
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #3b82f6;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        border-bottom: 2px solid #2d4170;
        padding-bottom: 8px;
        margin: 20px 0 16px 0;
    }

    /* Governance score ring */
    .gov-score-container {
        text-align: center;
        padding: 20px;
    }
    .gov-score-value {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Data tables */
    .stDataFrame {
        background: #1a2340 !important;
        border: 1px solid #2d4170;
        border-radius: 8px;
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #fca5a5;
    }
    .alert-warning {
        background: linear-gradient(135deg, #451a03, #78350f);
        border-left: 4px solid #f97316;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #fed7aa;
    }
    .alert-success {
        background: linear-gradient(135deg, #052e16, #065f46);
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #6ee7b7;
    }
    .alert-info {
        background: linear-gradient(135deg, #0c1a3a, #1e3a5f);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #93c5fd;
    }

    /* Pillar cards */
    .pillar-card {
        background: linear-gradient(135deg, #1a2340 0%, #1e2d52 100%);
        border: 1px solid #2d4170;
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        transition: all 0.3s ease;
    }
    .pillar-card:hover {
        border-color: #60a5fa;
        box-shadow: 0 0 24px rgba(96, 165, 250, 0.15);
    }
    .pillar-title {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .pillar-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }

    /* Sidebar nav */
    .stRadio > div > label {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }
    .stRadio > div > label:hover {
        color: #60a5fa !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }

    /* Toast/success messages */
    .stSuccess {
        background: #052e16 !important;
        color: #6ee7b7 !important;
    }
    .stError {
        background: #450a0a !important;
        color: #fca5a5 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# API Client helpers
# ──────────────────────────────────────────────────────────────────────────────

def api_get(path: str, params: Optional[dict] = None) -> Optional[Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: Optional[dict] = None) -> Optional[Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def backend_online() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def status_badge(status: str) -> str:
    status_upper = str(status).upper()
    if status_upper in ("HEALTHY", "STABLE", "COMPLIANT", "FAIR", "CLOSED", "OK"):
        return f'<span class="badge-healthy">✅ {status}</span>'
    elif status_upper in ("WARNING", "DEGRADED", "AT_RISK", "HALF_OPEN"):
        return f'<span class="badge-warning">⚠️ {status}</span>'
    elif status_upper in ("CRITICAL", "DRIFT", "BIAS_DETECTED", "OPEN", "FRAUD"):
        return f'<span class="badge-critical">🔴 {status}</span>'
    else:
        return f'<span class="badge-info">ℹ️ {status}</span>'


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size:2.2rem">🚄</div>
            <div style="font-size:1.1rem; font-weight:800; color:#60a5fa; line-height:1.2">
                AI Railway
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px">
                Governance Control Tower
            </div>
        </div>
        <hr style="border-color:#2d4170; margin:12px 0;">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "📊 Platform Health & SLA",
            "📉 Drift Detector",
            "⚖️ Bias & Fairness Auditor",
            "🧠 Hallucination Monitor",
            "🚨 Incidents & RCA",
            "🔄 Model Lifecycle",
            "📋 Governance Reports",
            "🧪 Chaos Engineering Lab",
            "🤖 AI Travel Assistant",
            "🎫 Ticket Predictor",
            "🚄 Train Recommender",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2d4170;'>", unsafe_allow_html=True)

    # API Status indicator
    online = backend_online()
    if online:
        st.markdown(
            '<div class="alert-success" style="font-size:0.8rem;">🟢 <b>API Online</b> — All systems operational</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-warning" style="font-size:0.8rem;">🟡 <b>API Offline</b> — Start the FastAPI server with <code>python run_platform.py</code></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="font-size:0.72rem; color:#475569; text-align:center; margin-top:16px;">
            Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page: Executive Overview
# ──────────────────────────────────────────────────────────────────────────────

if page == "🏠 Executive Overview":
    st.markdown(
        """
        <div style="padding: 8px 0 24px 0;">
            <h1 style="font-size:2.4rem; font-weight:900; color:#e2e8f0; margin:0;">
                🚄 AI Railway Autonomous Governance
            </h1>
            <p style="color:#94a3b8; font-size:1.05rem; margin-top:6px;">
                Enterprise-grade AI Governance Control Tower — Real-time visibility, autonomous remediation & compliance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dashboard = api_get("/governance/supervisor/dashboard")

    if dashboard:
        gov_score = dashboard.get("governance_score", 0)
        health = dashboard.get("platform_health", {})
        rca_sum = dashboard.get("rca_summary", {})
        lifecycle = dashboard.get("lifecycle", {})
        drift_sum = dashboard.get("drift_summary", {})
        hall_stats = dashboard.get("hallucination_stats", {})

        # ── KPI Row ───────────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)

        score_color = "#34d399" if gov_score >= 80 else "#fbbf24" if gov_score >= 60 else "#f87171"
        with col1:
            st.markdown(
                f"""
                <div class="kpi-card" style="text-align:center;">
                    <div class="kpi-value" style="background:linear-gradient(135deg,{score_color},{score_color}aa);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {gov_score}
                    </div>
                    <div class="kpi-label">Governance Score</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        platform_status = health.get("platform_status", "UNKNOWN")
        status_icon = "✅" if platform_status == "HEALTHY" else "⚠️" if platform_status == "DEGRADED" else "🔴"
        with col2:
            st.markdown(
                f"""
                <div class="kpi-card" style="text-align:center;">
                    <div style="font-size:2rem;">{status_icon}</div>
                    <div class="kpi-value" style="font-size:1.2rem;">{platform_status}</div>
                    <div class="kpi-label">Platform Status</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        total_req = health.get("total_requests", 0)
        with col3:
            st.markdown(
                f"""
                <div class="kpi-card" style="text-align:center;">
                    <div class="kpi-value">{total_req:,}</div>
                    <div class="kpi-label">Total Inferences</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        open_inc = rca_sum.get("open_incidents", 0)
        critical_inc = rca_sum.get("critical_incidents", 0)
        inc_color = "#f87171" if critical_inc > 0 else "#fbbf24" if open_inc > 0 else "#34d399"
        with col4:
            st.markdown(
                f"""
                <div class="kpi-card" style="text-align:center;">
                    <div class="kpi-value" style="background:linear-gradient(135deg,{inc_color},{inc_color}aa);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {open_inc}
                    </div>
                    <div class="kpi-label">Open Incidents</div>
                    <div style="font-size:0.75rem; color:#f87171; margin-top:4px;">{critical_inc} Critical</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        retrain_q = len(lifecycle.get("retraining_queue", []))
        with col5:
            st.markdown(
                f"""
                <div class="kpi-card" style="text-align:center;">
                    <div class="kpi-value">{retrain_q}</div>
                    <div class="kpi-label">Retraining Queue</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Governance Pillars Status ─────────────────────────────────────────
        st.markdown('<div class="section-header">🏛️ Governance Pillar Status</div>', unsafe_allow_html=True)

        pillars = [
            ("📊", "Health Monitor", "SLA & Circuit Breaker", platform_status),
            ("📉", "Drift Detector", f"{drift_sum.get('recent_drift_events', 0)} drift events", "STABLE"),
            ("⚖️", "Bias Auditor", "DIR / DPD / EOD", "FAIR"),
            ("🧠", "Hallucination Guard", f"{hall_stats.get('total_quarantined', 0)} quarantined", "STABLE"),
            ("🚨", "RCA Engine", f"{rca_sum.get('total_incidents', 0)} total incidents", "HEALTHY"),
            ("🔄", "Lifecycle Manager", f"{len(lifecycle.get('models_registered', []))} models tracked", "HEALTHY"),
            ("📋", "Report Generator", "Auto-report enabled", "HEALTHY"),
            ("🤖", "Supervisor Agent", "Running" if dashboard.get("supervisor_running") else "Idle", "HEALTHY"),
        ]

        cols = st.columns(4)
        for i, (icon, title, subtitle, status) in enumerate(pillars):
            with cols[i % 4]:
                status_str = status_badge(status)
                st.markdown(
                    f"""
                    <div class="pillar-card">
                        <div class="pillar-icon">{icon}</div>
                        <div class="pillar-title">{title}</div>
                        <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:8px;">{subtitle}</div>
                        {status_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if i % 4 == 3 or i == len(pillars) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)

        # ── Recent Telemetry ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📡 Live Telemetry Stream</div>', unsafe_allow_html=True)
        recent_tel = dashboard.get("recent_telemetry", [])
        if recent_tel:
            import pandas as pd
            tel_df = pd.DataFrame(recent_tel)
            if "module_id" in tel_df.columns:
                display_cols = ["module_id", "event_type", "severity", "timestamp"]
                show_cols = [c for c in display_cols if c in tel_df.columns]
                st.dataframe(
                    tel_df[show_cols].tail(15),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.markdown(
                '<div class="alert-info">ℹ️ No telemetry events yet. Start making API calls to generate events.</div>',
                unsafe_allow_html=True,
            )

        # ── Action Log ────────────────────────────────────────────────────────
        recent_actions = dashboard.get("action_log_recent", [])
        if recent_actions:
            st.markdown('<div class="section-header">🛡️ Recent Governance Actions</div>', unsafe_allow_html=True)
            import pandas as pd
            act_df = pd.DataFrame(recent_actions)
            st.dataframe(act_df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            """
            <div class="alert-warning">
                <b>⚠️ Cannot connect to the FastAPI backend.</b><br>
                Start the platform with: <code>python run_platform.py</code><br>
                Then navigate to <a href="http://localhost:8000/docs" style="color:#fbbf24">http://localhost:8000/docs</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show static demo KPIs when offline
        st.markdown('<div class="section-header">📋 Platform Architecture</div>', unsafe_allow_html=True)
        arch_cols = st.columns(3)
        with arch_cols[0]:
            st.info("**Stage 1** ✅\nSynthetic Data Generator\nML Training Pipeline\nDual-mode LLM Router")
        with arch_cols[1]:
            st.info("**Stage 2** ✅\nModules 1–7 (FastAPI)\nPredictive Engines\nMulti-Agent Planner")
        with arch_cols[2]:
            st.info("**Stage 3** ✅\nAutonomous Governance\nControl Tower Dashboard\nSupervisor Agent")

    # Quick report trigger
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        if st.button("📋 Generate Governance Report"):
            with st.spinner("Generating full audit report..."):
                report = api_post("/governance/supervisor/report")
            if report and "report_id" in report:
                st.success(f"✅ Report generated: **{report['report_id']}** — Risk: `{report.get('overall_risk_level', 'N/A')}`")
            else:
                st.error("Failed to generate report. Ensure the backend is running.")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Platform Health & SLA
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📊 Platform Health & SLA":
    st.markdown(
        '<h1 style="color:#e2e8f0;">📊 Platform Health & SLA Monitor</h1>'
        '<p style="color:#94a3b8;">Real-time circuit breakers, error rates, and latency percentiles per module.</p>',
        unsafe_allow_html=True,
    )

    health = api_get("/governance/health/platform")
    sla_report = api_get("/governance/health/sla") or []

    if health:
        import pandas as pd

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Platform Status", health.get("platform_status", "—"))
        col2.metric("Total Requests", f"{health.get('total_requests', 0):,}")
        col3.metric("Overall Error Rate", f"{health.get('overall_error_rate_pct', 0):.2f}%")
        open_cb = health.get("open_circuits", [])
        col4.metric("Open Circuit Breakers", len(open_cb), delta=f"{len(open_cb)} OPEN" if open_cb else None,
                    delta_color="inverse")

        if open_cb:
            st.markdown(
                f'<div class="alert-critical">⚡ <b>Circuit Breakers OPEN:</b> {", ".join(open_cb)}</div>',
                unsafe_allow_html=True,
            )

        # Module breakdown
        st.markdown('<div class="section-header">Module Health Breakdown</div>', unsafe_allow_html=True)
        modules = health.get("modules", {})
        if modules:
            module_rows = []
            for mod_id, stats in modules.items():
                module_rows.append({
                    "Module": mod_id,
                    "Requests": stats.get("total_requests", 0),
                    "Errors": stats.get("total_errors", 0),
                    "Error Rate %": stats.get("error_rate_pct", 0),
                    "Avg Latency (ms)": stats.get("avg_latency_ms", 0),
                    "P95 (ms)": stats.get("p95_ms", 0),
                    "P99 (ms)": stats.get("p99_ms", 0),
                    "Circuit": stats.get("circuit_state", "—"),
                    "SLA Breaches": stats.get("sla_breaches_total", 0),
                })
            df = pd.DataFrame(module_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No module health data yet. Make API calls to populate metrics.")

        # SLA compliance
        st.markdown('<div class="section-header">SLA Compliance Report</div>', unsafe_allow_html=True)
        if sla_report:
            for sla in sla_report:
                compliant = sla.get("compliant", True)
                icon = "✅" if compliant else "🔴"
                breaches = sla.get("breaches", [])
                st.markdown(
                    f"""
                    <div class="{'alert-success' if compliant else 'alert-critical'}">
                        {icon} <b>{sla.get('module_id', 'unknown')}</b> —
                        P95: {sla.get('p95_ms', 0)}ms |
                        Error Rate: {sla.get('error_rate_pct', 0)}% |
                        Breaches: {sla.get('sla_breaches_total', 0)}
                        {'<br>⚠️ ' + ' | '.join(breaches) if breaches else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No SLA data available yet.")
    else:
        st.error("❌ Cannot reach the backend. Start with: `python run_platform.py`")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Drift Detector
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📉 Drift Detector":
    st.markdown(
        '<h1 style="color:#e2e8f0;">📉 Statistical Drift Detector</h1>'
        '<p style="color:#94a3b8;">Kolmogorov-Smirnov test + Population Stability Index (PSI) across all models.</p>',
        unsafe_allow_html=True,
    )

    drift_results = api_get("/governance/drift/all") or []
    drift_history = api_get("/governance/drift/history", {"n": 10}) or []

    if drift_results:
        import pandas as pd

        # Summary row
        drifting = [d for d in drift_results if d.get("overall_drift_detected")]
        col1, col2, col3 = st.columns(3)
        col1.metric("Models Monitored", len(drift_results))
        col2.metric("Models Drifting", len(drifting), delta=f"{len(drifting)} DRIFT" if drifting else None,
                    delta_color="inverse")
        col3.metric("Total Drift Events", len(drift_history))

        for dr in drift_results:
            drift_badge = status_badge("DRIFT" if dr.get("overall_drift_detected") else "STABLE")
            with st.expander(
                f"🤖 {dr['model_id']} — {drift_badge}",
                expanded=dr.get("overall_drift_detected", False),
            ):
                st.markdown(f"**Status:** {dr.get('status', 'N/A')} &nbsp;|&nbsp; **Features Monitored:** {dr.get('features_monitored', 0)}")

                feat_drift = dr.get("feature_drift", [])
                if feat_drift:
                    feat_rows = []
                    for fd in feat_drift:
                        if isinstance(fd, dict) and "feature" in fd:
                            feat_rows.append({
                                "Feature": fd.get("feature", "—"),
                                "KS Statistic": fd.get("ks_statistic", "—"),
                                "KS p-value": fd.get("ks_pvalue", "—"),
                                "PSI": fd.get("psi", "—"),
                                "PSI Severity": fd.get("psi_severity", "—"),
                                "Mean Shift %": fd.get("mean_shift_pct", "—"),
                                "Drift": "🔴 YES" if fd.get("drift_detected") else "✅ NO",
                            })
                    if feat_rows:
                        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

                pred_drift = dr.get("prediction_drift")
                if pred_drift:
                    st.markdown(f"**Prediction Drift:** KS={pred_drift.get('ks_statistic', '—')} | Mean Shift={pred_drift.get('mean_shift', '—')}")
    else:
        st.markdown(
            '<div class="alert-info">ℹ️ No drift data available. Ensure models have been used for predictions first.</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Page: Bias & Fairness Auditor
# ──────────────────────────────────────────────────────────────────────────────

elif page == "⚖️ Bias & Fairness Auditor":
    st.markdown(
        '<h1 style="color:#e2e8f0;">⚖️ Fairness & Bias Auditor</h1>'
        '<p style="color:#94a3b8;">Disparate Impact Ratio (DIR), Demographic Parity Difference (DPD), and Equal Opportunity Difference (EOD).</p>',
        unsafe_allow_html=True,
    )

    bias_results = api_get("/governance/bias/audit/all") or []

    if bias_results:
        import pandas as pd
        fair_count = sum(1 for b in bias_results if b.get("overall_fair", True))
        biased_count = len(bias_results) - fair_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Modules Audited", len(bias_results))
        col2.metric("✅ Fair", fair_count)
        col3.metric("⚠️ Bias Detected", biased_count, delta=f"{biased_count} flagged" if biased_count else None,
                    delta_color="inverse")

        for b in bias_results:
            is_fair = b.get("overall_fair", True)
            badge = status_badge("FAIR" if is_fair else "BIAS_DETECTED")
            fairness_score = b.get("fairness_score", 100)
            with st.expander(
                f"📦 {b['module_id']} — {badge} (Score: {fairness_score}/100)",
                expanded=not is_fair,
            ):
                flags = b.get("adverse_impact_flags", [])
                if flags:
                    for f in flags:
                        st.markdown(f'<div class="alert-warning">⚠️ {f}</div>', unsafe_allow_html=True)

                for attr_report in b.get("attributes_audited", []):
                    st.markdown(f"**Protected Attribute: `{attr_report.get('attribute', '—')}`**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("DIR (≥0.8 = fair)", f"{attr_report.get('dir', '—')}")
                    col2.metric("DPD (≤0.1 = fair)", f"{attr_report.get('dpd', '—')}")
                    col3.metric("EOD (≤0.1 = fair)", str(attr_report.get("eod", "N/A")))

                    if attr_report.get("adverse_impact"):
                        st.markdown(f'<div class="alert-warning">💡 Recommendation: {attr_report.get("recommendation", "—")}</div>', unsafe_allow_html=True)

                    groups = attr_report.get("groups", {})
                    if groups:
                        group_rows = [
                            {"Group": gk, **{k: v for k, v in gv.items() if k != "group"}}
                            for gk, gv in groups.items()
                        ]
                        st.dataframe(pd.DataFrame(group_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="alert-info">ℹ️ No bias data yet. Module predictions with protected attribute metadata will populate this view.</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Page: Hallucination Monitor
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🧠 Hallucination Monitor":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🧠 LLM Hallucination Monitor</h1>'
        '<p style="color:#94a3b8;">Faithfulness scoring, groundedness evaluation, contradiction detection, and auto-quarantine.</p>',
        unsafe_allow_html=True,
    )

    hall_stats = api_get("/governance/hallucination/stats") or {}
    recent_evals = api_get("/governance/hallucination/recent", {"n": 20}) or []

    total_evals = hall_stats.get("total_evaluations", 0)
    total_quar = hall_stats.get("total_quarantined", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Evaluations", total_evals)
    col2.metric("Quarantined Responses", total_quar, delta=f"{total_quar} blocked" if total_quar else None,
                delta_color="inverse")
    col3.metric("Quarantine Rate", f"{(total_quar/total_evals*100):.1f}%" if total_evals > 0 else "0%")

    if hall_stats.get("modules"):
        st.markdown('<div class="section-header">Per-Module Faithfulness Stats</div>', unsafe_allow_html=True)
        import pandas as pd
        mod_rows = [
            {"Module": mid, **stats}
            for mid, stats in hall_stats["modules"].items()
        ]
        st.dataframe(pd.DataFrame(mod_rows), use_container_width=True, hide_index=True)

    if recent_evals:
        st.markdown('<div class="section-header">Recent Evaluations</div>', unsafe_allow_html=True)
        import pandas as pd
        eval_df = pd.DataFrame([
            {
                "Module": e.get("module_id", "—"),
                "Query": e.get("query", "—")[:60],
                "Faithfulness": e.get("faithfulness_score", "—"),
                "Composite": e.get("composite_score", "—"),
                "Quarantine": "🔴 YES" if e.get("quarantine") else "✅ NO",
                "Severity": e.get("severity", "—"),
                "Recommendation": e.get("recommendation", "—")[:80],
            }
            for e in recent_evals
        ])
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="alert-info">ℹ️ No hallucination evaluations yet. Use Module 6 (Support Assistant) to generate LLM responses.</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Page: Incidents & RCA
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🚨 Incidents & RCA":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🚨 Incidents & Root Cause Analysis</h1>'
        '<p style="color:#94a3b8;">Automated fault tree analysis, 5-Why reasoning, and remediation playbooks.</p>',
        unsafe_allow_html=True,
    )

    rca_sum = api_get("/governance/incidents/summary") or {}
    open_incidents = api_get("/governance/incidents/open") or []

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Incidents", rca_sum.get("total_incidents", 0))
    col2.metric("Open Incidents", rca_sum.get("open_incidents", 0))
    col3.metric("🔴 Critical", rca_sum.get("critical_incidents", 0))
    col4.metric("🟠 High", rca_sum.get("high_incidents", 0))

    if open_incidents:
        st.markdown('<div class="section-header">Open Incidents</div>', unsafe_allow_html=True)
        for inc in open_incidents:
            priority = inc.get("priority", "MEDIUM")
            priority_badge = status_badge(
                "CRITICAL" if priority == "CRITICAL" else "WARNING" if priority == "HIGH" else "INFO"
            )
            with st.expander(
                f"🚨 {inc.get('incident_id', '—')} | {priority_badge} — {inc.get('primary_event_type', '—')}",
                expanded=priority == "CRITICAL",
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Module:** `{inc.get('module_id', '—')}`")
                    st.markdown(f"**Priority:** `{priority}`")
                    st.markdown(f"**Timestamp:** {inc.get('timestamp', '—')}")
                with col2:
                    causes = inc.get("inferred_causes", [])
                    if causes:
                        st.markdown("**Inferred Causes:**")
                        for c in causes:
                            st.markdown(f"  - {c}")

                if inc.get("five_whys"):
                    st.markdown("**🔍 5-Why Analysis:**")
                    for i, why in enumerate(inc["five_whys"], 1):
                        st.markdown(f"  **Why {i}:** {why}")

                playbook = inc.get("remediation_playbook", "")
                if playbook:
                    st.markdown(
                        f'<div class="alert-info">💡 <b>Remediation Playbook:</b><br>{playbook}</div>',
                        unsafe_allow_html=True,
                    )

                if not inc.get("resolved"):
                    if st.button(f"✅ Resolve {inc.get('incident_id', '')}", key=f"resolve_{inc.get('incident_id')}"):
                        result = api_post(f"/governance/incidents/{inc.get('incident_id')}/resolve")
                        if result and result.get("success"):
                            st.success("Incident resolved!")
                            st.rerun()
    else:
        st.markdown(
            '<div class="alert-success">✅ No open incidents. All systems operating within normal parameters.</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Page: Model Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🔄 Model Lifecycle":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🔄 Model Lifecycle Manager</h1>'
        '<p style="color:#94a3b8;">Version registry, canary deployments, champion/challenger A-B testing, and rollback management.</p>',
        unsafe_allow_html=True,
    )

    lifecycle = api_get("/governance/lifecycle/platform") or {}
    retrain_q = api_get("/governance/lifecycle/retraining-queue") or []

    models_registered = lifecycle.get("models_registered", [])
    production_versions = lifecycle.get("production_versions", {})
    challenger_versions = lifecycle.get("challenger_versions", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Registered", len(models_registered))
    col2.metric("In Production", len(production_versions))
    col3.metric("Challenger Versions", len(challenger_versions))
    col4.metric("Retraining Queue", len(retrain_q), delta=f"{len(retrain_q)} queued" if retrain_q else None,
                delta_color="inverse")

    if retrain_q:
        st.markdown(
            f'<div class="alert-warning">⚠️ <b>Retraining Queue:</b> {", ".join(retrain_q)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Model Registry</div>', unsafe_allow_html=True)
    if models_registered:
        import pandas as pd
        rows = [
            {
                "Model ID": mid,
                "Production Version": production_versions.get(mid, "—"),
                "Challenger Version": challenger_versions.get(mid, "—"),
                "Retraining Queued": "⚠️ YES" if mid in retrain_q else "✅ NO",
            }
            for mid in models_registered
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Manual rollback
    st.markdown('<div class="section-header">Manual Rollback</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        rollback_model = st.selectbox("Select Model", options=models_registered or ["No models registered"])
        rollback_reason = st.text_input("Reason", value="manual_operator_request")
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔄 Trigger Rollback", disabled=not models_registered):
            result = api_post("/governance/supervisor/rollback", {"model_id": rollback_model, "reason": rollback_reason})
            if result and result.get("success"):
                st.success(f"✅ Rolled back `{rollback_model}` to `{result.get('restored_version', '—')}`")
            else:
                st.error(f"Rollback failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Governance Reports
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📋 Governance Reports":
    st.markdown(
        '<h1 style="color:#e2e8f0;">📋 Governance Audit Reports</h1>'
        '<p style="color:#94a3b8;">Automated JSON + Markdown audit reports with risk assessment and prioritized recommendations.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📋 Generate New Report"):
            with st.spinner("Running full audit across all governance pillars..."):
                report = api_post("/governance/supervisor/report")
            if report and "report_id" in report:
                st.session_state["latest_report"] = report
                st.success(f"Report {report['report_id']} generated!")
            else:
                st.error("Failed to generate report.")

    report_history = api_get("/governance/reports/history", {"n": 10}) or []
    if report_history:
        st.markdown('<div class="section-header">Report History</div>', unsafe_allow_html=True)
        for rpt in reversed(report_history):
            risk = rpt.get("risk", "UNKNOWN")
            risk_badge = status_badge(
                "CRITICAL" if risk == "HIGH" else "WARNING" if risk == "MEDIUM" else "HEALTHY"
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f'<div class="kpi-card" style="padding:12px 16px;">'
                    f'📋 <b>{rpt.get("report_id", "—")}</b> &nbsp; {risk_badge} &nbsp; '
                    f'<span style="color:#94a3b8; font-size:0.8rem;">{rpt.get("timestamp", "—")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("View", key=f"view_{rpt.get('report_id')}"):
                    full_report = api_get(f"/governance/reports/{rpt.get('report_id')}")
                    if full_report:
                        st.json(full_report)

    # Display latest report if available
    if "latest_report" in st.session_state:
        rpt = st.session_state["latest_report"]
        st.markdown('<div class="section-header">Latest Report</div>', unsafe_allow_html=True)
        st.markdown(f"**Report ID:** `{rpt.get('report_id')}` | **Risk:** `{rpt.get('overall_risk_level')}` | **Status:** `{rpt.get('platform_status')}`")

        flags = rpt.get("critical_flags", [])
        if flags:
            for f in flags:
                st.markdown(f'<div class="alert-critical">🚨 {f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">✅ No critical flags.</div>', unsafe_allow_html=True)

        recs = rpt.get("recommendations", [])
        if recs:
            st.markdown("**Recommendations:**")
            for rec in recs:
                urgency_icon = "🔴" if rec["urgency"] == "IMMEDIATE" else "🟠" if rec["urgency"] == "HIGH" else "🟡"
                st.markdown(f"**[P{rec['priority']}]** {urgency_icon} `{rec['urgency']}` — {rec['action']}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Chaos Engineering Lab
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🧪 Chaos Engineering Lab":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🧪 Chaos Engineering Lab</h1>'
        '<p style="color:#94a3b8;">Inject controlled faults to test autonomous governance, circuit breakers, and remediation loops.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="alert-warning">⚠️ <b>Chaos Engineering</b> — These injections simulate real faults to validate governance resilience. '
        'All injections are logged to the governance action audit trail.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🎛️ Supervisor Chaos Injection</div>', unsafe_allow_html=True)
        chaos_module = st.selectbox(
            "Target Module",
            ["m1_ticket_predictor", "m2_recommender", "m3_delay_predictor",
             "m4_planner", "m5_fare_predictor", "m6_support", "m7_fraud_detector"],
        )
        chaos_type = st.selectbox(
            "Chaos Type",
            [
                ("latency_spike", "🕐 Latency Spike — Simulates 5000ms responses"),
                ("error_surge", "💥 Error Surge — Injects 8 consecutive failures"),
                ("drift_simulation", "📉 Drift Simulation — Publishes synthetic drift event"),
            ],
            format_func=lambda x: x[1],
        )

        if st.button("🚀 Inject Chaos", type="primary"):
            with st.spinner("Injecting chaos..."):
                result = api_post("/governance/supervisor/chaos", {
                    "module_id": chaos_module,
                    "chaos_type": chaos_type[0],
                })
            if result and result.get("success"):
                st.success(f"✅ {result.get('message', 'Chaos injected.')}")
                st.markdown(
                    '<div class="alert-info">ℹ️ Monitor the <b>Platform Health</b> and <b>Incidents & RCA</b> pages to observe the governance response.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"Chaos injection failed: {result}")

    with col2:
        st.markdown('<div class="section-header">📡 Telemetry Chaos (Legacy)</div>', unsafe_allow_html=True)
        telemetry_module = st.selectbox(
            "Target Module (Telemetry)",
            ["m1_ticket_predictor", "m2_recommender", "m6_support", "m7_fraud_detector"],
            key="tel_module",
        )
        telemetry_chaos = st.selectbox(
            "Chaos Type (Telemetry)",
            ["latency", "drift", "schema_violation", "outlier"],
            key="tel_chaos",
        )
        magnitude = st.slider("Magnitude (ms)", 1000, 10000, 4000, step=500)

        if st.button("📡 Inject Telemetry Event"):
            with st.spinner("Injecting telemetry event..."):
                result = api_post("/governance/chaos/inject", {
                    "module": telemetry_module,
                    "chaos_type": telemetry_chaos,
                    "magnitude": magnitude,
                })
            if result and result.get("success"):
                st.success(f"✅ {result.get('message')}")
            else:
                st.error(f"Failed: {result}")

    # HITL pending
    hitl = api_get("/governance/supervisor/hitl/pending") or []
    if hitl:
        st.markdown('<div class="section-header">🔐 Pending HITL Approvals</div>', unsafe_allow_html=True)
        for req in hitl:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{req.get('request_id')}** — `{req.get('action_type')}`")
            with col2:
                if st.button("✅ Approve", key=f"approve_{req['request_id']}"):
                    api_post("/governance/supervisor/hitl/decide", {"request_id": req["request_id"], "approved": True})
                    st.rerun()
            with col3:
                if st.button("❌ Reject", key=f"reject_{req['request_id']}"):
                    api_post("/governance/supervisor/hitl/decide", {"request_id": req["request_id"], "approved": False})
                    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Page: AI Travel Assistant
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🤖 AI Travel Assistant":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🤖 AI Travel Assistant</h1>'
        '<p style="color:#94a3b8;">Agentic multi-tool travel planner with Human-in-the-Loop booking confirmation.</p>',
        unsafe_allow_html=True,
    )

    with st.form("planner_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin = st.text_input("Origin Station", value="Mumbai")
            destination = st.text_input("Destination Station", value="Delhi")
            travel_date = st.date_input("Travel Date")
        with col2:
            passengers = st.number_input("Passengers", min_value=1, max_value=10, value=1)
            travel_class = st.selectbox("Class", ["Sleeper", "3A", "2A", "1A", "CC"])
            persona = st.selectbox("Travel Persona", ["budget", "business", "family", "senior"])

        user_query = st.text_area(
            "Natural Language Request",
            value="I need to travel from Mumbai to Delhi next week with family. Looking for sleeper class with confirmed seats.",
            height=80,
        )
        submitted = st.form_submit_button("🤖 Plan My Journey")

    if submitted:
        with st.spinner("AI Agent planning your journey..."):
            payload = {
                "query": user_query,
                "origin": origin,
                "destination": destination,
                "travel_date": str(travel_date),
                "passengers": passengers,
                "travel_class": travel_class,
                "persona": persona,
            }
            result = api_post("/m4/plan", payload)

        if result and "error" not in result:
            st.markdown('<div class="alert-success">✅ Journey plan ready!</div>', unsafe_allow_html=True)
            st.json(result)
        elif result:
            st.warning(f"Partial result: {result}")
        else:
            st.error("Could not connect to the travel planner API.")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Ticket Predictor
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🎫 Ticket Predictor":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🎫 Ticket Confirmation Predictor</h1>'
        '<p style="color:#94a3b8;">ML-powered waitlist confirmation probability with SHAP explainability.</p>',
        unsafe_allow_html=True,
    )

    with st.form("ticket_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            train_id = st.text_input("Train ID", value="12951")
            origin = st.text_input("Origin", value="MMCT")
            destination = st.text_input("Destination", value="NDLS")
        with col2:
            pnr = st.text_input("PNR / Booking Ref", value="PNR123456")
            current_wl = st.number_input("Current Waitlist Number", min_value=1, max_value=500, value=15)
            quota_type = st.selectbox("Quota", ["GN", "TQ", "LD", "HO", "SS"])
        with col3:
            travel_class = st.selectbox("Class", ["SL", "3A", "2A", "1A", "CC"])
            days_to_travel = st.slider("Days to Departure", 1, 120, 7)
            booked_seats = st.number_input("Seats Already Booked", min_value=0, max_value=500, value=280)

        submitted = st.form_submit_button("🔮 Predict Confirmation")

    if submitted:
        with st.spinner("Running ML prediction with SHAP analysis..."):
            payload = {
                "pnr": pnr,
                "train_id": train_id,
                "origin_station": origin,
                "destination_station": destination,
                "travel_class": travel_class,
                "quota_type": quota_type,
                "current_wl_number": current_wl,
                "days_to_departure": days_to_travel,
                "seats_booked": booked_seats,
                "total_seats": 500,
                "historical_cancellation_rate": 0.12,
                "is_holiday": False,
                "festival_flag": False,
            }
            result = api_post("/m1/predict", payload)

        if result and "confirmation_probability" in result:
            prob = result.get("confirmation_probability", 0)
            verdict = result.get("verdict", "UNKNOWN")
            verdict_color = "#34d399" if "CONFIRM" in verdict else "#fbbf24" if "LIKELY" in verdict else "#f87171"

            col1, col2, col3 = st.columns(3)
            col1.metric("Confirmation Probability", f"{prob:.1%}")
            col2.metric("Verdict", verdict)
            col3.metric("Recommended Berth", result.get("berth_recommendation", "—"))

            st.progress(prob)

            if result.get("shap_explanation"):
                st.markdown('<div class="section-header">🔍 SHAP Explanation</div>', unsafe_allow_html=True)
                st.json(result["shap_explanation"])
        else:
            st.error(f"Prediction failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Train Recommender
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🚄 Train Recommender":
    st.markdown(
        '<h1 style="color:#e2e8f0;">🚄 Smart Train Recommender</h1>'
        '<p style="color:#94a3b8;">Multi-factor persona-based scoring with Disparate Impact Ratio fairness audit.</p>',
        unsafe_allow_html=True,
    )

    with st.form("recommender_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin = st.text_input("Origin Station", value="MMCT")
            destination = st.text_input("Destination", value="NDLS")
            travel_date = st.date_input("Travel Date", key="rec_date")
        with col2:
            persona = st.selectbox("Persona", ["budget", "business", "family", "senior", "tourist"])
            travel_class = st.selectbox("Preferred Class", ["SL", "3A", "2A", "1A", "CC"])
            max_results = st.slider("Max Results", 3, 10, 5)

        submitted = st.form_submit_button("🔍 Find Best Trains")

    if submitted:
        with st.spinner("Scoring trains with ML fairness engine..."):
            payload = {
                "origin_station": origin,
                "destination_station": destination,
                "travel_date": str(travel_date),
                "persona": persona,
                "preferred_class": travel_class,
                "max_results": max_results,
            }
            result = api_post("/m2/recommend", payload)

        if result and "recommendations" in result:
            recs = result["recommendations"]
            st.success(f"Found {len(recs)} trains")

            dir_score = result.get("fairness_audit", {}).get("disparate_impact_ratio", 1.0)
            fair_badge = "✅ FAIR" if dir_score >= 0.8 else "⚠️ REVIEW"
            st.markdown(
                f'<div class="alert-info">⚖️ <b>Fairness Audit:</b> DIR = {dir_score:.3f} — {fair_badge}</div>',
                unsafe_allow_html=True,
            )

            import pandas as pd
            rec_df = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Train": r.get("train_name", "—"),
                    "Train No": r.get("train_number", "—"),
                    "Departure": r.get("departure_time", "—"),
                    "Duration": r.get("duration", "—"),
                    "Score": f"{r.get('composite_score', 0):.3f}",
                    "Fare": f"₹{r.get('fare', 0)}",
                    "Delay Risk": r.get("delay_risk", "—"),
                }
                for i, r in enumerate(recs)
            ])
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Recommendation failed: {result}")

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #2d4170;
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
    ">
        🚄 <b style="color:#60a5fa">AI Railway Autonomous Governance Platform</b> —
        Enterprise ML Governance · Dual-Mode LLM Resilience · Zero-Crash Architecture<br>
        Built with FastAPI · Streamlit · LightGBM · SHAP · Pydantic V2 · SciPy
    </div>
    """,
    unsafe_allow_html=True,
)
