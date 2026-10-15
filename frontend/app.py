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

from dotenv import load_dotenv

# Ensure live .env credentials (e.g. Azure OpenAI) are loaded with override
load_dotenv(override=True)

import requests
import streamlit as st

try:
    from core.config import settings
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.config import settings

try:
    from frontend.components.telemetry_badge import render_telemetry_badge
    from frontend.components.shap_chart import render_shap_chart
    from frontend.components.chaos_simulator import render_chaos_simulator_panel
    from frontend.components.governance_sidebar import render_sidebar_governance_widget
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend.components.telemetry_badge import render_telemetry_badge
    from frontend.components.shap_chart import render_shap_chart
    from frontend.components.chaos_simulator import render_chaos_simulator_panel
    from frontend.components.governance_sidebar import render_sidebar_governance_widget

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

import os

API_BASE = os.getenv("BACKEND_URL", os.getenv("API_BASE", "http://127.0.0.1:8000")).rstrip("/")
API_V1 = f"{API_BASE}/api/v1"
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

    /* Prevent page titles clipping at top viewport edge */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
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

def api_get(path: str, params: Optional[dict] = None, base: str = API_V1) -> Optional[Any]:
    try:
        r = requests.get(f"{base}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: Optional[dict] = None, base: str = API_V1) -> Optional[Any]:
    try:
        r = requests.post(f"{base}{path}", json=payload or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Human-Friendly Station & Train Dictionaries and Aliases
# ──────────────────────────────────────────────────────────────────────────────

STATION_CHOICES = {
    "New Delhi (NDLS)": "NDLS",
    "Mumbai CSMT (CSTM)": "CSTM",
    "Chennai Central (MAS)": "MAS",
    "Howrah Jn (HWH)": "HWH",
    "Bengaluru (SBC)": "SBC",
    "Bhopal Jn (BPL)": "BPL",
    "Ahmedabad Jn (ADI)": "ADI",
    "Lucknow (LKO)": "LKO",
    "Kolkata (KOAA)": "KOAA",
    "Pune Jn (PUNE)": "PUNE",
    "Jaipur Jn (JP)": "JP",
}

STATION_ALIASES = {
    "BCT": "CSTM",
    "MMCT": "CSTM",
    "BDTS": "CSTM",
    "DLI": "NDLS",
    "NZM": "NDLS",
    "ANVT": "NDLS",
    "SDAH": "HWH",
    "MS": "MAS",
    "YPR": "SBC",
}

TRAIN_CHOICES = {
    "12951 - Mumbai Rajdhani Express (NDLS → CSTM)": "12951",
    "12002 - Bhopal Shatabdi Express (NDLS → BPL)": "12002",
    "12301 - Howrah Rajdhani Express (HWH → NDLS)": "12301",
    "22691 - KSR Bengaluru Rajdhani (SBC → NDLS)": "22691",
    "12622 - Tamil Nadu Express (NDLS → MAS)": "12622",
    "20901 - Vande Bharat Express (CSTM → ADI)": "20901",
    "12137 - Punjab Mail (CSTM → NDLS)": "12137",
    "12260 - Sealdah AC Duronto Express (NDLS → HWH)": "12260",
}


def backend_online() -> bool:
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=3)
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


# Auto-seed baseline telemetry on cold-start if empty
if "baseline_auto_seeded" not in st.session_state:
    try:
        events = api_get("/governance/telemetry", params={"limit": 5})
        if events is not None and len(events) == 0:
            api_post("/governance/telemetry/seed", {"count": 50})
        st.session_state["baseline_auto_seeded"] = True
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation: Role-Based Portal Switcher
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 6px 0;">
            <div style="font-size:2.2rem">🚄</div>
            <div style="font-size:1.1rem; font-weight:800; color:#60a5fa; line-height:1.2">
                AI Railway
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px">
                Intelligent Travel & Governance
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    portal_mode = st.radio(
        "SELECT PORTAL",
        ["🚆 Passenger AI Portal", "🏛️ AI Governance Control Tower"],
        index=0 if st.session_state.get("portal_mode") == "passenger" else 1,
        help="Switch between Passenger AI Services and AI Governance Control Tower",
    )
    st.session_state["portal_mode"] = "passenger" if "Passenger" in portal_mode else "governance"

    st.markdown("<hr style='border-color:#2d4170; margin:10px 0;'>", unsafe_allow_html=True)

    if portal_mode == "🚆 Passenger AI Portal":
        page_options = [
            "🎫 Ticket & WL Predictor",
            "🚆 Smart Train Recommender",
            "⏱️ Delay & Disruption Predictor",
            "🧭 AI Travel Planner",
            "💰 Dynamic Fare Predictor",
            "💬 Multilingual Support Concierge",
            "🛡️ Fraud & Bot Shield",
        ]
    else:
        page_options = [
            "🏠 Executive Overview",
            "📊 Platform Health & SLA",
            "📉 Statistical Drift Detector",
            "⚖️ Bias & Fairness Auditor",
            "🧠 Hallucination Monitor",
            "🚨 Incidents & RCA",
            "🔄 Model Lifecycle Manager",
            "📋 Governance Audit Reports",
            "🧪 Chaos Engineering Lab",
        ]

    page = st.radio("Navigation", page_options, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2d4170; margin:12px 0;'>", unsafe_allow_html=True)

    # 1-Click Telemetry Seeder button
    if st.button("⚡ Seed 50 Live Requests", use_container_width=True, help="Populate realistic baseline telemetry across M1-M7"):
        with st.spinner("Seeding 50 baseline telemetry events across M1–M7..."):
            res = api_post("/governance/telemetry/seed", {"count": 50})
            if res and res.get("success"):
                st.success("✅ Seeded 50 live events!")
                time.sleep(0.4)
                st.rerun()
            else:
                st.error("Seeding failed.")

    # Engine badge
    engine_label = (
        "⚙️ Engine: Azure OpenAI (gpt-4o-mini) ⚡"
        if getattr(settings, "llm_available", False)
        else "⚙️ Engine: Intelligent Fallback Engine 🛡️"
    )
    st.markdown(
        f"""
        <div style="background:#131c38; border:1px solid #1e2d52; border-radius:6px; padding:6px 10px; text-align:center; margin:8px 0; font-size:0.75rem; color:#93c5fd;">
            {engine_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Status indicator
    online = backend_online()
    render_sidebar_governance_widget(online)

    st.markdown(
        f"""
        <div style="font-size:0.72rem; color:#475569; text-align:center; margin-top:14px;">
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
        <div style="padding: 4px 0 16px 0;">
            <h1 style="font-size:2.3rem; font-weight:900; color:#e2e8f0; margin:0;">
                🚄 AI Railway Autonomous Governance Control Tower
            </h1>
            <p style="color:#94a3b8; font-size:1.02rem; margin-top:6px;">
                Enterprise-grade AI Governance Control Tower — Real-time telemetry, autonomous remediation & supervisory intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🎯 Quick Evaluator Guide — 3-Step Guided Demonstration", expanded=True):
        st.markdown(
            """
            Welcome to the **AI Railway Autonomous Governance Platform** evaluation. Follow this 3-step tour:
            1. **🚆 Step 1: Explore Passenger AI Services** — Switch to **`🚆 Passenger AI Portal`** in the top sidebar to test waitlist prediction, fair train ranking, arrival delays, dynamic fares, support concierge, and bot security.
            2. **📊 Step 2: Inspect Live Telemetry & Governance** — Switch to **`🏛️ AI Governance Control Tower`** to observe real-time SLA compliance, demographic parity audits (DIR), statistical drift detection, and hallucination monitoring.
            3. **🧪 Step 3: Inject Faults in Chaos Lab** — Navigate to **`🧪 Chaos Engineering Lab`**, trigger latency spikes or drift anomalies, and watch the platform detect, escalate, and self-heal automatically!
            """
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
                report = api_post("/governance/supervisor/report", base=API_V1)
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

elif page in ("📉 Drift Detector", "📉 Statistical Drift Detector"):
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
            is_drifting = bool(dr.get("overall_drift_detected"))
            tag = "🔴 DRIFT DETECTED" if is_drifting else "🟢 STABLE"
            with st.expander(
                f"🤖 {dr['model_id']} — {tag}",
                expanded=is_drifting,
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

elif page in ("🔄 Model Lifecycle", "🔄 Model Lifecycle Manager"):
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

    st.markdown('<div class="section-header">Model Registry (M1–M7)</div>', unsafe_allow_html=True)
    if models_registered:
        import pandas as pd
        rows = []
        for mid in models_registered:
            status_obj = api_get(f"/governance/lifecycle/model/{mid}") or {}
            versions = {v.get("version"): v for v in status_obj.get("versions", [])}
            prod_v = versions.get(production_versions.get(mid, ""), {})
            metrics = prod_v.get("metrics", {})
            metrics_summary = ", ".join([f"{k}: {v}" for k, v in list(metrics.items())[:2]]) if metrics else "Optimal"
            rows.append({
                "Model Identifier": mid,
                "Production Version": production_versions.get(mid, "—"),
                "Canary Challenger": challenger_versions.get(mid, "—"),
                "Evaluated Metrics": metrics_summary,
                "Retraining Queued": "⚠️ YES" if mid in retrain_q else "✅ NO",
            })
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

elif page in ("📋 Governance Reports", "📋 Governance Audit Reports"):
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

    r_col1, r_col2 = st.columns([1, 2])
    with r_col1:
        if st.button("🔄 Reset Platform to Healthy State", type="primary", use_container_width=True):
            with st.spinner("Resetting platform state, clearing circuits & resolving alerts..."):
                reset_res = api_post("/governance/supervisor/reset")
                if reset_res and reset_res.get("success"):
                    st.success("✅ Platform reset to Healthy! Alerts & incidents resolved.")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error("Reset failed.")

    render_chaos_simulator_panel(api_post, api_get)

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
            ["m1_ticket_predictor", "m2_train_recommender", "m3_delay_predictor",
             "m4_travel_planner", "m5_fare_predictor", "m6_support_assistant", "m7_fraud_detector"],
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
        st.markdown('<div class="section-header">📡 Real-Time Telemetry Event Injection</div>', unsafe_allow_html=True)
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

elif page in ("🤖 AI Travel Assistant", "🧭 AI Travel Planner"):
    st.markdown(
        '<h1 style="color:#e2e8f0;">🧭 AI Travel Planner & Agentic Assistant</h1>'
        '<p style="color:#94a3b8;">Agentic multi-tool travel planner with Human-in-the-Loop (HITL) booking confirmation.</p>',
        unsafe_allow_html=True,
    )

    with st.form("planner_form"):
        col1, col2 = st.columns(2)
        with col1:
            persona = st.selectbox("Traveler Persona", ["budget", "business", "family", "senior", "student", "premium"])
            max_budget = st.number_input("Max Budget (INR ₹, optional)", min_value=0, max_value=50000, value=3000, step=500)
        with col2:
            st.info(
                "💡 **Agentic Multi-Tool Workflow:** Automatically queries M1 (Waitlist Predictor), "
                "M3 (Delay Predictor), and M5 (Dynamic Fare Forecaster). "
                "Halts execution at a strict Human-in-the-Loop gate prior to booking preparation."
            )

        user_query = st.text_area(
            "Natural Language Journey Request",
            value="I need to travel from NDLS to CSTM next Friday with budget 2000. Looking for 3A with confirmed berth.",
            height=90,
        )
        submitted = st.form_submit_button("🤖 Generate Agentic Journey Plan")

    if submitted:
        with st.spinner("Multi-agent system synthesizing routes, delay risks, and fare projections..."):
            payload = {
                "query": user_query,
                "user_id": "passenger_control_tower",
                "persona": persona,
                "max_budget": float(max_budget) if max_budget > 0 else None,
            }
            result = api_post("/agent/plan", payload)

        if result and "recommended_itinerary" in result:
            st.session_state["active_approval"] = result
            st.session_state["approval_confirmed"] = False
        else:
            st.error(f"Agent travel planning failed: {result}")

    active_app = st.session_state.get("active_approval")
    if active_app:
        req_id = active_app.get("request_id")
        status = active_app.get("status", "PENDING_APPROVAL")
        rec = active_app.get("recommended_itinerary", {})

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #131c38 0%, #1a274e 100%);
                        border: 1px solid #3b82f6; border-radius: 12px; padding: 22px; margin: 16px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div>
                        <span style="font-size:1.25rem; font-weight:700; color:#60a5fa;">
                            🚄 {rec.get('train_name')} ({rec.get('train_number')})
                        </span>
                        <span style="font-size:0.85rem; color:#94a3b8; margin-left:12px;">
                            Request ID: <code>{req_id}</code>
                        </span>
                    </div>
                    <span class="{'badge-healthy' if status == 'APPROVED' else 'badge-warning'}">
                        {status}
                    </span>
                </div>
                <div style="font-size:0.95rem; color:#cbd5e1; margin-bottom:14px;">
                    <b>Route:</b> {rec.get('source')} ➔ {rec.get('destination')} &nbsp;|&nbsp;
                    <b>Departure:</b> {rec.get('departure_time')} &nbsp;|&nbsp;
                    <b>Arrival:</b> {rec.get('arrival_time')} ({rec.get('duration_hours')} hrs) &nbsp;|&nbsp;
                    <b>Class:</b> {rec.get('travel_class')}
                </div>
                <div style="display:flex; gap:20px; margin-bottom:16px;">
                    <div class="kpi-card" style="flex:1; padding:12px 16px;">
                        <div class="kpi-label">Estimated Fare</div>
                        <div style="font-size:1.4rem; font-weight:700; color:#34d399;">₹{rec.get('estimated_fare_inr', 0):,.0f}</div>
                    </div>
                    <div class="kpi-card" style="flex:1; padding:12px 16px;">
                        <div class="kpi-label">Confirmation Odds</div>
                        <div style="font-size:1.4rem; font-weight:700; color:#60a5fa;">{rec.get('confirmation_probability', 0):.1%}</div>
                    </div>
                    <div class="kpi-card" style="flex:1; padding:12px 16px;">
                        <div class="kpi-label">Expected Delay</div>
                        <div style="font-size:1.4rem; font-weight:700; color:#fde68a;">~{rec.get('predicted_delay_minutes', 0):.0f} min</div>
                    </div>
                </div>
                <div style="background:#0d1530; border:1px solid #1e2d52; border-radius:8px; padding:14px; margin-bottom:14px;">
                    <b style="color:#60a5fa;">🧠 Multi-Agent Reasoning:</b>
                    <p style="color:#94a3b8; font-size:0.9rem; margin-top:4px;">{active_app.get('agent_reasoning_summary')}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Human-in-the-Loop Actions
        if status == "PENDING_APPROVAL" and not st.session_state.get("approval_confirmed"):
            st.markdown("#### 🛡️ Human-in-the-Loop (HITL) Gate — Explicit Passenger Decision")
            hcol1, hcol2 = st.columns(2)
            with hcol1:
                if st.button("✅ Confirm Booking Itinerary (Execute Gate)", type="primary", use_container_width=True):
                    with st.spinner("Dispatching confirmed booking payload..."):
                        conf_res = api_post("/agent/confirm", {
                            "request_id": req_id,
                            "selected_option_id": rec.get("option_id", "opt-1"),
                            "passenger_name": "Demo Passenger",
                            "passenger_age": 30,
                            "passenger_gender": "M",
                        })
                    if conf_res and "booking_reference" in conf_res:
                        st.session_state["approval_confirmed"] = True
                        st.session_state["active_approval"]["status"] = "APPROVED"
                        st.success(f"🎉 Booking Confirmed! Ref: **{conf_res.get('booking_reference')}** | PNR: **{conf_res.get('pnr')}**")
                        st.rerun()
                    else:
                        st.error(f"Confirmation failed: {conf_res}")
            with hcol2:
                if st.button("❌ Reject Itinerary & Replan", use_container_width=True):
                    with st.spinner("Registering rejection and triggering replan..."):
                        rej_res = api_post("/agent/reject", {
                            "request_id": req_id,
                            "reason": "Passenger requested alternate route or timing",
                        })
                    st.warning("Itinerary rejected. Status changed to REPLANNING.")
                    st.session_state["active_approval"]["status"] = "REPLANNING"
                    st.rerun()

        render_telemetry_badge(active_app.get("governance_telemetry"))


# ──────────────────────────────────────────────────────────────────────────────
# Page: Ticket Predictor
# ──────────────────────────────────────────────────────────────────────────────

elif page in ("🎫 Ticket Predictor", "🎫 Ticket & WL Predictor"):
    st.markdown(
        '<h1 style="color:#e2e8f0;">🎫 Ticket Confirmation Predictor</h1>'
        '<p style="color:#94a3b8;">ML-powered waitlist confirmation probability with SHAP explainability.</p>',
        unsafe_allow_html=True,
    )

    from datetime import date, timedelta

    st.markdown("**⚡ 1-Click Scenario Presets:**")
    pcol1, pcol2, pcol3 = st.columns(3)
    if pcol1.button("🟢 Confirmed Ticket Case", use_container_width=True):
        st.session_state["ticket_train_idx"] = 0
        st.session_state["ticket_class"] = "3A"
        st.session_state["ticket_quota"] = "GN"
        st.session_state["ticket_wl"] = 2
        st.session_state["ticket_days"] = 14
        st.session_state["ticket_senior"] = False
        st.session_state["ticket_disability"] = False
        st.session_state["auto_predict_ticket"] = True
    if pcol2.button("🟠 High WL Risk Tatkal", use_container_width=True):
        st.session_state["ticket_train_idx"] = 0
        st.session_state["ticket_class"] = "2A"
        st.session_state["ticket_quota"] = "TQ"
        st.session_state["ticket_wl"] = 48
        st.session_state["ticket_days"] = 2
        st.session_state["ticket_senior"] = False
        st.session_state["ticket_disability"] = False
        st.session_state["auto_predict_ticket"] = True
    if pcol3.button("🔵 Peak Festival Waitlist", use_container_width=True):
        st.session_state["ticket_train_idx"] = 2
        st.session_state["ticket_class"] = "SL"
        st.session_state["ticket_quota"] = "GN"
        st.session_state["ticket_wl"] = 180
        st.session_state["ticket_days"] = 45
        st.session_state["ticket_senior"] = False
        st.session_state["ticket_disability"] = False
        st.session_state["auto_predict_ticket"] = True

    train_labels = list(TRAIN_CHOICES.keys())
    default_train_idx = st.session_state.get("ticket_train_idx", 0)

    with st.form("ticket_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_train_label = st.selectbox(
                "Select Train",
                train_labels,
                index=min(default_train_idx, len(train_labels) - 1),
            )
            train_id = TRAIN_CHOICES[selected_train_label]
            travel_class = st.selectbox(
                "Travel Class",
                ["3A", "2A", "1A", "SL", "CC"],
                index=["3A", "2A", "1A", "SL", "CC"].index(st.session_state.get("ticket_class", "3A")),
            )
        with col2:
            current_wl = st.number_input(
                "Current Waitlist / RAC Number",
                min_value=1,
                max_value=500,
                value=st.session_state.get("ticket_wl", 15),
            )
            quota_type = st.selectbox(
                "Quota",
                ["GN", "TQ", "LADIES", "DEFENCE", "SENIOR"],
                index=["GN", "TQ", "LADIES", "DEFENCE", "SENIOR"].index(st.session_state.get("ticket_quota", "GN")),
            )
        with col3:
            days_to_travel = st.slider(
                "Days to Departure",
                1,
                120,
                st.session_state.get("ticket_days", 7),
            )
            journey_date_input = date.today() + timedelta(days=days_to_travel)
            st.caption(f"Estimated Journey Date: **{journey_date_input.isoformat()}**")

        with st.expander("⚙️ Advanced Parameters"):
            acol1, acol2 = st.columns(2)
            with acol1:
                seniority = st.checkbox("Senior Citizen Quota (Age >= 60)", value=st.session_state.get("ticket_senior", False))
            with acol2:
                disability = st.checkbox("Divyangjan (Disability) Quota", value=st.session_state.get("ticket_disability", False))

        submitted = st.form_submit_button("🔮 Predict Confirmation")

    trigger_prediction = submitted or st.session_state.pop("auto_predict_ticket", False)

    if trigger_prediction:
        with st.spinner("Running ML prediction with SHAP analysis..."):
            payload = {
                "train_number": str(train_id),
                "journey_class": travel_class,
                "quota": quota_type,
                "journey_date": journey_date_input.isoformat(),
                "booking_date": date.today().isoformat(),
                "current_wl_number": int(current_wl),
                "seniority_flag": seniority,
                "disability_flag": disability,
            }
            result = api_post("/predict/ticket", payload)

        if result and "confirmation_probability" in result:
            prob = result.get("confirmation_probability", 0.0)
            verdict = result.get("pnr_status_prediction", "Borderline")
            berth = result.get("estimated_berth_type", "—")
            advisory = result.get("booking_advisory", "")
            confidence = result.get("confidence_score", 0.85)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Confirmation Probability", f"{prob:.1%}")
            col2.metric("Prediction Verdict", verdict)
            col3.metric("Estimated Berth", berth)
            col4.metric("Model Certainty", f"{confidence:.1%}")

            st.progress(prob)

            if advisory:
                alert_cls = "alert-success" if prob >= 0.75 else "alert-warning" if prob >= 0.4 else "alert-critical"
                st.markdown(f'<div class="{alert_cls}">📢 <b>Advisory:</b> {advisory}</div>', unsafe_allow_html=True)

            if result.get("shap_explanation"):
                render_shap_chart(result["shap_explanation"], title="Waitlist Confirmation Drivers (SHAP)")

            if result.get("berth_probability_breakdown"):
                with st.expander("📊 Berth Type Probability Distribution"):
                    b_df = pd.DataFrame([
                        {"Berth Type": k, "Probability": f"{v:.1%}"}
                        for k, v in result["berth_probability_breakdown"].items()
                    ])
                    st.dataframe(b_df, use_container_width=True, hide_index=True)

            render_telemetry_badge(result.get("governance_telemetry"))
        else:
            st.error(f"Prediction failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Train Recommender
# ──────────────────────────────────────────────────────────────────────────────

elif page in ("🚄 Train Recommender", "🚆 Smart Train Recommender"):
    st.markdown(
        '<h1 style="color:#e2e8f0;">🚆 Smart Train Recommender</h1>'
        '<p style="color:#94a3b8;">Multi-factor persona-based scoring with Disparate Impact Ratio fairness audit.</p>',
        unsafe_allow_html=True,
    )

    from datetime import date, timedelta

    station_labels = list(STATION_CHOICES.keys())

    with st.form("recommender_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin_label = st.selectbox("Origin Station", station_labels, index=0)
            dest_label = st.selectbox("Destination Station", station_labels, index=1)
            travel_date = st.date_input("Travel Date", value=date.today() + timedelta(days=5), key="rec_date")
        with col2:
            persona = st.selectbox("Traveler Persona", ["budget", "premium", "student", "senior", "business", "family"])
            travel_class = st.selectbox("Preferred Class", ["3A", "2A", "1A", "SL", "CC"])
            max_budget = st.number_input("Max Budget (INR ₹, 0 for no limit)", min_value=0, max_value=20000, value=0, step=500)

        submitted = st.form_submit_button("🔍 Find Best Trains")

    origin_code = STATION_CHOICES[origin_label]
    dest_code = STATION_CHOICES[dest_label]

    if submitted:
        with st.spinner("Scoring corridor trains with ML fairness engine..."):
            payload = {
                "source_station": origin_code,
                "destination_station": dest_code,
                "journey_date": str(travel_date),
                "persona": persona,
                "preferred_class": travel_class,
                "max_budget": float(max_budget) if max_budget > 0 else None,
            }
            result = api_post("/recommend/trains", payload)

        if result and "recommendations" in result:
            recs = result["recommendations"]
            st.success(f"Found {len(recs)} trains running on the {origin_code} ➔ {dest_code} corridor")

            dir_score = float(result.get("disparate_impact_ratio", 1.0))
            is_fair = dir_score >= 0.80
            if is_fair:
                st.markdown(
                    f'<div class="alert-success">⚖️ <b>Algorithmic Fairness Audit: PASS</b> — '
                    f'Disparate Impact Ratio = <b>{dir_score:.3f}</b> (Threshold ≥ 0.80) — <b>✅ FAIR & EQUITABLE ACCESS</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alert-warning">⚠️ <b>Algorithmic Fairness Audit: BIAS REVIEW</b> — '
                    f'Disparate Impact Ratio = <b>{dir_score:.3f}</b> (< 0.80 Threshold) — <b>⚠️ CLASS EQUITY ADJUSTMENT APPLIED</b></div>',
                    unsafe_allow_html=True,
                )

            import pandas as pd
            rec_df = pd.DataFrame([
                {
                    "Rank": r.get("rank", i + 1),
                    "Train": r.get("train_name", "—"),
                    "Train No": r.get("train_number", "—"),
                    "Route": f"{r.get('source', origin_code)} → {r.get('destination', dest_code)}",
                    "Departure": r.get("departure_time", "—"),
                    "Arrival": r.get("arrival_time", "—"),
                    "Duration": f"{r.get('duration_hours', 0):.1f} hrs",
                    "Score": f"{r.get('composite_score', 0):.3f}",
                    "Fare": f"₹{r.get('fare_estimate_inr', 0):,.0f}",
                    "Punctuality": f"{r.get('punctuality_score', 0):.1%}",
                    "Reason": r.get("recommendation_reason", "—"),
                }
                for i, r in enumerate(recs)
            ])
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
            render_telemetry_badge(result.get("governance_telemetry"))
        else:
            st.error(f"Recommendation failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Delay & Disruption Predictor (Module 3)
# ──────────────────────────────────────────────────────────────────────────────

elif page in ("⏱️ Delay Predictor", "⏱️ Delay & Disruption Predictor"):
    st.markdown(
        '<h1 style="color:#e2e8f0;">⏱️ Train Delay & Disruption Predictor</h1>'
        '<p style="color:#94a3b8;">Forecast intermediate arrival delays, platform congestion, and proactive alternative routes.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("**⚡ 1-Click Scenario Presets:**")
    dcol1, dcol2, dcol3 = st.columns(3)
    if dcol1.button("🟢 On-Time Clear Weather", use_container_width=True):
        st.session_state["delay_train_idx"] = 1
        st.session_state["delay_station_idx"] = 5
        st.session_state["delay_weather"] = 0.05
        st.session_state["delay_congestion"] = 0.15
        st.session_state["delay_distance"] = 450.0
        st.session_state["auto_predict_delay"] = True
    if dcol2.button("🟠 Monsoon Track Disruption", use_container_width=True):
        st.session_state["delay_train_idx"] = 0
        st.session_state["delay_station_idx"] = 1
        st.session_state["delay_weather"] = 0.70
        st.session_state["delay_congestion"] = 0.65
        st.session_state["delay_distance"] = 1380.0
        st.session_state["auto_predict_delay"] = True
    if dcol3.button("🔴 Dense Fog & Signal Congestion", use_container_width=True):
        st.session_state["delay_train_idx"] = 2
        st.session_state["delay_station_idx"] = 0
        st.session_state["delay_weather"] = 0.95
        st.session_state["delay_congestion"] = 0.90
        st.session_state["delay_distance"] = 1450.0
        st.session_state["auto_predict_delay"] = True

    train_labels = list(TRAIN_CHOICES.keys())
    station_labels = list(STATION_CHOICES.keys())

    with st.form("delay_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_train = st.selectbox(
                "Select Train",
                train_labels,
                index=min(st.session_state.get("delay_train_idx", 1), len(train_labels) - 1),
            )
            train_number = TRAIN_CHOICES[sel_train]
            sel_station = st.selectbox(
                "Station",
                station_labels,
                index=min(st.session_state.get("delay_station_idx", 5), len(station_labels) - 1),
            )
            station_code = STATION_CHOICES[sel_station]
        with col2:
            journey_date = st.date_input("Journey Date", key="delay_journey_date")
            weather_index = st.slider(
                "Adverse Weather Index",
                0.0,
                1.0,
                st.session_state.get("delay_weather", 0.25),
                step=0.05,
                help="0.0 = clear skies, 1.0 = heavy fog/flooding",
            )
        with col3:
            congestion_index = st.slider(
                "Station Congestion Index",
                0.0,
                1.0,
                st.session_state.get("delay_congestion", 0.45),
                step=0.05,
                help="0.0 = free platform, 1.0 = gridlock",
            )

        with st.expander("⚙️ Advanced Parameters"):
            distance_km = st.number_input(
                "Distance from Origin (km)",
                min_value=10.0,
                max_value=4000.0,
                value=float(st.session_state.get("delay_distance", 500.0)),
                step=50.0,
            )

        submitted = st.form_submit_button("⏱️ Forecast Arrival Delay")

    trigger_delay = submitted or st.session_state.pop("auto_predict_delay", False)

    if trigger_delay:
        with st.spinner("Calculating arrival delay forecast & bottleneck probability..."):
            payload = {
                "train_number": train_number,
                "station_code": station_code,
                "journey_date": str(journey_date),
                "weather_index": weather_index,
                "congestion_index": congestion_index,
                "distance_from_origin_km": distance_km,
            }
            result = api_post("/predict/delay", payload)

        if result and "predicted_delay_minutes" in result:
            delay_min = result.get("predicted_delay_minutes", 0.0)
            ci = result.get("confidence_interval", {})
            congestion = result.get("congestion_level", "Moderate")
            alert_text = result.get("proactive_alert", "On-schedule operation.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Delay", f"{delay_min:.1f} mins")
            col2.metric("Congestion Level", congestion)
            ci_str = f"[{ci.get('lower_bound', 0):.0f} – {ci.get('upper_bound', 0):.0f} min]" if ci else "±5 min"
            col3.metric("90% Confidence Interval", ci_str)

            alert_class = "alert-success" if delay_min <= 15 else "alert-warning" if delay_min <= 60 else "alert-critical"
            st.markdown(f'<div class="{alert_class}">📢 <b>Advisory:</b> {alert_text}</div>', unsafe_allow_html=True)

            alts = result.get("alternative_options", [])
            if alts:
                st.markdown('<div class="section-header">🔄 Recommended Alternative Trains</div>', unsafe_allow_html=True)
                for alt in alts:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <b style="color:#60a5fa;">{alt.get('train_name')} ({alt.get('train_number')})</b><br>
                                <span style="font-size:0.85rem; color:#94a3b8;">Departs: {alt.get('departure_time')} | Arrives: {alt.get('arrival_time')}</span><br>
                                <span style="font-size:0.8rem; color:#34d399;">💡 {alt.get('reason')}</span>
                            </div>
                            <div style="text-align:right;">
                                <span class="badge-healthy">Save ~{alt.get('delay_advantage_minutes', 0):.0f} mins</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if result.get("delay_drivers"):
                st.markdown('<div class="section-header">🔍 Delay Root-Cause Factors Breakdown</div>', unsafe_allow_html=True)
                drivers = result["delay_drivers"]
                if isinstance(drivers, dict):
                    import plotly.express as px
                    import pandas as pd
                    d_items = [
                        {"Factor": k.replace("_", " ").title(), "Contribution (min)": round(float(v), 1)}
                        for k, v in drivers.items()
                    ]
                    df_drivers = pd.DataFrame(d_items)
                    fig = px.bar(
                        df_drivers,
                        x="Contribution (min)",
                        y="Factor",
                        orientation="h",
                        color="Contribution (min)",
                        color_continuous_scale="Reds",
                        text="Contribution (min)",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#0d1530",
                        font=dict(color="#cbd5e1"),
                        height=230,
                        margin=dict(l=10, r=20, t=10, b=10),
                        xaxis=dict(gridcolor="#1e2d52", title="Delay Impact (Minutes)"),
                        yaxis=dict(gridcolor="#1e2d52", title=""),
                        coloraxis_showscale=False,
                    )
                    fig.update_traces(texttemplate="%{text} min", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            render_telemetry_badge(result.get("governance_telemetry"))
        else:
            st.error(f"Delay prediction failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Dynamic Fare Predictor (Module 5)
# ──────────────────────────────────────────────────────────────────────────────

elif page == "💰 Dynamic Fare Predictor":
    st.markdown(
        '<h1 style="color:#e2e8f0;">💰 Dynamic Fare Predictor & Booking Window Advisor</h1>'
        '<p style="color:#94a3b8;">Forecast dynamic pricing surges, 30-day price trajectories, and cost-saving alternatives.</p>',
        unsafe_allow_html=True,
    )

    with st.form("fare_form"):
        col1, col2 = st.columns(2)
        with col1:
            train_labels = list(TRAIN_CHOICES.keys())
            sel_train = st.selectbox("Select Train", train_labels, index=0)
            train_number = TRAIN_CHOICES[sel_train]
            travel_class = st.selectbox("Travel Class", ["3A", "2A", "1A", "SL", "CC"])
        with col2:
            journey_date = st.date_input("Journey Date", key="fare_journey_date")
            tatkal = st.checkbox("Book under Tatkal Quota", value=False)

        submitted = st.form_submit_button("💰 Forecast Dynamic Fare")

    if submitted:
        with st.spinner("Analyzing demand curves and dynamic fare trajectory..."):
            payload = {
                "train_number": train_number,
                "journey_class": travel_class,
                "journey_date": str(journey_date),
                "tatkal_flag": tatkal,
            }
            result = api_post("/predict/fare", payload)

        if result and "current_predicted_fare_inr" in result:
            pred_fare = result.get("current_predicted_fare_inr", 0.0)
            base_fare = result.get("base_fare_inr", 0.0)
            surge = result.get("surge_multiplier", 1.0)
            demand = result.get("demand_level", "Moderate")
            opt_window = result.get("optimal_booking_window", "14–21 days prior")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Dynamic Fare", f"₹{pred_fare:,.0f}")
            col2.metric("Base Benchmark Fare", f"₹{base_fare:,.0f}")
            col3.metric("Surge Multiplier", f"{surge:.2f}x")
            col4.metric("Demand Pressure", demand)

            st.markdown(
                f'<div class="alert-info">💡 <b>Optimal Booking Advisory:</b> Best purchase window is <b>{opt_window}</b>. Book early to lock in baseline rates.</div>',
                unsafe_allow_html=True,
            )

            # 30-Day Forecast Line Chart
            curve = result.get("forecast_curve", [])
            if curve:
                import plotly.graph_objects as go
                import pandas as pd
                c_df = pd.DataFrame(curve)
                if "forecast_date" in c_df.columns and "predicted_fare_inr" in c_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=c_df["forecast_date"],
                        y=c_df["predicted_fare_inr"],
                        mode="lines+markers",
                        name="Predicted Fare (₹)",
                        line=dict(color="#60a5fa", width=3),
                        marker=dict(size=6, color="#3b82f6"),
                    ))
                    fig.add_hline(y=base_fare, line_dash="dash", line_color="#34d399",
                                  annotation_text=f"Base Fare (₹{base_fare:.0f})", annotation_position="bottom right")
                    fig.update_layout(
                        title="30-Day Dynamic Fare Projection & Surge Horizon",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#0d1530",
                        xaxis=dict(title="Projection Date", gridcolor="#1e2d52", tickfont=dict(color="#94a3b8")),
                        yaxis=dict(title="Fare (INR ₹)", gridcolor="#1e2d52", tickfont=dict(color="#cbd5e1")),
                        margin=dict(l=10, r=20, t=40, b=30),
                        height=280,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            alts = result.get("cost_saving_alternatives", [])
            if alts:
                st.markdown('<div class="section-header">🏷️ Cost-Saving Alternatives</div>', unsafe_allow_html=True)
                for alt in alts:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <b style="color:#60a5fa;">{alt.get('alternative_type')}</b> — {alt.get('description')}<br>
                                <span style="font-size:0.85rem; color:#94a3b8;">Recommended Fare: ₹{alt.get('recommended_fare_inr', 0):,.0f}</span>
                            </div>
                            <div style="text-align:right;">
                                <span class="badge-healthy">Save ₹{alt.get('potential_savings_inr', 0):,.0f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if result.get("fare_drivers"):
                render_shap_chart(result["fare_drivers"], title="Dynamic Fare Drivers (SHAP Attribution)", target_metric="₹ Impact")

            render_telemetry_badge(result.get("governance_telemetry"))
        else:
            st.error(f"Fare prediction failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Multilingual Support Assistant (Module 6)
# ──────────────────────────────────────────────────────────────────────────────

elif page in ("💬 Multilingual Support", "💬 Multilingual Support Concierge"):
    st.markdown(
        '<h1 style="color:#e2e8f0;">💬 Multilingual Customer Support Assistant</h1>'
        '<p style="color:#94a3b8;">Policy & booking concierge powered by RAG with automated Hallucination & Faithfulness auditing.</p>',
        unsafe_allow_html=True,
    )

    col_lang, col_presets = st.columns([1, 3])
    with col_lang:
        selected_lang = st.selectbox("Response Language", ["auto", "en", "hi", "hinglish"], index=0)

    st.markdown("**Sample Railway Policy Queries (Click to test):**")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    preset_query = ""
    if pcol1.button("🚆 Refund on WL Ticket"):
        preset_query = "What happens if my waitlist ticket is not confirmed before chart preparation? Will I get full refund?"
    if pcol2.button("👨‍👩‍👧 Family Ticket Transfer"):
        preset_query = "Can I transfer my confirmed train ticket to a family member like father, mother, or spouse?"
    if pcol3.button("🇮🇳 Tatkal Policy (Hindi)"):
        preset_query = "तत्काल टिकट रद्द करने पर क्या कोई रिफंड मिलता है? IRCTC का क्या नियम है?"
    if pcol4.button("⚡ RAC Rules (Hinglish)"):
        preset_query = "Kya RAC passenger 3A coach me travel kar sakta hai aur berth kaise allot hoti hai?"

    query_input = st.text_area(
        "Enter your railway inquiry",
        value=preset_query or "Can I transfer my confirmed train ticket to a family member?",
        height=90,
    )

    if st.button("💬 Ask Support Concierge", type="primary"):
        with st.spinner("Retrieving verified IRCTC policy chunks & auditing response faithfulness..."):
            payload = {
                "query": query_input,
                "language": selected_lang,
                "user_id": "passenger_web_session",
            }
            result = api_post("/support/query", payload)

        if result and "response" in result:
            ans = result.get("response", "")
            det_lang = result.get("detected_language", "English")
            faith = result.get("faithfulness_score", 0.95)
            hallucinated = result.get("hallucination_flag", False)
            sentiment = result.get("sentiment", "Neutral")
            escalate = result.get("escalation_recommended", False)

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #131c38 0%, #1a274e 100%);
                            border: 1px solid #3b82f6; border-radius: 12px; padding: 20px; margin: 16px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <span class="badge-healthy">🌐 Language: {det_lang.upper()}</span>
                        <span class="{'badge-critical' if hallucinated else 'badge-healthy'}">
                            {'🔴 LOW FAITHFULNESS' if hallucinated else '✅ FAITHFUL & AUDITED'}
                        </span>
                    </div>
                    <div style="font-size: 1.05rem; line-height: 1.6; color: #f1f5f9;">
                        {ans}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness Score", f"{faith:.1%}")
            col2.metric("User Sentiment", sentiment)
            col3.metric("Escalation Gate", "🚨 Escalate to Human" if escalate else "✅ Self-Resolved")

            chunks = result.get("retrieved_contexts", [])
            if chunks:
                with st.expander(f"📚 Retrieved IRCTC Policy Evidence Chunks ({len(chunks)} verified sources)"):
                    for c in chunks:
                        sim = c.get("similarity_score", 0.0)
                        st.markdown(
                            f"""
                            <div style="background:#0d1530; border:1px solid #1e2d52; border-radius:8px; padding:12px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8; margin-bottom:4px;">
                                    <b>Source:</b> {c.get('source_file')} | <b>Chunk ID:</b> {c.get('chunk_id')}
                                    <span style="color:#34d399;">Cosine Sim: {sim:.3f}</span>
                                </div>
                                <div style="font-size:0.88rem; color:#cbd5e1;">{c.get('text')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            render_telemetry_badge(result.get("governance_telemetry"))
        else:
            st.error(f"Support query failed: {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Fraud & Bot Detection (Module 7)
# ──────────────────────────────────────────────────────────────────────────────

elif page in ("🔐 Fraud & Bot Detection", "🛡️ Fraud & Bot Shield"):
    st.markdown(
        '<h1 style="color:#e2e8f0;">🛡️ Real-Time Fraud & Bot Shield</h1>'
        '<p style="color:#94a3b8;">Real-time automated bot scoring, Tatkal abuse mitigation, and suspicious session isolation.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("**⚡ 1-Click Scenario Presets:**")
    fcol1, fcol2, fcol3 = st.columns(3)
    if fcol1.button("🟢 Legitimate Passenger", use_container_width=True):
        st.session_state["fraud_rpm"] = 4.0
        st.session_state["fraud_pnr"] = 1
        st.session_state["fraud_tatkal"] = False
        st.session_state["fraud_captchas"] = 0
        st.session_state["fraud_age"] = 365
        st.session_state["fraud_payments"] = 1
        st.session_state["fraud_ip_risk"] = 0.10
        st.session_state["fraud_bulk"] = False
        st.session_state["fraud_sess"] = "SESS-GENUINE-PASSENGER"
        st.session_state["auto_score_fraud"] = True
    if fcol2.button("🟠 High-Velocity Tatkal User", use_container_width=True):
        st.session_state["fraud_rpm"] = 35.0
        st.session_state["fraud_pnr"] = 4
        st.session_state["fraud_tatkal"] = True
        st.session_state["fraud_captchas"] = 1
        st.session_state["fraud_age"] = 45
        st.session_state["fraud_payments"] = 2
        st.session_state["fraud_ip_risk"] = 0.40
        st.session_state["fraud_bulk"] = False
        st.session_state["fraud_sess"] = "SESS-TATKAL-FREQUENT"
        st.session_state["auto_score_fraud"] = True
    if fcol3.button("🔴 Automated Scalper Bot", use_container_width=True):
        st.session_state["fraud_rpm"] = 120.0
        st.session_state["fraud_pnr"] = 25
        st.session_state["fraud_tatkal"] = True
        st.session_state["fraud_captchas"] = 6
        st.session_state["fraud_age"] = 1
        st.session_state["fraud_payments"] = 6
        st.session_state["fraud_ip_risk"] = 0.85
        st.session_state["fraud_bulk"] = True
        st.session_state["fraud_sess"] = "SESS-SCALPER-BOTNET-99"
        st.session_state["auto_score_fraud"] = True

    with st.form("fraud_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            rpm = st.slider(
                "Requests Per Minute",
                1.0,
                150.0,
                float(st.session_state.get("fraud_rpm", 85.0)),
                step=5.0,
            )
            unique_pnr = st.number_input(
                "Unique PNR Lookups",
                min_value=1,
                max_value=50,
                value=int(st.session_state.get("fraud_pnr", 12)),
            )
        with col2:
            tatkal = st.checkbox(
                "During 10:00–10:15 Tatkal Window",
                value=st.session_state.get("fraud_tatkal", True),
            )
            account_age = st.number_input(
                "Account Age (Days)",
                min_value=0,
                max_value=2000,
                value=int(st.session_state.get("fraud_age", 3)),
            )
            payments = st.number_input(
                "Payment Attempts",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("fraud_payments", 4)),
            )
        with col3:
            captchas = st.slider(
                "Failed Captchas",
                0,
                10,
                int(st.session_state.get("fraud_captchas", 3)),
            )

        with st.expander("⚙️ Advanced Parameters"):
            acol1, acol2 = st.columns(2)
            with acol1:
                session_id = st.text_input("Session ID", value=st.session_state.get("fraud_sess", "SESS-TATKAL-9921"))
                ip_hash = st.text_input("IP Hash", value="a9b8c7d6e5f4")
            with acol2:
                bulk_booking = st.checkbox("Bulk Passenger Count (>= 6)", value=st.session_state.get("fraud_bulk", True))
                ip_risk = st.slider(
                    "Subnet IP Risk Score",
                    0.0,
                    1.0,
                    float(st.session_state.get("fraud_ip_risk", 0.65)),
                    step=0.05,
                )

        submitted = st.form_submit_button("🔐 Score Session Security Risk")

    trigger_fraud = submitted or st.session_state.pop("auto_score_fraud", False)

    if trigger_fraud:
        with st.spinner("Executing real-time bot classification & behavioural audit..."):
            payload = {
                "session_id": session_id,
                "ip_hash": ip_hash,
                "requests_per_minute": rpm,
                "unique_pnr_attempts": unique_pnr,
                "tatkal_flag": tatkal,
                "account_age_days": account_age,
                "payment_attempts": payments,
                "failed_captcha_count": captchas,
                "booking_time_hour": 10,
                "is_bulk_booking": bulk_booking,
                "ip_risk_score": ip_risk,
            }
            result = api_post("/security/score", payload)

        if result and "fraud_risk_score" in result:
            score = result.get("fraud_risk_score", 0.0)
            category = result.get("risk_category", "Safe")
            action = result.get("recommended_action", "ALLOW")

            cat_badge = (
                '<span class="badge-critical">🔴 BOT DETECTED</span>' if "Bot" in category
                else '<span class="badge-critical">🟠 HIGH RISK</span>' if "High" in category
                else '<span class="badge-warning">🟡 SUSPICIOUS</span>' if "Suspicious" in category
                else '<span class="badge-healthy">🟢 SAFE</span>'
            )

            col1, col2 = st.columns(2)
            col1.metric("Fraud Risk Score", f"{score:.1f} / 100")
            col2.markdown(f"**Threat Category:**<br>{cat_badge}", unsafe_allow_html=True)

            action_cls = "badge-critical" if "BLOCK" in action else "badge-warning" if "CHALLENGE" in action or "RATE" in action else "badge-healthy"
            action_tag = "🛡️ IMMEDIATE ISOLATION & BLOCK" if "BLOCK" in action else "⚠️ IDENTITY CHALLENGE REQUIRED" if "CHALLENGE" in action else "✅ ACCESS PERMITTED"

            st.markdown(
                f"""
                <div style="background:#131c38; border:1px solid #2d4170; border-radius:10px; padding:14px 20px; margin: 14px 0; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em;">Automated Security Policy Action</span><br>
                        <span style="font-size:1.2rem; font-weight:800; color:#e2e8f0; font-family:monospace;">{action}</span>
                    </div>
                    <span class="{action_cls}" style="font-size:0.85rem; padding:6px 14px;">
                        {action_tag}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(1.0, score / 100.0))

            inv = result.get("investigation_features", [])
            if inv:
                st.markdown('<div class="section-header">🔍 Threat Investigation Dossier</div>', unsafe_allow_html=True)
                import pandas as pd
                idf = pd.DataFrame([
                    {
                        "Feature": it.get("feature"),
                        "Observed Value": it.get("value"),
                        "Anomaly Severity": it.get("severity"),
                        "Finding": it.get("description"),
                    }
                    for it in inv
                ])
                st.dataframe(idf, use_container_width=True, hide_index=True)

            if result.get("shap_risk_drivers"):
                render_shap_chart(result["shap_risk_drivers"], title="Fraud Risk Feature Contribution (SHAP)", target_metric="Risk Multiplier")

            render_telemetry_badge(result.get("governance_telemetry"))
        else:
            st.error(f"Fraud scoring failed: {result}")

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
