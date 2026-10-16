"""
frontend/app.py
───────────────
Railway Buddy — Streamlit entry point.

  • Passenger services (Modules 1–7) — redesigned, blue-and-white consumer UI
  • AI Governance Control Tower — a single scrollable operations page

Run with:  streamlit run streamlit_app.py   (or)   python run_platform.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)

import streamlit as st

from frontend import passenger, tower
from frontend.ui import BRAND, api_post, backend_online, engine_label, inject_theme

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title=f"{BRAND} — AI Railway Travel & Governance",
    page_icon=str(ASSETS / "icon.svg") if (ASSETS / "icon.svg").exists() else "🚆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"{BRAND}: AI-powered railway travel with an autonomous AI governance layer."},
)

try:
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
except Exception:
    pass

# Warm the platform once per session: push a little REAL traffic through the governed models so every
# Control Tower pillar has live data on a cold start (the cloud container boots with empty monitors).
if "boot_warm" not in st.session_state:
    st.session_state["boot_warm"] = True
    try:
        from frontend.ui import api_get
        h = api_get("/governance/health/platform") or {}
        if h and h.get("total_requests", 0) < 25:
            with st.spinner("Warming up Railway Buddy…"):
                tower.simulate_traffic(12, drifted=False, quiet=True)
    except Exception:
        pass

PAGES: dict = {}


def _home():
    passenger.home(PAGES)


PAGES["home"] = st.Page(_home, title="Home", icon="🏠", url_path="home", default=True)
PAGES["ticket"] = st.Page(passenger.ticket, title="Ticket Confirmation", icon="🎫", url_path="ticket")
PAGES["recommend"] = st.Page(passenger.recommend, title="Train Recommender", icon="🚆", url_path="trains")
PAGES["delay"] = st.Page(passenger.delay, title="Delay & Disruption", icon="⏱️", url_path="delay")
PAGES["planner"] = st.Page(passenger.planner, title="AI Travel Planner", icon="🧭", url_path="planner")
PAGES["fare"] = st.Page(passenger.fare, title="Fare Forecast", icon="💰", url_path="fares")
PAGES["support"] = st.Page(passenger.support, title="Support Assistant", icon="💬", url_path="support")
PAGES["fraud"] = st.Page(passenger.fraud, title="Fraud & Bot Shield", icon="🛡️", url_path="security")
PAGES["tower"] = st.Page(tower.tower, title="Governance Control Tower", icon="🏛️", url_path="governance-tower")

nav = st.navigation(
    {
        "Railway Buddy": [PAGES[k] for k in ("home", "ticket", "recommend", "delay", "planner", "fare", "support", "fraud")],
        "AI Governance": [PAGES["tower"]],
    }
)

with st.sidebar:
    online = backend_online()
    st.markdown(
        f'<div style="background:#F3F7FE;border:1px solid #D9E4F5;border-radius:12px;padding:10px 12px;font-size:.8rem;margin-top:8px">'
        f'<div style="font-weight:700;color:#0A2463;margin-bottom:4px">System status</div>'
        f'{"🟢 All services online" if online else "🔴 Backend starting…"}<br>'
        f'<span style="color:#5B6B82">Engine: {engine_label()}</span></div>',
        unsafe_allow_html=True,
    )

nav.run()

st.markdown(
    f'<div class="rb-foot">🚆 <b>{BRAND}</b> · AI-powered railway travel with autonomous AI governance<br>'
    "FastAPI · Streamlit · XGBoost · LightGBM · SHAP · SciPy</div>",
    unsafe_allow_html=True,
)
