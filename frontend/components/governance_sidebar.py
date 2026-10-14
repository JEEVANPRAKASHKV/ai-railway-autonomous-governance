"""
frontend/components/governance_sidebar.py
─────────────────────────────────────────
Sidebar Mini-Widget for Real-time Governance Telemetry & LLM Mode Status.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def render_sidebar_governance_widget(
    is_online: bool,
    governance_data: Optional[Dict[str, Any]] = None,
    llm_mode: str = "Fallback 🔧",
) -> None:
    """Renders the mini telemetry cockpit directly in the Streamlit sidebar."""
    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #131c38 0%, #1a274e 100%);
            border: 1px solid #2d4170;
            border-radius: 10px;
            padding: 12px;
            margin: 10px 0;
            font-size: 0.8rem;
        ">
            <div style="font-weight: 700; color: #60a5fa; margin-bottom: 6px; display:flex; align-items:center; gap:6px;">
                <span>🛡️</span> <span>CONTROL TOWER STATUS</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if is_online:
        st.sidebar.markdown(
            '<div style="color: #34d399; font-weight:600; margin-bottom:4px;">🟢 Systems: Operational</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div style="color: #fbbf24; font-weight:600; margin-bottom:4px;">🟡 Server: Offline (Mock/Local)</div>',
            unsafe_allow_html=True,
        )

    # LLM Mode indicator
    st.sidebar.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; margin:4px 0; color:#cbd5e1;">
            <span>Engine:</span>
            <span style="font-weight:600; color:#93c5fd;">{llm_mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if governance_data:
        score = governance_data.get("governance_score", 96)
        healthy_models = governance_data.get("healthy_models_count", "7/7")
        st.sidebar.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; margin:4px 0; color:#cbd5e1;">
                <span>Gov Score:</span>
                <span style="font-weight:700; color:#34d399;">{score}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin:4px 0; color:#cbd5e1;">
                <span>Active Models:</span>
                <span style="font-weight:600; color:#60a5fa;">{healthy_models}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
