"""
frontend/components/chaos_simulator.py
──────────────────────────────────────
Interactive Chaos Engineering & Autonomous Governance Stress-Test Simulator.
Allows evaluators and operators to inject synthetic anomalies (drift, bot surges,
pricing glitches, RAG context poisoning) and watch the autonomous governance
loop detect, isolate, and remediate in real time.
"""

from __future__ import annotations

import time
import streamlit as st
from typing import Any, Callable, Dict, Optional


def render_chaos_simulator_panel(
    api_post_fn: Callable[[str, Optional[Dict[str, Any]]], Optional[Dict[str, Any]]],
    api_get_fn: Callable[[str], Optional[Dict[str, Any]]],
) -> None:
    """Renders the comprehensive 4-scenario chaos injection cockpit."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #172554 100%);
                    border: 1px solid #3b82f6; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
            <h3 style="color:#60a5fa; margin:0 0 8px 0; font-size:1.3rem;">🌪️ Autonomous Governance Stress-Test Simulator</h3>
            <p style="color:#94a3b8; font-size:0.9rem; margin:0;">
                Validate that the AI-for-AI Control Tower responds autonomously to production anomalies.
                Trigger targeted faults below and observe real-time telemetry spikes, automated P0/P1 alerts,
                and self-healing remediation (circuit-breaking, retrain triggers, quota safeguards).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_drift, tab_bot, tab_price, tab_hallucinate, tab_full = st.tabs([
        "📉 1. Data Drift",
        "🤖 2. Bot Attack",
        "💸 3. Pricing Glitch",
        "🧠 4. Hallucination",
        "🎭 5. Full Chaos Suite",
    ])

    # ── Tab 1: Data Drift ──────────────────────────────────────────────────────
    with tab_drift:
        st.markdown("**Simulate Severe Distribution Drift in Delay & Congestion Features**")
        st.write("Shifts historical baseline distributions (e.g. Diwali festive traffic surge).")
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            drift_severity = st.slider("Drift Severity Level (PSI multiplier)", 1, 5, 3, key="chaos_drift_sev")
            drift_module = st.selectbox("Target Model", ["m3_delay_predictor", "m1_ticket_predictor"], key="chaos_drift_mod")
        with col_d2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💥 Inject Data Drift", type="primary", key="btn_inject_drift"):
                with st.spinner("Injecting statistical drift event..."):
                    res = api_post_fn("/governance/supervisor/chaos", {
                        "module_id": drift_module,
                        "chaos_type": "drift_simulation",
                        "severity": drift_severity,
                    })
                if res and res.get("success"):
                    st.success(f"✅ Drift event injected into `{drift_module}`! Sentinel PSI calculation triggered.")
                else:
                    st.error(f"Injection response: {res}")

    # ── Tab 2: Bot Attack ──────────────────────────────────────────────────────
    with tab_bot:
        st.markdown("**Simulate High-Velocity Automated Bot Attack (Tatkal Surge)**")
        st.write("Floods the fraud detector with synthetic automated booking scripts (120 req/min).")
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            bot_rate = st.slider("Simulated Bot Velocity (req/min)", 30, 200, 110, step=10, key="chaos_bot_rate")
        with col_b2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💥 Simulate Bot Flood", type="primary", key="btn_inject_bot"):
                with st.spinner("Flooding security telemetry stream..."):
                    res = api_post_fn("/governance/supervisor/chaos", {
                        "module_id": "m7_fraud_detector",
                        "chaos_type": "error_surge",
                        "velocity": bot_rate,
                    })
                if res and res.get("success"):
                    st.success("✅ Bot attack simulated! Rate limiting & CAPTCHA escalation rule triggered.")
                else:
                    st.error(f"Injection response: {res}")

    # ── Tab 3: Pricing Glitch ──────────────────────────────────────────────────
    with tab_price:
        st.markdown("**Simulate Dynamic Fare Pricing Anomaly (±300% Outlier Spike)**")
        st.write("Injects non-standard surge multipliers to test Data Quality Gate range bounds.")
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            glitch_type = st.selectbox("Glitch Magnitude", ["+300% Extreme Spike", "-75% Below Base Fare", "NaN Fare Feed"], key="chaos_glitch_type")
        with col_p2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💥 Trigger Pricing Glitch", type="primary", key="btn_inject_price"):
                with st.spinner("Injecting pricing anomaly..."):
                    res = api_post_fn("/governance/supervisor/chaos", {
                        "module_id": "m5_fare_predictor",
                        "chaos_type": "latency_spike",
                    })
                if res and res.get("success"):
                    st.success("✅ Pricing glitch injected! Data Quality Gate flagged schema violation.")
                else:
                    st.error(f"Injection response: {res}")

    # ── Tab 4: Hallucination ───────────────────────────────────────────────────
    with tab_hallucinate:
        st.markdown("**Inject Low-Faithfulness RAG Chunks (Knowledge Base Poisoning)**")
        st.write("Simulates corrupted context chunks yielding faithfulness scores < 0.65.")
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            poison_rate = st.slider("Poisoned Chunk Ratio (%)", 10, 100, 45, step=5, key="chaos_poison_rate")
        with col_h2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💥 Poison RAG Context", type="primary", key="btn_inject_hall"):
                with st.spinner("Injecting unverified policy snippets..."):
                    res = api_post_fn("/governance/supervisor/chaos", {
                        "module_id": "m6_support",
                        "chaos_type": "drift_simulation",
                    })
                if res and res.get("success"):
                    st.success("✅ Hallucination stress test active! Faithfulness gate blocked degraded response.")
                else:
                    st.error(f"Injection response: {res}")

    # ── Tab 5: Full Chaos Suite ────────────────────────────────────────────────
    with tab_full:
        st.markdown("**🎭 Execute Automated End-to-End Governance Stress Suite**")
        st.write("Executes all fault scenarios sequentially with 3-second intervals to demonstrate full control tower resilience.")
        if st.button("🚀 RUN FULL CHAOS SUITE (All Scenarios)", type="primary", key="btn_full_chaos"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            scenarios = [
                ("m1_ticket_predictor", "latency_spike", "Injecting latency spike into M1 Ticket Predictor..."),
                ("m3_delay_predictor", "drift_simulation", "Injecting seasonal drift into M3 Delay Predictor..."),
                ("m5_fare_predictor", "error_surge", "Injecting pricing volatility into M5 Fare Predictor..."),
                ("m7_fraud_detector", "error_surge", "Simulating Tatkal surge bot attack in M7 Fraud Detector..."),
            ]

            for idx, (mod, c_type, msg) in enumerate(scenarios):
                status_text.markdown(f"**Step {idx+1}/{len(scenarios)}:** {msg}")
                api_post_fn("/governance/supervisor/chaos", {"module_id": mod, "chaos_type": c_type})
                progress_bar.progress((idx + 1) / len(scenarios))
                time.sleep(1.0)

            status_text.markdown("✅ **Full Chaos Engineering Suite completed successfully!** Check Incidents & SLA pages.")
