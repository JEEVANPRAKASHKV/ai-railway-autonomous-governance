"""
frontend/components/telemetry_badge.py
──────────────────────────────────────
Governance receipt shown under every AI result: proves the answer was monitored.
Only real values returned by the backend are displayed — nothing is invented.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from frontend.ui import esc


def render_telemetry_badge(telemetry: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    d = {**(telemetry or {}), **kwargs}
    if not d:
        return
    cells = []
    if d.get("latency_ms") is not None:
        cells.append(("⏱ Latency", f'{float(d["latency_ms"]):.0f} ms'))
    if "schema_valid" in d:
        cells.append(("📋 Input check", "Passed" if d["schema_valid"] else "Flagged"))
    if d.get("confidence_score") is not None:
        cells.append(("🎯 Confidence", f'{float(d["confidence_score"]) * 100:.0f}%'))
    cells.append(("📉 Drift", "Flagged" if d.get("drift_flag") else "Clear"))
    cells.append(("⚖️ Bias", "Flagged" if d.get("bias_flag") else "Clear"))
    if d.get("llm_mode"):
        cells.append(("⚙️ Engine", "Smart engine" if "fallback" in str(d["llm_mode"]).lower() else esc(d["llm_mode"])))
    items = "".join(
        f'<div style="min-width:110px"><div style="font-size:.66rem;letter-spacing:.07em;text-transform:uppercase;color:#5B6B82;font-weight:700">{k}</div>'
        f'<div style="font-weight:700;color:#0A2463">{v}</div></div>' for k, v in cells
    )
    audit = esc(d.get("audit_id", "—"))
    st.markdown(
        f'<div style="background:#fff;border:1px solid #D9E4F5;border-left:4px solid var(--accent);border-radius:12px;'
        f'padding:12px 16px;margin:14px 0 6px;display:flex;flex-wrap:wrap;gap:22px;align-items:center">'
        f'<div style="font-weight:800;color:var(--deep);font-size:.82rem">🛡️ Governed by<br>AI Control Tower</div>{items}'
        f'<div style="margin-left:auto;font-size:.72rem;color:#7C8DA6">Audit ID<br><code>{audit}</code></div></div>',
        unsafe_allow_html=True,
    )
