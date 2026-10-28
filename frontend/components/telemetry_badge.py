"""
frontend/components/telemetry_badge.py
──────────────────────────────────────
Governance receipt shown under every AI result: proves the answer was monitored.
Only real values returned by the backend are displayed — nothing is invented.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from frontend.ui import details, esc


def render_telemetry_badge(telemetry: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    d = {**(telemetry or {}), **kwargs}
    if not d or not details():      # the receipt is for staff and reviewers, not passengers
        return
    cells = []
    if d.get("latency_ms") is not None:
        cells.append(("Latency", f'{float(d["latency_ms"]):.0f} ms'))
    if "schema_valid" in d:
        cells.append(("Input check", "Passed" if d["schema_valid"] else "Flagged"))
    if d.get("confidence_score") is not None:
        cells.append(("Confidence", f'{float(d["confidence_score"]) * 100:.0f}%'))
    cells.append(("Drift", "Flagged" if d.get("drift_flag") else "Clear"))
    cells.append(("Bias", "Flagged" if d.get("bias_flag") else "Clear"))
    if d.get("llm_mode") and str(d["llm_mode"]).lower() != "ml":
        mode = str(d["llm_mode"]).lower()
        cells.append(("Engine", {"sarvam": "Sarvam AI", "gemini": "Gemini", "azure_openai": "Azure OpenAI"}.get(
            mode, "Smart engine" if "fallback" in mode else esc(d["llm_mode"]))))
    items = "".join(
        f'<div class="c"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells
    )
    audit = esc(d.get("audit_id", "—"))
    st.markdown(
        '<style>.rb-receipt{display:flex;flex-wrap:wrap;border:1px solid #ddd;background:#fff;margin:12px 0 6px;font-size:.82rem}'
        '.rb-receipt .c{padding:6px 12px;border-right:1px solid #ddd}.rb-receipt .k{color:#555;font-size:.74rem}'
        '.rb-receipt .v{font-family:var(--mono);font-weight:600;color:#000}'
        '.rb-receipt .h{padding:6px 12px;border-right:1px solid #ddd;background:#000;color:#F5C400;font-weight:700;display:flex;align-items:center}'
        '.rb-receipt .a{margin-left:auto;border-right:0}</style>'
        f'<div class="rb-receipt"><div class="h">Governance receipt</div>{items}'
        f'<div class="c a"><div class="k">Audit ID</div><div class="v">{audit}</div></div></div>',
        unsafe_allow_html=True,
    )
