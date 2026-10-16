"""
frontend/components/shap_chart.py
─────────────────────────────────
"Why this prediction?" chart: SHAP feature attributions in the Railway Buddy blue theme.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import plotly.graph_objects as go
import streamlit as st

from frontend.ui import style_fig


def render_shap_chart(
    shap_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    title: str = "Why this prediction? (SHAP)",
    target_metric: str = "Contribution",
) -> None:
    if not shap_data:
        st.info("Explanation not available for this prediction.")
        return

    items: List[Dict[str, Any]] = []
    if isinstance(shap_data, dict):
        for k, v in shap_data.items():
            if isinstance(v, (int, float)):
                items.append({"feature": k, "contribution": float(v)})
            elif isinstance(v, dict) and "contribution" in v:
                items.append({"feature": k, "contribution": float(v["contribution"])})
    else:
        for e in shap_data:
            if isinstance(e, dict):
                name = e.get("feature_name") or e.get("feature") or e.get("name") or "Unknown"
                val = e.get("contribution", e.get("value", e.get("importance", 0.0)))
                try:
                    items.append({"feature": str(name), "contribution": float(val)})
                except (TypeError, ValueError):
                    continue
    if not items:
        st.info("No numeric feature contributions available.")
        return

    items.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top = list(reversed(items[:8]))
    labels = [t["feature"].replace("_", " ").title() if "_" in t["feature"] else t["feature"] for t in top]
    vals = [t["contribution"] for t in top]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=["#2563EB" if v >= 0 else "#F97316" for v in vals]),
        text=[f"{v:+.3f}" for v in vals], textposition="auto",
    ))
    fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
                      xaxis_title=f"{target_metric}  (blue ↑ raises, orange ↓ lowers)")
    style_fig(fig, height=max(240, len(labels) * 34 + 70))
    st.plotly_chart(fig, use_container_width=True)
