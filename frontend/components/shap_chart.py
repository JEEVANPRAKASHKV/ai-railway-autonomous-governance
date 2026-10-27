"""
frontend/components/shap_chart.py
─────────────────────────────────
"Why this prediction?" chart: SHAP feature attributions in the Railway Buddy blue theme.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import plotly.graph_objects as go
import streamlit as st

from frontend.ui import NEG, POS, style_fig


def _fmt(v: float, metric: str) -> str:
    if "₹" in metric:
        return f"{'+' if v >= 0 else '−'}₹{abs(v):,.0f}"
    if "min" in metric.lower():
        return f"{v:+.1f} min"
    return f"{v:+.3f}"


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
                if e.get("feature") == "base_value" or e.get("type") == "Baseline":
                    continue
                name = e.get("feature_name") or e.get("factor") or e.get("driver") or e.get("feature") or e.get("name") or "Unknown"
                val = next((e[k] for k in ("contribution", "contribution_inr", "impact", "impact_minutes", "importance")
                            if e.get(k) is not None), 0.0)
                try:
                    items.append({"feature": str(name), "contribution": float(val)})
                except (TypeError, ValueError):
                    continue
    if not items:
        st.info("No numeric feature contributions available.")
        return

    items = [i for i in items if abs(i["contribution"]) >= 1e-4] or items   # hide factors with no effect
    items.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top = list(reversed(items[:8]))
    labels = [t["feature"].replace("_", " ").title() if "_" in t["feature"] else t["feature"] for t in top]
    vals = [t["contribution"] for t in top]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=[POS if v >= 0 else NEG for v in vals]),
        text=[_fmt(v, target_metric) for v in vals], textposition="auto",
    ))
    fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
                      xaxis_title=f"{target_metric}  (black raises, gold lowers)")
    style_fig(fig, height=max(240, len(labels) * 34 + 70))
    st.plotly_chart(fig, width="stretch")
