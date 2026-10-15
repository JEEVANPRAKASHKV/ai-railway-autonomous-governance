"""
frontend/components/shap_chart.py
─────────────────────────────────
Reusable SHAP feature importance & waterfall visualization component.
Utilizes Plotly to render interactive, publication-quality explanations
for model predictions (M1 Ticket, M5 Fare, M7 Fraud).
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
import streamlit as st
import plotly.graph_objects as go


def render_shap_chart(
    shap_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    title: str = "SHAP Explainability — Feature Attribution Breakdown",
    target_metric: str = "Contribution",
) -> None:
    """
    Renders an interactive horizontal bar chart of feature contributions.
    Positive values push prediction higher (green), negative lower (red).
    """
    if not shap_data:
        st.info("ℹ️ SHAP attribution data not available for this prediction.")
        return

    # Normalize list format
    items: List[Dict[str, Any]] = []
    if isinstance(shap_data, dict):
        for k, v in shap_data.items():
            if isinstance(v, (int, float)):
                items.append({"feature": k, "contribution": float(v)})
            elif isinstance(v, dict) and "contribution" in v:
                items.append({"feature": k, "contribution": float(v["contribution"])})
    elif isinstance(shap_data, list):
        for entry in shap_data:
            if isinstance(entry, dict):
                feat = entry.get("feature", entry.get("name", "Unknown"))
                val = entry.get("contribution", entry.get("value", entry.get("importance", 0.0)))
                items.append({"feature": str(feat), "contribution": float(val)})

    if not items:
        st.info("No numeric feature contributions extracted.")
        return

    # Sort by absolute impact
    items.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top_items = items[:10]  # Show top 10 features

    features = [it["feature"].replace("_", " ").title() for it in reversed(top_items)]
    contributions = [it["contribution"] for it in reversed(top_items)]
    colors = ["#34d399" if c >= 0 else "#f87171" for c in contributions]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=features,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#2d4170", width=1)),
            text=[f"{c:+.3f}" for c in contributions],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(
            title=dict(text=f"{target_metric} (Impact on Prediction)", font=dict(color="#cbd5e1")),
            gridcolor="#1e2d52",
            zerolinecolor="#60a5fa",
            zerolinewidth=2,
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            tickfont=dict(color="#e2e8f0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1530",
        margin=dict(l=10, r=20, t=40, b=30),
        height=max(240, len(features) * 32),
    )

    st.plotly_chart(fig, use_container_width=True)
