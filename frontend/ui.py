"""
frontend/ui.py
──────────────
Railway Buddy design system — a light, blue-and-white theme.

Every screen gets its own shade of blue (``ACCENTS``) so the user always knows where
they are, while neutrals, spacing, cards and typography stay identical across the app.
Semantic status colours (green / amber / red) are reserved for health & risk signals.
"""

from __future__ import annotations

import html
import os
from typing import Any, Dict, List, Optional, Sequence

import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Backend access
# ──────────────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("BACKEND_URL", os.getenv("API_BASE", "http://127.0.0.1:8000")).rstrip("/")
API_V1 = f"{API_BASE}/api/v1"


def api_get(path: str, params: Optional[dict] = None) -> Optional[Any]:
    try:
        r = requests.get(f"{API_V1}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# The UI runs server-side, so it can hold the governance keys; browsers never see them.
OPERATOR_KEY = os.getenv("RB_OPERATOR_KEY") or "rb-dev-operator-key"
ADMIN_KEY = os.getenv("RB_ADMIN_KEY") or "rb-dev-admin-key"
_ADMIN_PATHS = ("/governance/supervisor/hitl/decide", "/governance/supervisor/reset")


def api_post(path: str, payload: Optional[dict] = None, timeout: int = 30) -> Optional[Any]:
    """POST to the API. Governance calls carry the operator (or admin) key and identify the UI as the actor.
    Errors come back as {"error": <human-readable message>, "status": <code>} instead of raising."""
    headers = {}
    if path.startswith("/governance"):
        headers = {"X-API-Key": ADMIN_KEY if path.startswith(_ADMIN_PATHS) else OPERATOR_KEY,
                   "X-Actor": "Control Tower UI"}
    try:
        r = requests.post(f"{API_V1}{path}", json=payload or {}, timeout=timeout, headers=headers)
    except Exception as exc:
        return {"error": f"Backend unreachable ({type(exc).__name__}).", "status": 0}
    if r.status_code >= 400:
        try:
            body = r.json()
            detail = body.get("detail") or body.get("message") or r.reason
            if isinstance(detail, list):   # FastAPI validation errors
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            if body.get("violations"):
                detail = f"{detail} — " + "; ".join(map(str, body["violations"][:3]))
        except Exception:
            detail = r.reason
        return {"error": str(detail), "status": r.status_code}
    try:
        return r.json()
    except Exception:
        return {"error": "Unexpected response from the API.", "status": r.status_code}


def send_outcome(audit_id: Optional[str], actual: Any) -> Optional[dict]:
    """Report what really happened for a prediction (closes the governance feedback loop)."""
    if not audit_id:
        return {"error": "This prediction has no audit id."}
    return api_post("/governance/outcomes", {"audit_id": audit_id, "actual": actual, "source": "passenger"})


def backend_online() -> bool:
    try:
        return requests.get(f"{API_BASE}/api/health", timeout=3).status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────

BRAND = "Railway Buddy"

# One blue per screen: (accent, deep, soft tint)
ACCENTS: Dict[str, tuple] = {
    "home":      ("#1D4ED8", "#1E3A8A", "#DBEAFE"),
    "ticket":    ("#2563EB", "#1E40AF", "#DBEAFE"),
    "recommend": ("#0284C7", "#075985", "#E0F2FE"),
    "delay":     ("#0891B2", "#155E75", "#CFFAFE"),
    "planner":   ("#4F46E5", "#3730A3", "#E0E7FF"),
    "fare":      ("#0EA5E9", "#0369A1", "#E0F2FE"),
    "support":   ("#3B82F6", "#1D4ED8", "#DBEAFE"),
    "fraud":     ("#1E40AF", "#172554", "#DBEAFE"),
    "tower":     ("#0B3B8C", "#0A2463", "#DBEAFE"),
}

TONES = {  # status tones -> (fg, bg, border)
    "ok":    ("#047857", "#ECFDF5", "#A7F3D0"),
    "warn":  ("#B45309", "#FFFBEB", "#FDE68A"),
    "bad":   ("#B91C1C", "#FEF2F2", "#FECACA"),
    "info":  ("#1D4ED8", "#EFF6FF", "#BFDBFE"),
    "mute":  ("#475569", "#F1F5F9", "#E2E8F0"),
}

CHART_BLUES = ["#1D4ED8", "#0EA5E9", "#6366F1", "#0891B2", "#60A5FA", "#1E3A8A"]


def esc(x: Any) -> str:
    return html.escape(str(x))


def tone_for(status: str) -> str:
    s = str(status).upper()
    if s in ("HEALTHY", "STABLE", "COMPLIANT", "FAIR", "CLOSED", "OK", "ACTIVE", "SAFE", "PASS", "CLEAR", "ALLOW",
             "ACCEPTED", "APPROVED", "INTACT", "PASSED", "WITHIN LIMITS"):
        return "ok"
    if s in ("WARNING", "DEGRADED", "AT_RISK", "HALF_OPEN", "RETRAINING", "SUSPICIOUS", "INCIDENT", "MEDIUM", "HIGH", "REVIEW",
             "WARN", "CANARY", "PENDING", "PENDING_APPROVAL", "SHIFT", "NEEDS_CLARIFICATION", "REJECTED"):
        return "warn"
    if s in ("CRITICAL", "DRIFT", "BIAS_DETECTED", "OPEN", "BOT DETECTED", "HIGH RISK", "BLOCK", "FAIL", "BLOCKED",
             "DEGRADED_QUALITY", "BROKEN", "FAILED"):
        return "bad"
    return "info"


# ──────────────────────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────────────────────

def inject_theme(screen: str = "home") -> None:
    accent, deep, soft = ACCENTS.get(screen, ACCENTS["home"])
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {{
  --accent: {accent}; --deep: {deep}; --soft: {soft};
  --ink: #0B1F3A; --muted: #5B6B82; --line: #D9E4F5; --bg: #F3F7FE; --card: #FFFFFF;
}}
html, body, [class*="css"], .stApp {{ font-family: 'Plus Jakarta Sans', system-ui, sans-serif; color: var(--ink); }}
.stApp {{ background: var(--bg); }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid var(--line); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
.rb-brand {{ display:flex; align-items:center; gap:10px; padding: 4px 4px 14px; }}
.rb-logo {{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:22px;
  background: linear-gradient(135deg,#2563EB,#0B3B8C); color:#fff; box-shadow: 0 6px 16px rgba(37,99,235,.35); }}
.rb-name {{ font-weight:800; font-size:1.15rem; line-height:1.1; color:#0A2463; }}
.rb-tag {{ font-size:.72rem; color: var(--muted); }}
.rb-nav-label {{ font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color:#7C8DA6; font-weight:700; margin:14px 4px 4px; }}

/* Hero */
.rb-hero {{ border-radius: 20px; padding: 26px 30px; color:#fff; margin-bottom: 18px; position:relative; overflow:hidden;
  background: linear-gradient(120deg, var(--deep) 0%, var(--accent) 100%);
  box-shadow: 0 14px 34px -14px var(--accent); }}
.rb-hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,.22), transparent 65%); }}
.rb-hero h1 {{ margin:0; font-size:1.75rem; font-weight:800; letter-spacing:-.01em; color:#fff; }}
.rb-hero p {{ margin:6px 0 0; font-size:.98rem; color: rgba(255,255,255,.88); max-width: 760px; }}
.rb-hero .rb-chips {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }}
.rb-chip {{ background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); padding:4px 11px; border-radius:999px; font-size:.76rem; font-weight:600; }}

/* Cards */
.rb-card {{ background: var(--card); border:1px solid var(--line); border-radius:16px; padding:18px 20px;
  box-shadow: 0 1px 2px rgba(11,31,58,.04), 0 8px 24px -16px rgba(11,31,58,.18); margin-bottom: 12px; }}
.rb-card h4 {{ margin:0 0 4px; font-size:1rem; font-weight:700; color: var(--deep); }}
.rb-sub {{ color: var(--muted); font-size:.84rem; }}
.rb-kpi {{ background: var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; min-height:104px; box-sizing:border-box;
  border-top: 3px solid var(--accent); }}
.rb-kpi .l {{ font-size:.7rem; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); font-weight:700; }}
.rb-kpi .v {{ font-size:1.65rem; font-weight:800; color: var(--deep); line-height:1.25; }}
.rb-kpi .s {{ font-size:.76rem; color: var(--muted); }}
.rb-kpi.ok {{ border-top-color:#10B981; }} .rb-kpi.warn {{ border-top-color:#F59E0B; }} .rb-kpi.bad {{ border-top-color:#EF4444; }}

/* Section headings */
.rb-sec {{ display:flex; align-items:flex-end; justify-content:space-between; gap:12px; margin: 30px 0 10px; padding-bottom:8px;
  border-bottom: 2px solid var(--soft); }}
.rb-sec .t {{ font-size:1.18rem; font-weight:800; color: var(--deep); }}
.rb-sec .d {{ font-size:.84rem; color: var(--muted); }}
.rb-sec .n {{ background: var(--accent); color:#fff; border-radius:8px; font-size:.72rem; font-weight:800; padding:2px 8px; margin-right:8px; }}

/* Badges & callouts */
.rb-badge {{ display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:999px; font-size:.74rem; font-weight:700; border:1px solid; white-space:nowrap; }}
.rb-call {{ border-radius:12px; padding:12px 16px; font-size:.92rem; border:1px solid; margin: 8px 0; }}

/* Stat rows inside cards */
.rb-row {{ display:flex; justify-content:space-between; gap:10px; padding:7px 0; border-bottom:1px dashed var(--line); font-size:.9rem; }}
.rb-row:last-child {{ border-bottom:0; }} .rb-row .k {{ color: var(--muted); }} .rb-row .v {{ font-weight:700; color:var(--ink); text-align:right; }}

/* Pillar tiles */
.rb-pillar {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px 14px; min-height:98px; box-sizing:border-box; }}
.rb-pillar .h {{ display:flex; justify-content:space-between; align-items:center; gap:6px; font-weight:700; font-size:.88rem; color:var(--deep); }}
.rb-pillar .m {{ font-size:.78rem; color:var(--muted); margin-top:6px; min-height:2.2em; }}
.rb-dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}

/* Anchor nav */
.rb-anchors {{ position:sticky; top:48px; z-index:50; background: rgba(243,247,254,.92); backdrop-filter: blur(8px);
  padding:8px 0; display:flex; flex-wrap:wrap; gap:8px; border-bottom:1px solid var(--line); margin-bottom:6px; }}
.rb-anchors a {{ text-decoration:none; font-size:.76rem; font-weight:700; color: var(--deep); background:#fff; border:1px solid var(--line);
  padding:4px 11px; border-radius:999px; }}
.rb-anchors a:hover {{ background: var(--accent); color:#fff; border-color: var(--accent); }}

/* Streamlit widgets */
div[data-testid="stMetric"] {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px 16px; border-top:3px solid var(--accent); }}
div[data-testid="stMetricValue"] {{ color: var(--deep); font-weight:800; }}
div[data-testid="stMetricLabel"] p {{ color: var(--muted); font-weight:600; }}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{ border-radius:11px; font-weight:700; border:1px solid var(--line); }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stFormSubmitButton > button, .stDownloadButton > button {{
  background: var(--accent); color:#fff; border-color: var(--accent); }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {{ background: var(--deep); border-color: var(--deep); color:#fff; }}
div[data-testid="stForm"] {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 20px; }}
div[data-testid="stExpander"] {{ background:#fff; border:1px solid var(--line); border-radius:14px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:16px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
.stTabs [data-baseweb="tab"] {{ background:#fff; border:1px solid var(--line); border-radius:10px 10px 0 0; padding:6px 14px; }}
.stTabs [aria-selected="true"] {{ background: var(--accent); color:#fff !important; }}
.stTabs [aria-selected="true"] p {{ color:#fff !important; }}
.stProgress > div > div > div > div {{ background: var(--accent); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding:7px 10px; border-radius:10px; width:100%; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: var(--soft); }}
.rb-foot {{ text-align:center; color:#7C8DA6; font-size:.78rem; margin-top:42px; padding-top:16px; border-top:1px solid var(--line); }}
</style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

def hero(title: str, subtitle: str, chips: Sequence[str] = ()) -> None:
    chip_html = "".join(f'<span class="rb-chip">{esc(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="rb-hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'{"<div class=rb-chips>" + chip_html + "</div>" if chips else ""}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, tone: Optional[str] = None) -> str:
    tone = tone or tone_for(text)
    fg, bg, bd = TONES.get(tone, TONES["info"])
    return f'<span class="rb-badge" style="color:{fg};background:{bg};border-color:{bd}">{esc(text)}</span>'


def callout(text: str, tone: str = "info", icon: str = "") -> None:
    fg, bg, bd = TONES.get(tone, TONES["info"])
    st.markdown(
        f'<div class="rb-call" style="color:{fg};background:{bg};border-color:{bd}">{icon + " " if icon else ""}{text}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: Any, sub: str = "", tone: str = "") -> str:
    return (f'<div class="rb-kpi {tone}"><div class="l">{esc(label)}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def kpi_row(items: Sequence[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def section(title: str, desc: str = "", anchor: str = "", num: str = "") -> None:
    a = f'<a id="{anchor}"></a>' if anchor else ""
    n = f'<span class="n">{num}</span>' if num else ""
    st.markdown(
        f'{a}<div class="rb-sec"><div><span class="t">{n}{title}</span></div><div class="d">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def rows_card(title: str, rows: Sequence[tuple], note: str = "") -> None:
    body = "".join(f'<div class="rb-row"><span class="k">{esc(k)}</span><span class="v">{v}</span></div>' for k, v in rows)
    st.markdown(f'<div class="rb-card"><h4>{title}</h4>{body}'
                f'{"<div class=rb-sub style=margin-top:8px>" + note + "</div>" if note else ""}</div>',
                unsafe_allow_html=True)


def style_fig(fig, height: int = 280):
    """Apply the light blue chart theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0B1F3A", size=12),
        margin=dict(l=10, r=14, t=36, b=10),
        colorway=CHART_BLUES,
        xaxis=dict(gridcolor="#E6EEF9", zerolinecolor="#C9D8EE"),
        yaxis=dict(gridcolor="#E6EEF9", zerolinecolor="#C9D8EE"),
        legend=dict(orientation="h", y=-0.28),
    )
    return fig


def engine_label() -> str:
    """Which language engine is answering right now (live from the backend /health)."""
    try:
        h = requests.get(f"{API_BASE}/api/health", timeout=3).json() or {}
        llm = h.get("llm") or {}
        if h.get("llm_mode") == "gemini":
            if llm.get("last_model"):
                return f"Gemini · {llm['last_model']}"
            if llm.get("fallback_used") and not llm.get("gemini_ok"):
                return "Gemini unreachable · built-in engine"
            return "Gemini connected"
        if h.get("llm_mode") == "azure_openai":
            return "Azure OpenAI"
    except Exception:
        pass
    return "Built-in smart engine"


def footer() -> None:
    st.markdown(
        f'<div class="rb-foot">🚆 <b>{BRAND}</b> · AI-powered railway travel with autonomous AI governance<br>'
        'FastAPI · Streamlit · XGBoost · LightGBM · SHAP · SciPy</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reference data (train & station pickers) and small chart helpers
# ──────────────────────────────────────────────────────────────────────────────

_CURATED_TRAINS = ["12951", "12002", "12301", "22691", "12622", "20901", "12137", "12260"]
_CURATED_STATIONS = ["NDLS", "CSTM", "MAS", "HWH", "SBC", "BPL", "ADI", "LKO", "KOAA", "PUNE", "JP"]


@st.cache_data(show_spinner=False)
def load_reference() -> Dict[str, Any]:
    """Train and station pickers built from the platform's own dataset (falls back to a curated list)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    trains: Dict[str, str] = {}
    stations: Dict[str, str] = {}
    try:
        import pandas as pd
        t = pd.read_csv(root / "data" / "raw" / "trains.csv", dtype={"train_number": str})
        s = pd.read_csv(root / "data" / "raw" / "stations.csv")
        names = dict(zip(s["station_code"], s["station_name"]))
        t["label"] = t.apply(lambda r: f'{r.train_number} · {r.train_name}  ({r.source} → {r.destination})', axis=1)
        order = {n: i for i, n in enumerate(_CURATED_TRAINS)}
        t["o"] = t["train_number"].map(order).fillna(999)
        for r in t.sort_values(["o", "train_name"]).itertuples():
            trains[r.label] = r.train_number
        codes = sorted(set(t["source"]) | set(t["destination"]))
        pri = {c: i for i, c in enumerate(_CURATED_STATIONS)}
        for c in sorted(codes, key=lambda c: (pri.get(c, 999), c)):
            nm = names.get(c, c)
            stations[f"{nm.title() if nm.isupper() else nm} ({c})"] = c
    except Exception:
        trains = {
            "12951 · Mumbai Rajdhani Express  (NDLS → CSTM)": "12951",
            "12002 · Bhopal Shatabdi Express  (NDLS → BPL)": "12002",
            "12301 · Howrah Rajdhani Express  (NDLS → HWH)": "12301",
        }
        stations = {"New Delhi (NDLS)": "NDLS", "Mumbai CSMT (CSTM)": "CSTM", "Chennai Central (MAS)": "MAS",
                    "Howrah Jn (HWH)": "HWH", "Bengaluru (SBC)": "SBC", "Pune Jn (PUNE)": "PUNE"}
    return {"trains": trains, "stations": stations}


def label_for(mapping: Dict[str, str], code: str, default_index: int = 0) -> str:
    for k, v in mapping.items():
        if v == code:
            return k
    return list(mapping.keys())[default_index]


def gauge(value: float, title: str, suffix: str = "%", lo: float = 0, hi: float = 100,
          steps: Optional[List[tuple]] = None, color: Optional[str] = None, height: int = 230):
    import plotly.graph_objects as go
    steps = steps or [(lo, (lo + hi) * 0.4, "#E0ECFF"), ((lo + hi) * 0.4, (lo + hi) * 0.75, "#BFD7FF"), ((lo + hi) * 0.75, hi, "#93BBFF")]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title, "font": {"size": 14, "color": "#5B6B82"}},
        number={"suffix": suffix, "font": {"size": 34, "color": "#0A2463"}},
        gauge={"axis": {"range": [lo, hi], "tickcolor": "#94A3B8"},
               "bar": {"color": color or "#1D4ED8", "thickness": 0.3},
               "bgcolor": "#fff", "borderwidth": 0,
               "steps": [{"range": [a, b], "color": c} for a, b, c in steps]},
    ))
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=46, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans, sans-serif"))
    return fig
