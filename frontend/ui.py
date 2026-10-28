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
import re
from typing import Any, Dict, List, Optional, Sequence

import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Backend access
# ──────────────────────────────────────────────────────────────────────────────

def api_base() -> str:
    """Read on every call: the launcher moves the API to a fresh process (and port) after a code update."""
    return os.getenv("BACKEND_URL", os.getenv("API_BASE", "http://127.0.0.1:8000")).rstrip("/")


API_BASE = api_base()   # kept for display / backwards compatibility


def api_get(path: str, params: Optional[dict] = None) -> Optional[Any]:
    try:
        r = requests.get(f"{api_base()}/api/v1{path}", params=params, timeout=8)
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
        r = requests.post(f"{api_base()}/api/v1{path}", json=payload or {}, timeout=timeout, headers=headers)
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
        return requests.get(f"{api_base()}/api/health", timeout=3).status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Design tokens — industrial transit dashboard
#   One bold element: the page header is an Indian Railways station name board
#   (signal-yellow plate, heavy black type, black code box like a platform sign).
#   Everything else is quiet: white panels, 1px #ddd rules, no radius, no shadow,
#   monospace for every number, time, train ID, amount and audit ID.
# ──────────────────────────────────────────────────────────────────────────────

BRAND = "Railway Buddy"

INK = "#000000"        # text, primary buttons, board type
BOARD = "#F5C400"      # station name-board yellow — headers and the active nav item only
CANVAS = "#F0F0EE"     # page background
PANEL = "#FFFFFF"      # panels, tables
RULE = "#DDDDDD"       # every separation line
MUTED = "#555555"      # secondary text
SIG_GREEN, SIG_AMBER, SIG_RED = "#13703A", "#A35F00", "#B3141B"

SANS = "'Archivo', 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace"

# Kept for compatibility with pages that call inject_theme(screen); every screen shares one palette now.
SCREEN_CODES: Dict[str, str] = {
    "home": "RB", "ticket": "M1", "recommend": "M2", "delay": "M3", "planner": "M4",
    "fare": "M5", "support": "M6", "fraud": "M7", "tower": "GOV",
}

TONES = {  # status tones -> (fg, bg, border)
    "ok":    (SIG_GREEN, "#EAF3EC", "#9CC7A8"),
    "warn":  (SIG_AMBER, "#FBF1E1", "#E2B877"),
    "bad":   (SIG_RED, "#FBEAEA", "#E3A0A3"),
    "info":  ("#1F1F1F", "#F4F4F2", RULE),
    "mute":  (MUTED, "#F4F4F2", RULE),
}
TONE_BAR = {"ok": SIG_GREEN, "warn": SIG_AMBER, "bad": SIG_RED}

# Chart series: black, dark gold, grey, signal green, signal red, light grey
CHART_COLORS = ["#111111", "#B08900", "#7A7A7A", SIG_GREEN, SIG_RED, "#B5B5B5"]
CHART_BLUES = CHART_COLORS          # old name, still imported in places
POS, NEG = "#111111", "#B08900"     # explanation bars: raises / lowers


def esc(x: Any) -> str:
    return html.escape(str(x))


def mono(x: Any) -> str:
    """Wrap a number / time / ID so it renders in the monospace face."""
    return f'<span class="rb-mono">{esc(x)}</span>'


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

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
:root {
  --ink: #000000; --board: #F5C400; --canvas: #F0F0EE; --panel: #FFFFFF; --rule: #DDDDDD; --muted: #555555;
  --green: #13703A; --amber: #A35F00; --red: #B3141B;
  --sans: 'Archivo', 'Helvetica Neue', Arial, sans-serif; --mono: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
  /* legacy names used by inline styles in the pages */
  --accent: #000000; --deep: #000000; --soft: #DDDDDD;
}
html, body, .stApp, [class*="css"], .stMarkdown, button, input, textarea, select { font-family: var(--sans); }
.stApp { background: var(--canvas); color: var(--ink); }
.stApp *, .stApp *::before, .stApp *::after { box-shadow: none !important; text-shadow: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1320px; }
code, .rb-mono, [data-testid="stMetricValue"], .stDataFrame { font-family: var(--mono) !important; font-variant-numeric: tabular-nums; }
.rb-mono { letter-spacing: -0.01em; }
.stApp code { color: var(--ink) !important; background: #EDEDEA !important; border-radius: 0 !important; padding: 0 4px; }
:focus-visible { outline: 2px solid var(--ink) !important; outline-offset: 1px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--rule); }
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] { border-radius: 0; border-left: 3px solid transparent; padding-left: 10px; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover { background: #F4F4F2; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--board); border-left-color: var(--ink); }
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span { color: var(--ink); font-weight: 700; }
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] { color: var(--muted); font-weight: 700; font-size: .8rem; border-top: 1px solid var(--rule); padding-top: 10px; }
.rb-status { border: 1px solid var(--rule); background: var(--panel); font-size: .8rem; margin-top: 8px; }
.rb-status > div { padding: 6px 10px; border-bottom: 1px solid var(--rule); display: flex; justify-content: space-between; gap: 8px; }
.rb-status > div:last-child { border-bottom: 0; }
.rb-status .k { color: var(--muted); }

/* Station name-board header (the one bold element) */
.rb-board { border: 2px solid var(--ink); background: var(--panel); margin-bottom: 14px; }
.rb-board .plate { background: var(--board); display: flex; align-items: stretch; border-bottom: 2px solid var(--ink); }
.rb-board .plate h1 { margin: 0; padding: 12px 18px 10px; font-family: var(--sans); font-weight: 900; font-size: 1.85rem;
  letter-spacing: -0.015em; line-height: 1.1; color: var(--ink); flex: 1; }
.rb-board .plate .code { background: var(--ink); color: var(--board); font-family: var(--mono); font-weight: 700; font-size: 1.35rem;
  display: flex; align-items: center; padding: 0 18px; min-width: 64px; justify-content: center; }
.rb-board .desc { padding: 10px 18px; font-size: .95rem; color: #1F1F1F; max-width: 80ch; line-height: 1.45; }
.rb-board .spec { display: flex; flex-wrap: wrap; border-top: 1px solid var(--rule); }
.rb-board .spec span { padding: 6px 12px; font-size: .78rem; color: var(--muted); border-right: 1px solid var(--rule); }

/* Panels */
.rb-card { background: var(--panel); border: 1px solid var(--rule); padding: 14px 16px; margin-bottom: 10px; }
.rb-card h4 { margin: 0 0 6px; font-size: 1rem; font-weight: 800; color: var(--ink); }
.rb-sub { color: var(--muted); font-size: .85rem; }

/* Stat cells: a single strip with shared borders */
.rb-grid { display: grid; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); border-top: 1px solid var(--rule); border-left: 1px solid var(--rule); background: var(--panel); margin-bottom: 10px; }
.rb-grid > .rb-kpi { border: 0; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 0; }
.rb-kpi { background: var(--panel); border: 1px solid var(--rule); padding: 10px 14px 11px; min-height: 92px; box-sizing: border-box; border-left: 4px solid transparent !important; }
.rb-grid > .rb-kpi { border-left: 4px solid transparent !important; }
.rb-kpi.ok { border-left-color: var(--green) !important; } .rb-kpi.warn { border-left-color: var(--amber) !important; } .rb-kpi.bad { border-left-color: var(--red) !important; }
.rb-kpi .l { font-size: .8rem; color: var(--muted); font-weight: 600; }
.rb-kpi .v { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.55rem; font-weight: 600; color: var(--ink); line-height: 1.3; letter-spacing: -0.02em; }
.rb-kpi .v.txt { font-family: var(--sans); font-weight: 800; letter-spacing: -0.01em; font-size: clamp(1.05rem, 1.5vw, 1.55rem); word-break: keep-all; overflow-wrap: normal; }
.rb-kpi .s { font-size: .76rem; color: var(--muted); }
@media (max-width: 760px) { .rb-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .rb-thead { display: none; } }

/* Section headings */
.rb-sec { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 4px 16px; margin: 26px 0 10px;
  padding-bottom: 6px; border-bottom: 1px solid var(--ink); }
.rb-sec > div:first-child { flex: none; }
.rb-sec .t { font-size: 1.15rem; font-weight: 800; color: var(--ink); white-space: nowrap; }
.rb-sec .d { font-size: .84rem; color: var(--muted); flex: 1 1 320px; text-align: right; }
.rb-sec .n { font-family: var(--mono); background: var(--ink); color: var(--board); font-size: .8rem; font-weight: 700; padding: 1px 7px; margin-right: 10px; }

/* Badges & callouts */
.rb-badge { display: inline-flex; align-items: center; gap: 5px; padding: 1px 8px; font-size: .75rem; font-weight: 600; border: 1px solid; white-space: nowrap; }
.rb-call { padding: 9px 14px; font-size: .92rem; border: 1px solid var(--rule); border-left: 4px solid; margin: 8px 0; background: var(--panel); color: #1F1F1F; }
.rb-call b { color: var(--ink); }

/* Key/value rows */
.rb-row { display: flex; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px solid var(--rule); font-size: .9rem; }
.rb-row:last-child { border-bottom: 0; }
.rb-row .k { color: #1F1F1F; } .rb-row .v { font-family: var(--mono); font-weight: 600; color: var(--ink); text-align: right; }
.rb-list { border: 1px solid var(--rule); background: var(--panel); margin-bottom: 10px; }
.rb-list .rb-row { padding: 8px 12px; align-items: center; }

/* Train result row (recommender) */
.rb-tkt { display: grid; grid-template-columns: 44px minmax(0, 2fr) auto auto auto auto minmax(90px, 1fr); align-items: stretch; }
.rb-tkt > div { padding: 9px 12px; border-right: 1px solid var(--rule); font-size: .9rem; white-space: nowrap; }
.rb-tkt > div:nth-child(2) { white-space: normal; }
.rb-tkt > div:last-child { border-right: 0; }
.rb-tkt .rk { font-family: var(--mono); font-weight: 700; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; }
.rb-tkt .rk.top { background: var(--board); }
.rb-tkt .fare { font-size: 1.3rem; font-weight: 600; text-align: right; display: flex; align-items: center; justify-content: flex-end; }
@media (max-width: 760px) { .rb-tkt { grid-template-columns: 40px 1fr 1fr; } .rb-tkt > div { border-bottom: 1px solid var(--rule); } }

/* Dense tables */
.rb-table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--rule); font-size: .86rem; margin-bottom: 10px; }
.rb-table th { background: var(--ink); color: var(--board); text-align: left; font-weight: 700; font-size: .78rem; padding: 6px 10px; }
.rb-table td { border-top: 1px solid var(--rule); padding: 6px 10px; vertical-align: top; }
.rb-table td.m, .rb-table th.m { font-family: var(--mono); white-space: nowrap; }
.rb-table .r { text-align: right; }
.rb-table tr:hover td { background: #FAFAF8; }

/* Pillar tiles */
.rb-pillar { background: var(--panel); padding: 10px 12px; min-height: 84px; box-sizing: border-box; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.rb-pillar .h { display: flex; justify-content: space-between; align-items: center; gap: 6px; font-weight: 700; font-size: .88rem; color: var(--ink); }
.rb-pillar .m { font-size: .78rem; color: var(--muted); margin-top: 6px; }

/* Readout meter (replaces gauges) */
.rb-meter { background: var(--panel); border: 1px solid var(--rule); padding: 12px 14px; }
.rb-meter .l { font-size: .85rem; color: var(--muted); font-weight: 600; }
.rb-meter .val { font-family: var(--mono); font-size: 2.6rem; font-weight: 600; line-height: 1.15; letter-spacing: -0.03em; color: var(--ink); }
.rb-meter .val small { font-size: 1.1rem; color: var(--muted); margin-left: 4px; }
.rb-meter .bar { display: grid; grid-template-columns: repeat(20, 1fr); gap: 2px; margin: 8px 0 4px; }
.rb-meter .bar i { height: 12px; background: #E6E6E3; display: block; }
.rb-meter .scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: .72rem; color: var(--muted); }

/* Data provenance strip: which inputs are real and which are simulated */
.rb-prov { display: flex; flex-wrap: wrap; border: 1px solid var(--rule); border-right: 0; margin: -4px 0 12px; background: var(--panel); }
.rb-prov span { border-right: 1px solid var(--rule); padding: 3px 9px; font-size: .74rem; color: var(--muted); }
.rb-prov b { color: var(--ink); font-weight: 600; }
.rb-prov .real b::before { content: "REAL "; font-family: var(--mono); font-size: .68rem; color: var(--green); }
.rb-prov .live b::before { content: "LIVE "; font-family: var(--mono); font-size: .68rem; color: var(--green); }
.rb-prov .sim b::before { content: "SIM "; font-family: var(--mono); font-size: .68rem; color: var(--amber); }

/* Anchor nav (Control Tower) */
.rb-anchors { position: sticky; top: 46px; z-index: 50; background: var(--canvas); display: flex; flex-wrap: wrap;
  border: 1px solid var(--rule); border-right: 0; margin: 10px 0 6px; }
.rb-anchors a { text-decoration: none; font-size: .78rem; font-weight: 600; color: var(--ink); background: var(--panel);
  border-right: 1px solid var(--rule); padding: 5px 10px; }
.rb-anchors a:hover { background: var(--board); }

/* Departure-board service table (home) */
[class*="st-key-svcrow"] { border: 1px solid var(--rule); border-top: 0; background: var(--panel); padding: 8px 10px 0; margin-top: -1rem; }
[class*="st-key-svcrow"] [data-testid="stHorizontalBlock"] { align-items: center; gap: 12px; }
[class*="st-key-svcrow"]:hover { background: #FAFAF8; }
.rb-thead { display: grid; grid-template-columns: var(--cols); gap: 12px; background: var(--ink); color: var(--board);
  font-size: .8rem; font-weight: 700; padding: 6px 10px; border: 1px solid var(--ink); margin-bottom: 14px; }

/* Streamlit widgets */
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--rule); padding: 10px 14px; }
div[data-testid="stMetricValue"] { color: var(--ink); }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a {
  border-radius: 0 !important; font-weight: 600; border: 1px solid var(--ink) !important; background: var(--panel); color: var(--ink); }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPageLink"] a:hover {
  background: var(--board) !important; color: var(--ink) !important; border-color: var(--ink) !important; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] { background: var(--ink); color: #fff; }
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primaryFormSubmit"] p { color: #fff; }
.stButton > button[kind="primary"]:hover p, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover p { color: var(--ink); }
div[data-testid="stForm"] { background: var(--panel); border: 1px solid var(--rule); border-radius: 0; padding: 14px 16px; }
div[data-testid="stExpander"] details { background: var(--panel); border: 1px solid var(--rule); border-radius: 0; }
div[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stContainer"] { border-radius: 0; }
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"], [data-testid="stNumberInputContainer"] { border-radius: 0 !important; }
input, textarea, [data-baseweb="select"] div { font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--ink); }
.stTabs [data-baseweb="tab"] { background: var(--panel); border: 1px solid var(--rule); border-bottom: 0; border-radius: 0; padding: 6px 14px; }
.stTabs [aria-selected="true"] { background: var(--board); }
[data-testid="stChatMessage"] { border-radius: 0; border-bottom: 1px solid var(--rule); background: transparent; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"], [data-testid*="stChatMessageAvatar"] { border-radius: 0; }
[data-testid="stPopoverBody"] { border-radius: 0 !important; border: 2px solid var(--ink) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--rule); }
.rb-foot { color: var(--muted); font-size: .78rem; margin-top: 36px; padding-top: 10px; border-top: 1px solid var(--rule); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


def inject_theme(screen: str = "home") -> None:
    st.session_state["_rb_screen"] = screen
    st.markdown(_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

_LEADING_SYMBOLS = re.compile(r"^[^\w(]+", re.UNICODE)
STAFF_SCREENS = {"tower"}          # governance screens always show the full technical layer


def details() -> bool:
    """Show the AI & governance layer (model names, demo scenarios, SHAP charts, governance receipts, feedback
    forms)? Off for passengers; railway staff and reviewers switch it on in the sidebar. Governance screens
    always show it."""
    return bool(st.session_state.get("rb_details", False)) or st.session_state.get("_rb_screen") in STAFF_SCREENS


def why_card(title: str, reasons: Sequence[tuple]) -> None:
    """Plain-language reasons: [(what, effect, good?)] → a short list a passenger can read in five seconds."""
    if not reasons:
        return
    rows = "".join(
        f'<div class="rb-row"><span class="k">{esc(what)}</span>'
        f'<span class="v" style="color:{"var(--green)" if good else "var(--amber)"}">{esc(effect)}</span></div>'
        for what, effect, good in reasons)
    st.markdown(f'<div class="rb-card"><h4>{esc(title)}</h4>{rows}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: Sequence[str] = (), code: Optional[str] = None) -> None:
    """Station name-board header: yellow plate with the page name, black code box, a plain description and a spec strip."""
    clean = _LEADING_SYMBOLS.sub("", title).strip() or title
    code = code or SCREEN_CODES.get(st.session_state.get("_rb_screen", ""), "")
    if not details():          # passengers don't need the model / technique strip or module codes
        chips, code = (), ""
    spec = "".join(f"<span>{esc(c)}</span>" for c in chips)
    st.markdown(
        f'<div class="rb-board"><div class="plate"><h1>{esc(clean)}</h1>'
        f'{f"<div class=code>{esc(code)}</div>" if code else ""}</div>'
        f'<div class="desc">{subtitle}</div>'
        f'{"<div class=spec>" + spec + "</div>" if chips else ""}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, tone: Optional[str] = None) -> str:
    tone = tone or tone_for(text)
    fg, bg, bd = TONES.get(tone, TONES["info"])
    return f'<span class="rb-badge" style="color:{fg};background:{bg};border-color:{bd}">{esc(text)}</span>'


def callout(text: str, tone: str = "info", icon: str = "") -> None:
    fg, _, _ = TONES.get(tone, TONES["info"])
    bar = TONE_BAR.get(tone, INK)
    # icons are accepted for compatibility but not drawn: the coloured edge carries the status
    st.markdown(f'<div class="rb-call" style="border-left-color:{bar}">{text}</div>',
                unsafe_allow_html=True)


def kpi(label: str, value: Any, sub: str = "", tone: str = "") -> str:
    # numbers, times and IDs in monospace; words ("Healthy", "Likely WL") in the sans face
    plain = re.sub(r"<[^>]+>", "", str(value))
    kind = "" if re.search(r"\d", plain) or plain.strip() in ("—", "-") else " txt"
    return (f'<div class="rb-kpi {tone}"><div class="l">{esc(label)}</div>'
            f'<div class="v{kind}">{value}</div><div class="s">{sub}</div></div>')


def kpi_row(items: Sequence[tuple], cols: Optional[int] = None) -> None:
    """One strip (or grid, with `cols`) of stat cells sharing their borders — data-dense, no gaps."""
    cells = "".join(kpi(*it) for it in items)
    st.markdown(f'<div class="rb-grid" style="--n:{cols or len(items)}">{cells}</div>', unsafe_allow_html=True)


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


PROVENANCE = {
    "timetable": ("real", "Timetable", "Indian Railways 2016 (datameet)"),
    "stations": ("real", "Stations", "2,719 real stations with coordinates"),
    "fares": ("real", "Fares", "Railway Board TAG 2025 tariff"),
    "weather": ("live", "Weather", "Open-Meteo forecast"),
    "outcomes": ("sim", "Outcomes", "booking, delay & demand history simulated"),
    "fraud": ("sim", "Sessions", "simulated booking traffic"),
}


_PLAIN_SOURCES = {"timetable": "train times from the 2016 Indian Railways timetable",
                  "fares": "fares from the official 2025 tariff", "weather": "the live weather forecast",
                  "outcomes": "estimates learned from past patterns"}


def provenance(*keys: str) -> None:
    """One strip under the header saying which inputs on this screen are real, live or simulated
    (passengers get a one-line note instead)."""
    if not details():
        parts = [_PLAIN_SOURCES[k] for k in keys if k in _PLAIN_SOURCES]
        if parts:
            joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
            st.caption("Uses " + joined + ". Always confirm on IRCTC before you travel.")
        return
    cells = "".join(f'<span class="{PROVENANCE[k][0]}"><b>{esc(PROVENANCE[k][1])}</b> {esc(PROVENANCE[k][2])}</span>'
                    for k in keys if k in PROVENANCE)
    st.markdown(f'<div class="rb-prov">{cells}</div>', unsafe_allow_html=True)


def meter(value: float, label: str, suffix: str = "%", lo: float = 0, hi: float = 100, tone: str = "",
          decimals: int = 1) -> str:
    """Readout instead of a gauge: big monospace value and a 20-segment bar in the signal colour."""
    v = float(value or 0)
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    lit = int(round(frac * 20))
    color = TONE_BAR.get(tone, INK)
    segs = "".join(f'<i style="background:{color}"></i>' if i < lit else "<i></i>" for i in range(20))
    num = f"{v:.{decimals}f}" if decimals else f"{v:.0f}"
    return (f'<div class="rb-meter"><div class="l">{esc(label)}</div>'
            f'<div class="val">{num}<small>{esc(suffix.strip())}</small></div>'
            f'<div class="bar" role="img" aria-label="{esc(label)}: {num}{esc(suffix)}">{segs}</div>'
            f'<div class="scale"><span>{lo:g}</span><span>{hi:g}</span></div></div>')


def style_fig(fig, height: int = 280):
    """Plotly theme: white panel, light grid, sans titles, monospace tick labels."""
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family=SANS, color=INK, size=12),
        title_font=dict(family=SANS, size=14, color=INK),
        margin=dict(l=10, r=14, t=40, b=10),
        colorway=CHART_COLORS,
        xaxis=dict(gridcolor="#EEEEEE", zerolinecolor="#999999", linecolor=RULE, tickfont=dict(family=MONO, size=11)),
        yaxis=dict(gridcolor="#EEEEEE", zerolinecolor="#999999", linecolor=RULE, tickfont=dict(family=SANS, size=12)),
        legend=dict(orientation="h", y=-0.28),
        hoverlabel=dict(font_family=MONO, bgcolor=PANEL, bordercolor=INK),
    )
    return fig


LLM_LABELS = {"sarvam": "Sarvam AI", "gemini": "Gemini", "azure_openai": "Azure OpenAI", "fallback": "Built-in smart engine"}


def engine_label() -> str:
    """Which language engine is answering right now (live from the backend /health)."""
    try:
        h = requests.get(f"{api_base()}/api/health", timeout=3).json() or {}
        llm = h.get("llm") or {}
        mode = h.get("llm_mode") or llm.get("mode") or "fallback"
        if mode == "fallback":
            return LLM_LABELS["fallback"]
        if llm.get("budget_reached"):
            return f"{LLM_LABELS.get(mode, mode)} · daily budget reached, built-in engine"
        last = llm.get("last_provider")
        if last:
            return f"{LLM_LABELS.get(last, last)} · {llm.get('last_model') or ''}".rstrip(" ·")
        if llm.get("fallback_used") and not llm.get("llm_ok"):
            return f"{LLM_LABELS.get(mode, mode)} unreachable · built-in engine"
        model = ((llm.get("per_provider") or {}).get(mode) or {}).get("model") or ""
        return f"{LLM_LABELS.get(mode, mode)} · {model}".rstrip(" ·") + " (ready)"
    except Exception:
        pass
    return LLM_LABELS["fallback"]


def footer() -> None:
    st.markdown(
        f'<div class="rb-foot"><span><b>{BRAND}</b> — '
        + ("AI-assisted railway travel with an autonomous governance layer</span>"
           "<span>FastAPI · Streamlit · XGBoost · LightGBM · SHAP · SciPy</span></div>" if details() else
           "an independent demo, not IRCTC. Book real tickets on irctc.co.in.</span></div>"),
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reference data (train & station pickers) and small chart helpers
# ──────────────────────────────────────────────────────────────────────────────

_CURATED_TRAINS = ["12951", "12952", "12002", "12301", "12138", "12627", "12622", "12260"]
_CURATED_STATIONS = ["NDLS", "CSTM", "BCT", "MAS", "HWH", "SBC", "BPL", "ADI", "LKO", "PUNE", "JP", "KOTA"]


@st.cache_data(show_spinner=False)
def load_reference() -> Dict[str, Any]:
    """Train and station pickers from the real network, plus each train's halts and classes."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    trains: Dict[str, str] = {}
    stations: Dict[str, str] = {}
    stops: Dict[str, list] = {}
    classes: Dict[str, list] = {}
    try:
        import pandas as pd
        t = pd.read_csv(root / "data" / "raw" / "trains.csv", dtype={"train_number": str})
        s = pd.read_csv(root / "data" / "raw" / "stations.csv")
        t["label"] = t.apply(lambda r: f'{r.train_number} · {r.train_name}  ({r.source} → {r.destination})', axis=1)
        order = {n: i for i, n in enumerate(_CURATED_TRAINS)}
        t["o"] = t["train_number"].map(order).fillna(999)
        for r in t.sort_values(["o", "train_number"]).itertuples():
            trains[r.label] = r.train_number
            classes[r.train_number] = str(r.classes).split("|")
        pri = {c: i for i, c in enumerate(_CURATED_STATIONS)}
        s["o"] = s["station_code"].map(pri).fillna(999)
        for r in s.sort_values(["o", "trains_halting"], ascending=[True, False]).itertuples():
            stations[f"{r.station_name} ({r.station_code})"] = r.station_code
        h = pd.read_csv(root / "data" / "real" / "train_stops.csv.gz", dtype={"train_number": str},
                        usecols=["train_number", "seq", "station_code", "arrival", "departure", "km_from_origin"])
        for num, g in h.sort_values(["train_number", "seq"]).groupby("train_number"):
            stops[num] = [(r.station_code, r.arrival if isinstance(r.arrival, str) else "",
                           r.departure if isinstance(r.departure, str) else "", float(r.km_from_origin))
                          for r in g.itertuples()]
    except Exception:
        trains = {
            "12951 · Mumbai Central-New Delhi Rajdhani Express  (BCT → NDLS)": "12951",
            "12002 · New Delhi-Bhopal Shatabdi Express  (NDLS → BPL)": "12002",
            "12301 · Howrah-New Delhi Rajdhani Express  (HWH → NDLS)": "12301",
        }
        stations = {"New Delhi (NDLS)": "NDLS", "Mumbai Cst (CSTM)": "CSTM", "Mumbai Central (BCT)": "BCT",
                    "Bhopal Jn (BPL)": "BPL", "Howrah Jn (HWH)": "HWH"}
    return {"trains": trains, "stations": stations, "stops": stops, "classes": classes}


def halt_labels(ref: Dict[str, Any], train_number: str) -> list:
    """Station-picker labels for one train's halts, in running order."""
    by_code = {v: k for k, v in ref["stations"].items()}
    return [by_code.get(c, c) for c, *_ in ref["stops"].get(str(train_number), [])] or list(ref["stations"].keys())


def label_for(mapping: Dict[str, str], code: str, default_index: int = 0) -> str:
    for k, v in mapping.items():
        if v == code:
            return k
    return list(mapping.keys())[default_index]


def gauge(value: float, title: str, suffix: str = "%", lo: float = 0, hi: float = 100, *_, **__) -> str:
    """Deprecated: gauges were replaced by monospace readouts. Returns meter() HTML."""
    return meter(value, title, suffix, lo, hi)
