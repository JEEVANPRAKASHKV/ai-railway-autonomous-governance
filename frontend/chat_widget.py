"""
frontend/chat_widget.py
───────────────────────
Floating Support Assistant (Module 6): a round chat button fixed to the bottom-right corner of every
page. Clicking it opens a compact chat panel with quick questions, grounded answers, sources and a
star rating that feeds the governance satisfaction monitor. English only.
"""

from __future__ import annotations

import uuid

import streamlit as st

from frontend.ui import api_post, badge, esc, send_outcome

QUICK_QUESTIONS = [
    ("🔎 PNR status", "What is the status of PNR 0000001 and my refund?"),
    ("💸 Waitlist refund", "What happens if my waitlist ticket is not confirmed at chart preparation? Will I get a full refund?"),
    ("⚡ Tatkal rules", "What are the Tatkal booking timings and is there a refund if I cancel a Tatkal ticket?"),
    ("🚆 Delhi → Mumbai trains", "Which trains from Delhi to Mumbai?"),
]

WIDGET_CSS = """
<style>
/* Floating launcher, bottom-right on every page */
.st-key-rb_chat_fab { position: fixed !important; right: 28px; bottom: 26px; z-index: 999990; width: auto !important; }
.st-key-rb_chat_fab button {
  width: 62px !important; height: 62px !important; min-height: 62px; padding: 0 !important; border-radius: 50% !important;
  background: linear-gradient(135deg, #2563EB 0%, #0B3B8C 100%) !important; color: #fff !important; border: 3px solid #fff !important;
  box-shadow: 0 10px 28px -6px rgba(29, 78, 216, .65), 0 2px 6px rgba(11, 31, 58, .25) !important;
  display: flex; align-items: center; justify-content: center; transition: transform .15s ease, box-shadow .15s ease;
}
.st-key-rb_chat_fab button:hover { transform: translateY(-2px) scale(1.04); box-shadow: 0 14px 32px -6px rgba(29, 78, 216, .8) !important; }
.st-key-rb_chat_fab button p { font-size: 28px !important; line-height: 1 !important; margin: 0 !important; }
.st-key-rb_chat_fab button [data-testid="stIconMaterial"], .st-key-rb_chat_fab button svg { display: none !important; }
.st-key-rb_chat_fab::before {
  content: "Need help? Ask me"; position: absolute; right: 74px; bottom: 17px; white-space: nowrap;
  background: #0A2463; color: #fff; font-size: .76rem; font-weight: 700; padding: 6px 11px; border-radius: 999px;
  box-shadow: 0 6px 16px -6px rgba(10, 36, 99, .6); pointer-events: none; opacity: .95;
}
.st-key-rb_chat_fab:hover::before { opacity: 1; }
/* Chat panel inside the popover */
.st-key-rb_chat_panel { width: min(400px, calc(100vw - 48px)); }
.rb-chat-head { display: flex; gap: 10px; align-items: center; padding: 2px 2px 10px; border-bottom: 1px solid #D9E4F5; margin-bottom: 8px; }
.rb-chat-head .av { width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center; font-size: 19px;
  background: linear-gradient(135deg, #2563EB, #0B3B8C); color: #fff; flex: none; }
.rb-chat-head .t { font-weight: 800; color: #0A2463; line-height: 1.15; }
.rb-chat-head .s { font-size: .74rem; color: #5B6B82; }
.rb-chat-head .on { width: 8px; height: 8px; border-radius: 50%; background: #10B981; display: inline-block; margin-right: 4px; }
.st-key-rb_chat_panel [data-testid="stChatMessage"] { padding: 6px 8px; }
.st-key-rb_chat_panel [data-testid="stChatMessageContent"] p { font-size: .9rem; }
.st-key-rb_chat_panel .rb-badge { font-size: .66rem; padding: 2px 8px; }
.st-key-rb_chat_panel .stButton > button { font-size: .78rem; padding: 4px 8px; min-height: 0; }
@media (max-width: 640px) {
  .st-key-rb_chat_fab { right: 16px; bottom: 16px; }
  .st-key-rb_chat_fab::before { display: none; }
}
</style>
"""


def _queue(q: str) -> None:
    st.session_state["chat_pending"] = q


def _clear() -> None:
    st.session_state["chat"] = []
    st.session_state["chat_sid"] = f"web-{uuid.uuid4().hex[:10]}"


def ask(q: str) -> None:
    """Send one question to the governed support service and keep the turn in this browser session."""
    q = (q or "").strip()
    if not q:
        return
    hist = st.session_state.setdefault("chat", [])
    hist.append({"role": "user", "text": q})
    res = api_post("/support/query", {"query": q, "user_id": "railway_buddy_user",
                                      "session_id": st.session_state.setdefault("chat_sid", f"web-{uuid.uuid4().hex[:10]}")})
    hist.append({"role": "assistant", "res": res})


def _answer(idx: int, res: dict) -> None:
    if not res or "response" not in res:
        detail = (res or {}).get("error", "") if isinstance(res, dict) else ""
        st.warning(f"Sorry, I couldn't answer that right now. {esc(str(detail)[:160])}")
        return
    st.markdown(res["response"].replace("\n", "  \n"))
    faith = float(res.get("faithfulness_score") or 0)
    flag = res.get("hallucination_flag")
    chips = [badge(res.get("answered_by") or "policy engine", "mute")]
    if res.get("answered_by") != "language guard":
        chips.append(badge(f"Grounded {faith:.0%}" if not flag else f"Weak grounding {faith:.0%}", "warn" if flag else "ok"))
    if res.get("escalation_recommended"):
        chips.append(badge("Escalated to a human", "warn"))
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    chunks = res.get("retrieved_contexts") or []
    if chunks:
        with st.expander(f"📚 Sources ({len(chunks)})"):
            for ch in chunks:
                st.markdown(f'**{esc(ch.get("source_file"))}** · match {ch.get("similarity_score", 0):.2f}')
                st.caption(ch.get("text", "")[:320])
    audit = (res.get("governance_telemetry") or {}).get("audit_id")
    if audit and res.get("answered_by") != "language guard":
        rated = st.session_state.setdefault("rated", {})
        if audit in rated:
            st.caption(f"Thanks for rating this answer {rated[audit]}/5.")
        else:
            stars = st.feedback("stars", key=f"fb_{idx}_{audit}")
            if stars is not None:
                r = send_outcome(audit, int(stars) + 1)
                if r and r.get("accepted"):
                    rated[audit] = int(stars) + 1
                    st.caption("Thanks — your rating feeds the satisfaction monitor.")


def render_chat_widget() -> None:
    """Draw the floating launcher and its chat panel. Call once per page run, after the page content."""
    st.markdown(WIDGET_CSS, unsafe_allow_html=True)
    with st.container(key="rb_chat_fab"):
        with st.popover("💬"):
            with st.container(key="rb_chat_panel"):
                st.markdown(
                    '<div class="rb-chat-head"><div class="av">🚆</div><div><div class="t">Railway Buddy Support</div>'
                    '<div class="s"><span class="on"></span>Online · PNR, refunds, Tatkal &amp; RAC rules · English</div></div></div>',
                    unsafe_allow_html=True)

                log = st.container(height=380, border=False)
                pending = st.session_state.pop("chat_pending", None)
                typed = st.chat_input("Type your question…", key="rb_chat_input")
                q = typed or pending
                if q:
                    with log:
                        with st.spinner("Checking booking records and IRCTC policy…"):
                            ask(q)

                hist = st.session_state.get("chat", [])
                with log:
                    if not hist:
                        st.markdown(
                            '<div style="text-align:center;padding:14px 6px 8px"><div style="font-size:1.8rem">👋</div>'
                            '<b style="color:#0A2463">Hi! How can I help with your journey?</b>'
                            '<div class="rb-sub">Ask in English, or pick a question below. Answers come from booking '
                            'records and official IRCTC policy, and are checked for accuracy.</div></div>',
                            unsafe_allow_html=True)
                        for i, (lbl, qq) in enumerate(QUICK_QUESTIONS):
                            st.button(lbl, key=f"rb_qq{i}", width="stretch", on_click=_queue, args=(qq,))
                    for idx, m in enumerate(hist):
                        if m["role"] == "user":
                            with st.chat_message("user", avatar="🧑"):
                                st.markdown(m["text"])
                        else:
                            with st.chat_message("assistant", avatar="🚆"):
                                _answer(idx, m["res"])
                if hist:
                    c1, c2 = st.columns([1, 1])
                    c1.button("🧹 New conversation", key="rb_chat_clear", width="stretch", on_click=_clear)
                    c2.caption("Urgent? Call RailMadad **139**")
