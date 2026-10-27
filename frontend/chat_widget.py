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
    ("PNR status", "What is the status of PNR 0000001 and my refund?"),
    ("Waitlist refund", "What happens if my waitlist ticket is not confirmed at chart preparation? Will I get a full refund?"),
    ("Tatkal rules", "What are the Tatkal booking timings and is there a refund if I cancel a Tatkal ticket?"),
    ("Delhi → Mumbai trains", "Which trains from Delhi to Mumbai?"),
]

WIDGET_CSS = """
<style>
/* Floating launcher, bottom-right on every page: a black round button with a signal-yellow ring */
.st-key-rb_chat_fab { position: fixed !important; right: 28px; bottom: 26px; z-index: 999990; width: auto !important; }
.st-key-rb_chat_fab button {
  width: 60px !important; height: 60px !important; min-height: 60px; padding: 0 !important; border-radius: 50% !important;
  background: #000 !important; color: #F5C400 !important; border: 3px solid #F5C400 !important; outline: 1px solid #000;
  display: flex; align-items: center; justify-content: center;
}
.st-key-rb_chat_fab button:hover { background: #F5C400 !important; border-color: #000 !important; }
.st-key-rb_chat_fab button p { font-size: 26px !important; line-height: 1 !important; margin: 0 !important; color: inherit !important; }
.st-key-rb_chat_fab button [data-testid="stIconMaterial"], .st-key-rb_chat_fab button svg { display: none !important; }
.st-key-rb_chat_fab::before {
  content: "Help — ask a question"; position: absolute; right: 72px; bottom: 17px; white-space: nowrap;
  background: #000; color: #F5C400; font-size: .78rem; font-weight: 700; padding: 5px 10px; border: 1px solid #000; pointer-events: none;
}
/* Chat panel inside the popover */
.st-key-rb_chat_panel { width: min(400px, calc(100vw - 48px)); }
.rb-chat-head { display: flex; gap: 10px; align-items: stretch; margin: -2px 0 8px; border: 2px solid #000; }
.rb-chat-head .av { width: 40px; display: grid; place-items: center; font-size: 18px; background: #F5C400; border-right: 2px solid #000; flex: none; }
.rb-chat-head .t { font-weight: 800; color: #000; line-height: 1.15; padding-top: 5px; }
.rb-chat-head .s { font-size: .74rem; color: #555; padding-bottom: 5px; }
.rb-chat-head .on { width: 8px; height: 8px; background: #13703A; display: inline-block; margin-right: 5px; }
.st-key-rb_chat_panel [data-testid="stChatMessage"] { padding: 6px 4px; }
.st-key-rb_chat_panel [data-testid="stChatMessageContent"] p { font-size: .9rem; }
.st-key-rb_chat_panel h1, .st-key-rb_chat_panel h2, .st-key-rb_chat_panel h3, .st-key-rb_chat_panel h4 { font-size: .95rem !important; font-weight: 800; margin: 6px 0 2px !important; padding: 0 !important; line-height: 1.3; }
.st-key-rb_chat_panel .rb-badge { font-size: .68rem; padding: 1px 6px; }
.st-key-rb_chat_panel .stButton > button { font-size: .8rem; padding: 4px 8px; min-height: 0; justify-content: flex-start; }
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
        with st.expander(f"Sources ({len(chunks)})"):
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
                    '<div class="rb-chat-head"><div class="av" style="font-family:var(--mono);font-weight:700;font-size:.9rem">M6</div><div><div class="t">Railway Buddy Support</div>'
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
                            '<div style="padding:6px 2px 8px">'
                            '<b>How can I help with your journey?</b>'
                            '<div class="rb-sub">Ask in English, or pick a question below. Answers come from booking '
                            'records and official IRCTC policy, and are checked for accuracy.</div></div>',
                            unsafe_allow_html=True)
                        for i, (lbl, qq) in enumerate(QUICK_QUESTIONS):
                            st.button(lbl, key=f"rb_qq{i}", width="stretch", on_click=_queue, args=(qq,))
                    for idx, m in enumerate(hist):
                        if m["role"] == "user":
                            with st.chat_message("user", avatar=":material/person:"):
                                st.markdown(m["text"])
                        else:
                            with st.chat_message("assistant", avatar=":material/train:"):
                                _answer(idx, m["res"])
                if hist:
                    c1, c2 = st.columns([1, 1])
                    c1.button("New conversation", key="rb_chat_clear", width="stretch", on_click=_clear)
                    c2.caption("Urgent? Call RailMadad **139**")
