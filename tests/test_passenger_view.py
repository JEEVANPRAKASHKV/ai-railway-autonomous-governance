"""
tests/test_passenger_view.py
============================
Two audiences, two views of the same travel pages:
- passengers (default) see answers in plain language — no demo scenarios, model names, SHAP charts,
  governance receipts or outcome forms;
- railway staff / reviewers switch on "Staff & reviewer view" (or open ?view=staff) and get the full
  AI and governance layer back.
"""

import pathlib

import pytest

from tests.test_ui_round2 import _page_script, live_api  # noqa: F401  (fixture re-used)

ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _run(page: str, staff: bool):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page_script, args=(page, ROOT), default_timeout=90)
    at.session_state["rb_details"] = staff
    at.run()
    return at


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _submit(at, label: str):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_passenger_ticket_page_is_plain(live_api):  # noqa: F811
    at = _submit(_run("ticket", staff=False), "Predict confirmation")
    text = _text(at)
    assert "Demo scenarios" not in text and "XGBoost" not in text
    assert "Governance receipt" not in text and "Model confidence" not in text
    assert "Why this chance" in text and "What happened to similar bookings" in text
    assert not any("Report the final status" in e.label for e in at.expander)


def test_staff_ticket_page_shows_the_ai_layer(live_api):  # noqa: F811
    at = _submit(_run("ticket", staff=True), "Predict confirmation")
    text = _text(at)
    assert "Demo scenarios" in text and "XGBoost" in text
    assert "Governance receipt" in text and "Model confidence" in text
    assert any("Report the final status" in e.label for e in at.expander)


@pytest.mark.parametrize("page,button", [("trains", "Find best trains"), ("fares", "Forecast fare"),
                                         ("delay", "Forecast delay")])
def test_passenger_results_have_no_governance_receipt(live_api, page, button):  # noqa: F811
    at = _submit(_run(page, staff=False), button)
    text = _text(at)
    assert "Governance receipt" not in text and "SHAP" not in text and "Demo scenarios" not in text


def test_passenger_delay_page_has_no_manual_weather_controls(live_api):  # noqa: F811
    at = _run("delay", staff=False)
    assert not at.toggle and not at.slider
    staff = _run("delay", staff=True)
    assert staff.toggle
