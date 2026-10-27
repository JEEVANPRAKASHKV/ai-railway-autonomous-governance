"""
tests/test_ui_round2.py
=======================
Regression tests for the second UI/product pass (October 2026):
English-only support, recommender connections and validation, timetable consistency,
and a headless smoke run of every Streamlit page (including the floating support chat).
"""

from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
NEXT_WEEK = str(date.today() + timedelta(days=7))

try:
    from streamlit.testing.v1.element_tree import ButtonGroup
    _orig_indices = ButtonGroup.indices.fget
    def _safe_indices(self):
        if getattr(self, "value", None) is None:
            return []
        return _orig_indices(self)
    ButtonGroup.indices = property(_safe_indices)
except Exception:
    pass


# ── Support assistant: English only ────────────────────────────────────────────────────

def test_support_answers_english_questions():
    r = client.post("/api/v1/support/query", json={"query": "What is the refund if my waitlist ticket is not confirmed?"})
    assert r.status_code == 200
    body = r.json()
    assert body["detected_language"] == "en"
    assert body["answered_by"] != "language guard"
    assert "refund" in body["response"].lower()


def test_support_replies_in_english_to_other_languages():
    r = client.post("/api/v1/support/query", json={"query": "तत्काल टिकट रद्द करने पर रिफंड मिलता है?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answered_by"] == "language guard"
    assert body["detected_language"] == "unsupported"
    assert "English" in body["response"]


def test_support_ignores_legacy_language_field():
    r = client.post("/api/v1/support/query", json={"query": "How do I check my PNR status?", "language": "hi"})
    assert r.status_code == 200
    assert r.json()["detected_language"] == "en"


def test_knowledge_base_is_english_only():
    import re
    text = open("data/raw/railway_faq.txt", encoding="utf-8").read()
    assert not re.search(r"[ऀ-ॿ]", text)
    assert "karna" not in text.lower()


# ── Train recommender: connections, same-station and time validation ──────────────────

def test_recommender_offers_connections_when_no_direct_train():
    # no train in the timetable runs Ranchi → Pune, but there are one-change journeys via a hub
    r = client.post("/api/v1/recommend/trains", json={"source_station": "RNC", "destination_station": "PUNE",
                                                       "journey_date": NEXT_WEEK})
    assert r.status_code == 200
    body = r.json()
    assert body["recommendations"] == []
    assert body["connections"], "expected at least one one-change connection"
    c = body["connections"][0]
    assert c["legs"][0]["source"] == "RNC" and c["legs"][-1]["destination"] == "PUNE"
    assert c["legs"][0]["destination"] == c["legs"][1]["source"] == c["via"]
    assert 45 <= c["layover_minutes"] <= 8 * 60
    assert c["total_fare_inr"] == pytest.approx(sum(l["fare_inr"] for l in c["legs"]))


def test_recommender_rejects_same_station():
    r = client.post("/api/v1/recommend/trains", json={"source_station": "NDLS", "destination_station": "NDLS",
                                                       "journey_date": NEXT_WEEK})
    assert r.status_code == 200
    assert r.json()["recommendations"] == [] and "same station" in r.json()["message"]


def test_recommender_validates_arrival_time():
    r = client.post("/api/v1/recommend/trains", json={"source_station": "NDLS", "destination_station": "CSTM",
                                                       "journey_date": NEXT_WEEK, "arrival_deadline": "late evening"})
    assert r.status_code == 422


# ── Timetable consistency: arrival time = departure + journey time ────────────────────

def test_timetable_arrival_matches_departure_plus_duration():
    t = pd.read_csv("data/raw/trains.csv")

    def minutes(s):
        h, m = map(int, s.split(":"))
        return h * 60 + m

    expected = (t["departure_time"].map(minutes) + (t["duration_hours"] * 60).round().astype(int)) % 1440
    assert (t["arrival_time"].map(minutes) == expected).all()


# ── Delay explanations are shown in passenger units ───────────────────────────────────

def test_delay_drivers_have_readable_values():
    r = client.post("/api/v1/predict/delay", json={"train_number": "12301", "station_code": "NDLS",
                                                   "journey_date": NEXT_WEEK, "weather_index": 0.9, "congestion_index": 0.8})
    assert r.status_code == 200
    drivers = {d["feature"]: d for d in r.json()["delay_drivers"] if d["feature"] != "base_value"}
    assert drivers["weather_index"]["display"] == "90% severity"
    assert drivers["congestion_at_station"]["display"] == "80/100"


# ── Headless smoke test of every page (Streamlit AppTest) ──────────────────────────────

PAGES = ["frontend/app.py"]


@pytest.fixture(scope="module")
def live_api():
    """Serve the API on 127.0.0.1:8000 so the UI under test talks to a real backend."""
    import socket
    import threading
    import time

    import uvicorn

    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8000)) == 0:
            yield
            return
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield
    server.should_exit = True
    th.join(timeout=5)


def _page_script(page: str, root: str) -> None:
    """Runs one page exactly as app.py does (page body + floating support chat), headless."""
    import sys
    sys.path.insert(0, root)
    from frontend import passenger, tower
    from frontend.chat_widget import render_chat_widget
    {"home": lambda: passenger.home({}), "ticket": passenger.ticket, "trains": passenger.recommend,
     "delay": passenger.delay, "planner": passenger.planner, "fares": passenger.fare,
     "governance-tower": lambda: tower.tower({}), "security": passenger.fraud}[page]()
    render_chat_widget()


@pytest.mark.parametrize("page", ["home", "ticket", "trains", "delay", "planner", "fares", "governance-tower", "security"])
def test_every_page_renders_without_errors(live_api, page):
    import pathlib
    from streamlit.testing.v1 import AppTest
    root = str(pathlib.Path(__file__).resolve().parents[1])
    at = AppTest.from_function(_page_script, args=(page, root), default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # the floating support chat is on every page
    assert any(getattr(c, "key", None) == "rb_chat_input" for c in at.chat_input)


def test_support_chat_quick_question_round_trip(live_api):
    import pathlib
    from streamlit.testing.v1 import AppTest
    root = str(pathlib.Path(__file__).resolve().parents[1])
    at = AppTest.from_function(_page_script, args=("home", root), default_timeout=90)
    at.run()
    at.button(key="rb_qq1").click().run()                   # "Waitlist refund" quick question
    assert not at.exception
    hist = at.session_state["chat"]
    assert hist[0]["role"] == "user" and hist[1]["res"]["response"]
    at.chat_input(key="rb_chat_input").set_value("Can I transfer my ticket to my father?").run()
    assert len(at.session_state["chat"]) == 4
