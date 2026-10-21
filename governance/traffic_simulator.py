"""
governance/traffic_simulator.py
───────────────────────────────
Generates passenger traffic through the real, governed services (demo / load-test helper).

"Normal" traffic replays held-out historical records, so its inputs follow the training
distribution and should NOT trigger drift. "Shifted" traffic changes the world on purpose (very
long waitlists, storms and crowded stations, last-minute bookings, bot-like sessions) so the drift
detector, data-quality gate and fraud monitoring have something real to react to.
"""

from __future__ import annotations

import os
import random
import threading
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from core.config import settings
from core.seasonality import season_factor

_lock = threading.Lock()
_cache: Dict[str, pd.DataFrame] = {}
SUPPORT_QS = [
    "How do I check my PNR status?", "What is the refund if my waitlist ticket is not confirmed?",
    "Can I transfer my ticket to a family member?", "What are the Tatkal booking timings?",
    "Can an RAC passenger travel in a 3A coach?", "Is there a refund if I cancel a Tatkal ticket?",
    "What is the status of PNR 0000002?", "Which trains from Delhi to Mumbai?",
]
PERSONAS = ["budget", "premium", "student", "senior", "business", "family"]


def _df(name: str, n: int = 4000) -> pd.DataFrame:
    with _lock:
        if name not in _cache:
            df = pd.read_csv(os.path.join(settings.data_raw_dir, name))
            _cache[name] = df.sample(min(n, len(df)), random_state=99).reset_index(drop=True)
        return _cache[name]


def simulate(rounds: int = 20, shifted: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    from modules.m1_ticket_predictor.schema import TicketPredictRequest
    from modules.m1_ticket_predictor.service import ticket_predictor_service
    from modules.m2_train_recommender.schema import RecommendRequest
    from modules.m2_train_recommender.service import train_recommender_service
    from modules.m3_delay_predictor.schema import DelayPredictRequest
    from modules.m3_delay_predictor.service import delay_predictor_service
    from modules.m5_fare_predictor.schema import FarePredictRequest
    from modules.m5_fare_predictor.service import fare_predictor_service
    from modules.m6_support_assistant.schema import SupportQueryRequest
    from modules.m6_support_assistant.service import support_assistant_service
    from modules.m7_fraud_detector.schema import FraudDetectRequest
    from modules.m7_fraud_detector.service import fraud_detector_service

    rnd = random.Random(seed)
    pnr, delay, price, fraud, trains = (_df("pnr_history.csv"), _df("delay_logs.csv"), _df("pricing_history.csv"),
                                        _df("fraud_events.csv"), _df("trains.csv", 10_000))
    routes = trains[["source", "destination"]].drop_duplicates().values.tolist()
    ok, failed = 0, 0
    today = date.today()

    def call(fn, *a):
        nonlocal ok, failed
        try:
            fn(*a)
            ok += 1
        except Exception:
            failed += 1

    for i in range(rounds):
        # M1 — replay a historical booking (or an extreme waitlist when shifted)
        r = pnr.iloc[rnd.randrange(len(pnr))]
        days = rnd.randint(1, 5) if shifted else max(1, int(r["days_to_departure"]))
        call(ticket_predictor_service.predict, TicketPredictRequest(
            train_number=str(r["train_number"]), journey_class=str(r["journey_class"]), quota=str(r["quota"]),
            journey_date=today + timedelta(days=days),
            current_wl_number=rnd.randint(160, 400) if shifted else int(r["initial_wl_number"]),
            seniority_flag=bool(r["seniority_flag"]), disability_flag=bool(r["disability_flag"])))

        # M3 — replay recorded conditions (inputs converted so the model sees the recorded values)
        d = delay.iloc[rnd.randrange(len(delay))]
        jd = today + timedelta(days=rnd.randint(0, 30))
        sf = season_factor(jd)
        weather = rnd.uniform(0.85, 1.0) if shifted else float(d["weather_index"]) / 10.0
        cong_model = float(d["congestion_at_station"])
        cong = rnd.uniform(0.9, 1.0) if shifted else min(1.0, max(0.0, (cong_model / sf - 10.0) / 0.8 / 100.0))
        call(delay_predictor_service.predict, DelayPredictRequest(
            train_number=str(d["train_number"]), station_code=str(d["station_code"]), journey_date=jd,
            weather_index=round(weather, 3), congestion_index=round(cong, 3),
            distance_from_origin_km=float(rnd.uniform(1800, 2600) if shifted else d["distance_from_origin_km"])))

        # M5 — replay a historical fare enquiry (last-minute Tatkal-style when shifted)
        p = price.iloc[rnd.randrange(len(price))]
        pdays = rnd.randint(0, 1) if shifted else int(p["days_to_departure"])
        call(fare_predictor_service.predict, FarePredictRequest(
            train_number=str(p["train_number"]), journey_class=str(p["journey_class"]),
            journey_date=today + timedelta(days=pdays), base_fare=float(p["base_fare"]),
            tatkal_flag=bool(shifted or p["tatkal_flag"])))

        # M7 — replay a recorded session (bot-heavy when shifted)
        f = fraud[fraud["label"] == "bot"].iloc[rnd.randrange(int((fraud["label"] == "bot").sum()))] if shifted \
            else fraud.iloc[rnd.randrange(len(fraud))]
        call(fraud_detector_service.score_session, FraudDetectRequest(
            session_id=f"SIM-{i}-{rnd.randint(1000, 9999)}", ip_hash=str(f["ip_hash"]),
            requests_per_minute=float(f["requests_per_minute"]), unique_pnr_attempts=int(f["unique_pnr_attempts"]),
            tatkal_flag=bool(f["tatkal_flag"]), account_age_days=int(f["account_age_days"]),
            payment_attempts=int(f["payment_attempts"]), failed_captcha_count=int(f["failed_captcha_count"]),
            booking_time_hour=int(f["booking_time_hour"]), is_bulk_booking=bool(f["is_bulk_booking"]),
            ip_risk_score=float(f["ip_risk_score"])))

        # M2 — a real route; persona mix shifts to business travellers when shifted
        src, dst = routes[rnd.randrange(len(routes))]
        call(train_recommender_service.recommend, RecommendRequest(
            source_station=src, destination_station=dst, journey_date=today + timedelta(days=rnd.randint(1, 60)),
            persona="business" if shifted else rnd.choice(PERSONAS)))

        if i % 3 == 0:
            call(support_assistant_service.query, SupportQueryRequest(query=rnd.choice(SUPPORT_QS)))
    return {"rounds": rounds, "shifted": shifted, "requests_ok": ok, "requests_failed": failed}
