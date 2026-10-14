"""
tests/test_modules.py
=====================
Direct inference unit tests for all 7 AI service modules (M1–M7).
"""

from datetime import date, timedelta
import pytest

from modules.m1_ticket_predictor.schema import TicketPredictRequest
from modules.m1_ticket_predictor.service import ticket_predictor_service
from modules.m2_train_recommender.schema import RecommendRequest
from modules.m2_train_recommender.service import train_recommender_service
from modules.m3_delay_predictor.schema import DelayPredictRequest
from modules.m3_delay_predictor.service import delay_predictor_service
from modules.m4_travel_planner.schema import BookingConfirmRequest, PlanRequest
from modules.m4_travel_planner.agent import travel_planner_agent
from modules.m4_travel_planner.hitl_gate import hitl_gate
from modules.m5_fare_predictor.schema import FarePredictRequest
from modules.m5_fare_predictor.service import fare_predictor_service
from modules.m6_support_assistant.schema import SupportQueryRequest
from modules.m6_support_assistant.service import support_assistant_service
from modules.m7_fraud_detector.schema import FraudDetectRequest
from modules.m7_fraud_detector.service import fraud_detector_service


def test_m1_ticket_predictor():
    req = TicketPredictRequest(
        train_number="12301",
        journey_class="3A",
        quota="GN",
        journey_date=date.today() + timedelta(days=20),
        current_wl_number=8,
    )
    res = ticket_predictor_service.predict(req)
    assert res.pnr_status_prediction in ["Likely Confirmed", "Borderline", "Likely WL"]
    assert 0.0 <= res.confirmation_probability <= 1.0
    assert res.estimated_berth_type is not None
    assert len(res.shap_explanation) > 0


def test_m2_train_recommender():
    req = RecommendRequest(
        source_station="NDLS",
        destination_station="CSTM",
        journey_date=date.today() + timedelta(days=10),
        persona="budget",
    )
    res = train_recommender_service.recommend(req)
    assert len(res.recommendations) > 0
    assert res.disparate_impact_ratio > 0.0
    assert res.persona_applied == "budget"


def test_m3_delay_predictor():
    req = DelayPredictRequest(
        train_number="12002",
        station_code="BPL",
        journey_date=date.today() + timedelta(days=5),
        weather_index=0.3,
        congestion_index=0.4,
    )
    res = delay_predictor_service.predict(req)
    assert res.predicted_delay_minutes >= 0.0
    assert res.congestion_level in ["Low", "Moderate", "Severe"]
    assert "lower_bound" in res.confidence_interval


def test_m4_travel_planner_and_hitl_gate():
    plan_req = PlanRequest(query="Travel from NDLS to CSTM next week in 3A with budget 2500")
    approval = travel_planner_agent.plan(plan_req)
    assert approval.status == "PENDING_APPROVAL"
    assert approval.recommended_itinerary is not None

    # Test HITL approval confirmation
    confirm_req = BookingConfirmRequest(
        request_id=approval.request_id,
        selected_option_id=approval.recommended_itinerary.option_id,
        passenger_name="Rohan Verma",
    )
    booking = hitl_gate.confirm_plan(confirm_req)
    assert booking.status == "SIMULATED_SUCCESS"
    assert booking.pnr_number.startswith("PNR")


def test_m5_fare_predictor():
    req = FarePredictRequest(
        train_number="12951",
        journey_class="3A",
        journey_date=date.today() + timedelta(days=15),
    )
    res = fare_predictor_service.predict(req)
    assert res.current_predicted_fare_inr > 0.0
    assert len(res.forecast_curve) > 0
    assert res.optimal_booking_days_before > 0


def test_m6_support_assistant():
    req = SupportQueryRequest(query="How do I get a refund on Tatkal ticket cancellation?")
    res = support_assistant_service.query(req)
    assert len(res.response) > 0
    assert res.faithfulness_score > 0.0
    assert len(res.retrieved_contexts) > 0


def test_m7_fraud_detector():
    req = FraudDetectRequest(
        session_id="TEST-SES-001",
        ip_hash="TEST-IP-999",
        requests_per_minute=120.0,
        failed_captcha_count=4,
        tatkal_flag=True,
    )
    res = fraud_detector_service.score_session(req)
    assert res.fraud_risk_score > 60.0
    assert res.risk_category in ["High Risk", "Bot Detected"]
    assert res.recommended_action in ["RATE_LIMIT", "BLOCK_AND_REPORT"]
