"""
tests/test_api.py
=================
Integration tests for FastAPI REST API endpoints using TestClient.
"""

from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_api_ticket_predictor():
    payload = {
        "train_number": "12301",
        "journey_class": "3A",
        "quota": "GN",
        "journey_date": (date.today() + timedelta(days=20)).isoformat(),
        "current_wl_number": 10,
    }
    res = client.post("/api/v1/predict/ticket", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "pnr_status_prediction" in data
    assert "confirmation_probability" in data


def test_api_train_recommender():
    payload = {
        "source_station": "NDLS",
        "destination_station": "CSTM",
        "journey_date": (date.today() + timedelta(days=12)).isoformat(),
        "persona": "budget",
    }
    res = client.post("/api/v1/recommend/trains", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["recommendations"]) > 0


def test_api_delay_predictor():
    payload = {
        "train_number": "12002",
        "station_code": "BPL",
        "journey_date": (date.today() + timedelta(days=5)).isoformat(),
        "weather_index": 0.2,
        "congestion_index": 0.5,
    }
    res = client.post("/api/v1/predict/delay", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_delay_minutes" in data


def test_api_agent_planner_and_confirm():
    plan_payload = {"query": "Find trains from Delhi to Mumbai next week in 3A with budget 2800"}
    res = client.post("/api/v1/agent/plan", json=plan_payload)
    assert res.status_code == 200
    plan_data = res.json()
    assert plan_data["status"] == "PENDING_APPROVAL"
    req_id = plan_data["request_id"]
    opt_id = plan_data["recommended_itinerary"]["option_id"]

    # Confirm via HITL gate
    confirm_payload = {
        "request_id": req_id,
        "selected_option_id": opt_id,
        "passenger_name": "Kavita Rao",
    }
    res_conf = client.post("/api/v1/agent/confirm", json=confirm_payload)
    assert res_conf.status_code == 200
    conf_data = res_conf.json()
    assert conf_data["status"] == "SIMULATED_SUCCESS"


def test_api_fare_predictor():
    payload = {
        "train_number": "12951",
        "journey_class": "3A",
        "journey_date": (date.today() + timedelta(days=15)).isoformat(),
    }
    res = client.post("/api/v1/predict/fare", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "current_predicted_fare_inr" in data
    assert len(data["forecast_curve"]) > 0


def test_api_support_query():
    payload = {"query": "How can I check my railway PNR status?"}
    res = client.post("/api/v1/support/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["response"]) > 0


def test_api_security_score():
    payload = {
        "session_id": "SES-API-TEST",
        "ip_hash": "IP-API-TEST",
        "requests_per_minute": 85.0,
        "failed_captcha_count": 3,
        "tatkal_flag": True,
    }
    res = client.post("/api/v1/security/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["fraud_risk_score"] > 50.0


def test_api_governance_telemetry_and_chaos():
    res_tel = client.get("/api/v1/governance/telemetry")
    assert res_tel.status_code == 200
    assert isinstance(res_tel.json(), list)

    res_met = client.get("/api/v1/governance/metrics")
    assert res_met.status_code == 200
    assert "total_inferences" in res_met.json()

    # Chaos injection
    chaos_payload = {
        "module": "m1_ticket_predictor",
        "chaos_type": "latency",
        "magnitude": 4200.0,
    }
    # Governance mutations are role-restricted: no key → 401, operator key → allowed
    assert client.post("/api/v1/governance/chaos/inject", json=chaos_payload).status_code == 401
    res_chaos = client.post("/api/v1/governance/chaos/inject", json=chaos_payload,
                            headers={"X-API-Key": "rb-dev-operator-key"})
    assert res_chaos.status_code == 200
    assert res_chaos.json()["success"] is True
