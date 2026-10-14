"""
tests/test_core.py
==================
Unit tests for core infrastructure, LLM router fallback, telemetry bus, and trained artifacts.
"""

import os
import json
import pytest

from core.config import settings
from core.llm_router import llm_router
from core.fallback_engine import FallbackEngine
from governance.telemetry_bus import telemetry_bus
from core.schemas import TelemetryEvent


def test_platform_settings():
    """Verify platform settings load default values properly."""
    assert os.path.exists(settings.project_root)
    assert settings.llm_mode in ["azure_openai", "fallback"]
    assert settings.fastapi_port == 8000
    assert settings.streamlit_port == 8501


def test_llm_router_fallback():
    """Verify LLMRouter functions in fallback mode without exceptions."""
    resp = llm_router.complete("How do I cancel my Tatkal ticket?")
    assert resp.content is not None
    assert len(resp.content) > 0
    assert resp.mode == "fallback"


def test_fallback_engine_travel_planner():
    """Verify FallbackEngine travel planner handles basic routes."""
    engine = FallbackEngine()
    resp = engine.plan_travel("Find train from NDLS to BCT on 2026-10-20 with budget 2000")
    assert resp is not None
    assert isinstance(resp, dict)
    assert "itineraries" in resp or "query" in resp


def test_telemetry_bus_emit_and_retrieve():
    """Verify in-memory ring buffer can emit and query events."""
    event = TelemetryEvent(
        event_id="test-event-001",
        module="m1_ticket_predictor",
        latency_ms=12.5,
        schema_valid=True,
        confidence_score=0.92,
        prediction="Confirmed",
    )
    telemetry_bus.emit(event)
    recent = telemetry_bus.get_recent(5)
    assert any(e.event_id == "test-event-001" for e in recent)

    module_events = telemetry_bus.get_by_module("m1_ticket_predictor")
    assert any(e.event_id == "test-event-001" for e in module_events)


def test_trained_model_artifacts_exist():
    """Verify all 5 ML models and feature schema exist in models/artifacts/."""
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
    expected_artifacts = [
        "wl_predictor_v1.0.pkl",
        "berth_estimator_v1.0.pkl",
        "delay_predictor_v1.0.pkl",
        "fare_predictor_v1.0.pkl",
        "fraud_detector_v1.0.pkl",
        "feature_columns.json",
    ]
    for art in expected_artifacts:
        path = os.path.join(artifacts_dir, art)
        assert os.path.exists(path), f"Missing model artifact: {art}"
        assert os.path.getsize(path) > 0, f"Model artifact is empty: {art}"


def test_model_registry_json():
    """Verify model registry JSON is valid and lists models."""
    registry_path = os.path.join(os.path.dirname(__file__), "..", "models", "registry", "model_registry.json")
    assert os.path.exists(registry_path), "model_registry.json missing"
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "models" in data or isinstance(data, list)
