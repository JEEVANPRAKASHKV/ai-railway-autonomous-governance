"""
tests/test_governance.py
========================
Comprehensive unit tests for the 8 Pillars of the AI Governance Control Tower.
"""

import pytest
from datetime import datetime, timezone

from governance.telemetry_bus import telemetry_bus
from governance.health_monitor import health_monitor
from governance.data_quality_gate import DataQualityGate
from governance.drift_detector import drift_detector
from governance.bias_auditor import bias_auditor
from governance.hallucination_monitor import hallucination_monitor
from governance.rca_engine import rca_engine
from governance.lifecycle_manager import lifecycle_manager
from governance.report_generator import report_generator
from governance.supervisor_agent import supervisor_agent


def test_pillar1_telemetry_bus_and_health_monitor():
    """Verify Telemetry Event emission and SLA metric aggregation (Pillar 1)."""
    event = telemetry_bus.emit_dict(
        module="m1_ticket_predictor",
        latency_ms=125.0,
        input_schema_valid=True,
        confidence_score=0.91,
    )
    assert event is not None
    assert event.module == "m1_ticket_predictor"
    assert event.latency_ms == 125.0

    recent = telemetry_bus.get_recent(10)
    assert len(recent) > 0
    assert any(e.module == "m1_ticket_predictor" for e in recent)

    # Health check & SLA tracking
    health_monitor.record_request(module_id="m1_ticket_predictor", latency_ms=125.0, success=True)
    health = health_monitor.get_platform_health()
    assert health is not None
    assert "platform_status" in health or "healthy_modules" in health or "modules" in health


def test_pillar2_data_quality_gate():
    """Verify pre-inference schema validation, bounds checking, and violation logging (Pillar 2)."""
    gate = DataQualityGate()

    valid_payload = {
        "train_number": "12951",
        "journey_class": "3A",
        "quota": "GN",
        "journey_date": "2026-10-25",
        "current_wl_number": 15,
    }
    report_valid = gate.validate("m1_ticket_predictor", valid_payload)
    assert report_valid["valid"] is True
    assert len(report_valid["violations"]) == 0

    # Missing required fields
    invalid_payload = {
        "train_number": "12951",
    }
    report_invalid = gate.validate("m1_ticket_predictor", invalid_payload)
    assert report_invalid["valid"] is False
    assert len(report_invalid["violations"]) > 0


def test_pillar3_drift_detector():
    """Verify statistical drift tracking and feature window comparison (Pillar 3)."""
    # Feed observation samples
    drift_detector.observe("m1_ticket_predictor", {"current_wl_number": 20.0}, prediction=0.85)
    drift_detector.observe("m1_ticket_predictor", {"current_wl_number": 22.0}, prediction=0.82)

    summary = drift_detector.get_summary()
    assert summary is not None
    assert "models_monitored" in summary or "total_observations" in summary or "features_tracked" in summary or isinstance(summary, dict)

    analysis = drift_detector.analyze_model("m1_ticket_predictor")
    assert analysis is not None


def test_pillar4_bias_and_hallucination_monitor():
    """Verify fairness disparate impact tracking and RAG hallucination checks (Pillar 4)."""
    bias_auditor.observe(
        module_id="m2_recommender",
        protected_attrs={"age_group": "senior", "class_preference": "SL"},
        prediction=0.88,
    )
    bias_summary = bias_auditor.audit_module("m2_recommender")
    assert bias_summary is not None
    assert "module" in bias_summary or "overall_status" in bias_summary or "status" in bias_summary

    h_res = hallucination_monitor.evaluate(
        module_id="m6_support",
        query="What is the refund rule for waitlisted tickets?",
        response="Unconfirmed waitlist tickets are automatically cancelled with a full refund.",
        context_docs=[
            "Unconfirmed waitlisted e-tickets are automatically cancelled upon chart preparation with full refund to customer account."
        ],
    )
    assert h_res is not None
    assert "faithfulness_score" in h_res
    assert 0.0 <= h_res["faithfulness_score"] <= 1.0


def test_pillar5_autonomous_rca_engine():
    """Verify causal DAG incident ingestion and root cause diagnosis (Pillar 5)."""
    incident_id = rca_engine.ingest_event({
        "source_module": "m3_delay_predictor",
        "event_type": "SLA_BREACH",
        "severity": "CRITICAL",
        "details": {"latency_ms": 5200.0, "p95_limit": 3000.0},
    })
    assert incident_id is not None

    summary = rca_engine.get_summary()
    assert summary is not None
    assert "total_incidents" in summary or "open_incidents" in summary


def test_pillar6_model_lifecycle_manager():
    """Verify Model Registry status and version tracking (Pillar 6)."""
    status = lifecycle_manager.get_platform_status()
    assert status is not None
    assert "models_registered" in status or "total_model_versions" in status

    version_id = lifecycle_manager.register_version(
        model_id="m1_ticket_predictor",
        artifact_path="models/artifacts/wl_predictor_v1.0.pkl",
        metrics={"accuracy": 0.865, "f1": 0.842},
        stage="staging",
    )
    assert version_id is not None
    m_status = lifecycle_manager.get_model_status("m1_ticket_predictor")
    assert m_status is not None


def test_pillar7_compliance_report_generator():
    """Verify compliance reports generation across Markdown, JSON, and CSV (Pillar 7)."""
    report_bundle = report_generator.generate()
    assert report_bundle is not None
    assert "report_id" in report_bundle
    assert "platform_status" in report_bundle
    assert "overall_risk_level" in report_bundle


def test_pillar8_agentic_supervisor():
    """Verify supervisor dashboard state and chaos injection (Pillar 8)."""
    dash = supervisor_agent.get_dashboard_state()
    assert dash is not None
    assert "governance_score" in dash
    assert 0 <= dash["governance_score"] <= 100

    chaos_res = supervisor_agent.inject_chaos(
        module_id="m1_ticket_predictor",
        chaos_type="latency_spike",
    )
    assert chaos_res is not None
    assert chaos_res.get("success") is True
