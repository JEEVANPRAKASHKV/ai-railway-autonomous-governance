"""
backend/routers/governance.py
=============================
FastAPI router for AI Governance Telemetry, Metrics, Alerts, and Chaos Simulator.
Stage 3 — Full Autonomous AI Governance Engine & Supervisory Control Tower endpoints.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query
import numpy as np
from pydantic import BaseModel, Field

from backend.dependencies import get_telemetry_bus
from core.schemas import SupervisorAlert, TelemetryEvent
from governance.telemetry_bus import TelemetryBus

# Stage 3 governance pillars
from governance.supervisor_agent import supervisor_agent
from governance.health_monitor import health_monitor
from governance.drift_detector import drift_detector
from governance.bias_auditor import bias_auditor
from governance.hallucination_monitor import hallucination_monitor
from governance.rca_engine import rca_engine
from governance.lifecycle_manager import lifecycle_manager
from governance.report_generator import report_generator

router = APIRouter(prefix="/governance", tags=["AI Governance Control Tower"])


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

class ChaosInjectionRequest(BaseModel):
    module: str = Field(default="m1_ticket_predictor", description="Target module for chaos simulation")
    chaos_type: str = Field(default="latency", description="'latency' | 'drift' | 'schema_violation' | 'outlier'")
    magnitude: float = Field(default=3500.0, description="Injected value (e.g. latency in ms)")


class ChaosInjectionResponse(BaseModel):
    success: bool
    message: str
    event_id: str
    injected_details: Dict[str, Any]


class RollbackRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to rollback")
    reason: str = Field(default="manual_operator_request")


class HITLDecisionRequest(BaseModel):
    request_id: str
    approved: bool


class SupervisorChaosRequest(BaseModel):
    module_id: str = Field(..., description="Module to inject chaos into")
    chaos_type: str = Field(
        default="latency_spike",
        description="'latency_spike' | 'error_surge' | 'drift_simulation'",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1-2 Legacy Endpoints (Telemetry Bus)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/telemetry",
    summary="Retrieve real-time telemetry events from in-memory ring buffer",
)
def get_recent_telemetry(
    limit: int = Query(default=50, ge=1, le=500),
    module: Optional[str] = Query(default=None),
    bus: TelemetryBus = Depends(get_telemetry_bus),
) -> List[Dict[str, Any]]:
    if module:
        events = bus.get_by_module(module, n=limit)
    else:
        events = bus.get_recent(n=limit)
    return [e.model_dump() for e in events]


@router.get(
    "/metrics",
    summary="Aggregated platform operational health, p95 latencies, and module error rates",
)
def get_governance_metrics(
    bus: TelemetryBus = Depends(get_telemetry_bus),
) -> Dict[str, Any]:
    events = bus.get_recent(n=500)
    total_calls = len(events)
    if total_calls == 0:
        return {
            "total_inferences": 0,
            "average_latency_ms": 0.0,
            "error_rate_pct": 0.0,
            "modules_active": [],
            "sla_breaches": 0,
        }

    latencies = [e.latency_ms for e in events]
    errors = sum(1 for e in events if e.error or not e.schema_valid)
    sla_breaches = sum(1 for e in events if e.latency_ms > 3000.0)

    by_mod: Dict[str, List[float]] = {}
    for e in events:
        by_mod.setdefault(e.module, []).append(e.latency_ms)

    mod_stats = {
        m: {
            "calls": len(lats),
            "avg_latency_ms": round(sum(lats) / len(lats), 1),
            "p95_latency_ms": round(float(np.percentile(lats, 95)) if len(lats) > 1 else lats[0], 1),
        }
        for m, lats in by_mod.items()
    }

    return {
        "total_inferences": total_calls,
        "average_latency_ms": round(sum(latencies) / total_calls, 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1) if len(latencies) > 1 else latencies[0],
        "error_rate_pct": round((errors / total_calls) * 100, 2),
        "sla_breaches": sla_breaches,
        "module_breakdown": mod_stats,
    }


@router.get(
    "/alerts",
    summary="Retrieve active autonomous supervisor alerts",
)
def get_supervisor_alerts(
    bus: TelemetryBus = Depends(get_telemetry_bus),
) -> List[Dict[str, Any]]:
    return [a.model_dump() for a in bus.get_alerts()]


@router.post(
    "/chaos/inject",
    response_model=ChaosInjectionResponse,
    summary="Inject simulated chaos to test autonomous governance alerting & remediation loops",
)
def inject_chaos(
    req: ChaosInjectionRequest,
    bus: TelemetryBus = Depends(get_telemetry_bus),
) -> ChaosInjectionResponse:
    event_id = f"CHAOS-{uuid.uuid4().hex[:6].upper()}"

    if req.chaos_type == "latency":
        bus.emit(
            TelemetryEvent(
                event_id=event_id,
                module=req.module,
                latency_ms=req.magnitude,
                schema_valid=True,
                extra={"chaos_injected": True, "type": "latency_spike"},
            )
        )
        msg = f"Injected artificial latency spike of {req.magnitude}ms into {req.module}."
    elif req.chaos_type == "drift":
        bus.emit(
            TelemetryEvent(
                event_id=event_id,
                module=req.module,
                latency_ms=45.0,
                schema_valid=True,
                drift_flag=True,
                extra={"chaos_injected": True, "psi": 0.28},
            )
        )
        msg = f"Injected synthetic feature distribution drift (PSI 0.28) into {req.module}."
    elif req.chaos_type == "schema_violation":
        bus.emit(
            TelemetryEvent(
                event_id=event_id,
                module=req.module,
                latency_ms=12.0,
                schema_valid=False,
                schema_violations=["Field 'current_wl_number' out of valid range: got 9999"],
                extra={"chaos_injected": True},
            )
        )
        msg = f"Injected schema constraint violation into {req.module}."
    else:
        bus.emit(
            TelemetryEvent(
                event_id=event_id,
                module=req.module,
                latency_ms=50.0,
                confidence_score=0.25,
                extra={"chaos_injected": True},
            )
        )
        msg = f"Injected low-confidence anomaly into {req.module}."

    return ChaosInjectionResponse(
        success=True,
        message=msg,
        event_id=event_id,
        injected_details={"module": req.module, "type": req.chaos_type, "magnitude": req.magnitude},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Supervisor Agent
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/supervisor/dashboard", summary="Full governance dashboard state — all pillars")
def get_supervisor_dashboard() -> Dict[str, Any]:
    return supervisor_agent.get_dashboard_state()


@router.get("/supervisor/actions", summary="Governance action audit log")
def get_action_log(n: int = Query(default=50, ge=1, le=200)) -> List[Dict[str, Any]]:
    return supervisor_agent.get_action_log(n=n)


@router.post("/supervisor/report", summary="Trigger a fresh comprehensive governance audit report")
def trigger_report() -> Dict[str, Any]:
    return supervisor_agent.trigger_report()


@router.post("/supervisor/chaos", summary="Inject controlled chaos via supervisor agent")
def supervisor_inject_chaos(req: SupervisorChaosRequest) -> Dict[str, Any]:
    return supervisor_agent.inject_chaos(req.module_id, req.chaos_type)


@router.post("/supervisor/rollback", summary="Manually trigger a model rollback")
def manual_rollback(req: RollbackRequest) -> Dict[str, Any]:
    return supervisor_agent.trigger_rollback(req.model_id, req.reason)


@router.get("/supervisor/hitl/pending", summary="List pending HITL approval requests")
def get_hitl_pending() -> List[Dict[str, Any]]:
    return supervisor_agent.get_hitl_pending()


@router.post("/supervisor/hitl/decide", summary="Approve or reject a HITL governance action")
def decide_hitl(req: HITLDecisionRequest) -> Dict[str, Any]:
    return supervisor_agent.approve_hitl(req.request_id, req.approved)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Health Monitor
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health/platform", summary="Platform-wide health metrics")
def get_platform_health() -> Dict[str, Any]:
    return health_monitor.get_platform_health()


@router.get("/health/module/{module_id}", summary="Per-module health metrics")
def get_module_health(module_id: str) -> Dict[str, Any]:
    return health_monitor.get_module_health(module_id)


@router.get("/health/sla", summary="SLA compliance report")
def get_sla_report() -> List[Dict[str, Any]]:
    return health_monitor.get_sla_report()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Drift Detector
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/drift/all", summary="KS-test + PSI drift analysis across all models")
def analyze_all_drift() -> List[Dict[str, Any]]:
    return drift_detector.analyze_all_models()


@router.get("/drift/model/{model_id}", summary="Drift analysis for a specific model")
def analyze_model_drift(model_id: str) -> Dict[str, Any]:
    return drift_detector.analyze_model(model_id)


@router.get("/drift/history", summary="Recent drift detection events")
def get_drift_history(n: int = Query(default=20, ge=1, le=100)) -> List[Dict[str, Any]]:
    return drift_detector.get_drift_history(n=n)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Bias Auditor
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/bias/audit/all", summary="Fairness audit across all modules")
def audit_all_bias() -> List[Dict[str, Any]]:
    return bias_auditor.audit_all_modules()


@router.get("/bias/audit/{module_id}", summary="Fairness audit for a specific module")
def audit_module_bias(module_id: str) -> Dict[str, Any]:
    return bias_auditor.audit_module(module_id)


@router.get("/bias/history", summary="Recent bias detection events")
def get_bias_history(n: int = Query(default=20, ge=1, le=100)) -> List[Dict[str, Any]]:
    return bias_auditor.get_audit_history(n=n)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Hallucination Monitor
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/hallucination/stats", summary="LLM faithfulness & hallucination statistics")
def get_hallucination_stats() -> Dict[str, Any]:
    return hallucination_monitor.get_platform_stats()


@router.get("/hallucination/recent", summary="Recent LLM faithfulness evaluations")
def get_recent_hallucination_evals(n: int = Query(default=20, ge=1, le=100)) -> List[Dict[str, Any]]:
    return hallucination_monitor.get_recent_evaluations(n=n)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — RCA Engine / Incidents
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/incidents/open", summary="List all open governance incidents")
def get_open_incidents() -> List[Dict[str, Any]]:
    return rca_engine.get_open_incidents()


@router.get("/incidents/summary", summary="RCA engine summary statistics")
def get_rca_summary() -> Dict[str, Any]:
    return rca_engine.get_summary()


@router.get("/incidents/{incident_id}", summary="Full RCA report for a specific incident")
def get_incident(incident_id: str) -> Dict[str, Any]:
    result = rca_engine.analyze_incident(incident_id)
    return result or {"error": "Incident not found", "incident_id": incident_id}


@router.post("/incidents/{incident_id}/resolve", summary="Mark an incident as resolved")
def resolve_incident(incident_id: str) -> Dict[str, Any]:
    success = rca_engine.resolve_incident(incident_id)
    return {"success": success, "incident_id": incident_id}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Model Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/lifecycle/platform", summary="Platform-wide model lifecycle status")
def get_lifecycle_status() -> Dict[str, Any]:
    return lifecycle_manager.get_platform_status()


@router.get("/lifecycle/model/{model_id}", summary="Lifecycle status for a specific model")
def get_model_lifecycle(model_id: str) -> Dict[str, Any]:
    return lifecycle_manager.get_model_status(model_id)


@router.get("/lifecycle/retraining-queue", summary="Models queued for retraining")
def get_retraining_queue() -> List[str]:
    return lifecycle_manager.get_retraining_queue()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Reports
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/reports/history", summary="List past governance audit reports")
def get_report_history(n: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    return report_generator.get_report_history(n=n)


@router.get("/reports/{report_id}", summary="Load a specific governance report by ID")
def get_report(report_id: str) -> Dict[str, Any]:
    result = report_generator.load_report(report_id)
    return result or {"error": "Report not found", "report_id": report_id}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Platform Health Reset & Telemetry Seeder
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/supervisor/reset", summary="Reset platform to healthy state")
def reset_platform(bus: TelemetryBus = Depends(get_telemetry_bus)) -> Dict[str, Any]:
    resolved_alerts = bus.resolve_all_alerts()
    resolved_incidents = rca_engine.resolve_all_incidents()
    health_monitor.reset_all()
    return {
        "success": True,
        "message": "Platform state successfully reset to Healthy.",
        "resolved_alerts": resolved_alerts,
        "resolved_incidents": resolved_incidents,
        "platform_status": "HEALTHY",
    }


@router.post("/telemetry/seed", summary="Seed 50 baseline telemetry events across M1-M7")
def seed_telemetry(
    count: int = Query(default=50, ge=10, le=200),
    bus: TelemetryBus = Depends(get_telemetry_bus),
) -> Dict[str, Any]:
    seeded = bus.seed_baseline(count=count)
    return {
        "success": True,
        "seeded_count": seeded,
        "message": f"Successfully seeded {seeded} realistic baseline telemetry events across M1–M7.",
    }
