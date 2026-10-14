"""
backend/routers/governance.py
=============================
FastAPI router for AI Governance Telemetry, Metrics, Alerts, and Chaos Simulator.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query
import numpy as np
from pydantic import BaseModel, Field

from backend.dependencies import get_telemetry_bus
from core.schemas import SupervisorAlert, TelemetryEvent
from governance.telemetry_bus import TelemetryBus

router = APIRouter(prefix="/governance", tags=["AI Governance Control Tower"])


class ChaosInjectionRequest(BaseModel):
    module: str = Field(default="m1_ticket_predictor", description="Target module for chaos simulation")
    chaos_type: str = Field(default="latency", description="'latency' | 'drift' | 'schema_violation' | 'outlier'")
    magnitude: float = Field(default=3500.0, description="Injected value (e.g. latency in ms)")


class ChaosInjectionResponse(BaseModel):
    success: bool
    message: str
    event_id: str
    injected_details: Dict[str, Any]


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

    # Per module latency breakdown
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
