"""
governance/telemetry_bus.py
============================
Thread-safe in-memory telemetry event bus.

All 7 passenger modules emit TelemetryEvent objects here via @governance_hook.
The Governance Cockpit reads from this bus in real-time.

Architecture:
  - Ring buffer (collections.deque) — maxlen=5000 events
  - Thread-safe via threading.Lock
  - Supports filtering by module, severity, time range
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.schemas import TelemetryEvent, SupervisorAlert, AuditEntry


class TelemetryBus:
    """
    Central telemetry event bus.
    Thread-safe ring buffer for inference telemetry, alerts, and audit entries.
    """

    def __init__(self, max_events: int = 5000) -> None:
        self._events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self._alerts: deque[SupervisorAlert] = deque(maxlen=1000)
        self._audit_log: deque[AuditEntry] = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._alert_lock = threading.Lock()
        self._audit_lock = threading.Lock()

    # ── Telemetry Events ──────────────────────────────────────────────────────

    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event to the bus. Thread-safe."""
        with self._lock:
            self._events.append(event)

    def emit_dict(self, module: str, latency_ms: float, **kwargs) -> TelemetryEvent:
        """Convenience method: create and emit a TelemetryEvent from kwargs."""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4())[:8],
            module=module,
            latency_ms=latency_ms,
            **kwargs,
        )
        self.emit(event)
        return event

    def publish(
        self,
        module_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
        latency_ms: float = 0.0,
        **kwargs,
    ) -> TelemetryEvent:
        """Alias for publishing structured events to the bus."""
        return self.emit_dict(
            module=module_id,
            latency_ms=latency_ms,
            event_type=event_type,
            severity=severity,
            payload=payload or {},
            **kwargs,
        )

    def get_recent(self, n: int = 100, limit: Optional[int] = None) -> List[TelemetryEvent]:
        """Get the n most recent events."""
        count = limit if limit is not None else n
        with self._lock:
            events = list(self._events)
        return events[-count:]

    def get_by_module(self, module: str, n: int = 200, limit: Optional[int] = None) -> List[TelemetryEvent]:
        """Get the n most recent events for a specific module."""
        count = limit if limit is not None else n
        with self._lock:
            events = [e for e in self._events if e.module == module]
        return events[-count:]

    def get_all(self) -> List[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def get_module_summary(self, module: str, window_minutes: int = 60) -> Dict:
        """Compute summary statistics for a module within a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        with self._lock:
            events = [e for e in self._events if e.module == module and e.timestamp >= cutoff]

        if not events:
            return {
                "module": module, "request_count": 0, "avg_latency_ms": 0,
                "p50_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0,
                "error_rate": 0, "schema_invalid_rate": 0, "health_status": "No Data",
            }

        latencies = sorted(e.latency_ms for e in events)
        n = len(latencies)
        avg_latency = sum(latencies) / n
        p50 = latencies[int(n * 0.50)]
        p95 = latencies[int(n * 0.95)] if n >= 20 else latencies[-1]
        p99 = latencies[int(n * 0.99)] if n >= 100 else latencies[-1]
        error_rate = sum(1 for e in events if e.error) / n
        schema_invalid_rate = sum(1 for e in events if not e.schema_valid) / n
        drift_rate = sum(1 for e in events if e.drift_flag) / n
        halluc_rate = sum(1 for e in events if e.hallucination_flag) / n

        # Health status
        if p95 > 5000 or error_rate > 0.05:
            health_status = "Critical"
        elif p95 > 3000 or error_rate > 0.02 or schema_invalid_rate > 0.05:
            health_status = "Degraded"
        else:
            health_status = "Healthy"

        return {
            "module": module,
            "request_count": n,
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "error_rate": round(error_rate, 4),
            "schema_invalid_rate": round(schema_invalid_rate, 4),
            "drift_rate": round(drift_rate, 4),
            "hallucination_rate": round(halluc_rate, 4),
            "health_status": health_status,
        }

    def get_all_module_summaries(self, window_minutes: int = 60) -> Dict[str, Dict]:
        modules = [
            "m1_ticket_predictor", "m2_train_recommender", "m3_delay_predictor",
            "m4_travel_planner", "m5_fare_predictor", "m6_support_assistant",
            "m7_fraud_detector",
        ]
        return {m: self.get_module_summary(m, window_minutes) for m in modules}

    # ── Alerts ────────────────────────────────────────────────────────────────

    def add_alert(self, alert: SupervisorAlert) -> None:
        with self._alert_lock:
            self._alerts.append(alert)

    def get_alerts(self, priority: Optional[str] = None, n: int = 50) -> List[SupervisorAlert]:
        with self._alert_lock:
            alerts = list(self._alerts)
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        return alerts[-n:]

    def get_active_alerts(self) -> List[SupervisorAlert]:
        """Get unresolved alerts."""
        with self._alert_lock:
            return [a for a in self._alerts if a.action_status not in ("Resolved",)]

    def resolve_alert(self, alert_id: str, resolution_log: str = "") -> bool:
        with self._alert_lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.action_status = "Resolved"
                    alert.resolution_log = resolution_log
                    return True
        return False

    # ── Audit Log ─────────────────────────────────────────────────────────────

    def log_audit(self, entry: AuditEntry) -> None:
        with self._audit_lock:
            self._audit_log.append(entry)

    def get_audit_log(
        self,
        module: Optional[str] = None,
        risk_level: Optional[str] = None,
        n: int = 1000,
    ) -> List[AuditEntry]:
        with self._audit_lock:
            entries = list(self._audit_log)
        if module:
            entries = [e for e in entries if e.module == module]
        if risk_level:
            entries = [e for e in entries if e.risk_level == risk_level]
        return entries[-n:]

    # ── Chaos Injection ───────────────────────────────────────────────────────

    def inject_chaos_drift(self, module: str = "m3_delay_predictor", severity: int = 3) -> None:
        """Inject synthetic drift events into the bus (for Chaos Simulator)."""
        import random
        for _ in range(severity * 20):
            event = TelemetryEvent(
                event_id=str(uuid.uuid4())[:8],
                module=module,
                latency_ms=random.uniform(200, 800),
                schema_valid=True,
                drift_flag=True,
                confidence_score=random.uniform(0.4, 0.65),
            )
            self.emit(event)

    def inject_chaos_bot_attack(self, intensity: int = 3) -> None:
        """Inject synthetic bot/fraud sessions (for Chaos Simulator)."""
        import random
        for _ in range(intensity * 15):
            event = TelemetryEvent(
                event_id=str(uuid.uuid4())[:8],
                module="m7_fraud_detector",
                latency_ms=random.uniform(50, 200),
                schema_valid=True,
                confidence_score=random.uniform(0.85, 0.99),
                extra={"fraud_risk_score": random.randint(75, 100), "chaos_injected": True},
            )
            self.emit(event)

    def inject_chaos_hallucination(self, rate: float = 0.5) -> None:
        """Inject synthetic hallucination events for M6 (for Chaos Simulator)."""
        import random
        count = max(5, int(rate * 20))
        for _ in range(count):
            event = TelemetryEvent(
                event_id=str(uuid.uuid4())[:8],
                module="m6_support_assistant",
                latency_ms=random.uniform(300, 1500),
                schema_valid=True,
                hallucination_flag=True,
                confidence_score=random.uniform(0.3, 0.60),
                extra={"faithfulness_score": random.uniform(0.2, 0.64), "chaos_injected": True},
            )
            self.emit(event)

    def inject_chaos_pricing_glitch(self) -> None:
        """Inject synthetic fare anomaly events (for Chaos Simulator)."""
        import random
        for _ in range(10):
            event = TelemetryEvent(
                event_id=str(uuid.uuid4())[:8],
                module="m5_fare_predictor",
                latency_ms=random.uniform(100, 400),
                schema_valid=True,
                extra={
                    "predicted_fare": random.uniform(5000, 25000),
                    "base_fare": 350,
                    "anomaly_flag": True,
                    "chaos_injected": True,
                },
            )
            self.emit(event)

    def seed_baseline(self, count: int = 50) -> int:
        """Seed realistic baseline telemetry events across all 7 modules (M1–M7)."""
        import random
        modules_cfg = [
            ("m1_ticket_predictor", "/api/v1/predict/confirmation", (25.0, 65.0), 0.88),
            ("m2_train_recommender", "/api/v1/recommend/trains", (35.0, 75.0), 0.94),
            ("m3_delay_predictor", "/api/v1/predict/delay", (30.0, 80.0), 0.85),
            ("m4_travel_planner", "/api/v1/plan/travel", (45.0, 110.0), 0.92),
            ("m5_fare_predictor", "/api/v1/predict/fare", (20.0, 55.0), 0.96),
            ("m6_support_assistant", "/api/v1/support/query", (180.0, 420.0), 0.94),
            ("m7_fraud_detector", "/api/v1/fraud/assess", (15.0, 40.0), 0.99),
        ]

        events_added = 0
        now = datetime.utcnow()
        for _ in range(count):
            mod, endpoint, lat_range, base_conf = random.choice(modules_cfg)
            lat = round(random.uniform(*lat_range), 2)
            conf = round(min(0.99, max(0.70, random.gauss(base_conf, 0.04))), 3)
            event_time = now - timedelta(minutes=random.uniform(1, 45))

            ev_id = f"{mod[:3].upper()}-{str(uuid.uuid4())[:8].upper()}"
            event = TelemetryEvent(
                event_id=ev_id,
                module=mod,
                endpoint=endpoint,
                latency_ms=lat,
                schema_valid=True,
                schema_violations=[],
                confidence_score=conf,
                drift_flag=False,
                bias_flag=False,
                hallucination_flag=False,
                error=None,
                timestamp=event_time,
                llm_mode="intelligent_fallback",
            )
            self.emit(event)

            audit = AuditEntry(
                entry_id=ev_id,
                timestamp=event_time,
                module=mod,
                action="baseline_inference",
                input_summary="seeded_baseline=True",
                output_summary=f"status=success, confidence={conf}",
                governance_flags=[],
                risk_level="Low",
            )
            self.log_audit(audit)
            events_added += 1

        return events_added

    def resolve_all_alerts(self) -> int:
        """Resolve all alerts in the bus."""
        with self._alert_lock:
            count = 0
            for alert in self._alerts:
                if alert.action_status != "Resolved":
                    alert.action_status = "Resolved"
                    alert.resolution_log = "Reset to healthy by operator"
                    count += 1
            return count

    def clear(self) -> None:
        """Clear all events (for testing)."""
        with self._lock:
            self._events.clear()
        with self._alert_lock:
            self._alerts.clear()


# Module-level singleton
telemetry_bus = TelemetryBus()
