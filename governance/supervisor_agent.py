"""
governance/supervisor_agent.py
────────────────────────────────
Pillar 8 – Autonomous AI Supervisory Meta-Agent.

The Supervisor is the "brain" of the governance layer. It:
  • Polls all governance pillars on a configurable schedule
  • Detects anomalies and routes events to RCA engine
  • Triggers automatic rollbacks via lifecycle manager
  • Enforces HITL (Human-in-the-Loop) approval gates for high-risk actions
  • Generates and stores governance reports
  • Maintains a governance action log (audit trail)
  • Exposes a decision API for the Streamlit Control Tower
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from governance.health_monitor import health_monitor
from governance.drift_detector import drift_detector
from governance.bias_auditor import bias_auditor
from governance.hallucination_monitor import hallucination_monitor
from governance.rca_engine import rca_engine
from governance.lifecycle_manager import lifecycle_manager
from governance.report_generator import report_generator
from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_SECONDS = 30          # Background polling interval
AUTO_ROLLBACK_ENABLED = True        # Global auto-rollback toggle
HITL_REQUIRED_FOR = {               # Actions requiring human approval
    "promote_to_production",
    "mass_rollback",
}
ACTION_LOG_SIZE = 500


# ──────────────────────────────────────────────────────────────────────────────
# Governance Action
# ──────────────────────────────────────────────────────────────────────────────

class GovernanceAction:
    def __init__(
        self,
        action_type: str,
        module_id: str,
        trigger: str,
        payload: dict,
        approved: bool = True,
        executor: str = "supervisor",
    ) -> None:
        self.action_type = action_type
        self.module_id = module_id
        self.trigger = trigger
        self.payload = payload
        self.approved = approved
        self.executor = executor
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.outcome: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "module_id": self.module_id,
            "trigger": self.trigger,
            "payload": self.payload,
            "approved": self.approved,
            "executor": self.executor,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Supervisor Agent (Singleton)
# ──────────────────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Autonomous governance supervisor. Orchestrates all governance pillars,
    enforces policies, triggers remediations, and maintains audit trail.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action_log: deque = deque(maxlen=ACTION_LOG_SIZE)
        self._hitl_pending: List[dict] = []    # Actions awaiting human approval
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable] = []

        # Governance state
        self.auto_rollback_enabled = AUTO_ROLLBACK_ENABLED
        self.poll_interval = POLL_INTERVAL_SECONDS
        self._last_report_id: Optional[str] = None
        self._governance_score: float = 100.0   # 0–100 overall governance health score

    # ── Background Polling ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background governance polling loop."""
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="GovernanceSupervisor"
        )
        self._poll_thread.start()

    def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    def _poll_loop(self) -> None:
        """Main governance loop — runs every POLL_INTERVAL_SECONDS."""
        while self._running:
            try:
                self._governance_cycle()
            except Exception as e:
                telemetry_bus.publish(
                    module_id="supervisor",
                    event_type="SUPERVISOR_ERROR",
                    payload={"error": str(e)},
                    severity="CRITICAL",
                )
            time.sleep(self.poll_interval)

    def _governance_cycle(self) -> None:
        """Single governance cycle: poll → analyze → act → report."""

        # 1. Pull recent telemetry events
        recent_events = telemetry_bus.get_recent(n=50)

        # 2. Route critical events to RCA engine
        for event in recent_events:
            if event.get("severity") in ("CRITICAL", "WARNING"):
                rca_engine.ingest_event(event)

        # 3. Check for rollback triggers
        if self.auto_rollback_enabled:
            for event in recent_events:
                result = lifecycle_manager.auto_rollback_on_event(event)
                if result and result.get("success"):
                    self._log_action(
                        GovernanceAction(
                            action_type="AUTO_ROLLBACK",
                            module_id=event.get("module_id", "unknown"),
                            trigger=event.get("event_type", "unknown"),
                            payload=result,
                            executor="supervisor_auto",
                        )
                    )

        # 4. Drift analysis across all models
        drift_results = drift_detector.analyze_all_models()
        for dr in drift_results:
            if dr.get("overall_drift_detected"):
                lifecycle_manager.queue_retraining(dr["model_id"])
                retrain_res = lifecycle_manager.execute_retraining(dr["model_id"])
                self._log_action(
                    GovernanceAction(
                        action_type="AUTONOMOUS_RETRAIN_DEPLOY",
                        module_id=dr["model_id"],
                        trigger="DRIFT_DETECTED",
                        payload=retrain_res,
                        executor="supervisor_auto",
                    )
                )

        # 5. Compute governance score
        self._update_governance_score()

    def _update_governance_score(self) -> None:
        """Recompute 0–100 platform governance health score."""
        score = 100.0
        health = health_monitor.get_platform_health()
        if health.get("platform_status") == "DEGRADED":
            score -= 20
        error_rate = health.get("overall_error_rate_pct", 0)
        score -= min(error_rate * 2, 20)

        open_incidents = rca_engine.get_open_incidents()
        score -= len([i for i in open_incidents if i.get("priority") == "CRITICAL"]) * 10
        score -= len([i for i in open_incidents if i.get("priority") == "HIGH"]) * 5

        lifecycle_st = lifecycle_manager.get_platform_status()
        score -= len(lifecycle_st.get("retraining_queue", [])) * 3

        self._governance_score = max(0.0, round(score, 1))

    # ── Manual / API-Triggered Actions ───────────────────────────────────────

    def trigger_rollback(self, model_id: str, reason: str = "manual") -> dict:
        """Manually trigger a model rollback."""
        result = lifecycle_manager.rollback(model_id, reason=reason)
        action = GovernanceAction(
            action_type="MANUAL_ROLLBACK",
            module_id=model_id,
            trigger=reason,
            payload=result,
            executor="human_operator",
        )
        action.outcome = "SUCCESS" if result.get("success") else "FAILED"
        self._log_action(action)
        return result

    def trigger_report(self) -> dict:
        """Generate a fresh governance report on demand."""
        health_data = health_monitor.get_platform_health()
        drift_data = drift_detector.analyze_all_models()
        bias_data = bias_auditor.audit_all_modules()
        hallucination_data = hallucination_monitor.get_platform_stats()
        incident_data = rca_engine.get_open_incidents()
        lifecycle_data = lifecycle_manager.get_platform_status()
        telemetry_summary = telemetry_bus.get_recent(n=20)

        report = report_generator.generate(
            health_data=health_data,
            drift_data=drift_data,
            bias_data=bias_data,
            hallucination_data=hallucination_data,
            incident_data=incident_data,
            lifecycle_data=lifecycle_data,
            telemetry_summary=telemetry_summary,
        )
        with self._lock:
            self._last_report_id = report.get("report_id")
        return report

    def inject_chaos(self, module_id: str, chaos_type: str) -> dict:
        """
        Inject a controlled fault for resilience testing.
        Supported: latency_spike | error_surge | drift_simulation
        """
        if chaos_type == "latency_spike":
            # Simulate 10 slow requests
            for _ in range(10):
                health_monitor.record_request(module_id, latency_ms=5000.0, success=True)
            msg = f"Injected 10 high-latency (5000ms) requests into {module_id}."
        elif chaos_type == "error_surge":
            for _ in range(8):
                health_monitor.record_request(module_id, latency_ms=200.0, success=False)
            msg = f"Injected 8 errors into {module_id}."
        elif chaos_type == "drift_simulation":
            # Publish a synthetic drift event
            telemetry_bus.publish(
                module_id=module_id,
                event_type="MODEL_DRIFT_DETECTED",
                payload={"simulated": True, "chaos_type": chaos_type},
                severity="WARNING",
            )
            msg = f"Published synthetic drift event for {module_id}."
        else:
            return {"success": False, "error": f"Unknown chaos_type: {chaos_type}"}

        self._log_action(
            GovernanceAction(
                action_type="CHAOS_INJECTION",
                module_id=module_id,
                trigger="manual_chaos_test",
                payload={"chaos_type": chaos_type},
                executor="operator",
            )
        )
        return {"success": True, "message": msg, "module_id": module_id, "chaos_type": chaos_type}

    def resolve_incident(self, incident_id: str) -> dict:
        """Mark an incident as resolved."""
        success = rca_engine.resolve_incident(incident_id)
        return {"success": success, "incident_id": incident_id}

    # ── HITL Gate ─────────────────────────────────────────────────────────────

    def request_hitl_approval(self, action_type: str, payload: dict) -> str:
        """Queue a high-risk action for human approval. Returns request_id."""
        request_id = f"HITL-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        with self._lock:
            self._hitl_pending.append({
                "request_id": request_id,
                "action_type": action_type,
                "payload": payload,
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "approved": None,
            })
        return request_id

    def approve_hitl(self, request_id: str, approved: bool) -> dict:
        """Approve or reject a pending HITL request."""
        with self._lock:
            for req in self._hitl_pending:
                if req["request_id"] == request_id:
                    req["approved"] = approved
                    req["decided_at"] = datetime.now(timezone.utc).isoformat()
                    return {"success": True, "request_id": request_id, "approved": approved}
        return {"success": False, "error": "Request not found."}

    def get_hitl_pending(self) -> List[dict]:
        with self._lock:
            return [r for r in self._hitl_pending if r.get("approved") is None]

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_action(self, action: GovernanceAction) -> None:
        with self._lock:
            self._action_log.append(action.to_dict())

    def get_action_log(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self._action_log)[-n:]

    # ── Dashboard State ───────────────────────────────────────────────────────

    def get_dashboard_state(self) -> dict:
        """Return comprehensive state dict for the Streamlit Control Tower."""
        health = health_monitor.get_platform_health()
        drift_summary = drift_detector.get_summary()
        rca_summary = rca_engine.get_summary()
        lifecycle_status = lifecycle_manager.get_platform_status()
        hallucination_stats = hallucination_monitor.get_platform_stats()
        telemetry_recent = telemetry_bus.get_recent(n=20)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "governance_score": self._governance_score,
            "supervisor_running": self._running,
            "platform_health": health,
            "drift_summary": drift_summary,
            "rca_summary": rca_summary,
            "lifecycle": lifecycle_status,
            "hallucination_stats": hallucination_stats,
            "hitl_pending_count": len(self.get_hitl_pending()),
            "action_log_recent": self.get_action_log(n=10),
            "recent_telemetry": telemetry_recent,
            "last_report_id": self._last_report_id,
        }

    def register_alert_callback(self, callback: Callable) -> None:
        """Register an external alert handler (e.g., Streamlit notification)."""
        with self._lock:
            self._alert_callbacks.append(callback)


# ── Singleton export ──────────────────────────────────────────────────────────
supervisor_agent = SupervisorAgent()
