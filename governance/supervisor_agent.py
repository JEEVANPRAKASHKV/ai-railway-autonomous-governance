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
from typing import Any, Callable, Dict, List, Optional

from governance.health_monitor import health_monitor
from governance.drift_detector import drift_detector
from governance.bias_auditor import bias_auditor
from governance.hallucination_monitor import hallucination_monitor
from governance.rca_engine import rca_engine
from governance.lifecycle_manager import lifecycle_manager
from governance.report_generator import report_generator
from governance.telemetry_bus import telemetry_bus
from governance.performance_monitor import performance_monitor
from governance.audit_store import audit_store


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_SECONDS = 30          # Background polling interval
AUTO_ROLLBACK_ENABLED = True        # Global auto-rollback toggle
HITL_REQUIRED_FOR = {               # Risk-INCREASING actions need a human; risk-reducing ones run automatically
    "promote_to_production",
}
ACTION_LOG_SIZE = 500
MIN_LABELLED_FOR_DRIFT_RETRAIN = 30  # input drift alone: retrain only once there are new labelled outcomes
AUTO_RESOLVE_CYCLES = 2              # an incident closes after its signal has been clear this many cycles


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
        self._seen_event_ids: set = set()
        self._seen_order: deque = deque(maxlen=20000)
        self._clear_streak: Dict[str, int] = {}
        self._promotion_requested: set = set()
        self._deferred_logged: Dict[str, float] = {}
        self._cycles = 0
        self._last_cycle_at: Optional[str] = None
        self._restore()

    def _restore(self) -> None:
        """Reload the action log and undecided approvals from the durable audit store after a restart."""
        try:
            for row in reversed(audit_store.query(kind="governance_action", limit=200)):
                self._action_log.append(row["payload"])
            decided = {r["ref"] for r in audit_store.query(kind="hitl_decision", limit=500)}
            for row in reversed(audit_store.query(kind="hitl_request", limit=100)):
                if row["ref"] not in decided:
                    req = dict(row["payload"]); req["approved"] = None
                    self._hitl_pending.append(req)
                    if req.get("action_type") == "promote_to_production":
                        self._promotion_requested.add(req.get("payload", {}).get("model_id"))
        except Exception:
            pass

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

    def _new_events(self, n: int = 300) -> List[dict]:
        """Events not processed by an earlier cycle (each event is acted on exactly once)."""
        fresh = []
        for ev in telemetry_bus.get_recent(n=n):
            key = ev.get("event_id") or (ev.get("module_id"), ev.get("event_type"), str(ev.get("timestamp")))
            if key in self._seen_event_ids:
                continue
            if len(self._seen_order) == self._seen_order.maxlen:
                self._seen_event_ids.discard(self._seen_order[0])
            self._seen_order.append(key)
            self._seen_event_ids.add(key)
            fresh.append(ev)
        return fresh

    def _governance_cycle(self) -> None:
        """One closed-loop cycle: observe → diagnose → act → verify → report."""
        self._cycles += 1
        self._last_cycle_at = datetime.now(timezone.utc).isoformat()

        # 1. Route NEW warning/critical events to the RCA engine
        for event in self._new_events():
            if event.get("severity") in ("CRITICAL", "WARNING"):
                rca_engine.ingest_event(event)

        # 2. Concept drift / quality degradation from real outcomes → retrain with those outcomes
        degraded = {r["module_id"]: r for r in performance_monitor.check_degradation()}
        for mid, rep in degraded.items():
            if lifecycle_manager.recently_promoted(mid) and self.auto_rollback_enabled:
                res = lifecycle_manager.rollback(mid, reason=f"post-promotion degradation: {rep['status_reason']}")
                if res.get("success"):
                    self._log_action(GovernanceAction("AUTO_ROLLBACK", mid, "PERFORMANCE_DEGRADED_AFTER_PROMOTION",
                                                      res, executor="supervisor_auto"))
                    continue
            lifecycle_manager.queue_retraining(mid)
            res = lifecycle_manager.execute_retraining(mid, reason=f"concept drift: {rep['status_reason']}")
            if res.get("started"):
                self._log_action(GovernanceAction("AUTONOMOUS_RETRAIN", mid, "PERFORMANCE_DEGRADED",
                                                  {**res, "evidence": rep["status_reason"]}, executor="supervisor_auto"))

        # 3. Input drift → retrain only when there is new labelled data to learn from
        drift_results = drift_detector.analyze_all_models()
        for dr in drift_results:
            mid = dr["model_id"]
            if not dr.get("overall_drift_detected") or mid in degraded:
                continue
            lifecycle_manager.queue_retraining(mid)
            n_lab = len(performance_monitor.labelled_rows(mid))
            if n_lab >= MIN_LABELLED_FOR_DRIFT_RETRAIN:
                res = lifecycle_manager.execute_retraining(mid, reason="input drift + new labelled outcomes")
                if res.get("started"):
                    self._log_action(GovernanceAction("AUTONOMOUS_RETRAIN", mid, "DRIFT_DETECTED",
                                                      {**res, "labelled_rows": n_lab}, executor="supervisor_auto"))
            elif time.time() - self._deferred_logged.get(mid, 0) > 600:
                self._deferred_logged[mid] = time.time()
                self._log_action(GovernanceAction(
                    "RETRAIN_DEFERRED", mid, "DRIFT_DETECTED",
                    {"labelled_rows": n_lab, "needed": MIN_LABELLED_FOR_DRIFT_RETRAIN,
                     "why": "Inputs shifted but there are no new labelled outcomes yet — retraining on the same "
                            "historical data would not adapt the model. Queued until outcomes arrive."},
                    executor="supervisor_auto"))

        # 4. Canary analysis: challenger vs champion on live outcomes
        lc = lifecycle_manager.get_platform_status()
        for mid, info in (lc.get("canary") or {}).items():
            cmp_ = performance_monitor.compare_versions(mid, lc["production_versions"].get(mid, "v1.0.0"), info["version"])
            if cmp_["decision"] == "challenger_worse":
                res = lifecycle_manager.stop_canary(mid, reason=f"canary worse on live {cmp_['metric']}")
                self._log_action(GovernanceAction("CANARY_STOPPED", mid, "CANARY_ANALYSIS", {**res, "analysis": cmp_},
                                                  executor="supervisor_auto"))
                self._promotion_requested.discard(mid)
            elif cmp_["decision"] == "challenger_ok" and mid not in self._promotion_requested:
                self.request_promotion(mid, requested_by="supervisor_auto",
                                       evidence={"canary_analysis": cmp_})

        # 5. User-behaviour change (request mix vs training mix)
        drift_detector.behaviour_report()

        # 6. Verify: close incidents whose signal has cleared (closed loop)
        self._auto_resolve(drift_results, degraded)

        # 7. Score
        self._update_governance_score()

    def _signal_cleared(self, inc: dict, drift_by_model: Dict[str, dict], degraded: Dict[str, dict]) -> bool:
        mid, et = inc.get("module_id"), inc.get("primary_event_type")
        if et == "MODEL_DRIFT_DETECTED":
            d = drift_by_model.get(mid)
            return bool(d) and not d.get("overall_drift_detected")
        if et == "PERFORMANCE_DEGRADED":
            return mid not in degraded
        if et in ("SLA_BREACH", "CIRCUIT_OPEN"):
            h = health_monitor.get_module_health(mid)
            return bool(h.get("sla", {}).get("compliant", True)) and h.get("circuit_state", "CLOSED") == "CLOSED"
        if et == "BIAS_DETECTED":
            try:
                return not bias_auditor.audit_module(mid).get("bias_detected", False)
            except Exception:
                return False
        if et in ("DATA_QUALITY_VIOLATION", "HALLUCINATION_DETECTED", "PREDICTION_FAILURE"):
            last = inc.get("last_seen") or inc.get("timestamp")
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
            except Exception:
                age = 0
            return age > 600   # no recurrence for 10 minutes
        return False

    def _auto_resolve(self, drift_results: List[dict], degraded: Dict[str, dict]) -> None:
        drift_by_model = {d["model_id"]: d for d in drift_results}
        for inc in rca_engine.get_open_incidents():
            iid = inc["incident_id"]
            if self._signal_cleared(inc, drift_by_model, degraded):
                self._clear_streak[iid] = self._clear_streak.get(iid, 0) + 1
            else:
                self._clear_streak[iid] = 0
            if self._clear_streak[iid] >= AUTO_RESOLVE_CYCLES:
                if rca_engine.resolve_incident(iid, resolution=f"Auto-resolved: signal clear for {AUTO_RESOLVE_CYCLES} cycles"):
                    self._log_action(GovernanceAction("AUTO_RESOLVED", inc.get("module_id", "?"),
                                                      inc.get("primary_event_type", "?"),
                                                      {"incident_id": iid}, executor="supervisor_auto"))
                self._clear_streak.pop(iid, None)

    def _update_governance_score(self) -> None:
        """Recompute 0–100 platform governance health score."""
        score = 100.0
        health = health_monitor.get_platform_health()
        if health.get("platform_status") == "DEGRADED":
            score -= 20
        error_rate = health.get("overall_error_rate_pct", 0)
        score -= min(error_rate * 2, 20)

        open_incidents = rca_engine.get_open_incidents()
        n_crit = len([i for i in open_incidents if i.get("priority") == "CRITICAL"])
        n_high = len([i for i in open_incidents if i.get("priority") == "HIGH"])
        n_med = len([i for i in open_incidents if i.get("priority") == "MEDIUM"])
        # Capped so a burst of incidents degrades the score without flat-lining it.
        score -= min(n_crit * 10, 35) + min(n_high * 5, 15) + min(n_med * 2, 6)

        lifecycle_st = lifecycle_manager.get_platform_status()
        score -= min(len(lifecycle_st.get("retraining_queue", [])) * 3, 9)
        try:
            score -= min(sum(1 for r in performance_monitor.all_reports() if r["degraded"]) * 8, 16)
        except Exception:
            pass

        self._governance_score = max(0.0, round(score, 1))

    # ── Manual / API-Triggered Actions ───────────────────────────────────────

    def trigger_rollback(self, model_id: str, reason: str = "manual", actor: str = "human_operator") -> dict:
        """Manually trigger a model rollback (risk-reducing: runs immediately, fully audited)."""
        result = lifecycle_manager.rollback(model_id, reason=reason)
        action = GovernanceAction(
            action_type="MANUAL_ROLLBACK",
            module_id=model_id,
            trigger=reason,
            payload=result,
            executor=actor,
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

    def request_hitl_approval(self, action_type: str, payload: dict, requested_by: str = "supervisor") -> str:
        """Queue a high-risk action for human approval. Returns request_id."""
        from uuid import uuid4
        request_id = f"HITL-{uuid4().hex[:8].upper()}"
        req = {
            "request_id": request_id,
            "action_type": action_type,
            "payload": payload,
            "requested_by": requested_by,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "approved": None,
        }
        with self._lock:
            self._hitl_pending.append(req)
        audit_store.append("hitl_request", req, module=payload.get("model_id", ""), ref=request_id, actor=requested_by)
        telemetry_bus.publish(module_id=payload.get("model_id", "supervisor"), event_type="HITL_APPROVAL_REQUESTED",
                              payload={"request_id": request_id, "action_type": action_type}, severity="INFO")
        return request_id

    def request_promotion(self, model_id: str, requested_by: str = "operator", evidence: Optional[dict] = None) -> dict:
        """Promotion of a challenger to 100% traffic is risk-increasing → it waits for a human decision."""
        lc = lifecycle_manager.get_platform_status()
        ch = lc.get("challenger_versions", {}).get(model_id)
        if not ch:
            return {"success": False, "error": "No challenger registered for this model."}
        for r in self.get_hitl_pending():
            if r["action_type"] == "promote_to_production" and r["payload"].get("model_id") == model_id:
                return {"success": True, "status": "PENDING_APPROVAL", "request_id": r["request_id"],
                        "note": "A promotion request for this model is already waiting for approval."}
        rid = self.request_hitl_approval("promote_to_production",
                                         {"model_id": model_id, "version": ch,
                                          "from_version": lc.get("production_versions", {}).get(model_id),
                                          "evidence": evidence or {}}, requested_by=requested_by)
        self._promotion_requested.add(model_id)
        self._log_action(GovernanceAction("PROMOTION_REQUESTED", model_id, "HITL_GATE",
                                          {"request_id": rid, "version": ch}, executor=requested_by))
        return {"success": True, "status": "PENDING_APPROVAL", "request_id": rid, "version": ch}

    def approve_hitl(self, request_id: str, approved: bool, decided_by: str = "human_approver",
                     note: str = "") -> dict:
        """Record a human decision on a pending request and, if approved, execute it."""
        with self._lock:
            req = next((r for r in self._hitl_pending if r["request_id"] == request_id), None)
            if req is None:
                return {"success": False, "error": "Request not found."}
            if req.get("approved") is not None:
                return {"success": False, "error": f"Request already decided ({'approved' if req['approved'] else 'rejected'})."}
            req["approved"] = bool(approved)
            req["decided_at"] = datetime.now(timezone.utc).isoformat()
            req["decided_by"] = decided_by
        outcome: dict = {}
        if approved and req["action_type"] == "promote_to_production":
            mid = req["payload"]["model_id"]
            current = lifecycle_manager.get_platform_status().get("challenger_versions", {}).get(mid)
            if current != req["payload"].get("version"):
                outcome = {"success": False, "error": "The challenger changed since the request was made; not executed."}
            else:
                outcome = lifecycle_manager.promote_challenger(mid)
            self._promotion_requested.discard(mid)
        elif not approved and req["action_type"] == "promote_to_production":
            self._promotion_requested.discard(req["payload"]["model_id"])
        req["outcome"] = outcome
        audit_store.append("hitl_decision", {"request_id": request_id, "approved": bool(approved), "note": note,
                                             "action_type": req["action_type"], "outcome": outcome},
                           module=req["payload"].get("model_id", ""), ref=request_id, actor=decided_by)
        self._log_action(GovernanceAction("HITL_APPROVED" if approved else "HITL_REJECTED",
                                          req["payload"].get("model_id", "?"), req["action_type"],
                                          {"request_id": request_id, "outcome": outcome, "note": note},
                                          executor=decided_by))
        return {"success": True, "request_id": request_id, "approved": bool(approved), "execution": outcome}

    def get_hitl_pending(self) -> List[dict]:
        with self._lock:
            return [r for r in self._hitl_pending if r.get("approved") is None]

    def get_hitl_history(self, n: int = 20) -> List[dict]:
        with self._lock:
            return [r for r in self._hitl_pending if r.get("approved") is not None][-n:]

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_action(self, action: GovernanceAction) -> None:
        d = action.to_dict()
        with self._lock:
            self._action_log.append(d)
        try:
            audit_store.append("governance_action", d, module=action.module_id, ref=action.action_type,
                               actor=action.executor)
        except Exception:
            pass

    def get_action_log(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self._action_log)[-n:]

    # ── Dashboard State ───────────────────────────────────────────────────────

    def get_dashboard_state(self) -> dict:
        """Return comprehensive state dict for the Streamlit Control Tower."""
        health = health_monitor.get_platform_health()
        drift_summary = drift_detector.get_summary()
        bias_summary = bias_auditor.get_summary()
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
            "bias_summary": bias_summary,
            "rca_summary": rca_summary,
            "lifecycle": lifecycle_status,
            "hallucination_stats": hallucination_stats,
            "hitl_pending_count": len(self.get_hitl_pending()),
            "hitl_pending": self.get_hitl_pending(),
            "performance": performance_monitor.all_reports(),
            "cycles_completed": self._cycles,
            "last_cycle_at": self._last_cycle_at,
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
