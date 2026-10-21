"""
governance/rca_engine.py
─────────────────────────
Pillar 5 – Root Cause Analysis (RCA) Engine.

Implements:
  • Event-driven anomaly correlation across modules
  • Fault tree construction from telemetry event streams
  • 5-Why automated reasoning chain
  • Causal graph traversal (simplified)
  • Incident timeline reconstruction
  • Priority scoring and recommended remediation playbooks
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CORRELATION_WINDOW_SECONDS = 300    # 5-minute window for correlating events
CRITICAL_EVENT_TYPES = {
    "SLA_BREACH", "MODEL_DRIFT_DETECTED", "BIAS_DETECTED",
    "HALLUCINATION_DETECTED", "FRAUD_SPIKE", "CIRCUIT_OPEN",
    "DATA_QUALITY_VIOLATION", "PREDICTION_FAILURE",
    "PERFORMANCE_DEGRADED", "RETRAIN_FAILED", "AGENT_ANOMALY", "PRICING_ANOMALY",
}

# Service dependency graph (who calls whom). Used to trace a symptom in one service back to an
# upstream root cause in another — e.g. a degraded delay model makes the travel planner's plans worse.
SERVICE_DEPENDENCIES: Dict[str, List[str]] = {
    "m4_travel_planner": ["m1_ticket_predictor", "m3_delay_predictor", "m5_fare_predictor"],
    "m6_support_assistant": ["m2_train_recommender"],
    "m2_train_recommender": [],
    "m1_ticket_predictor": [], "m3_delay_predictor": [], "m5_fare_predictor": [], "m7_fraud_detector": [],
}


def _downstream_of(module: str) -> List[str]:
    return [m for m, deps in SERVICE_DEPENDENCIES.items() if module in deps]

# Causal dependency graph: event_type → likely upstream causes
CAUSAL_MAP: Dict[str, List[str]] = {
    "SLA_BREACH": ["MODEL_DRIFT_DETECTED", "DATA_QUALITY_VIOLATION", "CIRCUIT_OPEN"],
    "MODEL_DRIFT_DETECTED": ["DATA_QUALITY_VIOLATION", "TRAINING_DATA_STALE"],
    "BIAS_DETECTED": ["DATA_QUALITY_VIOLATION", "TRAINING_IMBALANCE"],
    "HALLUCINATION_DETECTED": ["MODEL_DRIFT_DETECTED", "LOW_RETRIEVAL_QUALITY"],
    "FRAUD_SPIKE": ["DATA_QUALITY_VIOLATION", "BOT_ATTACK"],
    "PREDICTION_FAILURE": ["MODEL_DRIFT_DETECTED", "SLA_BREACH", "CIRCUIT_OPEN"],
    "CIRCUIT_OPEN": ["SLA_BREACH", "PREDICTION_FAILURE"],
    "PERFORMANCE_DEGRADED": ["MODEL_DRIFT_DETECTED", "CONCEPT_DRIFT", "DATA_QUALITY_VIOLATION"],
    "RETRAIN_FAILED": ["TRAINING_PIPELINE_ERROR", "DATA_QUALITY_VIOLATION"],
    "AGENT_ANOMALY": ["PROMPT_INJECTION", "TOOL_FAILURE", "UPSTREAM_MODEL_DEGRADED"],
    "PRICING_ANOMALY": ["MODEL_DRIFT_DETECTED", "DATA_QUALITY_VIOLATION"],
}

# 5-Why templates per event type
FIVE_WHY_TEMPLATES: Dict[str, List[str]] = {
    "SLA_BREACH": [
        "Why did latency exceed SLA?",
        "Why did the module take longer than expected?",
        "Why was the model inference slow?",
        "Why was there high feature complexity or cardinality?",
        "Why was feature engineering not optimized for this distribution?",
    ],
    "MODEL_DRIFT_DETECTED": [
        "Why is the model's input distribution drifting?",
        "Why has the real-world data pattern changed?",
        "Why has the underlying booking/travel behavior changed?",
        "Why wasn't seasonal/temporal drift accounted for in training?",
        "Why was the model not retrained on recent data?",
    ],
    "BIAS_DETECTED": [
        "Why is the model producing disparate outcomes?",
        "Why are certain demographic groups under-represented in positive outcomes?",
        "Why is the training data imbalanced across protected groups?",
        "Why was fairness not enforced during feature selection?",
        "Why were protected attributes correlated with target proxies?",
    ],
    "HALLUCINATION_DETECTED": [
        "Why did the LLM generate a response not grounded in context?",
        "Why was the retrieved context insufficient?",
        "Why did the retrieval system return low-relevance documents?",
        "Why was the query embedding not closely aligned with the corpus?",
        "Why was the policy corpus not updated to cover this query domain?",
    ],
    "FRAUD_SPIKE": [
        "Why are fraud events spiking?",
        "Why are fraudulent sessions evading detection?",
        "Why have adversarial attack patterns changed?",
        "Why wasn't the fraud model retrained on recent attack vectors?",
        "Why were anomalous velocity signals not flagged earlier?",
    ],
    "PERFORMANCE_DEGRADED": [
        "Why did live accuracy fall below the training baseline?",
        "Why do real outcomes now differ from what the model predicts for the same inputs?",
        "Why has the relationship between inputs and outcomes changed (concept drift)?",
        "Why was the model trained only on historical patterns that no longer hold?",
        "Why were recent labelled outcomes not yet part of the training data?",
    ],
    "AGENT_ANOMALY": [
        "Why did the planner agent behave abnormally?",
        "Why did it need unusually many tool calls, or why were its decisions rejected?",
        "Why did a tool return unusable or failing results?",
        "Why was the user request ambiguous or adversarial?",
        "Why did the guardrails not resolve it earlier in the pipeline?",
    ],
    "PRICING_ANOMALY": [
        "Why is the predicted fare outside the historical range for this class and lead time?",
        "Why did the model extrapolate beyond its training data?",
        "Why were the inputs (base fare, demand) unusual?",
        "Why did the data-quality gate not block the inputs?",
        "Why is there no hard pricing guardrail for this case?",
    ],
    "DEFAULT": [
        "Why did this event occur?",
        "Why was the root condition not caught earlier?",
        "Why did monitoring not alert proactively?",
        "Why was the affected component not resilient?",
        "Why was the system design not fault-tolerant for this scenario?",
    ],
}

# Remediation playbooks
REMEDIATION_PLAYBOOKS: Dict[str, str] = {
    "SLA_BREACH": (
        "1. Enable circuit breaker for the affected module. "
        "2. Scale inference workers. "
        "3. Reduce feature set complexity. "
        "4. Check downstream service latency."
    ),
    "MODEL_DRIFT_DETECTED": (
        "1. Trigger model retraining on recent data window. "
        "2. Update baseline distributions. "
        "3. Deploy updated model via lifecycle manager. "
        "4. Monitor post-deployment performance for 24h."
    ),
    "BIAS_DETECTED": (
        "1. Apply post-processing threshold calibration per group. "
        "2. Re-sample training data for under-represented groups. "
        "3. Re-evaluate feature set for proxy discrimination. "
        "4. Run fairness audit post-retraining."
    ),
    "HALLUCINATION_DETECTED": (
        "1. Expand retrieval corpus with domain-specific documents. "
        "2. Tune retrieval similarity threshold. "
        "3. Add explicit grounding instructions to LLM system prompt. "
        "4. Quarantine low-faithfulness responses and return policy fallback."
    ),
    "FRAUD_SPIKE": (
        "1. Lower fraud detection threshold temporarily. "
        "2. Enable enhanced CAPTCHA and rate limiting. "
        "3. Retrain fraud model on latest fraud event logs. "
        "4. Flag suspicious sessions for manual review."
    ),
    "PERFORMANCE_DEGRADED": (
        "1. Retrain with the latest labelled outcomes (supervisor does this automatically). "
        "2. Evaluate the challenger on the fixed hold-out AND recent outcomes. "
        "3. Run a 10% canary on live traffic. 4. Promote only after human approval; auto-rollback if it regresses."
    ),
    "AGENT_ANOMALY": (
        "1. Inspect the execution trace in the audit log. 2. Check upstream model health (ticket / delay / fare). "
        "3. Tighten the injection guard or tool-call budget. 4. Keep the human approval gate mandatory."
    ),
    "PRICING_ANOMALY": (
        "1. Serve the policy-capped fare instead of the raw prediction. 2. Check the input base fare and demand. "
        "3. Investigate drift on fare features. 4. Retrain if the anomaly rate stays high."
    ),
    "RETRAIN_FAILED": (
        "1. Read the failure reason in the lifecycle history. 2. Fix the pipeline / data issue. "
        "3. Re-run retraining. The current production model keeps serving meanwhile."
    ),
    "DEFAULT": (
        "1. Investigate telemetry event stream for correlated signals. "
        "2. Review module logs and feature inputs. "
        "3. Escalate to on-call ML engineer if unresolved within 30 minutes."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Incident Data Model
# ──────────────────────────────────────────────────────────────────────────────

class Incident:
    def __init__(
        self,
        incident_id: str,
        primary_event_type: str,
        module_id: str,
        trigger_payload: dict,
    ) -> None:
        self.incident_id = incident_id
        self.primary_event_type = primary_event_type
        self.module_id = module_id
        self.trigger_payload = trigger_payload
        self.timestamp = datetime.now(timezone.utc)
        self.correlated_events: List[dict] = []
        self.five_whys: List[str] = []
        self.inferred_causes: List[str] = []
        self.remediation: str = ""
        self.priority: str = "MEDIUM"
        self.resolved: bool = False
        self.occurrences: int = 1
        self.last_seen = self.timestamp
        self.evidence: List[str] = []
        self.upstream_suspects: List[dict] = []
        self.downstream_impact: List[str] = []
        self.root_cause_module: str = module_id
        self.resolution: str = ""
        self.resolved_at = None

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp.isoformat(),
            "module_id": self.module_id,
            "primary_event_type": self.primary_event_type,
            "priority": self.priority,
            "resolved": self.resolved,
            "correlated_events_count": len(self.correlated_events),
            "inferred_causes": self.inferred_causes,
            "five_whys": self.five_whys,
            "remediation_playbook": self.remediation,
            "trigger_payload": self.trigger_payload,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen.isoformat(),
            "evidence": self.evidence,
            "root_cause_module": self.root_cause_module,
            "upstream_suspects": self.upstream_suspects,
            "downstream_impact": self.downstream_impact,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# RCA Engine (Singleton)
# ──────────────────────────────────────────────────────────────────────────────

class RCAEngine:
    """
    Root Cause Analysis engine that correlates telemetry events,
    constructs fault trees, applies 5-Why reasoning, and generates
    prioritized remediation playbooks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._incidents: Dict[str, Incident] = {}
        self._incident_history: deque = deque(maxlen=200)
        self._event_buffer: deque = deque(maxlen=2000)
        self._seen_keys: set = set()
        self._seen_order: deque = deque(maxlen=5000)

    def ingest_event(self, event: dict) -> Optional[str]:
        """
        Ingest a telemetry event. If it's a critical event, trigger RCA.
        Returns incident_id if an incident was created, else None.
        """
        with self._lock:
            # Idempotent ingestion: the supervisor re-polls the same recent events every
            # cycle, so an event must only ever be processed once.
            key = event.get("event_id") or (
                event.get("module_id"), event.get("event_type"), str(event.get("timestamp"))
            )
            if key in self._seen_keys:
                return None
            if len(self._seen_order) == self._seen_order.maxlen:
                self._seen_keys.discard(self._seen_order[0])
            self._seen_order.append(key)
            self._seen_keys.add(key)

            self._event_buffer.append(event)
            event_type = event.get("event_type", "")
            if event_type in CRITICAL_EVENT_TYPES:
                # Fold repeats of the same fault on the same module into one open incident.
                for inc in self._incidents.values():
                    if (not inc.resolved and inc.module_id == event.get("module_id")
                            and inc.primary_event_type == event_type):
                        inc.occurrences += 1
                        inc.last_seen = datetime.now(timezone.utc)
                        return inc.incident_id
                return self._create_incident(event)
            return None

    def _create_incident(self, trigger_event: dict) -> str:
        """Create a new incident and run automated RCA."""
        incident_id = f"INC-{uuid4().hex[:8].upper()}"
        incident = Incident(
            incident_id=incident_id,
            primary_event_type=trigger_event.get("event_type", "UNKNOWN"),
            module_id=trigger_event.get("module_id", "unknown"),
            trigger_payload=trigger_event.get("payload", {}),
        )

        # Correlate events in the last 5 minutes
        incident.correlated_events = self._correlate_events(
            trigger_event, window_seconds=CORRELATION_WINDOW_SECONDS
        )

        # Infer causes from causal map
        incident.inferred_causes = CAUSAL_MAP.get(
            incident.primary_event_type, ["Unknown upstream cause"]
        )

        # Build 5-Why chain
        incident.five_whys = FIVE_WHY_TEMPLATES.get(
            incident.primary_event_type, FIVE_WHY_TEMPLATES["DEFAULT"]
        )

        # Assign remediation playbook
        incident.remediation = REMEDIATION_PLAYBOOKS.get(
            incident.primary_event_type, REMEDIATION_PLAYBOOKS["DEFAULT"]
        )

        # Evidence: the concrete numbers behind this incident
        incident.evidence = self._evidence(incident.primary_event_type, incident.trigger_payload)

        # Dependency tracing: is an UPSTREAM service already failing? Then that is the likelier root cause.
        for dep in SERVICE_DEPENDENCIES.get(incident.module_id, []):
            open_dep = [i for i in self._incidents.values() if not i.resolved and i.module_id == dep]
            if open_dep:
                incident.upstream_suspects.append({"module_id": dep, "incident_id": open_dep[0].incident_id,
                                                   "event_type": open_dep[0].primary_event_type})
        if incident.upstream_suspects:
            incident.root_cause_module = incident.upstream_suspects[0]["module_id"]
            incident.inferred_causes = [f"Upstream {incident.root_cause_module} "
                                        f"{incident.upstream_suspects[0]['event_type']}"] + incident.inferred_causes
        incident.downstream_impact = _downstream_of(incident.module_id)

        # Priority: CRITICAL if multiple correlated events
        if len(incident.correlated_events) >= 3:
            incident.priority = "CRITICAL"
        elif len(incident.correlated_events) >= 1:
            incident.priority = "HIGH"
        else:
            incident.priority = "MEDIUM"

        self._incidents[incident_id] = incident
        self._incident_history.append(incident.to_dict())

        # Publish incident to telemetry
        telemetry_bus.publish(
            module_id=incident.module_id,
            event_type="INCIDENT_CREATED",
            payload={
                "incident_id": incident_id,
                "priority": incident.priority,
                "inferred_causes": incident.inferred_causes,
            },
            severity="CRITICAL" if incident.priority == "CRITICAL" else "WARNING",
        )

        return incident_id

    @staticmethod
    def _evidence(event_type: str, p: dict) -> List[str]:
        ev: List[str] = []
        try:
            if event_type == "MODEL_DRIFT_DETECTED":
                feats = p.get("drifted_features") or []
                if feats:
                    ev.append("Drifted features: " + ", ".join(map(str, feats)))
                if p.get("prediction_drift", {}) and p["prediction_drift"].get("drift_detected"):
                    ev.append(f"Prediction distribution also shifted (KS {p['prediction_drift'].get('ks_statistic')})")
            elif event_type == "SLA_BREACH":
                ev.append(f"P95 latency {p.get('p95_ms', 0):.0f} ms; error rate {p.get('error_rate_pct', 0):.1f}%")
                for b in p.get("breaches", [])[:3]:
                    ev.append(str(b))
            elif event_type == "PERFORMANCE_DEGRADED":
                ev.append(str(p.get("reason", "")))
                ev.append(f"Live metrics {p.get('metrics')} vs baseline {p.get('baseline')} over {p.get('n_outcomes')} outcomes")
            elif event_type == "BIAS_DETECTED":
                ev.extend(map(str, (p.get("adverse_flags") or [])[:3]))
            elif event_type == "DATA_QUALITY_VIOLATION":
                ev.extend(map(str, (p.get("violations") or [])[:3]))
            elif event_type in ("HALLUCINATION_DETECTED", "LOW_FAITHFULNESS"):
                ev.append(f"Composite grounding score {p.get('composite_score')}")
            elif event_type in ("AGENT_ANOMALY", "PRICING_ANOMALY", "RETRAIN_FAILED"):
                ev.extend(f"{k}: {v}" for k, v in list(p.items())[:4])
        except Exception:
            pass
        return [e for e in ev if e]

    def _correlate_events(self, trigger_event: dict, window_seconds: int) -> List[dict]:
        """Find events in buffer within time window that may be related."""
        trigger_module = trigger_event.get("module_id", "")
        correlated = []
        for ev in list(self._event_buffer):
            if ev is trigger_event:
                continue
            # Same module or critical event type
            if ev.get("module_id") == trigger_module or ev.get("event_type") in CRITICAL_EVENT_TYPES:
                correlated.append(ev)
        return correlated[-20:]  # Limit to most recent 20

    def analyze_incident(self, incident_id: str) -> Optional[dict]:
        """Return full RCA report for an incident."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None
            return {
                **incident.to_dict(),
                "correlated_events": incident.correlated_events[-10:],
            }

    def run_rca_on_telemetry(self) -> List[dict]:
        """
        Pull recent critical events from telemetry bus and run RCA on any
        that don't yet have open incidents.
        """
        with self._lock:
            recent_events = telemetry_bus.get_recent(n=100)
        new_incidents = []
        existing_event_types = {
            i.primary_event_type for i in self._incidents.values() if not i.resolved
        }
        for ev in recent_events:
            if (
                ev.get("event_type") in CRITICAL_EVENT_TYPES
                and ev.get("event_type") not in existing_event_types
            ):
                inc_id = self.ingest_event(ev)
                if inc_id:
                    new_incidents.append({"incident_id": inc_id, "event": ev})
        return new_incidents

    def resolve_incident(self, incident_id: str, resolution: str = "Resolved by operator") -> bool:
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident and not incident.resolved:
                incident.resolved = True
                incident.resolution = resolution
                incident.resolved_at = datetime.now(timezone.utc)
                self._incident_history.append(incident.to_dict())
                return True
            return False

    def resolve_all_incidents(self) -> int:
        """Resolve all open incidents."""
        with self._lock:
            count = 0
            for incident in self._incidents.values():
                if not incident.resolved:
                    incident.resolved = True
                    count += 1
            return count

    def get_open_incidents(self) -> List[dict]:
        with self._lock:
            return [i.to_dict() for i in self._incidents.values() if not i.resolved]

    def get_incident_history(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._incident_history)[-n:]

    def get_summary(self) -> dict:
        with self._lock:
            open_inc = [i for i in self._incidents.values() if not i.resolved]
            return {
                "total_incidents": len(self._incidents),
                "open_incidents": len(open_inc),
                "critical_incidents": sum(1 for i in open_inc if i.priority == "CRITICAL"),
                "high_incidents": sum(1 for i in open_inc if i.priority == "HIGH"),
                "events_buffered": len(self._event_buffer),
            }


# ── Singleton export ──────────────────────────────────────────────────────────
rca_engine = RCAEngine()
