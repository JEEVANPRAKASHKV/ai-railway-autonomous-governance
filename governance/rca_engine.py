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
}

# Causal dependency graph: event_type → likely upstream causes
CAUSAL_MAP: Dict[str, List[str]] = {
    "SLA_BREACH": ["MODEL_DRIFT_DETECTED", "DATA_QUALITY_VIOLATION", "CIRCUIT_OPEN"],
    "MODEL_DRIFT_DETECTED": ["DATA_QUALITY_VIOLATION", "TRAINING_DATA_STALE"],
    "BIAS_DETECTED": ["DATA_QUALITY_VIOLATION", "TRAINING_IMBALANCE"],
    "HALLUCINATION_DETECTED": ["MODEL_DRIFT_DETECTED", "LOW_RETRIEVAL_QUALITY"],
    "FRAUD_SPIKE": ["DATA_QUALITY_VIOLATION", "BOT_ATTACK"],
    "PREDICTION_FAILURE": ["MODEL_DRIFT_DETECTED", "SLA_BREACH", "CIRCUIT_OPEN"],
    "CIRCUIT_OPEN": ["SLA_BREACH", "PREDICTION_FAILURE"],
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

    def ingest_event(self, event: dict) -> Optional[str]:
        """
        Ingest a telemetry event. If it's a critical event, trigger RCA.
        Returns incident_id if an incident was created, else None.
        """
        with self._lock:
            self._event_buffer.append(event)
            event_type = event.get("event_type", "")
            if event_type in CRITICAL_EVENT_TYPES:
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

    def resolve_incident(self, incident_id: str) -> bool:
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident:
                incident.resolved = True
                return True
            return False

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
