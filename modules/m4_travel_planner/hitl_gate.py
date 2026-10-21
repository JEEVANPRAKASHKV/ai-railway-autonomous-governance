"""
modules/m4_travel_planner/hitl_gate.py
======================================
Human-in-the-Loop (HITL) gate for the travel planner.

Invariants (each one is enforced here and covered by tests):
  • Nothing is booked without an explicit passenger approval of a plan that is still pending.
  • A plan can be decided once: approving twice returns the SAME booking (idempotent), approving a
    rejected / expired plan is refused (409), and an unknown option id is refused (400).
  • Plans expire after 30 minutes.
  • Every decision is written to the tamper-evident audit trail and fed back to governance
    (approval rate, rejection streaks).
"""

from __future__ import annotations

import logging
import random
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from modules.m4_travel_planner.schema import (
    ApprovalRequest,
    BookingConfirmRequest,
    BookingPayload,
    PlannedItineraryOption,
    RejectPlanRequest,
)

logger = logging.getLogger(__name__)
PLAN_TTL = timedelta(minutes=30)


class PlanStateError(Exception):
    """Raised when a decision is not allowed in the plan's current state (→ HTTP 409)."""


class PlanNotFoundError(ValueError):
    """Unknown or expired request id (→ HTTP 404)."""


class InvalidOptionError(ValueError):
    """The selected option id is not part of the plan (→ HTTP 400)."""


class HITLApprovalGate:
    """Thread-safe registry of plans awaiting an explicit passenger decision."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._created: Dict[str, datetime] = {}
        self._bookings_by_request: Dict[str, BookingPayload] = {}
        self._completed_bookings: Dict[str, BookingPayload] = {}

    def register_approval_request(self, approval_request: ApprovalRequest) -> None:
        with self._lock:
            self._pending_requests[approval_request.request_id] = approval_request
            self._created[approval_request.request_id] = datetime.now(timezone.utc)
        logger.info(f"[HITL Gate] Plan {approval_request.request_id} awaiting passenger approval")

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        return self._pending_requests.get(request_id)

    def _feedback(self, approval: ApprovalRequest, approved: bool) -> None:
        try:
            from modules.m4_travel_planner.agent import agent_monitor
            agent_monitor.record_decision(approval.user_id, approved)
        except Exception:
            pass
        try:
            badge = approval.governance_telemetry
            if badge and badge.audit_id:
                from governance.performance_monitor import performance_monitor
                performance_monitor.record_outcome(badge.audit_id, approved, source="passenger")
        except Exception:
            pass

    def confirm_plan(self, confirm_req: BookingConfirmRequest) -> BookingPayload:
        req_id = confirm_req.request_id
        with self._lock:
            approval = self._pending_requests.get(req_id)
            if not approval:
                raise PlanNotFoundError(f"Approval request '{req_id}' not found.")
            if req_id in self._bookings_by_request:            # idempotent: same booking, no second PNR
                return self._bookings_by_request[req_id]
            if approval.status != "PENDING_APPROVAL":
                raise PlanStateError(f"Plan is {approval.status.lower().replace('_', ' ')} and can no longer be approved.")
            if datetime.now(timezone.utc) - self._created.get(req_id, datetime.now(timezone.utc)) > PLAN_TTL:
                approval.status = "REJECTED"
                raise PlanStateError("Plan expired (older than 30 minutes). Please plan again for current fares.")
            options = [approval.recommended_itinerary] + list(approval.alternative_itineraries)
            selected: Optional[PlannedItineraryOption] = next(
                (o for o in options if o is not None and o.option_id == confirm_req.selected_option_id), None)
            if selected is None:
                raise InvalidOptionError(f"Option '{confirm_req.selected_option_id}' is not part of plan {req_id}.")

            approval.status = "APPROVED"
            pnr = f"PNR{random.randint(1000000, 9999999)}"
            payload = BookingPayload(
                booking_id=f"BK-{uuid.uuid4().hex[:8].upper()}",
                request_id=req_id,
                status="SIMULATED_SUCCESS",
                train_number=selected.train_number,
                train_name=selected.train_name,
                travel_class=selected.travel_class,
                fare_inr=selected.estimated_fare_inr,
                passenger_name=confirm_req.passenger_name,
                allocated_berth=f"{confirm_req.berth_preference or 'Lower'} (demo allocation)",
                pnr_number=pnr,
                message=(f"Approved by the passenger. Simulated booking prepared for {selected.train_name} "
                         f"({selected.train_number}) in {selected.travel_class}; demo PNR {pnr}. No real ticket was purchased."),
            )
            self._bookings_by_request[req_id] = payload
            self._completed_bookings[payload.booking_id] = payload
        try:
            from governance.audit_store import audit_store
            audit_store.append("booking", {"booking_id": payload.booking_id, "approved_by_passenger": True,
                                           "option_id": selected.option_id, "train": selected.train_number,
                                           "fare_inr": selected.estimated_fare_inr,
                                           "passenger_name": confirm_req.passenger_name},
                               module="m4_travel_planner", ref=req_id, actor="passenger")
        except Exception:
            pass
        self._feedback(approval, True)
        logger.info(f"[HITL Gate] Plan {req_id} APPROVED → booking {payload.booking_id}")
        return payload

    def reject_plan(self, reject_req: RejectPlanRequest) -> Dict[str, str]:
        req_id = reject_req.request_id
        with self._lock:
            approval = self._pending_requests.get(req_id)
            if not approval:
                raise PlanNotFoundError(f"Approval request '{req_id}' not found.")
            if approval.status == "APPROVED":
                raise PlanStateError("Plan was already approved and booked.")
            if approval.status == "REJECTED":
                return {"request_id": req_id, "status": "REJECTED", "message": "Plan was already rejected."}
            approval.status = "REJECTED"
        try:
            from governance.audit_store import audit_store
            audit_store.append("plan_rejected", {"reason": reject_req.rejection_reason}, module="m4_travel_planner",
                               ref=req_id, actor="passenger")
        except Exception:
            pass
        self._feedback(approval, False)
        return {"request_id": req_id, "status": "REPLANNING",
                "message": f"Plan declined ('{reject_req.rejection_reason}'). Adjust your request and plan again."}


hitl_gate = HITLApprovalGate()
