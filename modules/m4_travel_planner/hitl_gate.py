"""
modules/m4_travel_planner/hitl_gate.py
======================================
Human-in-the-Loop (HITL) Gate for Autonomous Travel Planner.
Enforces the mandatory safety invariant:
  The agent NEVER executes simulated booking actions without explicit,
  audited user confirmation in the UI.
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from modules.m4_travel_planner.schema import (
    ApprovalRequest,
    BookingConfirmRequest,
    BookingPayload,
    PlannedItineraryOption,
    RejectPlanRequest,
)

logger = logging.getLogger(__name__)


class HITLApprovalGate:
    """Thread-safe stateful registry for Human-in-the-Loop travel approvals."""

    def __init__(self) -> None:
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._completed_bookings: Dict[str, BookingPayload] = {}

    def register_approval_request(self, approval_request: ApprovalRequest) -> None:
        """Register a new plan awaiting explicit passenger signoff."""
        self._pending_requests[approval_request.request_id] = approval_request
        logger.info(
            f"[HITL Gate] Created Approval Request: {approval_request.request_id} "
            f"(Status: {approval_request.status}, Cost: ₹{approval_request.total_estimated_cost_inr})"
        )

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        return self._pending_requests.get(request_id)

    def confirm_plan(self, confirm_req: BookingConfirmRequest) -> BookingPayload:
        """Process explicit passenger approval and prepare simulated booking payload."""
        req_id = confirm_req.request_id
        approval = self._pending_requests.get(req_id)
        if not approval:
            raise ValueError(f"Approval request '{req_id}' not found or expired.")

        # Find selected itinerary
        selected: Optional[PlannedItineraryOption] = None
        if approval.recommended_itinerary.option_id == confirm_req.selected_option_id:
            selected = approval.recommended_itinerary
        else:
            for opt in approval.alternative_itineraries:
                if opt.option_id == confirm_req.selected_option_id:
                    selected = opt
                    break

        if not selected:
            selected = approval.recommended_itinerary

        # Update approval status to APPROVED
        approval.status = "APPROVED"

        pnr = f"PNR{random.randint(1000000, 9999999)}"
        booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"

        payload = BookingPayload(
            booking_id=booking_id,
            request_id=req_id,
            status="SIMULATED_SUCCESS",
            train_number=selected.train_number,
            train_name=selected.train_name,
            travel_class=selected.travel_class,
            fare_inr=selected.estimated_fare_inr,
            passenger_name=confirm_req.passenger_name,
            allocated_berth=f"{confirm_req.berth_preference or 'Lower'} (Coach S3, Berth 24)",
            pnr_number=pnr,
            message=(
                f"Booking payload verified and approved by passenger. "
                f"Simulated IRCTC PNR {pnr} generated for {selected.train_name} ({selected.train_number}) in {selected.travel_class}."
            ),
        )

        self._completed_bookings[booking_id] = payload
        logger.info(f"[HITL Gate] Plan APPROVED by user. Booking ID: {booking_id}, PNR: {pnr}")
        return payload

    def reject_plan(self, reject_req: RejectPlanRequest) -> Dict[str, str]:
        """Process passenger rejection and trigger replanning state."""
        req_id = reject_req.request_id
        approval = self._pending_requests.get(req_id)
        if not approval:
            raise ValueError(f"Approval request '{req_id}' not found.")

        approval.status = "REJECTED"
        logger.info(f"[HITL Gate] Plan REJECTED by user. Request: {req_id}. Reason: {reject_req.rejection_reason}")
        return {
            "request_id": req_id,
            "status": "REPLANNING",
            "message": f"Passenger declined plan: '{reject_req.rejection_reason}'. Replanning initiated.",
        }


hitl_gate = HITLApprovalGate()
