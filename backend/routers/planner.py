"""
backend/routers/planner.py
==========================
FastAPI router for Module 4: Agentic AI Travel Planner with HITL Gate.
"""

from __future__ import annotations

from typing import Dict
from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_hitl_gate, get_planner_agent
from modules.m4_travel_planner.agent import TravelPlannerAgent
from modules.m4_travel_planner.hitl_gate import HITLApprovalGate
from modules.m4_travel_planner.schema import (
    ApprovalRequest,
    BookingConfirmRequest,
    BookingPayload,
    PlanRequest,
    RejectPlanRequest,
)

router = APIRouter(prefix="/agent", tags=["Module 4: Agentic Travel Planner"])


@router.post(
    "/plan",
    response_model=ApprovalRequest,
    summary="Generate multi-agent travel plan and stop at the Human-in-the-Loop (HITL) gate",
)
def plan_travel_itinerary(
    request: PlanRequest,
    agent: TravelPlannerAgent = Depends(get_planner_agent),
) -> ApprovalRequest:
    try:
        return agent.plan(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent travel planning failed: {exc}")


@router.post(
    "/confirm",
    response_model=BookingPayload,
    summary="Explicit passenger confirmation executing simulated ticket booking preparation",
)
def confirm_itinerary_booking(
    request: BookingConfirmRequest,
    gate: HITLApprovalGate = Depends(get_hitl_gate),
) -> BookingPayload:
    try:
        return gate.confirm_plan(request)
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Booking confirmation failed: {exc}")


@router.post(
    "/reject",
    summary="Passenger rejection triggering replanning state",
)
def reject_itinerary(
    request: RejectPlanRequest,
    gate: HITLApprovalGate = Depends(get_hitl_gate),
) -> Dict[str, str]:
    try:
        return gate.reject_plan(request)
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rejection handling failed: {exc}")


@router.get(
    "/request/{request_id}",
    response_model=ApprovalRequest,
    summary="Retrieve current state of an active approval request",
)
def get_approval_status(
    request_id: str,
    gate: HITLApprovalGate = Depends(get_hitl_gate),
) -> ApprovalRequest:
    approval = gate.get_request(request_id)
    if not approval:
        raise HTTPException(status_code=404, detail=f"Approval request '{request_id}' not found.")
    return approval
