"""
modules/m4_travel_planner/schema.py
===================================
Pydantic schemas for Module 4: Agentic AI Travel Planner with HITL Gate.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class PlanRequest(BaseModel):
    query: str = Field(..., description="Natural language travel query (e.g., 'Travel from NDLS to CSTM on Friday with budget 2000')")
    user_id: str = Field(default="passenger_default", description="User identifier for session tracking")
    persona: Optional[str] = Field(default="budget", description="Traveler persona: budget, premium, senior, student, family, business")
    max_budget: Optional[float] = Field(default=None, description="Maximum budget in INR")


class PlannedItineraryOption(BaseModel):
    option_id: str
    train_number: str
    train_name: str
    source: str
    destination: str
    departure_time: str
    arrival_time: str
    duration_hours: float
    travel_class: str
    estimated_fare_inr: float
    confirmation_probability: float
    predicted_delay_minutes: float
    confidence_score: float
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    request_id: str
    user_id: str
    status: Literal["PENDING_APPROVAL", "APPROVED", "REJECTED", "REPLANNING"]
    recommended_itinerary: PlannedItineraryOption
    alternative_itineraries: List[PlannedItineraryOption] = Field(default_factory=list)
    total_estimated_cost_inr: float
    agent_reasoning_summary: str
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    governance_telemetry: Optional[GovernanceBadge] = None


class BookingConfirmRequest(BaseModel):
    request_id: str
    selected_option_id: str
    passenger_name: str = Field(default="Demo Passenger")
    passenger_age: int = Field(default=30)
    passenger_gender: str = Field(default="M")
    berth_preference: Optional[str] = Field(default="Lower")


class BookingPayload(BaseModel):
    booking_id: str
    request_id: str
    status: Literal["PREPARATION_APPROVED", "SIMULATED_SUCCESS", "FAILED"]
    train_number: str
    train_name: str
    travel_class: str
    fare_inr: float
    passenger_name: str
    allocated_berth: str
    pnr_number: str
    message: str


class RejectPlanRequest(BaseModel):
    request_id: str
    rejection_reason: str = Field(default="Fare too high or timing inconvenient")
