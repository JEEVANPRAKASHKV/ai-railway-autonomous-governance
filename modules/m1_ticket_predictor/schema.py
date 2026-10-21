"""
modules/m1_ticket_predictor/schema.py
=====================================
Pydantic schemas for Module 1: Intelligent Ticket Confirmation Predictor.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class TicketPredictRequest(BaseModel):
    train_number: str = Field(..., description="5-digit train number (e.g. '12301')")
    journey_class: str = Field(default="3A", description="Travel class: 1A, 2A, 3A, SL, CC, EC")
    quota: str = Field(default="GN", description="Booking quota: GN, TQ, LADIES, DEFENCE, SENIOR")
    journey_date: date = Field(..., description="Date of journey (YYYY-MM-DD)")
    booking_date: date = Field(default_factory=date.today, description="Booking date (defaults to today)")
    current_wl_number: int = Field(..., ge=1, le=500, description="Current Waitlist or RAC position")
    seniority_flag: bool = Field(default=False, description="Whether passenger is a senior citizen (age >= 60)")
    disability_flag: bool = Field(default=False, description="Whether passenger has Divyangjan quota entitlement")


class ShapFeatureContribution(BaseModel):
    feature: str
    feature_name: str
    value: Any
    contribution: float
    impact: str  # "Increases Confirmation Chance" | "Decreases Confirmation Chance"


class TicketPredictResponse(BaseModel):
    pnr_status_prediction: str = Field(..., description="'Likely Confirmed' | 'Borderline' | 'Likely WL'")
    confirmation_probability: float = Field(..., description="Estimated probability of confirmation (0.0 to 1.0)")
    estimated_berth_type: str = Field(..., description="Estimated berth allocation (Lower/Middle/Upper/SL/SU/RAC)")
    berth_probability_breakdown: Dict[str, float] = Field(default_factory=dict)
    shap_explanation: List[ShapFeatureContribution] = Field(default_factory=list)
    booking_advisory: str = Field(..., description="Actionable travel advisory based on confirmation odds")
    confidence_score: float = Field(default=0.85, description="Model certainty score")
    status_probabilities: Dict[str, float] = Field(default_factory=dict, description="P(Confirmed / RAC / WL)")
    historical_insights: Dict[str, Any] = Field(default_factory=dict, description="Booking & cancellation trends from history")
    berth_estimate_basis: str = Field(default="", description="How the berth estimate was derived")
    model_version: str = "v1.0.0"
    governance_telemetry: Optional[GovernanceBadge] = None
