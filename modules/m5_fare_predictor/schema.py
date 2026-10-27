"""
modules/m5_fare_predictor/schema.py
===================================
Pydantic schemas for Module 5: Dynamic Fare Prediction Engine.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class FarePredictRequest(BaseModel):
    train_number: str = Field(..., description="5-digit train number (e.g. '12951')")
    journey_class: str = Field(default="3A", description="Travel class: 1A, 2A, 3A, SL, CC")
    journey_date: date = Field(..., description="Date of journey (YYYY-MM-DD)")
    base_fare: Optional[float] = Field(default=None, description="Starting base fare in INR (auto-looked up if omitted)")
    booking_date: Optional[date] = Field(default_factory=date.today, description="Booking date (defaults to today)")
    tatkal_flag: bool = Field(default=False, description="Whether booking under Tatkal quota dynamic pricing")
    source_station: Optional[str] = Field(default=None, description="Boarding station (defaults to the train's origin)")
    destination_station: Optional[str] = Field(default=None, description="Alighting station (defaults to its terminus)")


class FareForecastPoint(BaseModel):
    days_to_departure: int
    forecast_date: date
    predicted_fare_inr: float
    demand_level: str  # "Low" | "Moderate" | "High" | "Surge"


class CostSavingAlternative(BaseModel):
    alternative_type: str  # "Date Shift" | "Class Shift" | "Off-Peak Route"
    description: str
    potential_savings_inr: float
    recommended_fare_inr: float


class FarePredictResponse(BaseModel):
    train_number: str
    journey_class: str
    current_predicted_fare_inr: float
    base_fare_inr: float
    surge_multiplier: float
    demand_level: str
    optimal_booking_window: str
    optimal_booking_days_before: int
    optimal_booking_date: date
    forecast_curve: List[FareForecastPoint] = Field(default_factory=list)
    cost_saving_alternatives: List[CostSavingAlternative] = Field(default_factory=list)
    fare_drivers: List[Dict[str, Any]] = Field(default_factory=list, description="SHAP contributions in ₹")
    demand_surge_from_days: Optional[int] = Field(default=None, description="Days before departure when demand turns High/Surge")
    pricing_anomaly: Optional[str] = Field(default=None, description="Set when the raw prediction is outside the historical range")
    policy_note: Optional[str] = None
    fare_breakdown: Dict[str, Any] = Field(default_factory=dict,
                                           description="Official base fare + reservation + superfast + GST for the journey")
    pricing_rule: str = Field(default="", description="flexi | fixed | tatkal — how Indian Railways prices this ticket")
    source_station: Optional[str] = None
    destination_station: Optional[str] = None
    model_version: str = "v1.0.0"
    confidence_score: float = Field(default=0.94)
    governance_telemetry: Optional[GovernanceBadge] = None
