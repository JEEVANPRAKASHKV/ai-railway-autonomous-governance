"""
modules/m3_delay_predictor/schema.py
====================================
Pydantic schemas for Module 3: Train Delay & Disruption Prediction.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class DelayPredictRequest(BaseModel):
    train_number: str = Field(..., description="5-digit train number (e.g. '12002')")
    station_code: str = Field(..., description="Intermediate or arrival station code (e.g. 'BPL')")
    journey_date: date = Field(..., description="Date of journey (YYYY-MM-DD)")
    weather_index: Optional[float] = Field(
        default=0.25, ge=0.0, le=1.0, description="Adverse weather index: 0.0 (clear) to 1.0 (heavy fog/flood)"
    )
    congestion_index: Optional[float] = Field(
        default=0.45, ge=0.0, le=1.0, description="Station platform congestion score (0.0 to 1.0)"
    )
    distance_from_origin_km: Optional[float] = Field(
        default=500.0, ge=0.0, description="Distance traversed from train origin in km"
    )


class AlternativeTravelOption(BaseModel):
    train_number: str
    train_name: str
    departure_time: str
    arrival_time: str
    delay_advantage_minutes: float
    reason: str


class DelayPredictResponse(BaseModel):
    train_number: str
    station_code: str
    predicted_delay_minutes: float = Field(..., description="Estimated arrival delay in minutes")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict, description="e.g. {'lower_bound': 10.0, 'upper_bound': 22.0}"
    )
    congestion_level: str = Field(..., description="'Low' | 'Moderate' | 'Severe'")
    proactive_alert: str = Field(..., description="Passenger disruption guidance or on-time notice")
    alternative_options: List[AlternativeTravelOption] = Field(
        default_factory=list, description="Populated when predicted delay exceeds 60 minutes"
    )
    delay_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(default=0.88)
    governance_telemetry: Optional[GovernanceBadge] = None
