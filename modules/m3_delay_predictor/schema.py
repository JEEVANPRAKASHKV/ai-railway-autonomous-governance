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
        default=None, ge=0.0, le=1.0,
        description="Adverse weather 0.0 (clear) – 1.0 (heavy fog/flood). Omit to use the month's climatology.")
    congestion_index: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Station congestion 0.0 – 1.0. Omit to estimate it from this station's own history.")
    distance_from_origin_km: Optional[float] = Field(
        default=None, ge=0.0, le=5000.0,
        description="Distance from the train's origin. Omit to derive it from the timetable.")


class AlternativeTravelOption(BaseModel):
    train_number: str
    train_name: str
    departure_time: str
    arrival_time: str
    delay_advantage_minutes: float
    reason: str


class DelayAlert(BaseModel):
    severity: str = Field(..., description="INFO | WARNING | CRITICAL")
    title: str
    message: str
    action: str = ""


class DelayPredictResponse(BaseModel):
    train_number: str
    station_code: str
    predicted_delay_minutes: float = Field(..., description="Estimated arrival delay at the station (minutes)")
    departure_delay_minutes: Optional[float] = Field(default=None, description="Estimated departure delay from the origin")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict, description="e.g. {'lower_bound': 10.0, 'upper_bound': 22.0}"
    )
    congestion_level: str = Field(..., description="'Low' | 'Moderate' | 'High' station congestion")
    station_congestion_index: Optional[float] = Field(default=None, description="0–1 station congestion used")
    congestion_source: str = Field(default="provided", description="provided | station history | network average")
    weather_source: str = Field(default="provided", description="provided | Open-Meteo forecast | station climatology (…)")
    weather_index_used: Optional[float] = Field(default=None, description="Adverse-weather index 0–10 fed to the model")
    weather_conditions: Dict[str, Any] = Field(default_factory=dict, description="Forecast rain, visibility, storm, wind")
    distance_from_origin_km: Optional[float] = Field(default=None, description="Real distance of this halt from the origin")
    distance_source: str = Field(default="timetable")
    scheduled_time: Optional[str] = Field(default=None, description="Scheduled arrival at the station (timetable)")
    disruption_probability: Optional[float] = Field(default=None, description="P(delay > 60 min)")
    proactive_alert: str = Field(..., description="Passenger disruption guidance or on-time notice")
    alerts: List[DelayAlert] = Field(default_factory=list)
    alternative_options: List[AlternativeTravelOption] = Field(
        default_factory=list, description="Populated when predicted delay exceeds 60 minutes"
    )
    delay_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    explanation_method: str = Field(default="SHAP (TreeExplainer)")
    model_version: str = Field(default="v1.0.0")
    confidence_score: float = Field(default=0.8, description="Nominal coverage of the prediction interval")
    governance_telemetry: Optional[GovernanceBadge] = None
