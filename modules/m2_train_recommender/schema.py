"""
modules/m2_train_recommender/schema.py
======================================
Pydantic schemas for Module 2: Smart Train Recommendation Engine.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class RecommendRequest(BaseModel):
    source_station: str = Field(..., description="Origin station code (e.g. 'NDLS')")
    destination_station: str = Field(..., description="Destination station code (e.g. 'BCT')")
    journey_date: date = Field(..., description="Date of journey (YYYY-MM-DD)")
    persona: str = Field(
        default="budget",
        description="User persona: 'budget', 'premium', 'student', 'senior', 'business', 'family'",
    )
    preferred_class: Optional[str] = Field(default=None, description="Preferred class: 1A, 2A, 3A, SL, CC")
    max_budget: Optional[float] = Field(default=None, ge=100, description="Maximum budget in INR")
    arrival_deadline: Optional[str] = Field(default=None, pattern=r"^([01]?\d|2[0-3]):[0-5]\d$", description="Desired arrival by HH:MM (24 h)")
    user_id: Optional[str] = Field(default=None, description="Passenger id (e.g. USR00001) — personalises using travel history")


class TrainRecommendation(BaseModel):
    rank: int
    train_number: str
    train_name: str
    departure_time: str
    arrival_time: str
    duration_hours: float
    available_classes: List[str]
    fare_estimate_inr: float
    punctuality_score: float = Field(..., description="Historical on-time score (0.0 to 1.0)")
    comfort_score: float = Field(..., description="Coach cleanliness & amenities score (0.0 to 1.0)")
    composite_score: float = Field(..., description="Multi-factor weighted match score (0.0 to 1.0)")
    recommendation_reason: str
    source: str = Field(default="", description="Origin station")
    destination: str = Field(default="", description="Destination station")
    tags: List[str] = Field(default_factory=list)
    factor_scores: Dict[str, float] = Field(default_factory=dict, description="Normalised 0–1 score per factor")
    factor_contributions: Dict[str, float] = Field(default_factory=dict, description="weight × score per factor")
    confirmation_odds: Optional[float] = Field(default=None, description="Historical confirmation rate on this train & class")
    meets_constraints: bool = True


class ConnectionLeg(BaseModel):
    train_number: str
    train_name: str
    source: str
    destination: str
    departure_time: str
    arrival_time: str
    duration_hours: float
    fare_inr: float


class ConnectionOption(BaseModel):
    """A one-change journey, offered when a corridor has few or no direct trains."""
    via: str
    legs: List[ConnectionLeg]
    layover_minutes: int
    total_hours: float = Field(..., description="Door-to-door time including the layover")
    total_fare_inr: float
    within_budget: bool = True


class RecommendResponse(BaseModel):
    recommendations: List[TrainRecommendation] = Field(default_factory=list)
    persona_applied: str
    weights_used: Dict[str, float]
    disparate_impact_ratio: float = Field(
        default=1.0,
        description="Fairness ratio between budget and premium train representation (target >= 0.80)",
    )
    bias_flag: bool = Field(default=False, description="Flagged if Disparate Impact Ratio < 0.80")
    personalisation: Dict[str, object] = Field(default_factory=dict, description="What was taken from the passenger's history")
    weights_version: int = Field(default=0, description="Increments each time weights are re-optimised from feedback")
    message: Optional[str] = None
    connections: List[ConnectionOption] = Field(default_factory=list,
                                                description="One-change options when fewer than 3 direct trains exist")
    governance_telemetry: Optional[GovernanceBadge] = None
