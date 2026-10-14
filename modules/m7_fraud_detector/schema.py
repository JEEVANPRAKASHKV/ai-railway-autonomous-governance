"""
modules/m7_fraud_detector/schema.py
===================================
Pydantic schemas for Module 7: Fraud, Security & Bot Detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class FraudDetectRequest(BaseModel):
    session_id: str = Field(..., description="Unique web/app session identifier")
    ip_hash: str = Field(..., description="Anonymized hash of client IP address")
    requests_per_minute: float = Field(..., ge=0.0, description="Observed query/submission velocity")
    unique_pnr_attempts: int = Field(default=1, ge=0, description="Count of distinct PNR or booking lookups")
    tatkal_flag: bool = Field(default=False, description="Whether session is during 10:00-10:15 Tatkal surge")
    account_age_days: int = Field(default=180, ge=0, description="Age of user IRCTC account in days")
    payment_attempts: int = Field(default=1, ge=0, description="Count of payment attempts in session")
    failed_captcha_count: int = Field(default=0, ge=0, description="Failed captcha verifications")
    booking_time_hour: int = Field(default=10, ge=0, le=23, description="Hour of booking submission (0-23)")
    is_bulk_booking: bool = Field(default=False, description="Whether booking attempts >= 6 passengers in one transaction")
    ip_risk_score: Optional[float] = Field(default=0.08, ge=0.0, le=1.0, description="Reputation score of subnet (0.0 to 1.0)")


class FraudInvestigativeFeature(BaseModel):
    feature: str
    value: Any
    severity: Literal["Normal", "Suspicious", "High Anomaly"]
    description: str


class FraudDetectResponse(BaseModel):
    session_id: str
    ip_hash: str
    fraud_risk_score: float = Field(..., description="Risk score from 0.0 (benign) to 100.0 (critical bot)")
    risk_category: Literal["Safe", "Suspicious", "High Risk", "Bot Detected"]
    recommended_action: Literal["ALLOW", "STEP_UP_CHALLENGE", "RATE_LIMIT", "BLOCK_AND_REPORT"]
    investigation_features: List[FraudInvestigativeFeature] = Field(default_factory=list)
    incident_dossier: Dict[str, Any] = Field(default_factory=dict)
    shap_risk_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(default=0.99)
    governance_telemetry: Optional[GovernanceBadge] = None
