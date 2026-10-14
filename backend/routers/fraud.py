"""
backend/routers/fraud.py
========================
FastAPI router for Module 7: Fraud, Security & Bot Detection.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_fraud_service
from modules.m7_fraud_detector.schema import FraudDetectRequest, FraudDetectResponse
from modules.m7_fraud_detector.service import FraudDetectorService

router = APIRouter(prefix="/security", tags=["Module 7: Fraud & Security Detector"])


@router.post(
    "/score",
    response_model=FraudDetectResponse,
    summary="Compute real-time fraud risk score, classify threat category, and recommend security action",
)
def score_session_fraud_risk(
    request: FraudDetectRequest,
    service: FraudDetectorService = Depends(get_fraud_service),
) -> FraudDetectResponse:
    try:
        return service.score_session(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fraud scoring failed: {exc}")
