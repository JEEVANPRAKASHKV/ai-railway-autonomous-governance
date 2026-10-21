"""
backend/routers/fraud.py
========================
FastAPI router for Module 7: Fraud, Security & Bot Detection.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_fraud_service
from modules.m7_fraud_detector.schema import FraudDetectRequest, FraudDetectResponse
from modules.m7_fraud_detector.service import FraudDetectorService

from governance.hooks import CircuitOpenError, DataQualityError

logger = logging.getLogger(__name__)

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
    except (CircuitOpenError, DataQualityError, HTTPException):
        raise
    except Exception:
        logger.exception("Fraud scoring failed")
        raise HTTPException(status_code=500, detail="Fraud scoring failed. The error has been logged.")


@router.get("/alerts", summary="Recent explainable fraud alerts for investigation")
def recent_fraud_alerts(n: int = 30):
    from modules.m7_fraud_detector.service import fraud_detector_service
    return fraud_detector_service.recent_alerts(n=min(max(n, 1), 200))
