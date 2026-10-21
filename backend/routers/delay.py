"""
backend/routers/delay.py
========================
FastAPI router for Module 3: Train Delay Predictor.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_delay_service
from modules.m3_delay_predictor.schema import DelayPredictRequest, DelayPredictResponse
from modules.m3_delay_predictor.service import DelayPredictorService

from governance.hooks import CircuitOpenError, DataQualityError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Module 3: Delay Predictor"])


@router.post(
    "/delay",
    response_model=DelayPredictResponse,
    summary="Forecast arrival delay, station bottleneck severity, and rerouting alternatives",
)
def predict_train_delay(
    request: DelayPredictRequest,
    service: DelayPredictorService = Depends(get_delay_service),
) -> DelayPredictResponse:
    try:
        return service.predict(request)
    except (CircuitOpenError, DataQualityError, HTTPException):
        raise
    except Exception:
        logger.exception("Delay prediction failed")
        raise HTTPException(status_code=500, detail="Delay prediction failed. The error has been logged.")
