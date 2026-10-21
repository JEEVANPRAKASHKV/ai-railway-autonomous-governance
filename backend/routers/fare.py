"""
backend/routers/fare.py
=======================
FastAPI router for Module 5: Dynamic Fare Predictor.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_fare_service
from modules.m5_fare_predictor.schema import FarePredictRequest, FarePredictResponse
from modules.m5_fare_predictor.service import FarePredictorService

from governance.hooks import CircuitOpenError, DataQualityError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Module 5: Dynamic Fare Predictor"])


@router.post(
    "/fare",
    response_model=FarePredictResponse,
    summary="Forecast dynamic railway fares, 30-day trajectory curve, and optimal booking window",
)
def predict_dynamic_fare(
    request: FarePredictRequest,
    service: FarePredictorService = Depends(get_fare_service),
) -> FarePredictResponse:
    try:
        return service.predict(request)
    except (CircuitOpenError, DataQualityError, HTTPException):
        raise
    except Exception:
        logger.exception("Dynamic fare prediction failed")
        raise HTTPException(status_code=500, detail="Dynamic fare prediction failed. The error has been logged.")
