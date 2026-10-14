"""
backend/routers/delay.py
========================
FastAPI router for Module 3: Train Delay Predictor.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_delay_service
from modules.m3_delay_predictor.schema import DelayPredictRequest, DelayPredictResponse
from modules.m3_delay_predictor.service import DelayPredictorService

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delay prediction failed: {exc}")
