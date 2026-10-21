"""
backend/routers/ticket.py
=========================
FastAPI router for Module 1: Ticket Confirmation & Berth Prediction.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_ticket_service
from modules.m1_ticket_predictor.schema import TicketPredictRequest, TicketPredictResponse
from modules.m1_ticket_predictor.service import TicketPredictorService

from governance.hooks import CircuitOpenError, DataQualityError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Module 1: Ticket Predictor"])


@router.post(
    "/ticket",
    response_model=TicketPredictResponse,
    summary="Predict PNR Waitlist/RAC confirmation probability and berth type",
)
def predict_ticket_confirmation(
    request: TicketPredictRequest,
    service: TicketPredictorService = Depends(get_ticket_service),
) -> TicketPredictResponse:
    try:
        return service.predict(request)
    except (CircuitOpenError, DataQualityError, HTTPException):
        raise
    except Exception:
        logger.exception("Ticket prediction failed")
        raise HTTPException(status_code=500, detail="Ticket prediction failed. The error has been logged.")
