"""
backend/routers/support.py
==========================
FastAPI router for Module 6: Multilingual Customer Support Assistant.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_support_service
from modules.m6_support_assistant.schema import SupportQueryRequest, SupportQueryResponse
from modules.m6_support_assistant.service import SupportAssistantService

from governance.hooks import CircuitOpenError, DataQualityError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/support", tags=["Module 6: Customer Support Assistant"])


@router.post(
    "/query",
    response_model=SupportQueryResponse,
    summary="Answer passenger policy & travel queries using multilingual RAG with hallucination audit",
)
def handle_support_query(
    request: SupportQueryRequest,
    service: SupportAssistantService = Depends(get_support_service),
) -> SupportQueryResponse:
    try:
        return service.query(request)
    except (CircuitOpenError, DataQualityError, HTTPException):
        raise
    except Exception:
        logger.exception("Support query processing failed")
        raise HTTPException(status_code=500, detail="Support query processing failed. The error has been logged.")
