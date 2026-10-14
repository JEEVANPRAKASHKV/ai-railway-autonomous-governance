"""
backend/routers/support.py
==========================
FastAPI router for Module 6: Multilingual Customer Support Assistant.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_support_service
from modules.m6_support_assistant.schema import SupportQueryRequest, SupportQueryResponse
from modules.m6_support_assistant.service import SupportAssistantService

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Support query processing failed: {exc}")
