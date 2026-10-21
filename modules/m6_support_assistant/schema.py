"""
modules/m6_support_assistant/schema.py
======================================
Pydantic schemas for Module 6: Customer Support Assistant (English).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import GovernanceBadge


class SupportQueryRequest(BaseModel):
    query: str = Field(..., max_length=2000, description="Passenger question in English")
    user_id: str = Field(default="passenger_guest", description="User ID for session context")
    session_id: Optional[str] = Field(default=None, description="Chat session identifier")


class RetrievedContextChunk(BaseModel):
    chunk_id: str
    source_file: str
    text: str
    similarity_score: float


class SupportQueryResponse(BaseModel):
    query: str
    response: str
    detected_language: str = Field(default="en", description="'en', or 'unsupported' when the question was not in English")
    faithfulness_score: float = Field(..., description="RAG faithfulness score (0.0 to 1.0, target >= 0.65)")
    hallucination_flag: bool = Field(default=False, description="Flagged if faithfulness score < 0.65")
    retrieved_contexts: List[RetrievedContextChunk] = Field(default_factory=list)
    sentiment: str = Field(default="Neutral", description="'Positive' | 'Neutral' | 'Frustrated'")
    escalation_recommended: bool = Field(default=False, description="True if sentiment frustrated or query unresolved")
    confidence_score: float = Field(default=0.90)
    answered_by: str = Field(default="", description="Gemini model / rule-based engine / tool that produced the answer")
    governance_telemetry: Optional[GovernanceBadge] = None
