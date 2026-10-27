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
    source_file: str = Field(..., description="Knowledge-base entry title, or the booking-record tool used")
    text: str
    similarity_score: float
    sources: List[str] = Field(default_factory=list, description="Where the entry was verified (URLs or references)")


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
    answered_by: str = Field(default="", description="LLM / verified knowledge base / tool that produced the answer")
    route: str = Field(default="", description="language | tool | smalltalk | offtopic | knowledge | no_match")
    match_score: Optional[float] = Field(default=None, description="Knowledge-base match confidence (0–1) when retrieval ran")
    kb_version: str = Field(default="", description="Version (verification date) of the knowledge base used")
    governance_telemetry: Optional[GovernanceBadge] = None
