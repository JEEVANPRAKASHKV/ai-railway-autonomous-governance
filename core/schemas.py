"""
core/schemas.py
===============
Shared Pydantic schemas used across multiple modules and the API layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Governance Badge — embedded in every module response
# =============================================================================

class GovernanceBadge(BaseModel):
    """Inline governance telemetry badge attached to every prediction response."""
    module: str
    latency_ms: float
    schema_valid: bool
    confidence_score: Optional[float] = None
    drift_flag: bool = False
    bias_flag: bool = False
    hallucination_flag: bool = False
    llm_mode: str = "fallback"          # "sarvam" | "gemini" | "azure_openai" | "fallback"
    audit_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Telemetry Event — emitted by @governance_hook to the telemetry bus
# =============================================================================

class TelemetryEvent(BaseModel):
    event_id: str
    module: str
    endpoint: str = ""
    latency_ms: float
    event_type: str = "INFERENCE"
    severity: str = "INFO"
    payload: Dict[str, Any] = {}
    schema_valid: bool = True
    schema_violations: List[str] = []
    confidence_score: Optional[float] = None
    prediction: Optional[Any] = None
    drift_flag: bool = False
    bias_flag: bool = False
    hallucination_flag: bool = False
    error: Optional[str] = None
    llm_mode: str = "fallback"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access compatibility for legacy governance handlers."""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        if key == "module_id":
            return self.module
        return self.extra.get(key, default)


# =============================================================================
# Supervisor Alert
# =============================================================================

class SupervisorAlert(BaseModel):
    alert_id: str
    priority: Literal["P0-Critical", "P1-Warning", "P2-Info"]
    module: str
    description: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    root_cause: Optional[str] = None
    action_taken: str = "Pending"
    action_status: Literal["Pending", "In Progress", "Resolved", "Escalated"] = "Pending"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_log: Optional[str] = None


# =============================================================================
# LLM Response wrapper
# =============================================================================

class LLMResponse(BaseModel):
    content: str
    mode: Literal["sarvam", "gemini", "azure_openai", "fallback"]
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# Generic API envelope
# =============================================================================

class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Optional[Any] = None
    errors: List[str] = []


# =============================================================================
# Audit Trail Entry
# =============================================================================

class AuditEntry(BaseModel):
    entry_id: str
    module: str
    action: str
    input_summary: str = ""
    output_summary: str = ""
    input_hash: str = ""
    output_hash: str = ""
    governance_flags: List[str] = []
    risk_level: Literal["Low", "Medium", "High", "Critical"] = "Low"
    user_id: str = "anonymous"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = {}
