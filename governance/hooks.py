"""
governance/hooks.py
====================
@governance_hook decorator — Golden Rule 4: Embedded Governance Telemetry.

Wraps every module service function with:
  1. Pre-inference Data Quality Gate check
  2. Latency measurement
  3. TelemetryEvent emission to the bus

Usage:
    from governance.hooks import governance_hook

    @governance_hook("m1_ticket_predictor")
    def predict_ticket(request: TicketPredictRequest) -> TicketPredictResponse:
        ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Optional

from core.config import settings
from core.schemas import AuditEntry, GovernanceBadge, TelemetryEvent
from governance.telemetry_bus import telemetry_bus

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_data_quality_gate: Optional[Any] = None



# Request-field → baseline-feature aliases used by live drift monitoring
_FEATURE_ALIASES = {
    "current_wl_number": "initial_wl_number",
    "congestion_index": "congestion_at_station",
}

# Request fields treated as protected attributes by the fairness auditor
_PROTECTED_FIELDS = {
    "seniority_flag":   ("age_group", lambda v: "senior_60plus" if v else "under_60"),
    "disability_flag":  ("disability_status", lambda v: "divyangjan" if v else "none"),
    "quota":            ("quota", str),
    "journey_class":    ("travel_class", str),
    "preferred_class":  ("travel_class", str),
    "persona":          ("traveller_persona", str),
    "passenger_gender": ("gender", str),
    "gender":           ("gender", str),
}

def _get_quality_gate():
    global _data_quality_gate
    if _data_quality_gate is None:
        from governance.data_quality_gate import DataQualityGate
        _data_quality_gate = DataQualityGate()
    return _data_quality_gate


def _hash_payload(obj: Any) -> str:
    try:
        raw = json.dumps(obj, default=str, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    except Exception:
        return "hash_error"


def governance_hook(module_name: str, endpoint: str = ""):
    """
    Decorator factory for governance telemetry wrapping.

    Wraps the decorated function with:
      - Pre-inference Data Quality Gate validation
      - Latency measurement (wall-clock)
      - TelemetryEvent emission
      - AuditEntry logging
      - GovernanceBadge injection into response (if response has .governance_telemetry)

    Args:
        module_name: Module identifier (e.g., "m1_ticket_predictor")
        endpoint: Optional API endpoint string for logging

    Returns:
        Decorator that wraps the service function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            audit_id = f"{module_name[:3].upper()}-{str(uuid.uuid4())[:8].upper()}"
            t_start = time.perf_counter()
            error_str: Optional[str] = None
            schema_valid = True
            schema_violations: list = []
            result = None

            # ── Step 1: Pre-inference Data Quality Gate ───────────────────────
            try:
                gate = _get_quality_gate()
                payload = dict(kwargs)
                for arg in args:
                    if hasattr(arg, "model_dump"):
                        payload.update(arg.model_dump())
                        break
                    elif isinstance(arg, dict):
                        payload.update(arg)
                        break
                quality_report = gate.validate(module_name, payload)
                schema_valid = quality_report.get("valid", True)
                schema_violations = quality_report.get("violations", [])
                if not schema_valid:
                    logger.warning(
                        f"[Governance/{module_name}] Schema violations: {schema_violations}"
                    )
            except Exception as gate_err:
                logger.debug(f"[Governance/{module_name}] Quality gate error (non-blocking): {gate_err}")

            # ── Step 2: Execute the service function ──────────────────────────
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                error_str = str(exc)
                logger.error(f"[Governance/{module_name}] Inference error: {exc}")
                raise
            finally:
                # ── Step 3: Measure latency ───────────────────────────────────
                latency_ms = (time.perf_counter() - t_start) * 1000

                # ── Step 4: Extract confidence from result ────────────────────
                confidence: Optional[float] = None
                drift_flag = False
                bias_flag = False
                hallucination_flag = False

                if result is not None:
                    for attr in ("confirmation_probability", "confidence_score",
                                 "fraud_risk_score", "faithfulness_score"):
                        val = getattr(result, attr, None)
                        if val is not None:
                            if attr == "fraud_risk_score":
                                confidence = float(val) / 100.0
                            else:
                                confidence = float(val)
                            break

                    # Recommender returns a ranked list rather than a single score
                    if confidence is None and getattr(result, "recommendations", None):
                        try:
                            confidence = float(result.recommendations[0].composite_score)
                        except Exception:
                            pass

                    if hasattr(result, "bias_flag"):
                        bias_flag = bool(getattr(result, "bias_flag"))
                    if hasattr(result, "drift_detected"):
                        drift_flag = bool(getattr(result, "drift_detected"))
                    elif hasattr(result, "drift_flag"):
                        drift_flag = bool(getattr(result, "drift_flag"))
                    if hasattr(result, "hallucination_flag"):
                        hallucination_flag = bool(getattr(result, "hallucination_flag"))
                    elif hasattr(result, "is_hallucination"):
                        hallucination_flag = bool(getattr(result, "is_hallucination"))

                # ── Step 5: Emit TelemetryEvent ───────────────────────────────
                event = TelemetryEvent(
                    event_id=audit_id,
                    module=module_name,
                    endpoint=endpoint,
                    latency_ms=round(latency_ms, 3),
                    schema_valid=schema_valid,
                    schema_violations=schema_violations,
                    confidence_score=confidence,
                    drift_flag=drift_flag,
                    bias_flag=bias_flag,
                    hallucination_flag=hallucination_flag,
                    error=error_str,
                    llm_mode=settings.llm_mode,
                )
                telemetry_bus.emit(event)

                # ── Step 5b: Wire Live Pillar Telemetry ────────────────────────
                try:
                    from governance.health_monitor import health_monitor
                    health_monitor.record_request(
                        module_id=module_name,
                        latency_ms=latency_ms,
                        success=(error_str is None),
                    )
                except Exception as hm_err:
                    logger.debug(f"[Governance/{module_name}] Health monitor error: {hm_err}")

                try:
                    from governance.drift_detector import drift_detector
                    num_features = {
                        k: float(v) for k, v in payload.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                    # Map request field names onto the feature names the baselines were trained on
                    for src, dst in _FEATURE_ALIASES.items():
                        if src in num_features:
                            num_features[dst] = num_features[src]
                    jd, bd = payload.get("journey_date"), payload.get("booking_date")
                    if jd is not None and "days_to_departure" not in num_features:
                        try:
                            from datetime import date as _date
                            num_features["days_to_departure"] = float(max(0, (jd - (bd or _date.today())).days))
                        except Exception:
                            pass
                    if num_features:
                        drift_detector.observe(
                            model_id=module_name,
                            features=num_features,
                            prediction=confidence,
                        )
                except Exception as dd_err:
                    logger.debug(f"[Governance/{module_name}] Drift detector error: {dd_err}")

                try:
                    from governance.bias_auditor import bias_auditor
                    matched_prot = {}
                    for key, (attr, fn) in _PROTECTED_FIELDS.items():
                        if key in payload and payload[key] is not None:
                            matched_prot[attr] = fn(payload[key])
                    if matched_prot and confidence is not None:
                        bias_auditor.observe(
                            module_id=module_name,
                            protected_attrs=matched_prot,
                            prediction=confidence,
                        )
                except Exception as ba_err:
                    logger.debug(f"[Governance/{module_name}] Bias auditor error: {ba_err}")

                # ── Step 6: Log AuditEntry ────────────────────────────────────
                flags = []
                if not schema_valid:
                    flags.append("SCHEMA_VIOLATION")
                if drift_flag:
                    flags.append("DRIFT_DETECTED")
                if bias_flag:
                    flags.append("BIAS_FLAGGED")
                if hallucination_flag:
                    flags.append("HALLUCINATION_FLAGGED")
                if error_str:
                    flags.append("INFERENCE_ERROR")

                risk_level = "Low"
                if error_str or (confidence is not None and confidence < 0.5):
                    risk_level = "High"
                elif not schema_valid or latency_ms > settings.governance_p95_latency_threshold_ms:
                    risk_level = "Medium"

                input_hash = _hash_payload(kwargs)
                output_summary = ""
                output_hash = ""
                if result is not None:
                    try:
                        output_summary = str(result)[:200]
                        output_hash = _hash_payload(result.model_dump() if hasattr(result, "model_dump") else str(result))
                    except Exception:
                        pass

                audit_entry = AuditEntry(
                    entry_id=audit_id,
                    module=module_name,
                    action=func.__name__,
                    input_summary=f"kwargs_keys={list(kwargs.keys())}",
                    output_summary=output_summary,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    governance_flags=flags,
                    risk_level=risk_level,
                )
                telemetry_bus.log_audit(audit_entry)

                # ── Step 7: Inject GovernanceBadge into result ────────────────
                if result is not None and hasattr(result, "governance_telemetry"):
                    badge = GovernanceBadge(
                        module=module_name,
                        latency_ms=round(latency_ms, 2),
                        schema_valid=schema_valid,
                        confidence_score=confidence,
                        drift_flag=drift_flag,
                        bias_flag=bias_flag,
                        hallucination_flag=hallucination_flag,
                        llm_mode=settings.llm_mode,
                        audit_id=audit_id,
                    )
                    object.__setattr__(result, "governance_telemetry", badge)

                # ── Step 8: Log latency warning if SLA breached ───────────────
                if latency_ms > settings.governance_p95_latency_threshold_ms:
                    logger.warning(
                        f"[Governance/{module_name}] SLA breach: {latency_ms:.0f}ms > "
                        f"{settings.governance_p95_latency_threshold_ms:.0f}ms threshold"
                    )

            return result

        return wrapper

    return decorator


def emit_manual_event(
    module: str,
    latency_ms: float,
    schema_valid: bool = True,
    confidence: Optional[float] = None,
    drift_flag: bool = False,
    bias_flag: bool = False,
    hallucination_flag: bool = False,
    error: Optional[str] = None,
    extra: dict = None,
) -> str:
    """
    Manually emit a telemetry event (for use in async/streaming contexts
    where the @governance_hook decorator cannot wrap the full call).
    Returns the generated event_id.
    """
    event_id = f"MAN-{str(uuid.uuid4())[:8].upper()}"
    event = TelemetryEvent(
        event_id=event_id,
        module=module,
        latency_ms=round(latency_ms, 3),
        schema_valid=schema_valid,
        confidence_score=confidence,
        drift_flag=drift_flag,
        bias_flag=bias_flag,
        hallucination_flag=hallucination_flag,
        error=error,
        llm_mode=settings.llm_mode,
        extra=extra or {},
    )
    telemetry_bus.emit(event)
    return event_id
