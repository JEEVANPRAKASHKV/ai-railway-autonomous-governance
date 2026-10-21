"""
governance/context.py
─────────────────────
Per-request governance context.

A service calls :func:`report_inference` with the features *exactly as the model saw them* (after
scaling / encoding), the prediction it made and the model version that served it. The
``@governance_hook`` wrapper picks this up after the call, so drift detection, performance
monitoring, canary analysis and the audit trail all work in the model's own feature space —
not on raw request fields that may use different units.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional

_current: ContextVar[Optional[Dict[str, Any]]] = ContextVar("rb_governance_ctx", default=None)


def begin() -> object:
    """Start a fresh context for one governed call. Returns a token for :func:`end`."""
    return _current.set({})


def end(token: object) -> Dict[str, Any]:
    """Close the context and return everything the service reported."""
    data = _current.get() or {}
    try:
        _current.reset(token)  # type: ignore[arg-type]
    except Exception:
        _current.set(None)
    return data


def report_inference(
    features: Optional[Dict[str, float]] = None,
    prediction: Any = None,
    version: Optional[str] = None,
    **extra: Any,
) -> None:
    """Called from inside a service to describe the inference it just made."""
    ctx = _current.get()
    if ctx is None:
        return
    if features is not None:
        ctx["features"] = {k: float(v) for k, v in features.items() if _is_number(v)}
    if prediction is not None:
        ctx["prediction"] = prediction
    if version is not None:
        ctx["version"] = version
    ctx.update(extra)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
