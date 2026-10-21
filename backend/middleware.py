"""
backend/middleware.py
=====================
ASGI HTTP middleware for request latency logging, correlation tracking, and exception containment.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Measures end-to-end HTTP processing latency and injects correlation headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        corr_id = request.headers.get("X-Correlation-ID", f"REQ-{uuid.uuid4().hex[:8].upper()}")
        t_start = time.perf_counter()

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
            response.headers["X-Correlation-ID"] = corr_id
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.error(f"[Middleware] Unhandled exception on {request.url.path}: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal Platform Error — the details were logged with this correlation id.",
                    "correlation_id": corr_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
                headers={
                    "X-Process-Time-Ms": f"{elapsed_ms:.2f}",
                    "X-Correlation-ID": corr_id,
                },
            )
