"""
backend/security.py
===================
Role-based access control, rate limiting and safe error responses.

Roles (least → most privileged)
  viewer    read-only governance data (no key needed for GET endpoints)
  operator  run chaos tests, retrain, roll back, stop a canary, resolve incidents, request promotion
  admin     approve / reject human-in-the-loop requests (e.g. promote a model to 100% traffic), reset

Callers authenticate with the ``X-API-Key`` header. Keys come from ``RB_OPERATOR_KEY`` and
``RB_ADMIN_KEY``. When they are not set (local development / the public demo) well-known development
keys are used and a warning is logged — set real keys in any shared deployment. The Streamlit UI
runs server-side, so it holds the key; browsers never see it.

Risk-reducing actions (rollback, stopping a canary) are operator actions and run immediately;
risk-increasing ones (promotion) always go through an admin approval.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

ROLE_RANK = {"viewer": 0, "operator": 1, "admin": 2}
DEV_OPERATOR_KEY = "rb-dev-operator-key"
DEV_ADMIN_KEY = "rb-dev-admin-key"


def _configured_keys() -> Dict[str, str]:
    op = os.getenv("RB_OPERATOR_KEY") or DEV_OPERATOR_KEY
    ad = os.getenv("RB_ADMIN_KEY") or DEV_ADMIN_KEY
    return {op: "operator", ad: "admin"}


def using_dev_keys() -> bool:
    return not (os.getenv("RB_OPERATOR_KEY") and os.getenv("RB_ADMIN_KEY"))


if using_dev_keys():
    logger.warning("[Security] RB_OPERATOR_KEY / RB_ADMIN_KEY not set — using development keys. "
                   "Set real keys before sharing this deployment.")


@dataclass
class Principal:
    role: str
    actor: str


def require_role(min_role: str) -> Callable:
    """FastAPI dependency: reject the call unless the API key carries at least `min_role`."""

    def _dep(x_api_key: str = Header(default=""), x_actor: str = Header(default="")) -> Principal:
        role = "viewer"
        for key, r in _configured_keys().items():
            if x_api_key and secrets.compare_digest(x_api_key, key):
                role = r
                break
        if ROLE_RANK[role] < ROLE_RANK[min_role]:
            raise HTTPException(
                status_code=401 if role == "viewer" else 403,
                detail=f"This action needs the '{min_role}' role. Send a valid X-API-Key header.",
            )
        actor = (x_actor or role).strip()[:60]
        return Principal(role=role, actor=f"{actor} ({role})" if x_actor else role)

    return _dep


# ── Rate limiting (threat response for passenger endpoints) ───────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window limit per client on /api/v1 passenger endpoints. Loopback traffic (the Streamlit
    server calling its own API) is exempt; in production this sits behind a gateway that forwards the
    real client address in X-Forwarded-For, which is what is keyed here.
    """

    def __init__(self, app, limit_per_minute: int = 120) -> None:
        super().__init__(app)
        self.limit = int(os.getenv("RB_RATE_LIMIT_PER_MIN", limit_per_minute))
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _client(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if not path.startswith("/api/v1") or path.startswith("/api/v1/governance"):
            return await call_next(request)
        client = self._client(request)
        if client in ("127.0.0.1", "::1", "localhost", "testclient") and not request.headers.get("x-forwarded-for"):
            return await call_next(request)
        now = time.time()
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.limit:
                retry = max(1, int(60 - (now - q[0])))
                blocked = True
            else:
                q.append(now)
                blocked = False
        if blocked:
            try:
                from governance.telemetry_bus import telemetry_bus
                telemetry_bus.publish(module_id="m7_fraud_detector", event_type="RATE_LIMITED",
                                      payload={"client": client[:3] + "***", "path": path}, severity="WARNING")
            except Exception:
                pass
            return JSONResponse(status_code=429, headers={"Retry-After": str(retry)},
                                content={"success": False, "message": "Too many requests — please slow down.",
                                         "retry_after_s": retry})
        return await call_next(request)
