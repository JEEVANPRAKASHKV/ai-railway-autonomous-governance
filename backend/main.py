"""
backend/main.py
===============
FastAPI Application Entry Point.
Serves Passenger Experience Plane (/api/v1/*) and AI Governance Control Tower (/api/v1/governance/*).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv

# Ensure live .env credentials (e.g. Azure OpenAI) are loaded with override
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi import Request
from fastapi.exceptions import RequestValidationError

from backend.middleware import TelemetryMiddleware
from backend.security import RateLimitMiddleware, using_dev_keys
from governance.hooks import CircuitOpenError, DataQualityError
from backend.routers import (
    delay,
    fare,
    fraud,
    governance,
    planner,
    recommend,
    support,
    ticket,
)
from core.build_info import fingerprint
from core.config import settings
from governance.telemetry_bus import telemetry_bus

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup preflight check, baseline seeding, and shutdown cleanup hooks."""
    logger.info("=" * 60)
    logger.info(f" Starting {settings.project_name if hasattr(settings, 'project_name') else 'AI Railway Platform'}")
    logger.info(f" LLM mode: {settings.llm_mode}")
    logger.info(f" FastAPI Port: {settings.fastapi_port} | Streamlit Port: {settings.streamlit_port}")
    logger.info("=" * 60)

    # Verify models artifacts directory
    art_dir = settings.models_artifacts_dir
    models_found = [f for f in os.listdir(art_dir) if f.endswith(".pkl")] if os.path.exists(art_dir) else []
    logger.info(f" Pre-trained ML artifacts loaded: {len(models_found)} ({', '.join(models_found)})")

    # No synthetic telemetry is seeded: every number on the Control Tower comes from real requests.

    # Start autonomous AI Supervisor Agent loop (Pillar 8)
    try:
        from governance.supervisor_agent import supervisor_agent
        supervisor_agent.start()
        logger.info(" Autonomous AI Supervisor Agent background polling loop started.")
    except Exception as sa_err:
        logger.warning(f" Could not start supervisor agent: {sa_err}")

    yield

    try:
        from governance.supervisor_agent import supervisor_agent
        supervisor_agent.stop()
        logger.info(" Autonomous AI Supervisor Agent stopped.")
    except Exception:
        pass

    logger.info(" Shutting down AI Railway Platform gracefully.")


app = FastAPI(
    title="AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform",
    description=(
        "Enterprise-grade dual-plane platform: 7 Passenger Experience AI Services "
        "+ Autonomous AI-for-AI Governance Control Tower."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
_origins = [o.strip() for o in os.getenv("RB_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Actor", "X-Correlation-ID"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(TelemetryMiddleware)


@app.exception_handler(CircuitOpenError)
async def _circuit_open(request: Request, exc: CircuitOpenError) -> JSONResponse:
    return JSONResponse(status_code=503, headers={"Retry-After": "30"},
                        content={"success": False, "detail": f"{exc.module} is temporarily paused by its circuit "
                                 "breaker after repeated failures. Please retry shortly.", "module": exc.module})


@app.exception_handler(DataQualityError)
async def _data_quality(request: Request, exc: DataQualityError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"success": False, "detail": "Input rejected by the data-quality "
                        "gate, so no unreliable prediction was made.", "violations": exc.violations})

# ── Router Registration ────────────────────────────────────────────────────────
BUILD = fingerprint()          # code + data + models this process loaded (see core/build_info.py)
API_V1_PREFIX = "/api/v1"
app.include_router(ticket.router, prefix=API_V1_PREFIX)
app.include_router(recommend.router, prefix=API_V1_PREFIX)
app.include_router(delay.router, prefix=API_V1_PREFIX)
app.include_router(planner.router, prefix=API_V1_PREFIX)
app.include_router(fare.router, prefix=API_V1_PREFIX)
app.include_router(support.router, prefix=API_V1_PREFIX)
app.include_router(fraud.router, prefix=API_V1_PREFIX)
app.include_router(governance.router, prefix=API_V1_PREFIX)


# ── System Health Endpoints ────────────────────────────────────────────────────

@app.get("/", summary="Root Health & Platform Manifest")
def read_root() -> Dict[str, str]:
    return {
        "status": "online",
        "platform": "AI Railway Autonomous Governance",
        "api_docs": "/docs",
        "llm_mode": settings.llm_mode,
        "version": "1.0.0",
    }


def _llm_status() -> Dict[str, Any]:
    try:
        from core.llm_router import llm_router
        st = llm_router.status()
        st.pop("model_chain", None)
        return st
    except Exception:
        return {}


@app.get("/api/health", summary="Comprehensive subsystem health check")
@app.get("/health", summary="Health check alias for monitoring tools")
def health_check() -> JSONResponse:
    art_dir = settings.models_artifacts_dir
    models_ready = os.path.exists(art_dir) and any(f.endswith(".pkl") for f in os.listdir(art_dir))
    data_ready = os.path.exists(os.path.join(settings.data_raw_dir, "trains.csv"))

    status = "healthy" if (models_ready and data_ready) else "degraded"
    return JSONResponse(
        status_code=200 if status == "healthy" else 503,
        content={
            "status": status,
            "build": BUILD,
            "models_loaded": models_ready,
            "synthetic_data_ready": data_ready,
            "llm_mode": settings.llm_mode,
            "llm": _llm_status(),
            "security": {"rbac": True, "dev_keys_in_use": using_dev_keys()},
            "port": settings.fastapi_port,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=settings.fastapi_port, reload=False)
