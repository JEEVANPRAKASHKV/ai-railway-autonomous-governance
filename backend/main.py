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
from typing import AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.middleware import TelemetryMiddleware
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
from core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup preflight check and shutdown cleanup hooks."""
    logger.info("=" * 60)
    logger.info(f" Starting {settings.project_name if hasattr(settings, 'project_name') else 'AI Railway Platform'}")
    logger.info(f" Mode: {'Azure OpenAI GPT-4o-mini' if settings.llm_available else 'Intelligent Rule-Based Fallback'}")
    logger.info(f" FastAPI Port: {settings.fastapi_port} | Streamlit Port: {settings.streamlit_port}")
    logger.info("=" * 60)

    # Verify models artifacts directory
    art_dir = settings.models_artifacts_dir
    models_found = [f for f in os.listdir(art_dir) if f.endswith(".pkl")] if os.path.exists(art_dir) else []
    logger.info(f" Pre-trained ML artifacts loaded: {len(models_found)} ({', '.join(models_found)})")

    yield

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TelemetryMiddleware)

# ── Router Registration ────────────────────────────────────────────────────────
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
            "models_loaded": models_ready,
            "synthetic_data_ready": data_ready,
            "llm_mode": settings.llm_mode,
            "port": settings.fastapi_port,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=settings.fastapi_port, reload=False)
