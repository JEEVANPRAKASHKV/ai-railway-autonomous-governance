"""
streamlit_app.py
────────────────
Root cloud deployment entrypoint for Streamlit Community Cloud, Render, and Hugging Face.
Automatically bootstraps the dual-plane architecture:
1. Launches FastAPI backend in a background daemon thread on 127.0.0.1:8000 (if not already running).
2. Verifies health and seeds realistic baseline telemetry.
3. Hands execution to the Streamlit Control Tower & Passenger AI Portal.
"""

from __future__ import annotations

import os
import sys
import time
import runpy
import logging
import threading
import requests

from dotenv import load_dotenv

# Ensure live credentials and environment variables are loaded
load_dotenv(override=True)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("streamlit_app")
logging.basicConfig(level=logging.INFO)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _is_backend_alive(url: str = BACKEND_URL) -> bool:
    try:
        r = requests.get(f"{url}/api/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _start_background_backend():
    """Start uvicorn serving FastAPI in a background daemon thread."""
    if _is_backend_alive():
        return

    try:
        import uvicorn
        from backend.main import app as fastapi_app

        def _serve():
            uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

        server_thread = threading.Thread(target=_serve, daemon=True, name="FastAPI-Daemon")
        server_thread.start()

        # Wait up to 10 seconds for backend to become healthy
        for _ in range(20):
            time.sleep(0.5)
            if _is_backend_alive():
                break
    except Exception as e:
        logger.error(f"Failed to launch background FastAPI backend: {e}")


# Initialize background backend once per server process
if "backend_started" not in os.environ:
    _start_background_backend()
    os.environ["backend_started"] = "1"
elif not _is_backend_alive():
    _start_background_backend()

# Execute frontend app
frontend_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
runpy.run_path(frontend_path, run_name="__main__")
