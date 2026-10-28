"""
streamlit_app.py
────────────────
Root cloud deployment entrypoint for Streamlit Community Cloud, Render, and Hugging Face.
Automatically bootstraps the dual-plane architecture:
1. Runs the FastAPI backend as a child process on 127.0.0.1 (first free port from 8000).
2. Keeps it in step with the code: Streamlit Cloud pulls every push into the running app, but an API that is
   already running would keep serving the old modules, models and timetable. The API reports a build
   fingerprint (core/build_info.py); when it no longer matches the checkout, a fresh API is started in the
   background and traffic switches to it once it is healthy, then the old one is stopped.
3. Hands execution to the Streamlit Control Tower & Passenger AI Portal.
"""

from __future__ import annotations

import atexit
import logging
import os
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

import requests

from dotenv import load_dotenv

# Ensure live credentials and environment variables are loaded
load_dotenv(override=True)

# Bridge Streamlit Cloud secrets into os.environ so FastAPI daemon & core settings can read them
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ[str(_k)] = str(_v)
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.build_info import fingerprint  # noqa: E402

logger = logging.getLogger("streamlit_app")
logging.basicConfig(level=logging.INFO)

EXTERNAL_BACKEND = "BACKEND_URL" in os.environ and os.environ.get("RB_BACKEND_MANAGED") != "1"
CHECK_EVERY_S = 15


def _health(url: str) -> Optional[dict]:
    """The API's /health reply ({} if it answers without JSON), or None if nothing is listening."""
    try:
        r = requests.get(f"{url}/api/health", timeout=1.5)
        try:
            return r.json() or {}
        except Exception:
            return {}
    except Exception:
        return None


def _is_backend_alive(url: Optional[str] = None) -> bool:
    return _health(url or os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")) is not None


def _ensure_artifacts() -> None:
    """
    data/raw/ and models/artifacts/ are git-ignored, so a fresh cloud checkout has neither.
    Generate the synthetic datasets and train the models once, BEFORE the backend imports
    the services (they load artifacts at import time).
    """
    raw = os.path.join(PROJECT_ROOT, "data", "raw")
    art = os.path.join(PROJECT_ROOT, "models", "artifacts")
    need_data = not all(
        os.path.exists(os.path.join(raw, f)) for f in ("trains.csv", "pnr_history.csv", "railway_faq.txt")
    )
    need_models = not all(
        os.path.exists(os.path.join(art, f))
        for f in ("wl_predictor_v1.0.pkl", "delay_predictor_v1.0.pkl",
                  "fare_predictor_v1.0.pkl", "fraud_detector_v1.0.pkl")
    )
    if need_data:
        logger.info("Generating synthetic datasets (first launch)...")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "data", "generate_synthetic_data.py")],
                       cwd=PROJECT_ROOT, check=True)
    if need_models:
        logger.info("Training ML models (first launch)...")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "models", "train_all_models.py")],
                       cwd=PROJECT_ROOT, check=True)


def _free_port(start: int = 8000) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port for the API")


def _spawn(port: int) -> subprocess.Popen:
    """Start the API in its own process (so it can be replaced when the code changes)."""
    _ensure_artifacts()
    log_dir = os.path.join(PROJECT_ROOT, "runtime")
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, f"backend_{port}.log"), "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    logger.info(f"API starting on port {port} (pid {proc.pid})")
    return proc


def _wait_ready(url: str, proc: subprocess.Popen, build: str, timeout_s: float) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        h = _health(url)
        if h is not None and (not build or h.get("build") in (build, None)):
            return True
        time.sleep(0.5)
    return False


def _use(url: str, proc: subprocess.Popen) -> None:
    old = os.environ.get("RB_BACKEND_PID")
    os.environ.update(BACKEND_URL=url, RB_BACKEND_PID=str(proc.pid), RB_BACKEND_MANAGED="1")
    if old and old != str(proc.pid):
        def _stop_old(pid: int) -> None:
            time.sleep(10)                      # let requests already in flight finish
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(20):                 # reap it so no zombie is left behind
                    if os.waitpid(pid, os.WNOHANG)[0]:
                        break
                    time.sleep(0.5)
            except Exception:
                pass
        threading.Thread(target=_stop_old, args=(int(old),), daemon=True).start()


def _swap(build: str) -> None:
    """Start an up-to-date API next to the stale one and switch to it once it is healthy."""
    try:
        port = _free_port(8001)
        proc = _spawn(port)
        url = f"http://127.0.0.1:{port}"
        if _wait_ready(url, proc, build, 240):
            _use(url, proc)
            logger.info(f"Switched to the updated API on port {port} (build {build})")
        else:
            logger.error("Updated API did not become healthy; still serving the previous one")
            if proc.poll() is None:
                proc.terminate()
    except Exception as e:
        logger.error(f"API update failed: {e}")
    finally:
        os.environ.pop("RB_BACKEND_SWAP", None)


def ensure_backend() -> None:
    url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
    if EXTERNAL_BACKEND:                        # someone else runs the API (e.g. docker-compose)
        return
    now = time.time()
    if now - float(os.environ.get("RB_BACKEND_CHECKED", "0")) < CHECK_EVERY_S:
        return
    os.environ["RB_BACKEND_CHECKED"] = str(now)
    build = fingerprint()
    h = _health(url)
    if h is None:                               # nothing serving: start one and wait for it
        try:
            port = _free_port(8000)
            proc = _spawn(port)
            new_url = f"http://127.0.0.1:{port}"
            if _wait_ready(new_url, proc, build, 240):
                _use(new_url, proc)
            else:
                logger.error("API failed to start; see runtime/backend_*.log")
        except Exception as e:
            logger.error(f"Failed to launch the API: {e}")
    elif h.get("build") != build and not os.environ.get("RB_BACKEND_SWAP"):
        logger.info(f"Code changed (API build {h.get('build')} → {build}); starting an updated API")
        os.environ["RB_BACKEND_SWAP"] = "1"
        threading.Thread(target=_swap, args=(build,), daemon=True, name="api-update").start()


ensure_backend()

# Execute frontend app
frontend_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
runpy.run_path(frontend_path, run_name="__main__")
