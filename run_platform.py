"""
run_platform.py
================
Single-Command Platform Launcher — Golden Rule 3.

Orchestrates the full startup sequence:
  1. Validates Python version and dependencies
  2. Generates synthetic data (if data/raw/ is empty)
  3. Trains ML models (if models/artifacts/ is empty)
  4. Starts FastAPI backend on :8000 (background daemon thread)
  5. Launches Streamlit frontend on :8501

Usage:
    python run_platform.py
"""

import os
import sys
import subprocess
import threading
import time
import logging

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     🚆  AI-Powered Intelligent Railway Travel &                             ║
║         Autonomous AI Governance Platform                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • 7 Passenger AI Modules (Ticket, Delay, Fare, Planner, Support, Fraud)   ║
║  • AI-for-AI Governance Cockpit (8 Pillars + Chaos Simulator)              ║
║  • Dual-Mode LLM: Azure OpenAI ☁  or  Rule-Based Fallback 🔧              ║
║  • Single-command launch: python run_platform.py                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

REQUIRED_PACKAGES = [
    "fastapi", "uvicorn", "streamlit", "pydantic", "dotenv",
    "sklearn", "xgboost", "lightgbm", "pandas", "numpy", "shap",
    "scipy", "plotly",
]

REQUIRED_MIN_PYTHON = (3, 10)


def check_python_version() -> None:
    version = sys.version_info
    if version < REQUIRED_MIN_PYTHON:
        logger.error(
            f"Python {REQUIRED_MIN_PYTHON[0]}.{REQUIRED_MIN_PYTHON[1]}+ required. "
            f"You have Python {version.major}.{version.minor}."
        )
        sys.exit(1)
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")


def check_packages() -> None:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(
            f"Missing packages: {missing}\n"
            "Please run: pip install -r requirements.txt"
        )
        sys.exit(1)
    logger.info(f"✓ All required packages are installed")


def check_env() -> None:
    env_path = os.path.join(ROOT, ".env")
    example_path = os.path.join(ROOT, ".env.example")
    if not os.path.exists(env_path):
        logger.info(
            "ℹ  No .env file found — platform will run in intelligent rule-based fallback mode.\n"
            f"   To enable Azure OpenAI LLM features: copy {example_path} to .env and fill in credentials."
        )
    else:
        logger.info("✓ .env file found")

    # Load env into environment
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        from core.config import settings
        if settings.llm_available:
            logger.info("✓ Azure OpenAI credentials detected — LLM mode: Azure OpenAI ☁")
        else:
            logger.info("ℹ  LLM mode: Intelligent Rule-Based Fallback 🔧 (full functionality)")
    except Exception:
        pass


def ensure_data() -> None:
    raw_dir = os.path.join(ROOT, "data", "raw")
    required_files = ["trains.csv", "stations.csv", "pnr_history.csv",
                      "fraud_events.csv", "railway_faq.txt"]

    existing = os.listdir(raw_dir) if os.path.exists(raw_dir) else []
    missing = [f for f in required_files if f not in existing]

    if missing:
        logger.info(f"⚙  Generating synthetic data (missing: {missing})...")
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "data", "generate_synthetic_data.py")],
            cwd=ROOT,
        )
        if result.returncode != 0:
            logger.error("❌ Synthetic data generation failed. Check above for errors.")
            sys.exit(1)
        logger.info("✓ Synthetic data generated successfully")
    else:
        logger.info(f"✓ Synthetic data found ({len(existing)} files in data/raw/)")


def ensure_models() -> None:
    artifacts_dir = os.path.join(ROOT, "models", "artifacts")
    required_models = [
        "wl_predictor_v1.0.pkl",
        "delay_predictor_v1.0.pkl",
        "fare_predictor_v1.0.pkl",
        "fraud_detector_v1.0.pkl",
    ]

    existing = (
        [f for f in os.listdir(artifacts_dir) if f.endswith(".pkl")]
        if os.path.exists(artifacts_dir) else []
    )
    missing = [m for m in required_models if m not in existing]

    if missing:
        logger.info(f"⚙  Training ML models (missing: {missing})...")
        logger.info("   This may take 2–5 minutes on first run...")
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "models", "train_all_models.py")],
            cwd=ROOT,
        )
        if result.returncode != 0:
            logger.error("❌ Model training failed. Check above for errors.")
            sys.exit(1)
        logger.info("✓ All ML models trained and saved")
    else:
        logger.info(f"✓ ML model artifacts found ({len(existing)} files)")


def start_fastapi() -> threading.Thread:
    """Start FastAPI backend in a daemon background thread."""

    def run():
        try:
            import uvicorn
            uvicorn.run(
                "backend.main:app",
                host="0.0.0.0",
                port=8000,
                log_level="warning",
                reload=False,
            )
        except Exception as e:
            logger.error(f"FastAPI server error: {e}")

    thread = threading.Thread(target=run, daemon=True, name="FastAPI-Thread")
    thread.start()
    return thread


def wait_for_fastapi(timeout: int = 30) -> bool:
    """Poll until FastAPI is ready or timeout."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_streamlit() -> None:
    """Launch Streamlit (this blocks — keeps the process alive)."""
    streamlit_app = os.path.join(ROOT, "frontend", "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", streamlit_app,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#4F8BF9",
    ]
    logger.info(f"\n{'═'*78}")
    logger.info(" 🌐 Opening Streamlit App at:  http://localhost:8501")
    logger.info(" 📡 FastAPI Swagger Docs at:   http://localhost:8000/docs")
    logger.info(f"{'═'*78}\n")
    subprocess.run(cmd, cwd=ROOT)


def main():
    print(BANNER)
    logger.info("Starting AI Railway Governance Platform...\n")

    # 1. Pre-flight checks
    logger.info("━━━ Pre-flight Checks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    check_python_version()
    check_packages()
    check_env()

    # 2. Data generation
    logger.info("\n━━━ Data Layer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ensure_data()

    # 3. Model training
    logger.info("\n━━━ ML Model Layer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ensure_models()

    # 4. Start FastAPI
    logger.info("\n━━━ Starting Services ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("⚙  Starting FastAPI backend on :8000 ...")
    start_fastapi()

    # Give FastAPI a moment to initialize
    time.sleep(3)
    logger.info("✓ FastAPI backend started (http://localhost:8000/docs)")

    # 5. Launch Streamlit (blocking)
    logger.info("⚙  Launching Streamlit frontend on :8501 ...")
    start_streamlit()


if __name__ == "__main__":
    main()
