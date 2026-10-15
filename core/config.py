"""
core/config.py
==============
Central platform configuration using pydantic-settings.
All modules import `settings` from here — single source of truth.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class PlatformSettings(BaseSettings):
    # ── Google Gemini ────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    # ── Azure OpenAI ────────────────────────────────────────────────────────
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(default="", alias="AZURE_OPENAI_KEY")
    azure_openai_deployment_name: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION")

    # ── Platform ─────────────────────────────────────────────────────────────
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Server Ports ──────────────────────────────────────────────────────────
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    # ── Governance Thresholds ─────────────────────────────────────────────────
    governance_p95_latency_threshold_ms: float = Field(
        default=3000.0, alias="GOVERNANCE_ALERT_THRESHOLD_P95_LATENCY_MS"
    )
    governance_drift_psi_threshold: float = Field(
        default=0.15, alias="GOVERNANCE_DRIFT_PSI_THRESHOLD"
    )
    governance_accuracy_min_threshold: float = Field(
        default=0.80, alias="GOVERNANCE_ACCURACY_MIN_THRESHOLD"
    )
    governance_hallucination_threshold: float = Field(
        default=0.65, alias="GOVERNANCE_HALLUCINATION_THRESHOLD"
    )
    governance_bias_dir_threshold: float = Field(
        default=0.80, alias="GOVERNANCE_BIAS_DIR_THRESHOLD"
    )
    governance_fraud_fp_rate_threshold: float = Field(
        default=0.12, alias="GOVERNANCE_FRAUD_FP_RATE_THRESHOLD"
    )

    # ── Computed Properties ────────────────────────────────────────────────────
    @property
    def gemini_available(self) -> bool:
        """Returns True if GEMINI_API_KEY is configured and non-placeholder."""
        key = self.gemini_api_key.strip()
        return bool(key and not key.startswith("your-") and not key.startswith("AIzaSyYourKey"))

    @property
    def azure_available(self) -> bool:
        """Returns True only if all three Azure OpenAI variables are set and non-empty."""
        return bool(
            self.azure_openai_endpoint.strip()
            and self.azure_openai_key.strip()
            and self.azure_openai_deployment_name.strip()
            and not self.azure_openai_endpoint.startswith("https://your-resource")
            and not self.azure_openai_key.startswith("your-")
        )

    @property
    def llm_available(self) -> bool:
        """Returns True if either Gemini or Azure OpenAI is configured."""
        return self.gemini_available or self.azure_available

    @property
    def llm_mode(self) -> str:
        """Returns active LLM provider mode: 'gemini', 'azure_openai', or 'fallback'."""
        if self.gemini_available:
            return "gemini"
        if self.azure_available:
            return "azure_openai"
        return "fallback"

    # ── Path helpers ───────────────────────────────────────────────────────────
    @property
    def project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def data_raw_dir(self) -> str:
        return os.path.join(self.project_root, "data", "raw")

    @property
    def models_artifacts_dir(self) -> str:
        return os.path.join(self.project_root, "models", "artifacts")

    @property
    def model_registry_path(self) -> str:
        return os.path.join(self.project_root, "models", "registry", "model_registry.json")

    @property
    def baseline_distributions_path(self) -> str:
        return os.path.join(self.project_root, "data", "baseline_distributions.pkl")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    return PlatformSettings()


# Module-level singleton for easy import
settings = get_settings()
