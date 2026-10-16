"""
governance/drift_detector.py
─────────────────────────────
Pillar 2 – Statistical Data & Model Drift Detector.

Implements:
  • Kolmogorov-Smirnov (KS) test for continuous feature drift
  • Population Stability Index (PSI) for categorical/binned drift
  • Prediction drift (output distribution shift)
  • Rolling window comparison vs. baseline distributions
  • Automatic alerting via telemetry bus when drift threshold exceeded
"""

from __future__ import annotations

import pickle
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

KS_DRIFT_THRESHOLD = 0.1          # KS statistic > this → drift
PSI_WARNING_THRESHOLD = 0.1       # PSI 0.1–0.2 → slight drift
PSI_CRITICAL_THRESHOLD = 0.2      # PSI > 0.2 → significant drift
PREDICTION_DRIFT_THRESHOLD = 0.15 # Mean shift in predictions
WINDOW_SIZE = 200                  # Number of recent samples to compare
BASELINE_PATH = Path("data/baseline_distributions.pkl")

PSI_BINS = 10


# ──────────────────────────────────────────────────────────────────────────────
# PSI Calculation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index (PSI) — measures distributional shift."""
    # Create bins from expected distribution
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Avoid division by zero
    expected_pct = np.where(expected_counts == 0, 1e-6, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 1e-6, actual_counts / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 4))


# ──────────────────────────────────────────────────────────────────────────────
# Feature Window Tracker
# ──────────────────────────────────────────────────────────────────────────────

class _FeatureWindow:
    def __init__(self, feature_name: str, baseline: np.ndarray) -> None:
        self.feature_name = feature_name
        self.baseline = baseline
        self.window: deque = deque(maxlen=WINDOW_SIZE)

    def add(self, value: float) -> None:
        self.window.append(value)

    def analyze(self) -> dict:
        if len(self.window) < 30:
            return {"status": "INSUFFICIENT_DATA", "n_samples": len(self.window)}

        current = np.array(self.window)

        # KS-test
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(self.baseline, current)

        # PSI
        psi = _compute_psi(self.baseline, current)

        # Mean & std shift
        baseline_mean = float(np.mean(self.baseline))
        current_mean = float(np.mean(current))
        mean_shift_pct = (
            abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-9) * 100
        )

        drift_detected = bool(ks_stat > KS_DRIFT_THRESHOLD or psi > PSI_WARNING_THRESHOLD)
        psi = float(psi)

        return {
            "feature": self.feature_name,
            "n_samples": len(self.window),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
            "psi": psi,
            "psi_severity": (
                "CRITICAL" if psi > PSI_CRITICAL_THRESHOLD
                else "WARNING" if psi > PSI_WARNING_THRESHOLD
                else "STABLE"
            ),
            "baseline_mean": round(baseline_mean, 4),
            "current_mean": round(current_mean, 4),
            "mean_shift_pct": round(mean_shift_pct, 2),
            "drift_detected": drift_detected,
            "status": "DRIFT" if drift_detected else "STABLE",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Drift Detector (Singleton)
# ──────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Autonomous drift detector. Compares live inference feature distributions
    against training-time baselines using KS-test and PSI.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._baselines: Dict[str, Dict[str, np.ndarray]] = {}
        self._feature_windows: Dict[str, Dict[str, _FeatureWindow]] = {}
        self._prediction_windows: Dict[str, deque] = {}
        self._prediction_baselines: Dict[str, np.ndarray] = {}
        self._drift_history: deque = deque(maxlen=1000)
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load pre-computed baseline distributions from training."""
        if BASELINE_PATH.exists():
            try:
                with open(BASELINE_PATH, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if "." in k and isinstance(v, dict):
                            model_id, feat_name = k.split(".", 1)
                            mean = float(v.get("mean", 0.0))
                            std = float(v.get("std", 1.0)) or 1.0
                            low = float(v.get("min", mean - 3 * std))
                            high = float(v.get("max", mean + 3 * std))
                            rng = np.random.default_rng(42)
                            samples = np.clip(rng.normal(mean, std, 300), low, high)
                            self._baselines.setdefault(model_id, {})[feat_name] = samples
                        elif isinstance(v, dict):
                            for feat_name, feat_val in v.items():
                                if isinstance(feat_val, (list, np.ndarray)):
                                    self._baselines.setdefault(k, {})[feat_name] = np.array(feat_val)
                                elif isinstance(feat_val, dict):
                                    mean = float(feat_val.get("mean", 0.0))
                                    std = float(feat_val.get("std", 1.0)) or 1.0
                                    low = float(feat_val.get("min", mean - 3 * std))
                                    high = float(feat_val.get("max", mean + 3 * std))
                                    rng = np.random.default_rng(42)
                                    self._baselines.setdefault(k, {})[feat_name] = np.clip(rng.normal(mean, std, 300), low, high)
            except Exception as e:
                logger.debug(f"[DriftDetector] Baseline loading notice: {e}")

    def register_baseline(
        self,
        model_id: str,
        feature_distributions: Dict[str, np.ndarray],
        prediction_distribution: Optional[np.ndarray] = None,
    ) -> None:
        """Register or update baseline distributions for a model."""
        with self._lock:
            self._baselines[model_id] = {
                k: np.asarray(v) for k, v in feature_distributions.items()
            }
            if prediction_distribution is not None:
                self._prediction_baselines[model_id] = np.asarray(prediction_distribution)

    def _get_feature_window(self, model_id: str, feature: str) -> Optional[_FeatureWindow]:
        """Get or create a feature window if baseline exists."""
        if model_id not in self._baselines:
            return None
        if feature not in self._baselines[model_id]:
            return None
        if model_id not in self._feature_windows:
            self._feature_windows[model_id] = {}
        if feature not in self._feature_windows[model_id]:
            self._feature_windows[model_id][feature] = _FeatureWindow(
                feature, self._baselines[model_id][feature]
            )
        return self._feature_windows[model_id][feature]

    def observe(
        self,
        model_id: str,
        features: Dict[str, float],
        prediction: Optional[float] = None,
    ) -> None:
        """
        Feed a new inference observation. Updates rolling windows.
        Call this after every model prediction.
        """
        with self._lock:
            for feat_name, feat_val in features.items():
                try:
                    fw = self._get_feature_window(model_id, feat_name)
                    if fw:
                        fw.add(float(feat_val))
                except (TypeError, ValueError):
                    pass

            if prediction is not None:
                if model_id not in self._prediction_windows:
                    self._prediction_windows[model_id] = deque(maxlen=WINDOW_SIZE)
                try:
                    self._prediction_windows[model_id].append(float(prediction))
                except (TypeError, ValueError):
                    pass

    def analyze_model(self, model_id: str) -> dict:
        """Run full drift analysis for a model and return report."""
        with self._lock:
            if model_id not in self._baselines:
                return {
                    "model_id": model_id,
                    "status": "NO_BASELINE",
                    "feature_drift": [],
                    "prediction_drift": None,
                    "overall_drift_detected": False,
                }

            feature_reports = []
            any_drift = False

            for feature, fw in self._feature_windows.get(model_id, {}).items():
                report = fw.analyze()
                feature_reports.append(report)
                if report.get("drift_detected"):
                    any_drift = True

            # Prediction drift
            pred_drift = None
            if model_id in self._prediction_windows and model_id in self._prediction_baselines:
                pred_win = np.array(self._prediction_windows[model_id])
                pred_base = self._prediction_baselines[model_id]
                if len(pred_win) >= 30:
                    ks_stat, ks_pval = scipy_stats.ks_2samp(pred_base, pred_win)
                    pred_shift = abs(np.mean(pred_win) - np.mean(pred_base))
                    pred_drift_detected = bool(ks_stat > PREDICTION_DRIFT_THRESHOLD)
                    pred_drift = {
                        "ks_statistic": round(float(ks_stat), 4),
                        "ks_pvalue": round(float(ks_pval), 4),
                        "mean_shift": round(float(pred_shift), 4),
                        "drift_detected": pred_drift_detected,
                    }
                    if pred_drift_detected:
                        any_drift = True

            result = {
                "model_id": model_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_drift_detected": any_drift,
                "status": "DRIFT" if any_drift else "STABLE",
                "feature_drift": feature_reports,
                "prediction_drift": pred_drift,
                "features_monitored": len(feature_reports),
            }

            now = time.time()
            last_pub = getattr(self, "_last_drift_pub", {}).get(model_id, 0.0)
            if any_drift and now - last_pub >= 60:      # one alert per model per minute, not one per poll
                if not hasattr(self, "_last_drift_pub"):
                    self._last_drift_pub = {}
                self._last_drift_pub[model_id] = now
                self._drift_history.append(result)
                telemetry_bus.publish(
                    module_id=model_id,
                    event_type="MODEL_DRIFT_DETECTED",
                    payload={
                        "drifted_features": [
                            r["feature"] for r in feature_reports if r.get("drift_detected")
                        ],
                        "prediction_drift": pred_drift,
                    },
                    severity="CRITICAL" if any(
                        r.get("psi_severity") == "CRITICAL" for r in feature_reports
                    ) else "WARNING",
                )

            return result

    def analyze_all_models(self) -> List[dict]:
        """Run drift analysis across all registered models."""
        model_ids = list(self._baselines.keys())
        return [self.analyze_model(mid) for mid in model_ids]

    def reset(self) -> None:
        """Clear live observation windows (baselines are kept). Used by 'Reset to healthy'."""
        with self._lock:
            self._feature_windows.clear()
            self._prediction_windows.clear()
            self._drift_history.clear()
            if hasattr(self, "_last_drift_pub"):
                self._last_drift_pub.clear()

    def get_drift_history(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._drift_history)[-n:]

    def get_summary(self) -> dict:
        """Quick summary: which models are drifting."""
        with self._lock:
            return {
                "models_registered": list(self._baselines.keys()),
                "models_with_windows": list(self._feature_windows.keys()),
                "recent_drift_events": len(self._drift_history),
            }


# ── Singleton export ──────────────────────────────────────────────────────────
drift_detector = DriftDetector()
