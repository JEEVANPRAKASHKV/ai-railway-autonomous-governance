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

import logging
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

KS_DRIFT_THRESHOLD = 0.20         # KS effect size that matters…
KS_PVALUE_THRESHOLD = 0.001       # …and is statistically significant (both required)
PSI_WARNING_THRESHOLD = 0.10      # small-sample-corrected PSI 0.10–0.25 → watch
PSI_CRITICAL_THRESHOLD = 0.25     # corrected PSI > 0.25 → significant shift
PREDICTION_DRIFT_THRESHOLD = 0.25 # KS on the prediction distribution
MIN_SAMPLES = 50                  # never judge drift on fewer live samples
WINDOW_SIZE = 300                 # Number of recent samples to compare
ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "data" / "baseline_distributions.pkl"
RAW_DIR = ROOT / "data" / "raw"

PSI_BINS = 10
logger = logging.getLogger(__name__)

# Features that are NOT monitored for input drift: targets (never seen at inference) and values the
# service derives deterministically from the calendar (their "drift" is just the season changing).
NOT_MONITORED = {
    "m1_ticket_predictor": {"season_factor"},
    "m3_delay_predictor": {"delay_minutes"},
    "m5_fare_predictor": {"dynamic_fare", "demand_index"},
}

# Training-data columns used as the empirical reference for each monitored feature. Using the actual
# training rows (not a histogram re-sample) keeps heavily skewed features — request rates, fares — exact.
EMPIRICAL_BASELINES = {
    "m1_ticket_predictor": ("pnr_history.csv", ["days_to_departure", "initial_wl_number"]),
    "m3_delay_predictor": ("delay_logs.csv", ["weather_index", "congestion_at_station", "distance_from_origin_km"]),
    "m5_fare_predictor": ("pricing_history.csv", ["days_to_departure", "base_fare"]),
    "m7_fraud_detector": ("fraud_events.csv", ["requests_per_minute", "unique_pnr_attempts", "account_age_days",
                                               "payment_attempts", "failed_captcha_count"]),
}

# Categorical request fields → (training file, column) used as the "normal behaviour" reference
BEHAVIOUR_BASELINES = {
    "m1_ticket_predictor": {"journey_class": ("pnr_history.csv", "journey_class"),
                            "quota": ("pnr_history.csv", "quota")},
    "m5_fare_predictor": {"journey_class": ("pricing_history.csv", "journey_class")},
    "m2_train_recommender": {"persona": ("user_profiles.csv", "persona"),
                             "preferred_class": ("user_profiles.csv", "preferred_class")},
}


# ──────────────────────────────────────────────────────────────────────────────
# PSI Calculation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS,
                 edges: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Population Stability Index on quantile bins of the baseline.
    Returns (raw PSI, small-sample-corrected PSI). With n live samples and k bins, PSI has a positive
    bias of roughly (k-1)/n even when nothing changed; the corrected value subtracts it so that a
    50-request window does not look like drift.
    """
    if edges is None:
        qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges = qs if len(qs) > 2 else np.linspace(expected.min(), expected.max() + 1e-9, bins + 1)
    edges = np.array(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    expected_counts = np.histogram(expected, bins=edges)[0]
    actual_counts = np.histogram(actual, bins=edges)[0]
    e = np.clip(expected_counts / max(1, expected_counts.sum()), 1e-4, None)
    a = np.clip(actual_counts / max(1, actual_counts.sum()), 1e-4, None)
    psi = float(np.sum((a - e) * np.log(a / e)))
    k = int((expected_counts > 0).sum())
    corrected = max(0.0, psi - (k - 1) / max(1, len(actual)))
    return round(psi, 4), round(corrected, 4)


def _sample_from_histogram(counts: List[float], edges: List[float], n: int = 3000, seed: int = 42) -> np.ndarray:
    """Draw a faithful sample of the training distribution from its stored histogram."""
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    rng = np.random.default_rng(seed)
    probs = counts / counts.sum()
    idx = rng.choice(len(counts), size=n, p=probs)
    return rng.uniform(edges[idx], edges[idx + 1])


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
        if len(self.window) < MIN_SAMPLES:
            return {"feature": self.feature_name, "status": "INSUFFICIENT_DATA",
                    "n_samples": len(self.window), "drift_detected": False}

        current = np.array(self.window)
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(self.baseline, current)
        psi_raw, psi = _compute_psi(self.baseline, current)

        baseline_mean = float(np.mean(self.baseline))
        current_mean = float(np.mean(current))
        mean_shift_pct = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-9) * 100

        ks_drift = bool(ks_stat > KS_DRIFT_THRESHOLD and ks_pvalue < KS_PVALUE_THRESHOLD)
        drift_detected = bool(ks_drift or psi > PSI_CRITICAL_THRESHOLD)

        return {
            "feature": self.feature_name,
            "n_samples": len(self.window),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "psi": psi,
            "psi_raw": psi_raw,
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


class _CategoryWindow:
    """Tracks the mix of a categorical request field (user-behaviour change detection)."""

    def __init__(self, field: str, baseline_props: Dict[str, float]) -> None:
        self.field = field
        self.baseline = baseline_props
        self.window: deque = deque(maxlen=WINDOW_SIZE)

    def analyze(self) -> dict:
        n = len(self.window)
        if n < MIN_SAMPLES:
            return {"field": self.field, "status": "INSUFFICIENT_DATA", "n_samples": n, "shift_detected": False}
        cats = sorted(set(self.baseline) | set(self.window))
        cur = {c: sum(1 for x in self.window if x == c) / n for c in cats}
        psi = 0.0
        for c in cats:
            e = max(self.baseline.get(c, 0.0), 1e-4)
            a = max(cur.get(c, 0.0), 1e-4)
            psi += (a - e) * np.log(a / e)
        k = sum(1 for c in cats if self.baseline.get(c, 0) > 0)
        corrected = max(0.0, psi - (k - 1) / n)
        top = sorted(cats, key=lambda c: abs(cur.get(c, 0) - self.baseline.get(c, 0)), reverse=True)[:3]
        return {"field": self.field, "n_samples": n, "psi": round(float(corrected), 4),
                "shift_detected": bool(corrected > PSI_CRITICAL_THRESHOLD),
                "status": "SHIFT" if corrected > PSI_CRITICAL_THRESHOLD else "STABLE",
                "largest_changes": [{"category": c, "baseline_pct": round(self.baseline.get(c, 0) * 100, 1),
                                     "live_pct": round(cur.get(c, 0) * 100, 1)} for c in top]}


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
        self._cat_baselines: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._cat_windows: Dict[str, Dict[str, _CategoryWindow]] = {}
        self._load_baselines()
        self._load_empirical_baselines()
        self._load_behaviour_baselines()

    def _load_empirical_baselines(self) -> None:
        try:
            import pandas as pd
        except Exception:
            return
        for model_id, (fname, cols) in EMPIRICAL_BASELINES.items():
            try:
                df = pd.read_csv(RAW_DIR / fname, usecols=cols)
                sample = df.sample(min(5000, len(df)), random_state=42)
                for col in cols:
                    self._baselines.setdefault(model_id, {})[col] = sample[col].dropna().to_numpy(dtype=float)
            except Exception as e:
                logger.debug(f"[DriftDetector] empirical baseline {model_id}: {e}")

    def _load_behaviour_baselines(self) -> None:
        """Reference mix of categorical request fields, taken from the training data itself."""
        try:
            import pandas as pd
        except Exception:
            return
        cache: Dict[str, "pd.DataFrame"] = {}
        for model_id, fields in BEHAVIOUR_BASELINES.items():
            for field, (fname, col) in fields.items():
                try:
                    if fname not in cache:
                        cache[fname] = pd.read_csv(RAW_DIR / fname, usecols=None)
                    vc = cache[fname][col].astype(str).value_counts(normalize=True)
                    self._cat_baselines.setdefault(model_id, {})[field] = {k: float(v) for k, v in vc.items()}
                except Exception as e:
                    logger.debug(f"[DriftDetector] behaviour baseline {model_id}.{field}: {e}")

    def observe_categories(self, model_id: str, values: Dict[str, str]) -> None:
        with self._lock:
            for field, val in values.items():
                base = self._cat_baselines.get(model_id, {}).get(field)
                if not base:
                    continue
                win = self._cat_windows.setdefault(model_id, {}).setdefault(field, _CategoryWindow(field, base))
                win.window.append(str(val))

    def behaviour_report(self) -> List[dict]:
        """User-behaviour change: has the mix of classes / quotas / personas people ask about shifted?"""
        out = []
        with self._lock:
            for model_id, wins in self._cat_windows.items():
                fields = [w.analyze() for w in wins.values()]
                out.append({"model_id": model_id, "fields": fields,
                            "shift_detected": any(f.get("shift_detected") for f in fields)})
        for rep in out:
            if rep["shift_detected"]:
                now = time.time()
                key = "beh:" + rep["model_id"]
                if not hasattr(self, "_last_drift_pub"):
                    self._last_drift_pub = {}
                if now - self._last_drift_pub.get(key, 0) >= 120:
                    self._last_drift_pub[key] = now
                    telemetry_bus.publish(module_id=rep["model_id"], event_type="USER_BEHAVIOUR_SHIFT",
                                          payload={"fields": [f["field"] for f in rep["fields"] if f.get("shift_detected")]},
                                          severity="INFO")
        return out

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
                            if v.get("hist_counts") and v.get("hist_edges"):
                                # Faithful to the real (often uniform or heavily skewed) training distribution
                                samples = _sample_from_histogram(v["hist_counts"], v["hist_edges"])
                            else:
                                mean = float(v.get("mean", 0.0))
                                std = float(v.get("std", 1.0)) or 1.0
                                rng = np.random.default_rng(42)
                                samples = np.clip(rng.normal(mean, std, 3000),
                                                  float(v.get("min", mean - 3 * std)), float(v.get("max", mean + 3 * std)))
                            if feat_name in NOT_MONITORED.get(model_id, set()):
                                continue
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
            self._cat_windows.clear()
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
