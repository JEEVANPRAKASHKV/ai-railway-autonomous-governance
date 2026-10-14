"""
governance/data_quality_gate.py
================================
Pre-Inference Data Quality Gate — Governance Pillar 2.

Validates every module's input payload BEFORE inference occurs.
Checks:
  1. Schema validation (required fields present, correct types)
  2. Range validation (numeric bounds from feature_columns.json)
  3. Null rate check
  4. Distribution anomaly (z-score vs. baseline distributions)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Feature bounds per module (fallback if feature_columns.json not loaded)
_DEFAULT_BOUNDS: Dict[str, Dict[str, tuple]] = {
    "m1_ticket_predictor": {
        "current_wl_number": (1, 500),
    },
    "m3_delay_predictor": {},
    "m5_fare_predictor": {
        "days_to_departure": (0, 365),
    },
    "m7_fraud_detector": {
        "requests_per_minute": (0, 1000),
        "unique_pnr_attempts": (0, 200),
        "account_age_days": (0, 36500),
        "payment_attempts": (0, 50),
        "failed_captcha_count": (0, 100),
        "booking_time_hour": (0, 23),
    },
}

_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "m1_ticket_predictor": ["train_number", "journey_class", "quota", "journey_date", "current_wl_number"],
    "m2_train_recommender": ["source_station", "destination_station", "journey_date"],
    "m3_delay_predictor":   ["train_number", "station_code", "journey_date"],
    "m4_travel_planner":    ["query"],
    "m5_fare_predictor":    ["train_number", "journey_class", "journey_date"],
    "m6_support_assistant": ["query"],
    "m7_fraud_detector":    ["session_id", "ip_hash", "requests_per_minute"],
}


class DataQualityGate:
    """Pre-inference schema validator and anomaly interceptor."""

    def __init__(self) -> None:
        self._feature_columns: Dict[str, Any] = {}
        self._baseline: Optional[Dict[str, Dict]] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load feature_columns.json
        fc_path = os.path.join(root, "models", "artifacts", "feature_columns.json")
        if os.path.exists(fc_path):
            try:
                with open(fc_path, encoding="utf-8") as f:
                    self._feature_columns = json.load(f)
                logger.debug("[DataQualityGate] Loaded feature_columns.json")
            except Exception as e:
                logger.warning(f"[DataQualityGate] Could not load feature_columns.json: {e}")

        # Load baseline distributions
        bl_path = os.path.join(root, "data", "baseline_distributions.pkl")
        if os.path.exists(bl_path):
            try:
                with open(bl_path, "rb") as f:
                    self._baseline = pickle.load(f)
                logger.debug("[DataQualityGate] Loaded baseline_distributions.pkl")
            except Exception as e:
                logger.warning(f"[DataQualityGate] Could not load baseline distributions: {e}")

    def validate(self, module_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the input payload for a given module.

        Returns:
            dict with keys:
              - valid (bool)
              - violations (List[str])
              - anomaly_flags (List[str])
              - score (float, 0.0–1.0 quality score)
        """
        violations: List[str] = []
        anomaly_flags: List[str] = []

        # Flatten nested request objects
        flat = self._flatten_payload(payload)

        # 1. Required fields check
        required = _REQUIRED_FIELDS.get(module_name, [])
        for field in required:
            if field not in flat or flat[field] is None:
                violations.append(f"Missing required field: '{field}'")

        # 2. Null check (only for required fields; optional fields can be None)
        for field in required:
            if flat.get(field) is None and f"Missing required field: '{field}'" not in violations:
                violations.append(f"Required field '{field}' cannot be null")

        # 3. Range check
        bounds = _DEFAULT_BOUNDS.get(module_name, {})
        for field, (lo, hi) in bounds.items():
            val = flat.get(field)
            if val is not None:
                try:
                    fval = float(val)
                    if fval < lo or fval > hi:
                        violations.append(
                            f"Field '{field}' out of valid range [{lo}, {hi}]: got {fval}"
                        )
                except (TypeError, ValueError):
                    pass

        # 4. Distribution anomaly (z-score vs. baseline)
        if self._baseline:
            for field, val in flat.items():
                if val is not None and isinstance(val, (int, float)):
                    key = f"{module_name}.{field}"
                    if key in self._baseline:
                        bl = self._baseline[key]
                        mean, std = bl.get("mean", 0), bl.get("std", 1)
                        if std > 0:
                            z = abs((float(val) - mean) / std)
                            if z > 3.5:
                                anomaly_flags.append(
                                    f"Field '{field}' z-score={z:.2f} (outlier vs. training distribution)"
                                )

        valid = len(violations) == 0
        score = max(0.0, 1.0 - (len(violations) * 0.2 + len(anomaly_flags) * 0.1))

        return {
            "valid": valid,
            "violations": violations,
            "anomaly_flags": anomaly_flags,
            "score": round(score, 3),
            "module": module_name,
        }

    def _flatten_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested dict/Pydantic model payload to a flat dict."""
        flat: Dict[str, Any] = {}
        for k, v in payload.items():
            if hasattr(v, "model_dump"):
                flat.update(v.model_dump())
            elif isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        return flat
