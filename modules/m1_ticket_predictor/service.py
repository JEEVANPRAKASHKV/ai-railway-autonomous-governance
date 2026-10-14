"""
modules/m1_ticket_predictor/service.py
======================================
Inference service for Module 1: Intelligent Ticket Confirmation Predictor.
Wrapped with @governance_hook for telemetry, data quality gate, and latency audit.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import settings
from governance.hooks import governance_hook
from modules.m1_ticket_predictor.schema import (
    ShapFeatureContribution,
    TicketPredictRequest,
    TicketPredictResponse,
)

logger = logging.getLogger(__name__)

FEATURE_NAME_MAP = {
    "initial_wl_number": "Current Waitlist Number",
    "class_enc": "Travel Class",
    "quota_enc": "Booking Quota",
    "days_to_departure": "Days Ahead of Departure",
    "month": "Travel Month",
    "day_of_week": "Day of the Week",
    "is_peak_season": "Peak Festival / Holiday Period",
    "season_factor": "Seasonal Demand Multiplier",
    "seniority_flag": "Senior Citizen Concession",
    "disability_flag": "Divyangjan Quota Privilege",
}


class TicketPredictorService:
    """Service to predict ticket confirmation probability, berth allocation, and SHAP explainability."""

    def __init__(self) -> None:
        self._wl_bundle: Optional[Dict[str, Any]] = None
        self._berth_bundle: Optional[Dict[str, Any]] = None
        self._explainer: Optional[Any] = None
        self._load_models()

    def _load_models(self) -> None:
        art_dir = settings.models_artifacts_dir
        wl_path = os.path.join(art_dir, "wl_predictor_v1.0.pkl")
        berth_path = os.path.join(art_dir, "berth_estimator_v1.0.pkl")
        shap_path = os.path.join(art_dir, "shap_explainers", "shap_wl_predictor.pkl")

        if os.path.exists(wl_path):
            try:
                with open(wl_path, "rb") as f:
                    self._wl_bundle = pickle.load(f)
            except Exception as e:
                logger.error(f"[M1 Service] Error loading wl_predictor: {e}")

        if os.path.exists(berth_path):
            try:
                with open(berth_path, "rb") as f:
                    self._berth_bundle = pickle.load(f)
            except Exception as e:
                logger.error(f"[M1 Service] Error loading berth_estimator: {e}")

        if os.path.exists(shap_path):
            try:
                with open(shap_path, "rb") as f:
                    self._explainer = pickle.load(f)
            except Exception as e:
                logger.warning(f"[M1 Service] Could not load SHAP explainer: {e}")

    def _encode_safe(self, encoder: Any, val: str, default: str = "SL") -> int:
        try:
            return int(encoder.transform([str(val)])[0])
        except Exception:
            try:
                return int(encoder.transform([default])[0])
            except Exception:
                return 0

    @governance_hook("m1_ticket_predictor", endpoint="/api/v1/predict/ticket")
    def predict(self, request: TicketPredictRequest) -> TicketPredictResponse:
        """Run ML inference for ticket confirmation, berth estimate, and explainability."""
        b_date = request.booking_date or date.today()
        days_to_dept = max(0, (request.journey_date - b_date).days)
        j_month = request.journey_date.month
        j_dow = request.journey_date.weekday()
        is_peak = 1 if j_month in [5, 6, 10, 11, 12] else 0
        season_factor = 1.4 if is_peak else 1.0

        class_enc_val = 0
        quota_enc_val = 0
        if self._wl_bundle and "class_encoder" in self._wl_bundle:
            class_enc_val = self._encode_safe(self._wl_bundle["class_encoder"], request.journey_class, "SL")
        if self._wl_bundle and "quota_encoder" in self._wl_bundle:
            quota_enc_val = self._encode_safe(self._wl_bundle["quota_encoder"], request.quota, "GN")

        # Feature vector
        feature_dict = {
            "initial_wl_number": float(request.current_wl_number),
            "class_enc": float(class_enc_val),
            "quota_enc": float(quota_enc_val),
            "days_to_departure": float(days_to_dept),
            "month": float(j_month),
            "day_of_week": float(j_dow),
            "is_peak_season": float(is_peak),
            "season_factor": float(season_factor),
            "seniority_flag": float(int(request.seniority_flag)),
            "disability_flag": float(int(request.disability_flag)),
        }

        # ── Predict confirmation probability ──────────────────────────────────
        confirm_prob = 0.50
        pnr_status = "Borderline"
        confidence_score = 0.85

        if self._wl_bundle and "model" in self._wl_bundle:
            model = self._wl_bundle["model"]
            target_enc = self._wl_bundle.get("target_encoder")
            feature_cols = self._wl_bundle.get("feature_cols", list(feature_dict.keys()))

            input_df = pd.DataFrame([[feature_dict[c] for c in feature_cols]], columns=feature_cols)
            probs = model.predict_proba(input_df)[0]
            classes = list(target_enc.classes_) if target_enc else ["Confirmed", "RAC", "WL"]

            prob_map = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
            # Probability that ticket becomes usable (Confirmed + 0.65*RAC)
            raw_confirm = prob_map.get("Confirmed", 0.0) + 0.65 * prob_map.get("RAC", 0.0)
            confirm_prob = round(float(np.clip(raw_confirm, 0.02, 0.98)), 4)

            confidence_score = round(float(np.max(probs)), 4)
        else:
            # Deterministic heuristic fallback if artifact unmounted
            base = 0.85 - (request.current_wl_number * 0.015) + (days_to_dept * 0.005)
            if request.journey_class in ["1A", "2A"]:
                base -= 0.10
            confirm_prob = round(float(np.clip(base, 0.05, 0.95)), 4)

        if confirm_prob >= 0.72:
            pnr_status = "Likely Confirmed"
        elif confirm_prob >= 0.40:
            pnr_status = "Borderline"
        else:
            pnr_status = "Likely WL"

        # ── Predict Berth Type ────────────────────────────────────────────────
        estimated_berth = "Side Lower" if request.seniority_flag else "Middle"
        berth_breakdown: Dict[str, float] = {
            "Lower": 0.20, "Middle": 0.20, "Upper": 0.20,
            "Side Lower": 0.20, "Side Upper": 0.15, "RAC": 0.05
        }

        if self._berth_bundle and "model" in self._berth_bundle:
            b_model = self._berth_bundle["model"]
            b_enc = self._berth_bundle.get("berth_encoder")
            b_cols = self._berth_bundle.get("feature_cols", [
                "initial_wl_number", "class_enc", "quota_enc",
                "days_to_departure", "month", "is_peak_season",
                "seniority_flag", "disability_flag",
            ])
            b_input = pd.DataFrame([[feature_dict[c] for c in b_cols]], columns=b_cols)
            b_probs = b_model.predict_proba(b_input)[0]
            b_classes = list(b_enc.classes_) if b_enc else list(berth_breakdown.keys())

            berth_breakdown = {b_cls: round(float(b_probs[i]), 3) for i, b_cls in enumerate(b_classes)}
            if request.seniority_flag or request.disability_flag:
                # Prioritize lower berth quota policy
                berth_breakdown["Lower"] = min(0.85, berth_breakdown.get("Lower", 0.2) + 0.35)
                # Re-normalize
                total = sum(berth_breakdown.values())
                berth_breakdown = {k: round(v / total, 3) for k, v in berth_breakdown.items()}

            estimated_berth = max(berth_breakdown, key=berth_breakdown.get)

        # ── Compute SHAP Explainability ───────────────────────────────────────
        shap_explanations: List[ShapFeatureContribution] = []
        if self._explainer is not None and self._wl_bundle:
            try:
                feature_cols = self._wl_bundle.get("feature_cols", list(feature_dict.keys()))
                input_df = pd.DataFrame([[feature_dict[c] for c in feature_cols]], columns=feature_cols)
                shap_vals = self._explainer.shap_values(input_df)

                # For multiclass, take Confirmed class if index available or average
                if isinstance(shap_vals, list) and len(shap_vals) > 0:
                    vals = shap_vals[0][0]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    vals = shap_vals[0, :, 0]
                else:
                    vals = shap_vals[0]

                for col, val in zip(feature_cols, vals):
                    val_float = float(val)
                    shap_explanations.append(
                        ShapFeatureContribution(
                            feature=col,
                            feature_name=FEATURE_NAME_MAP.get(col, col.replace("_", " ").title()),
                            value=feature_dict[col],
                            contribution=round(val_float, 4),
                            impact="Increases Confirmation Chance" if val_float > 0 else "Decreases Confirmation Chance",
                        )
                    )
                # Sort by absolute impact
                shap_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)
            except Exception as e:
                logger.debug(f"[M1 Service] SHAP computation fallback: {e}")

        if not shap_explanations:
            # Deterministic feature contribution fallback
            shap_explanations = [
                ShapFeatureContribution(
                    feature="days_to_departure",
                    feature_name="Days Ahead of Departure",
                    value=days_to_dept,
                    contribution=round(0.005 * days_to_dept, 3),
                    impact="Increases Confirmation Chance" if days_to_dept > 10 else "Decreases Confirmation Chance",
                ),
                ShapFeatureContribution(
                    feature="initial_wl_number",
                    feature_name="Current Waitlist Number",
                    value=request.current_wl_number,
                    contribution=round(-0.02 * request.current_wl_number, 3),
                    impact="Decreases Confirmation Chance",
                ),
                ShapFeatureContribution(
                    feature="seniority_flag",
                    feature_name="Senior Citizen Concession",
                    value=int(request.seniority_flag),
                    contribution=0.15 if request.seniority_flag else 0.0,
                    impact="Increases Confirmation Chance" if request.seniority_flag else "Neutral",
                ),
            ]

        # ── Booking Advisory Generation ───────────────────────────────────────
        if pnr_status == "Likely Confirmed":
            advisory = (
                f"High confirmation odds ({confirm_prob * 100:.1f}%). "
                f"Estimated seat allocation: {estimated_berth}. No immediate cancellation needed; "
                "chart preparation usually clears remaining WL positions."
            )
        elif pnr_status == "Borderline":
            advisory = (
                f"Borderline confirmation odds ({confirm_prob * 100:.1f}%). "
                f"Recommended Action: Book an alternate train or keep Tatkal booking ready "
                "24 hours before journey date at 10:00 AM (AC) / 11:00 AM (Non-AC)."
            )
        else:
            advisory = (
                f"Low probability of confirmation ({confirm_prob * 100:.1f}%). "
                "High probability of remaining on waitlist after chart preparation. "
                "Strongly advised to book an alternate train immediately to guarantee travel."
            )

        return TicketPredictResponse(
            pnr_status_prediction=pnr_status,
            confirmation_probability=confirm_prob,
            estimated_berth_type=estimated_berth,
            berth_probability_breakdown=berth_breakdown,
            shap_explanation=shap_explanations[:5],
            booking_advisory=advisory,
            confidence_score=confidence_score,
        )


ticket_predictor_service = TicketPredictorService()
