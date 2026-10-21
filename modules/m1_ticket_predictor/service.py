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
from core.seasonality import is_peak, season_factor
from governance.context import report_inference
from governance.hooks import governance_hook
from governance.lifecycle_manager import lifecycle_manager
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
        self._history: Optional[pd.DataFrame] = None
        self._load_models()
        self._load_history()

    def _load_history(self) -> None:
        try:
            h = pd.read_csv(os.path.join(settings.data_raw_dir, "pnr_history.csv"),
                            usecols=["train_number", "journey_class", "quota", "journey_date", "days_to_departure",
                                     "initial_wl_number", "final_status", "berth_type", "seniority_flag",
                                     "cancellation_date"])
            h["month"] = pd.to_datetime(h["journey_date"]).dt.month
            h["cancelled"] = h["cancellation_date"].notna()
            self._history = h
        except Exception as e:
            logger.debug(f"[M1 Service] booking history unavailable: {e}")

    def historical_insights(self, req: TicketPredictRequest, days_to_dept: int) -> Dict[str, Any]:
        """Booking & cancellation trends from comparable historical PNRs (same class & quota, similar WL)."""
        h = self._history
        if h is None:
            return {}
        lo, hi = max(1, req.current_wl_number - 10), req.current_wl_number + 10
        peers = h[(h["journey_class"] == req.journey_class) & (h["quota"] == req.quota)
                  & h["initial_wl_number"].between(lo, hi)]
        scope = f"{req.journey_class} / {req.quota}, WL {lo}–{hi}"
        if len(peers) < 30:
            peers = h[(h["journey_class"] == req.journey_class) & h["initial_wl_number"].between(lo, hi)]
            scope = f"{req.journey_class}, any quota, WL {lo}–{hi}"
        if not len(peers):
            return {}
        rates = peers["final_status"].value_counts(normalize=True)
        train_peers = h[h["train_number"].astype(str) == str(req.train_number)]
        monthly = (peers.groupby("month")["final_status"].apply(lambda s: float((s == "Confirmed").mean()))
                   .round(3).to_dict())
        closer = peers[peers["days_to_departure"] <= days_to_dept]
        return {
            "comparable_bookings": int(len(peers)),
            "scope": scope,
            "historical_confirmed_pct": round(float(rates.get("Confirmed", 0)) * 100, 1),
            "historical_rac_pct": round(float(rates.get("RAC", 0)) * 100, 1),
            "historical_wl_pct": round(float(rates.get("WL", 0)) * 100, 1),
            "cancellation_rate_pct": round(float(peers["cancelled"].mean()) * 100, 1),
            "confirmed_pct_when_booked_this_close": (round(float((closer["final_status"] == "Confirmed").mean()) * 100, 1)
                                                     if len(closer) >= 20 else None),
            "this_train_bookings": int(len(train_peers)),
            "this_train_confirmed_pct": (round(float((train_peers["final_status"] == "Confirmed").mean()) * 100, 1)
                                         if len(train_peers) >= 20 else None),
            "confirmed_pct_by_month": {int(k): v for k, v in monthly.items()},
        }

    def empirical_berth(self, req: TicketPredictRequest) -> Dict[str, Any]:
        """Berth split of comparable CONFIRMED bookings — calibrated, unlike the weak berth classifier."""
        h = self._history
        if h is None:
            return {}
        conf = h[(h["final_status"] == "Confirmed") & (h["journey_class"] == req.journey_class)]
        if req.seniority_flag:
            sen = conf[conf["seniority_flag"] == 1]
            conf = sen if len(sen) >= 50 else conf
        conf = conf[conf["berth_type"].isin(["Lower", "Middle", "Upper", "Side Lower", "Side Upper", "RAC"])]
        if len(conf) < 30:
            return {}
        dist = conf["berth_type"].value_counts(normalize=True).round(3).to_dict()
        return {"distribution": dist, "n": int(len(conf))}

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
        peak = is_peak(request.journey_date)
        sf = season_factor(request.journey_date)

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
            "is_peak_season": float(peak),
            "season_factor": float(sf),
            "seniority_flag": float(int(request.seniority_flag)),
            "disability_flag": float(int(request.disability_flag)),
        }

        # ── Predict confirmation probability ──────────────────────────────────
        confirm_prob = 0.50
        pnr_status = "Borderline"
        confidence_score = 0.85

        wl_bundle, version = lifecycle_manager.serving_bundle("m1_ticket_predictor", self._wl_bundle)
        prob_map: Dict[str, float] = {}
        if wl_bundle and "model" in wl_bundle:
            model = wl_bundle["model"]
            target_enc = wl_bundle.get("target_encoder")
            feature_cols = wl_bundle.get("feature_cols", list(feature_dict.keys()))

            input_df = pd.DataFrame([[feature_dict[c] for c in feature_cols]], columns=feature_cols)
            probs = model.predict_proba(input_df)[0]
            classes = list(target_enc.classes_) if target_enc else ["Confirmed", "RAC", "WL"]

            prob_map = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
            # Probability that ticket becomes usable (Confirmed + 0.65*RAC)
            raw_confirm = prob_map.get("Confirmed", 0.0) + 0.65 * prob_map.get("RAC", 0.0)
            confirm_prob = round(float(np.clip(raw_confirm, 0.02, 0.98)), 4)

            confidence_score = round(float(np.max(probs)), 4)
            report_inference(features=feature_dict,
                             prediction={"label": max(prob_map, key=prob_map.get), "p_confirm": confirm_prob,
                                         "probs": {k: round(v, 4) for k, v in prob_map.items()}},
                             version=version)
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

        # ── Berth estimate ────────────────────────────────────────────────────
        # The trained berth classifier scores 27.7% accuracy (below always guessing the most common berth),
        # so it is not served. The estimate is the berth split of comparable confirmed bookings instead.
        emp = self.empirical_berth(request)
        berth_breakdown: Dict[str, float] = emp.get("distribution") or {
            "Lower": 0.2, "Middle": 0.2, "Upper": 0.2, "Side Lower": 0.15, "Side Upper": 0.15, "RAC": 0.1}
        berth_basis = (f"Share of {emp['n']:,} comparable confirmed bookings" if emp
                       else "Default split (no comparable history)")
        estimated_berth = max(berth_breakdown, key=berth_breakdown.get)

        # ── Compute SHAP Explainability ───────────────────────────────────────
        shap_explanations: List[ShapFeatureContribution] = []
        explainer = self._explainer if wl_bundle is self._wl_bundle else None
        if explainer is None and wl_bundle is not None:
            try:
                import shap
                explainer = shap.TreeExplainer(wl_bundle["model"])
            except Exception:
                explainer = None
        if explainer is not None and wl_bundle:
            try:
                feature_cols = wl_bundle.get("feature_cols", list(feature_dict.keys()))
                input_df = pd.DataFrame([[feature_dict[c] for c in feature_cols]], columns=feature_cols)
                shap_vals = explainer.shap_values(input_df)

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

        insights = self.historical_insights(request, days_to_dept)
        return TicketPredictResponse(
            historical_insights=insights,
            berth_estimate_basis=berth_basis,
            status_probabilities={k: round(v, 3) for k, v in prob_map.items()},
            model_version=version if prob_map else "heuristic",
            pnr_status_prediction=pnr_status,
            confirmation_probability=confirm_prob,
            estimated_berth_type=estimated_berth,
            berth_probability_breakdown=berth_breakdown,
            shap_explanation=shap_explanations[:5],
            booking_advisory=advisory,
            confidence_score=confidence_score,
        )


ticket_predictor_service = TicketPredictorService()
