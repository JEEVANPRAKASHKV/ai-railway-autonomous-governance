"""
modules/m2_train_recommender/service.py
======================================
Inference service for Module 2: Smart Train Recommendation Engine.
Implements multi-factor persona scoring, bias-aware ranking, and Disparate Impact Ratio tracking.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import settings
from governance.hooks import governance_hook
from modules.m2_train_recommender.schema import (
    RecommendRequest,
    RecommendResponse,
    TrainRecommendation,
)

logger = logging.getLogger(__name__)

# Persona-specific scoring weights
PERSONA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "budget": {
        "price": 0.40,
        "punctuality": 0.20,
        "availability": 0.20,
        "duration": 0.10,
        "comfort": 0.10,
    },
    "premium": {
        "price": 0.05,
        "punctuality": 0.25,
        "availability": 0.10,
        "duration": 0.25,
        "comfort": 0.35,
    },
    "senior": {
        "price": 0.10,
        "punctuality": 0.30,
        "availability": 0.10,
        "duration": 0.15,
        "comfort": 0.35,
    },
    "student": {
        "price": 0.45,
        "punctuality": 0.15,
        "availability": 0.20,
        "duration": 0.10,
        "comfort": 0.10,
    },
    "family": {
        "price": 0.15,
        "punctuality": 0.20,
        "availability": 0.25,
        "duration": 0.10,
        "comfort": 0.30,
    },
    "business": {
        "price": 0.05,
        "punctuality": 0.35,
        "availability": 0.05,
        "duration": 0.40,
        "comfort": 0.15,
    },
}

TRAIN_TYPE_COMFORT = {
    "Vande Bharat Express": 0.95,
    "Rajdhani Express": 0.92,
    "Shatabdi Express": 0.90,
    "Duronto Express": 0.85,
    "Garib Rath Express": 0.70,
    "Superfast Express": 0.75,
    "Mail/Express": 0.65,
    "Passenger": 0.50,
}


class TrainRecommenderService:
    """Service to score, rank, and fairly recommend trains based on traveler persona."""

    def __init__(self) -> None:
        self._trains_df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        path = os.path.join(settings.data_raw_dir, "trains.csv")
        if os.path.exists(path):
            try:
                self._trains_df = pd.read_csv(path)
            except Exception as e:
                logger.error(f"[M2 Service] Failed to load trains.csv: {e}")

    @governance_hook("m2_train_recommender", endpoint="/api/v1/recommend/trains")
    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Filter candidate trains, score according to persona, audit fairness, and rank."""
        persona = request.persona.lower()
        if persona not in PERSONA_WEIGHTS:
            persona = "budget"
        weights = PERSONA_WEIGHTS[persona]

        src = request.source_station.strip().upper()
        dst = request.destination_station.strip().upper()

        if self._trains_df is None or len(self._trains_df) == 0:
            self._load_data()

        candidates: pd.DataFrame
        if self._trains_df is not None:
            mask = (
                (self._trains_df["source"].str.upper() == src)
                & (self._trains_df["destination"].str.upper() == dst)
            )
            candidates = self._trains_df[mask].copy()

            if len(candidates) == 0:
                # Fallback: match either source or pick representative network trains
                src_mask = self._trains_df["source"].str.upper() == src
                if src_mask.sum() > 0:
                    candidates = self._trains_df[src_mask].head(6).copy()
                else:
                    candidates = self._trains_df.head(6).copy()
        else:
            candidates = pd.DataFrame()

        if len(candidates) == 0:
            return RecommendResponse(
                recommendations=[],
                persona_applied=persona,
                weights_used=weights,
                disparate_impact_ratio=1.0,
                bias_flag=False,
            )

        # ── Compute factor scores per train ───────────────────────────────────
        scored_trains: List[Dict[str, Any]] = []

        # Find fare based on preferred class or default 3A / SL
        pref_class = (request.preferred_class or "3A").upper()
        fare_col = f"base_fare_{pref_class.lower()}"
        if fare_col not in candidates.columns:
            fare_col = "base_fare_3a"

        fares = candidates[fare_col].replace(0, np.nan).fillna(candidates["base_fare_sl"] + 250).values
        durations = candidates["duration_hours"].values
        punctualities = candidates["punctuality_pct"].values / 100.0

        min_fare, max_fare = np.min(fares), max(np.max(fares), np.min(fares) + 1.0)
        min_dur, max_dur = np.min(durations), max(np.max(durations), np.min(durations) + 1.0)

        for idx, row in candidates.iterrows():
            f_val = float(row.get(fare_col, 0))
            if f_val <= 0:
                f_val = float(row.get("base_fare_sl", 350)) + 200

            # Filter out budget violations if requested
            if request.max_budget and f_val > request.max_budget:
                continue

            dur_val = float(row.get("duration_hours", 12.0))
            punc_val = float(row.get("punctuality_pct", 80)) / 100.0
            t_type = str(row.get("train_type", "Mail/Express"))
            comfort_val = TRAIN_TYPE_COMFORT.get(t_type, 0.70)
            avail_val = 0.85 if "3A" in str(row.get("classes", "")) else 0.70

            # Normalized factors (0.0 to 1.0, higher is better)
            norm_price = 1.0 - ((f_val - min_fare) / (max_fare - min_fare + 1e-5))
            norm_duration = 1.0 - ((dur_val - min_dur) / (max_dur - min_dur + 1e-5))
            norm_punctuality = punc_val
            norm_comfort = comfort_val
            norm_avail = avail_val

            composite = (
                weights["price"] * norm_price
                + weights["punctuality"] * norm_punctuality
                + weights["duration"] * norm_duration
                + weights["comfort"] * norm_comfort
                + weights["availability"] * norm_avail
            )

            # Build tag list
            tags = []
            if norm_price > 0.75:
                tags.append("Economical")
            if punc_val >= 0.90:
                tags.append("Highly Punctual")
            if comfort_val >= 0.90:
                tags.append("Executive Comfort")
            if dur_val <= min_dur * 1.15:
                tags.append("Fastest")

            reason = (
                f"Optimized for {persona.title()} traveler: "
                f"{'affordable fare' if norm_price > 0.5 else 'speed and comfort'} "
                f"with {int(punc_val * 100)}% punctuality rating."
            )

            classes_list = str(row.get("classes", "SL|3A|2A")).split("|")

            scored_trains.append({
                "train_number": str(row.get("train_number", "10001")),
                "train_name": str(row.get("train_name", "Express")),
                "departure_time": str(row.get("departure_time", "08:00")),
                "arrival_time": str(row.get("arrival_time", "20:00")),
                "duration_hours": round(dur_val, 1),
                "available_classes": classes_list,
                "fare_estimate_inr": round(f_val, 0),
                "punctuality_score": round(punc_val, 2),
                "comfort_score": round(comfort_val, 2),
                "composite_score": round(composite, 4),
                "is_affordable": bool(f_val <= np.median(fares)),
                "recommendation_reason": reason,
                "tags": tags,
            })

        if not scored_trains:
            return RecommendResponse(
                recommendations=[],
                persona_applied=persona,
                weights_used=weights,
                disparate_impact_ratio=1.0,
                bias_flag=False,
            )

        # Sort by composite score
        scored_trains.sort(key=lambda x: x["composite_score"], reverse=True)

        # ── Bias & Fairness Audit (Disparate Impact Ratio) ────────────────────
        # Measure selection rate of affordable trains in top 3 vs total population
        top_k = min(3, len(scored_trains))
        top_affordable = sum(1 for t in scored_trains[:top_k] if t["is_affordable"]) / float(top_k)
        pop_affordable = sum(1 for t in scored_trains if t["is_affordable"]) / float(len(scored_trains))

        if pop_affordable > 0:
            dir_ratio = round(top_affordable / pop_affordable, 3)
        else:
            dir_ratio = 1.0

        bias_flag = bool(dir_ratio < 0.80 or dir_ratio > 1.25)

        # If disparate impact threshold breached, apply fairness re-ranking
        if bias_flag and len(scored_trains) > 3:
            # Promote best affordable train into top 3 to guarantee fair socio-economic access
            for i in range(top_k, len(scored_trains)):
                if scored_trains[i]["is_affordable"]:
                    promoted = scored_trains.pop(i)
                    promoted["recommendation_reason"] += " (Fairness Equity Adjustment Applied)"
                    scored_trains.insert(top_k - 1, promoted)
                    break

        recommendations = [
            TrainRecommendation(
                rank=i + 1,
                train_number=t["train_number"],
                train_name=t["train_name"],
                departure_time=t["departure_time"],
                arrival_time=t["arrival_time"],
                duration_hours=t["duration_hours"],
                available_classes=t["available_classes"],
                fare_estimate_inr=t["fare_estimate_inr"],
                punctuality_score=t["punctuality_score"],
                comfort_score=t["comfort_score"],
                composite_score=t["composite_score"],
                recommendation_reason=t["recommendation_reason"],
                tags=t["tags"],
            )
            for i, t in enumerate(scored_trains[:5])
        ]

        return RecommendResponse(
            recommendations=recommendations,
            persona_applied=persona,
            weights_used=weights,
            disparate_impact_ratio=dir_ratio,
            bias_flag=bias_flag,
        )


train_recommender_service = TrainRecommenderService()
