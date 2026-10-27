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
from core.network import network
from governance.context import report_inference
from governance.hooks import governance_hook
from modules.m2_train_recommender.schema import (
    ConnectionLeg,
    ConnectionOption,
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
    "Vande Bharat": 0.95, "Rajdhani": 0.92, "Tejas": 0.9, "Shatabdi": 0.9, "Duronto": 0.85, "Humsafar": 0.82,
    "Superfast": 0.75, "Sampark Kranti": 0.74, "Jan Shatabdi": 0.72, "Garib Rath": 0.7,
    "Express": 0.66, "Mail": 0.64,
}
FACTORS = ("price", "punctuality", "availability", "duration", "comfort")
LEARNING_RATE = 0.05
MIN_WEIGHT = 0.03


def _hhmm(t: str) -> int:
    try:
        h, m = str(t).split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return -1


class TrainRecommenderService:
    """Scores, ranks and fairly recommends trains; personalises from history; learns from passenger choices."""

    def __init__(self) -> None:
        self._trains_df: Optional[pd.DataFrame] = None
        self._profiles: Dict[str, dict] = {}
        self._confirm_rate: Dict[tuple, float] = {}
        self._weights: Dict[str, Dict[str, float]] = {p: dict(w) for p, w in PERSONA_WEIGHTS.items()}
        self._weights_version = 0
        self._last_shown: Dict[str, dict] = {}
        self._load_data()

    def _load_data(self) -> None:
        raw = settings.data_raw_dir
        try:
            self._trains_df = pd.read_csv(os.path.join(raw, "trains.csv"))
        except Exception as e:
            logger.error(f"[M2 Service] Failed to load trains.csv: {e}")
        try:
            up = pd.read_csv(os.path.join(raw, "user_profiles.csv"))
            self._profiles = {r["user_id"]: r for r in up.to_dict(orient="records")}
        except Exception as e:
            logger.debug(f"[M2 Service] user_profiles.csv: {e}")
        try:
            h = pd.read_csv(os.path.join(raw, "pnr_history.csv"), usecols=["train_number", "journey_class", "final_status"])
            g = h.assign(ok=(h["final_status"] == "Confirmed")).groupby(["train_number", "journey_class"])["ok"]
            for k, v in g.agg(["mean", "count"]).iterrows():
                if v["count"] >= 10:
                    self._confirm_rate[(str(k[0]), str(k[1]))] = float(v["mean"])
            for k, v in h.assign(ok=(h["final_status"] == "Confirmed")).groupby("journey_class")["ok"].mean().items():
                self._confirm_rate[("*", str(k))] = float(v)
        except Exception as e:
            logger.debug(f"[M2 Service] confirmation history: {e}")
        try:
            from governance.performance_monitor import performance_monitor
            performance_monitor.add_listener("m2_train_recommender", self._learn_from_choice)
        except Exception:
            pass

    # ── Continuous optimisation from relevance feedback ───────────────────────

    def _learn_from_choice(self, record: dict, chosen: str) -> None:
        """Nudge the persona's weights towards the factors of the train the passenger actually picked."""
        pred = record.get("prediction") or {}
        factors = pred.get("factors") or {}
        persona = pred.get("persona")
        if str(chosen) not in factors or persona not in self._weights or len(factors) < 2:
            return
        mean = {f: float(np.mean([v[f] for v in factors.values()])) for f in FACTORS}
        w = self._weights[persona]
        for f in FACTORS:
            w[f] = max(MIN_WEIGHT, w[f] + LEARNING_RATE * (factors[str(chosen)][f] - mean[f]))
        total = sum(w.values())
        for f in FACTORS:
            w[f] = round(w[f] / total, 4)
        self._weights_version += 1

    def connections(self, src: str, dst: str, pref_class: str, max_budget: Optional[float],
                    limit: int = 3) -> List[ConnectionOption]:
        """One-change journeys src → hub → dst over the real network with a 45 min – 8 h layover (same rule as
        the travel planner), ordered by door-to-door time."""

        def fare(row) -> float:
            v = float(row.get(f"base_fare_{pref_class.lower()}", 0) or 0)
            if v > 0:
                return v
            offered = [float(row.get(f"base_fare_{c.lower()}", 0) or 0) for c in str(row["classes"]).split("|")]
            return min([f for f in offered if f > 0] or [400.0])

        out = []
        for a, b, lay, total_h in network.connections(src, dst, nearby=True, limit=limit * 2):
            legs = [ConnectionLeg(train_number=str(r["train_number"]), train_name=str(r["train_name"]),
                                  source=str(r["source"]), destination=str(r["destination"]),
                                  departure_time=str(r["departure_time"]), arrival_time=str(r["arrival_time"]),
                                  duration_hours=round(float(r["duration_hours"]), 1), fare_inr=round(fare(r), 0))
                    for r in (a, b)]
            total_fare = sum(l.fare_inr for l in legs)
            out.append(ConnectionOption(via=str(a["destination"]), legs=legs, layover_minutes=int(lay),
                                        total_hours=round(total_h, 1), total_fare_inr=round(total_fare, 0),
                                        within_budget=not (max_budget and total_fare > max_budget)))
        out.sort(key=lambda c: (not c.within_budget, c.total_hours))
        return out[:limit]

    def current_weights(self) -> dict:
        return {"version": self._weights_version, "weights": self._weights, "initial": PERSONA_WEIGHTS}

    # ── Inference ─────────────────────────────────────────────────────────────

    @governance_hook("m2_train_recommender", endpoint="/api/v1/recommend/trains")
    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Filter candidate trains, score according to persona and history, audit fairness, and rank."""
        src = request.source_station.strip().upper()
        dst = request.destination_station.strip().upper()

        # ── Personalisation from the passenger's profile / travel history ────
        persona = (request.persona or "budget").lower()
        pref_class = (request.preferred_class or "").upper() or None
        max_budget = request.max_budget
        personal: Dict[str, Any] = {}
        prof = self._profiles.get((request.user_id or "").strip().upper()) if request.user_id else None
        if prof:
            personal = {"user_id": prof["user_id"], "persona_from_history": prof["persona"],
                        "usual_class": prof["preferred_class"], "usual_budget_max": prof["budget_max"],
                        "frequent_routes": prof["frequent_routes"]}
            if not request.persona or request.persona.lower() == "budget":
                persona = str(prof["persona"]).lower()
            pref_class = pref_class or str(prof["preferred_class"]).upper()
            max_budget = max_budget or float(prof["budget_max"])
        if persona not in self._weights:
            persona = "budget"
        weights = self._weights[persona]
        pref_class = pref_class or "3A"
        frequent = set(str(personal.get("frequent_routes", "")).split("|")) if personal else set()

        # every train that halts at both stations (or at another station in the same city), in order
        candidates = network.segments(src, dst, nearby=True) if src != dst else pd.DataFrame()

        conns = self.connections(src, dst, pref_class, max_budget) if len(candidates) < 3 and src != dst else []
        if conns and len(candidates):   # only worth showing if not much slower than the fastest direct train
            fastest = float(candidates["duration_hours"].min())
            conns = [c for c in conns if c.total_hours <= fastest * 1.5]

        def empty(msg: str) -> RecommendResponse:
            return RecommendResponse(recommendations=[], persona_applied=persona, weights_used=weights,
                                     disparate_impact_ratio=1.0, bias_flag=False, personalisation=personal,
                                     weights_version=self._weights_version, message=msg, connections=conns)

        if src == dst:
            return empty("The origin and destination are the same station — pick two different stations.")
        if len(candidates) == 0:
            return empty(f"No direct train in the timetable stops at {src} and then at {dst}."
                         + (f" Showing {len(conns)} one-change connection(s) instead." if conns else
                            " No one-change connection was found either."))

        fare_col = f"base_fare_{pref_class.lower()}"
        if fare_col not in candidates.columns:
            fare_col = "base_fare_3a"
        cheapest = candidates[[c for c in candidates.columns if c.startswith("base_fare_")]].replace(0, np.nan).min(axis=1)
        fares = candidates[fare_col].replace(0, np.nan).fillna(cheapest).fillna(400).values
        durations = candidates["duration_hours"].values
        min_fare, max_fare = np.min(fares), max(np.max(fares), np.min(fares) + 1.0)
        min_dur, max_dur = np.min(durations), max(np.max(durations), np.min(durations) + 1.0)
        deadline = _hhmm(request.arrival_deadline) if request.arrival_deadline else -1

        scored: List[Dict[str, Any]] = []
        over_budget = 0
        for _, row in candidates.iterrows():
            f_val = float(row.get(fare_col, 0) or 0)
            fare_class = pref_class
            if f_val <= 0:      # preferred class not offered: quote the cheapest class it has, labelled as such
                offered = {c[len("base_fare_"):].upper(): float(row[c]) for c in candidates.columns
                           if c.startswith("base_fare_") and float(row[c] or 0) > 0}
                fare_class, f_val = min(offered.items(), key=lambda kv: kv[1]) if offered else (pref_class, 400.0)
            if max_budget and f_val > max_budget:
                over_budget += 1
                continue
            dur_val = float(row.get("duration_hours", 12.0))
            punc_val = float(row.get("punctuality_pct", 80)) / 100.0
            comfort_val = TRAIN_TYPE_COMFORT.get(str(row.get("train_type", "Express")), 0.66)
            classes_list = str(row.get("classes", "SL|3A|2A")).split("|")
            odds = self._confirm_rate.get((str(row["train_number"]), pref_class),
                                          self._confirm_rate.get(("*", pref_class)))
            avail_val = float(odds) if odds is not None else 0.5
            if pref_class not in classes_list:
                avail_val *= 0.4   # preferred class not offered on this train

            price_score = float(np.clip(1.0 - ((f_val - min_fare) / (max_fare - min_fare + 1e-5)), 0.0, 1.0))
            if fare_class != pref_class:
                price_score *= 0.5   # a cheaper fare in a different class is not a like-for-like saving
            fs = {"price": price_score,
                  "punctuality": punc_val,
                  "availability": avail_val,
                  "duration": 1.0 - ((dur_val - min_dur) / (max_dur - min_dur + 1e-5)),
                  "comfort": comfort_val}
            contrib = {f: round(weights[f] * fs[f], 4) for f in FACTORS}
            composite = sum(contrib.values())
            tags, meets = [], True
            if deadline >= 0 and dur_val < 24 and _hhmm(row.get("arrival_time")) > deadline:
                composite *= 0.7
                meets = False
                tags.append(f"Arrives after {request.arrival_deadline}")
            route_key = f"{src}-{dst}"
            if route_key in frequent:
                composite += 0.02
                tags.append("On your frequent route")
            if fs["price"] > 0.75:
                tags.append("Economical")
            if punc_val >= 0.90:
                tags.append("Highly Punctual")
            if comfort_val >= 0.90:
                tags.append("Premium comfort")
            if dur_val <= min_dur * 1.15:
                tags.append("Fastest")
            if pref_class not in classes_list:
                tags.append(f"No {pref_class}")

            top2 = sorted(contrib, key=contrib.get, reverse=True)[:2]
            reason = (f"Best for a {persona} traveller mainly because of {top2[0]} "
                      f"({fs[top2[0]]:.0%}) and {top2[1]} ({fs[top2[1]]:.0%}); "
                      f"{int(punc_val * 100)}% on time, {avail_val:.0%} historical confirmation in {pref_class}.")
            scored.append({
                "train_number": str(row.get("train_number")), "train_name": str(row.get("train_name")),
                "source": str(row.get("source", src)), "destination": str(row.get("destination", dst)),
                "departure_time": str(row.get("departure_time", "08:00")), "arrival_time": str(row.get("arrival_time", "20:00")),
                "duration_hours": round(dur_val, 1), "available_classes": classes_list,
                "fare_estimate_inr": round(f_val, 0), "fare_class": fare_class, "punctuality_score": round(punc_val, 2),
                "comfort_score": round(comfort_val, 2), "composite_score": round(composite, 4),
                "is_affordable": bool(f_val <= np.median(fares)), "recommendation_reason": reason, "tags": tags,
                "factor_scores": {f: round(v, 3) for f, v in fs.items()}, "factor_contributions": contrib,
                "confirmation_odds": round(avail_val, 3) if odds is not None else None, "meets_constraints": meets,
            })

        if not scored:
            return empty(f"All {over_budget} trains on this route cost more than ₹{max_budget:.0f} in {pref_class}.")

        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # ── Fairness audit (affordable options must not be crowded out of the top 3) ──
        top_k = min(3, len(scored))
        top_aff = sum(1 for t in scored[:top_k] if t["is_affordable"]) / float(top_k)
        pop_aff = sum(1 for t in scored if t["is_affordable"]) / float(len(scored))
        dir_ratio = round(top_aff / pop_aff, 3) if pop_aff > 0 else 1.0
        bias_flag = bool(dir_ratio < 0.80)
        if bias_flag and len(scored) > 3:
            for i in range(top_k, len(scored)):
                if scored[i]["is_affordable"]:
                    promoted = scored.pop(i)
                    promoted["recommendation_reason"] += " (Promoted by the affordability fairness rule.)"
                    promoted["tags"].append("Fairness adjustment")
                    scored.insert(top_k - 1, promoted)
                    break

        top = scored[:5]
        report_inference(prediction={"shown": [t["train_number"] for t in top], "persona": persona,
                                     "factors": {t["train_number"]: t["factor_scores"] for t in top}},
                         version=f"weights-v{self._weights_version}")
        recommendations = [TrainRecommendation(rank=i + 1, **{k: v for k, v in t.items() if k != "is_affordable"})
                           for i, t in enumerate(top)]
        return RecommendResponse(
            recommendations=recommendations, persona_applied=persona, weights_used=weights,
            disparate_impact_ratio=dir_ratio, bias_flag=bias_flag, personalisation=personal,
            weights_version=self._weights_version,
            message=(f"{over_budget} train(s) hidden because they exceed your budget." if over_budget else None),
            connections=conns,
        )


train_recommender_service = TrainRecommenderService()
