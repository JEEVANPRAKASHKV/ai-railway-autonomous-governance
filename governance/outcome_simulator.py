"""
governance/outcome_simulator.py
───────────────────────────────
Demo helper: generates *simulated* ground-truth outcomes for recent real predictions.

In production the outcomes arrive from railway systems (final chart status, actual arrival time,
charged fare, fraud analyst verdicts, clicks, satisfaction ratings) through
``POST /api/v1/governance/outcomes``. In this self-contained demo there is no such feed, so the
Control Tower can generate them — either *consistent* with what the models learned (normal) or
reflecting a changed world (concept drift). Every simulated outcome is tagged ``source="simulated"``
in the audit trail and in the metrics, so it is never mistaken for real data.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from governance.performance_monitor import TASKS, performance_monitor

MODES = ("normal", "concept_drift")


def _true_delay(f: dict, rng: random.Random) -> float:
    """The synthetic world's delay rule (same structure as data/generate_synthetic_data.py), from the INPUTS."""
    month, sf = int(f.get("month", 1)), float(f.get("season_factor", 1.0))
    base = 12.0 if month in (6, 7, 8, 9) else 10.0 if month in (12, 1) else 6.0
    return (rng.expovariate(1 / base) * sf + 0.45 * float(f.get("weather_index", 0)) ** 1.6
            + 0.22 * max(0.0, float(f.get("congestion_at_station", 0)) - 30) + float(f.get("distance_from_origin_km", 0)) / 120.0
            + rng.gauss(0, 4))


def _outcome(mid: str, pred: Any, mode: str, rng: random.Random, feats: dict = None) -> Any:
    """Outcomes are generated from the request INPUTS (the world), never from the prediction — otherwise a
    better challenger could never look better than the champion."""
    drift = mode == "concept_drift"
    feats = feats or {}
    if mid == "m1_ticket_predictor":
        probs = pred.get("probs") or {"Confirmed": 0.5, "RAC": 0.2, "WL": 0.3}
        if drift:   # cancellations dried up: far fewer waitlisted tickets clear
            return rng.choices(["Confirmed", "RAC", "WL"], weights=[0.1, 0.15, 0.75])[0]
        labels = list(probs)
        return rng.choices(labels, weights=[max(probs[k], 1e-3) for k in labels])[0]
    if mid == "m3_delay_predictor":
        if "weather_index" in feats:
            d = _true_delay(feats, rng)
            if drift:   # e.g. a new maintenance block: delays are much longer for the same conditions
                d = d * 1.8 + 18
            return round(max(0.0, d), 1)
        p = float(pred)
        return round(max(0.0, (p * 1.8 + 18) if drift else p + rng.gauss(0, 9)), 1)
    if mid == "m5_fare_predictor":
        if "base_fare" in feats:
            base = float(feats["base_fare"])
            fare = base * float(feats.get("demand_index", 1.0)) * (1 + rng.gauss(0, 0.01))
            return round(fare * (1.22 if drift else 1.0), 2)
        p = float(pred)
        return round(p * (1.22 if drift else 1.0) * (1 + rng.gauss(0, 0.012)), 2)
    if mid == "m7_fraud_detector":
        p = str(pred)
        if drift and p == "legitimate":     # attackers adapted: bots now look like humans
            return "bot" if rng.random() < 0.45 else "legitimate"
        if rng.random() < 0.97:
            return p
        return rng.choice([x for x in ("legitimate", "suspicious", "bot") if x != p])
    if mid == "m2_train_recommender":
        shown = list((pred.get("shown") if isinstance(pred, dict) else pred) or [])
        if not shown:
            return None
        w = [0.08, 0.08, 0.14, 0.35, 0.35] if drift else [0.55, 0.22, 0.1, 0.08, 0.05]
        return rng.choices(shown, weights=w[: len(shown)])[0]
    if mid == "m6_support_assistant":
        return rng.choice([1, 2, 2, 3]) if drift else rng.choice([4, 4, 5, 5, 5, 3])
    if mid == "m4_travel_planner":
        return rng.random() < (0.3 if drift else 0.78)
    return None


def simulate(mode: str = "normal", modules: List[str] = None, max_per_module: int = 200, seed: int = None) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = random.Random(seed)
    summary: Dict[str, Any] = {"mode": mode, "modules": {}}
    for mid in modules or list(TASKS):
        n = 0
        for aid in performance_monitor.pending_ids(mid, limit=max_per_module):
            rec = performance_monitor.get_prediction(aid)
            if not rec:
                continue
            actual = _outcome(mid, rec["prediction"], mode, rng, rec.get("features"))
            if actual is None:
                continue
            if performance_monitor.record_outcome(aid, actual, source="simulated").get("accepted"):
                n += 1
        summary["modules"][mid] = n
    summary["total"] = sum(summary["modules"].values())
    return summary
