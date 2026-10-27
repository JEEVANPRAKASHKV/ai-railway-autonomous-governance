"""
modules/m5_fare_predictor/service.py
====================================
Inference service for Module 5: Dynamic Fare Prediction Engine.
Uses trained XGBoost regressor to project 30-day dynamic pricing curves,
detect surge multiplier thresholds, and locate optimal booking windows.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import settings
from core.fares import TARIFF_SOURCE, fare_breakdown
from core.network import network
from core.seasonality import is_peak, season_factor
from governance.context import report_inference
from governance.hooks import governance_hook
from governance.lifecycle_manager import lifecycle_manager
from governance.telemetry_bus import telemetry_bus
from modules.m5_fare_predictor.schema import (
    CostSavingAlternative,
    FareForecastPoint,
    FarePredictRequest,
    FarePredictResponse,
)

logger = logging.getLogger(__name__)

ROUTE_POPULARITY_MEAN = 0.65   # route_popularity is uniform(0.3, 1.0) in the training data
FLEXI_TYPES = {"Rajdhani", "Duronto", "Shatabdi"}
FLEXI_CAP = {"3A": 1.4}        # flexi-fare ceiling: 1.5× base, 1.4× in 3A; 1A and EC are not flexi
ARP_DAYS = 60                  # advance reservation period
FARE_FEATURE_LABELS = {
    "base_fare": "Official fare for this class & distance",
    "flexi_flag": "Flexi-fare train (Rajdhani/Duronto/Shatabdi)",
    "days_to_departure": "How early you book",
    "demand_index": "Expected demand",
    "season_factor": "Festival / holiday season",
    "tatkal_flag": "Tatkal quota",
    "class_enc": "Travel class",
    "route_popularity": "Route popularity",
    "month": "Month of travel",
    "day_of_week": "Day of week",
    "is_peak_season": "Peak season",
}


class FarePredictorService:
    """Service to forecast dynamic railway fares, recommend cost-saving booking windows, and analyze fare drivers."""

    def __init__(self) -> None:
        self._bundle: Optional[Dict[str, Any]] = None
        self._explainer: Optional[Any] = None
        self._trains_df: Optional[pd.DataFrame] = None
        self._ratio_range: Dict[tuple, tuple] = {}
        self._load_resources()
        self._load_price_ranges()

    def _load_resources(self) -> None:
        art_dir = settings.models_artifacts_dir
        model_path = os.path.join(art_dir, "fare_predictor_v1.0.pkl")
        shap_path = os.path.join(art_dir, "shap_explainers", "shap_fare_predictor.pkl")
        trains_path = os.path.join(settings.data_raw_dir, "trains.csv")

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._bundle = pickle.load(f)
            except Exception as e:
                logger.error(f"[M5 Service] Failed to load fare_predictor bundle: {e}")

        if os.path.exists(shap_path):
            try:
                with open(shap_path, "rb") as f:
                    self._explainer = pickle.load(f)
            except Exception as e:
                logger.warning(f"[M5 Service] Could not load fare SHAP explainer: {e}")

        if os.path.exists(trains_path):
            try:
                self._trains_df = pd.read_csv(trains_path)
            except Exception as e:
                logger.debug(f"[M5 Service] Trains CSV load error: {e}")

    def _segment(self, train_number: str, src: Optional[str], dst: Optional[str]) -> Optional[pd.Series]:
        """The passenger's journey on this train (src/dst default to the train's origin and terminus)."""
        t = network.train(train_number)
        if t is None:
            return None
        src = (src or t["source"]).strip().upper()
        dst = (dst or t["destination"]).strip().upper()
        seg = network.segments(src, dst)
        seg = seg[seg["train_number"].astype(str) == str(train_number)] if len(seg) else seg
        return seg.iloc[0] if len(seg) else None

    def _lookup_base_fare(self, train_number: str, travel_class: str, src: Optional[str] = None,
                          dst: Optional[str] = None) -> float:
        cls_key = f"base_fare_{travel_class.lower()}"
        seg = self._segment(train_number, src, dst)
        if seg is not None and float(seg.get(cls_key, 0) or 0) > 0:
            return float(seg[cls_key])
        if self._trains_df is not None:
            match = self._trains_df[self._trains_df["train_number"].astype(str) == str(train_number)]
            if len(match) > 0 and cls_key in match.columns:
                val = float(match.iloc[0][cls_key])
                if val > 0:
                    return val
        # Defaults by class (≈ 1,000 km Mail/Express, TAG 2025)
        defaults = {"SL": 509.0, "3A": 1345.0, "2A": 1909.0, "1A": 3217.0, "CC": 1092.0, "EC": 2200.0, "2S": 288.0}
        return defaults.get(travel_class.upper(), 1345.0)

    @staticmethod
    def is_flexi(train_type: str, travel_class: str) -> bool:
        return str(train_type) in FLEXI_TYPES and travel_class.upper() not in ("1A", "EC")

    # ── Feature engineering consistent with the training data ────────────────

    @staticmethod
    def expected_demand_index(days_ahead: int, sf: float) -> float:
        """Expected demand for a lead time & season — the generator's own rule without its noise term."""
        return round(float(np.clip(1.0 + 0.5 * (1 - min(days_ahead, ARP_DAYS) / ARP_DAYS) * sf, 0.5, 3.0)), 3)

    def _features(self, base_fare: float, class_enc: int, days_ahead: int, jdate: date, tatkal: bool,
                  flexi: bool = False) -> Dict[str, float]:
        sf = season_factor(jdate)
        return {
            "base_fare": float(base_fare),
            "class_enc": float(class_enc),
            "days_to_departure": float(days_ahead),
            "demand_index": self.expected_demand_index(days_ahead, sf),
            "flexi_flag": float(int(flexi)),
            "tatkal_flag": float(int(tatkal)),
            "season_factor": float(sf),
            "route_popularity": ROUTE_POPULARITY_MEAN,
            "month": float(jdate.month),
            "day_of_week": float(jdate.weekday()),
            "is_peak_season": float(is_peak(jdate)),
        }

    @staticmethod
    def _model_fare(bundle: Dict[str, Any], feats: Dict[str, float], base_fare: float) -> float:
        cols = bundle.get("feature_cols", list(feats))
        x = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
        return float(max(base_fare * 0.9, bundle["model"].predict(x)[0]))

    def _load_price_ranges(self) -> None:
        """Historical fare / base-fare ratio range per class and lead-time bucket (pricing anomaly guard)."""
        try:
            ph = pd.read_csv(os.path.join(settings.data_raw_dir, "pricing_history.csv"),
                             usecols=["journey_class", "days_to_departure", "base_fare", "dynamic_fare"])
            ph["ratio"] = ph["dynamic_fare"] / ph["base_fare"].clip(lower=1)
            ph["bucket"] = pd.cut(ph["days_to_departure"], [-1, 1, 7, 30, 400], labels=["0-1", "2-7", "8-30", "31+"])
            for (cls, bk), g in ph.groupby(["journey_class", "bucket"], observed=True)["ratio"]:
                self._ratio_range[(str(cls), str(bk))] = (float(g.quantile(0.005)), float(g.quantile(0.995)))
        except Exception as e:
            logger.debug(f"[M5 Service] price ranges unavailable: {e}")

    @staticmethod
    def _bucket(days: int) -> str:
        return "0-1" if days <= 1 else "2-7" if days <= 7 else "8-30" if days <= 30 else "31+"

    def _shap(self, bundle: Dict[str, Any], feats: Dict[str, float]) -> List[Dict[str, Any]]:
        explainer = self._explainer if bundle is self._bundle else None
        try:
            if explainer is None:
                import shap
                explainer = shap.TreeExplainer(bundle["model"])
            cols = bundle.get("feature_cols", list(feats))
            x = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
            vals = np.asarray(explainer.shap_values(x))[0]
            base = float(np.ravel(explainer.expected_value)[0])
            rows = [{"driver": FARE_FEATURE_LABELS.get(c, c), "feature": c, "value": round(float(feats[c]), 3),
                     "contribution_inr": round(float(v), 0), "type": "SHAP"} for c, v in zip(cols, vals)]
            rows.sort(key=lambda r: abs(r["contribution_inr"]), reverse=True)
            rows.append({"driver": "Average fare in training data", "feature": "base_value", "value": None,
                         "contribution_inr": round(base, 0), "type": "Baseline"})
            return rows
        except Exception as e:
            logger.debug(f"[M5 Service] SHAP failed: {e}")
            return []

    @governance_hook("m5_fare_predictor", endpoint="/api/v1/predict/fare")
    def predict(self, request: FarePredictRequest) -> FarePredictResponse:
        """Forecast the fare, the curve until departure, demand surges and the best day to book."""
        b_date = request.booking_date or date.today()
        days_to_dept = max(0, (request.journey_date - b_date).days)
        src, dst = request.source_station, request.destination_station
        seg = self._segment(request.train_number, src, dst)
        train = network.train(request.train_number)
        ttype = str(train["train_type"]) if train is not None else "Express"
        base_fare = request.base_fare or self._lookup_base_fare(request.train_number, request.journey_class, src, dst)
        km = float(seg["distance_km"]) if seg is not None else (float(train["distance_km"]) if train is not None else 1000.0)
        breakdown = fare_breakdown(ttype, request.journey_class, km, request.train_number,
                                   tatkal=bool(request.tatkal_flag)) or {}
        if breakdown:
            breakdown["source"] = TARIFF_SOURCE
        flexi = self.is_flexi(ttype, request.journey_class)
        cap = FLEXI_CAP.get(request.journey_class.upper(), 1.5) if flexi else 1.0

        class_enc_val = 1
        if self._bundle and "class_encoder" in self._bundle:
            enc = self._bundle["class_encoder"]
            try:
                class_enc_val = int(enc.transform([request.journey_class])[0])
            except Exception:
                try:
                    class_enc_val = int(enc.transform(["3A"])[0])
                except Exception:
                    class_enc_val = 1

        tatkal = bool(request.tatkal_flag)
        feats = self._features(base_fare, class_enc_val, days_to_dept, request.journey_date, tatkal, flexi)
        bundle, version = lifecycle_manager.serving_bundle("m5_fare_predictor", self._bundle)
        use_model = bool(bundle and "model" in bundle)

        def fare_at(days_ahead: int) -> float:
            f = self._features(base_fare, class_enc_val, days_ahead, request.journey_date, tatkal, flexi)
            if not flexi and not tatkal:
                return float(base_fare)          # fixed fare: Indian Railways does not vary it by demand
            if use_model:
                return self._model_fare(bundle, f, base_fare)
            return base_fare * min(cap, f["demand_index"])

        raw_fare = round(fare_at(days_to_dept), 2)
        report_inference(features=feats, prediction=raw_fare, version=version if use_model else "heuristic")

        # ── Pricing-anomaly guard (governance requirement) ──────────────────────
        ratio = raw_fare / max(1.0, base_fare)
        lo, hi = self._ratio_range.get((request.journey_class.upper(), self._bucket(days_to_dept)), (0.85, 3.5))
        anomaly = None
        if not (lo * 0.95 <= ratio <= hi * 1.05):
            anomaly = (f"Predicted fare is {ratio:.2f}× base, outside the historical range "
                       f"{lo:.2f}–{hi:.2f}× for {request.journey_class} at {self._bucket(days_to_dept)} days ahead.")
            telemetry_bus.publish(module_id="m5_fare_predictor", event_type="PRICING_ANOMALY",
                                  payload={"ratio": round(ratio, 3), "range": [round(lo, 3), round(hi, 3)],
                                           "train": request.train_number, "class": request.journey_class},
                                  severity="WARNING")
        current_fare = raw_fare
        policy_note = None
        if not tatkal and current_fare > base_fare * cap:
            current_fare = round(base_fare * cap, 2)
            policy_note = f"Capped at {cap}× the base fare (flexi-fare ceiling for {request.journey_class})."
        if not flexi and not tatkal:
            why = (f"{request.journey_class} is outside the flexi-fare scheme on {ttype} trains"
                   if ttype in FLEXI_TYPES else f"{ttype} trains are not on flexi pricing")
            policy_note = (f"Fixed fare: {why}, so the fare is the same whenever you book. "
                           f"Book early for a better chance of a confirmed berth.")

        surge_mult = round(current_fare / max(1.0, base_fare), 2)
        def level(mult: float) -> str:
            return "Surge" if mult >= 1.4 else "High" if mult >= 1.2 else "Moderate" if mult >= 1.05 else "Low"
        demand_level = level(surge_mult)

        # ── Forecast curve: from the booking date up to departure (no repeated dates) ──
        curve: List[FareForecastPoint] = []
        horizon = min(days_to_dept, 60)
        step = 1 if horizon <= 30 else 2
        best_days, best_fare = days_to_dept, float("inf")
        surge_from: Optional[int] = None
        for d_ahead in range(horizon, -1, -step):
            d_date = request.journey_date - timedelta(days=d_ahead)
            if d_date < b_date:
                continue
            f_pred = fare_at(d_ahead)
            if not tatkal:
                f_pred = min(f_pred, base_fare * cap)
            f_pred = round(f_pred, 0)
            lvl = level(f_pred / max(1.0, base_fare))
            if lvl in ("High", "Surge") and surge_from is None:
                surge_from = d_ahead
            if f_pred < best_fare:
                best_fare, best_days = f_pred, d_ahead
            curve.append(FareForecastPoint(days_to_departure=d_ahead, forecast_date=d_date,
                                           predicted_fare_inr=f_pred, demand_level=lvl))

        optimal_date = request.journey_date - timedelta(days=best_days)
        flat = bool(curve) and max(p.predicted_fare_inr for p in curve) - min(p.predicted_fare_inr for p in curve) < 1
        if days_to_dept <= 0:
            optimal_window_str = f"Book now — departing today ({request.journey_date.strftime('%d %b %Y')})"
        elif flat and not flexi and not tatkal:
            best_days, optimal_date = days_to_dept, b_date
            optimal_window_str = "Any day — the fare does not change; book early to get a confirmed berth"
        elif flat and surge_mult >= cap - 0.01:
            best_days, optimal_date = days_to_dept, b_date
            optimal_window_str = (f"Today — the fare is already at the flexi ceiling ({cap}×) and stays there "
                                  f"until departure; waiting will not make it cheaper")
        elif flat:
            best_days, optimal_date = days_to_dept, b_date
            optimal_window_str = "Any day before departure — demand is not expected to push the flexi fare up"
        elif best_days == days_to_dept:
            optimal_window_str = f"Book today ({b_date.strftime('%d %b %Y')}) — fares only rise from here"
        else:
            optimal_window_str = f"{best_days} days before journey ({optimal_date.strftime('%d %b %Y')})"

        # ── Cost-saving recommendations ─────────────────────────────────────────
        savings: List[CostSavingAlternative] = []
        if best_fare < current_fare - 1:
            savings.append(CostSavingAlternative(
                alternative_type="Booking window",
                description=f"Book on {optimal_date.strftime('%d %b')} instead of today.",
                potential_savings_inr=round(current_fare - best_fare, 0), recommended_fare_inr=round(best_fare, 0)))
        elif surge_from is not None and days_to_dept > surge_from:
            savings.append(CostSavingAlternative(
                alternative_type="Book before the surge",
                description=f"Demand surge expected from {surge_from} days before departure — book before then.",
                potential_savings_inr=round(max(0.0, fare_at(surge_from) - current_fare), 0),
                recommended_fare_inr=round(current_fare, 0)))
        if request.journey_class.upper() in ("2A", "1A"):
            lower_fare = self._lookup_base_fare(request.train_number, "3A", src, dst)
            if lower_fare < current_fare:
                savings.append(CostSavingAlternative(
                    alternative_type="Class change", description="3-Tier AC (3A) on the same train.",
                    potential_savings_inr=round(current_fare - lower_fare * surge_mult, 0),
                    recommended_fare_inr=round(lower_fare * surge_mult, 0)))
        alt = self._cheaper_train(request.train_number, request.journey_class, base_fare, seg)
        if alt is not None:
            savings.append(CostSavingAlternative(
                alternative_type="Cheaper train",
                description=f"{alt['train_name']} ({alt['train_number']}) on the same route, departs {alt['departure_time']}.",
                potential_savings_inr=round((base_fare - alt["fare"]) * surge_mult, 0),
                recommended_fare_inr=round(alt["fare"] * surge_mult, 0)))

        # a fixed fare is set by the tariff, not by the model — there is nothing for SHAP to explain
        drivers = self._shap(bundle, feats) if use_model and (flexi or tatkal) else []

        return FarePredictResponse(
            train_number=request.train_number,
            journey_class=request.journey_class,
            current_predicted_fare_inr=current_fare,
            base_fare_inr=base_fare,
            surge_multiplier=surge_mult,
            demand_level=demand_level,
            optimal_booking_window=optimal_window_str,
            optimal_booking_days_before=best_days,
            optimal_booking_date=optimal_date,
            demand_surge_from_days=surge_from,
            forecast_curve=curve,
            cost_saving_alternatives=savings,
            fare_drivers=drivers,
            pricing_anomaly=anomaly,
            policy_note=policy_note,
            fare_breakdown=breakdown,
            pricing_rule="tatkal" if tatkal else "flexi" if flexi else "fixed",
            source_station=str(seg["source"]) if seg is not None else None,
            destination_station=str(seg["destination"]) if seg is not None else None,
            model_version=version if use_model else "heuristic",
            confidence_score=0.94,
        )

    def _cheaper_train(self, train_number: str, travel_class: str, base_fare: float,
                       seg: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        """The cheapest other train between the same two stations in this class."""
        if seg is None:
            return None
        col = f"base_fare_{travel_class.lower()}"
        same = network.segments(str(seg["source"]), str(seg["destination"]))
        if not len(same) or col not in same.columns:
            return None
        same = same[(same["train_number"].astype(str) != str(train_number)) & (same[col] > 0) & (same[col] < base_fare * 0.95)]
        if not len(same):
            return None
        r = same.sort_values(col).iloc[0]
        return {"train_number": str(r["train_number"]), "train_name": str(r["train_name"]),
                "departure_time": str(r["departure_time"]), "fare": float(r[col])}


fare_predictor_service = FarePredictorService()
