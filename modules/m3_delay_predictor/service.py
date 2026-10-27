"""
modules/m3_delay_predictor/service.py
====================================
Inference service for Module 3: Train Delay & Disruption Prediction.

  • Arrival delay at the chosen station and departure delay from the origin (LightGBM)
  • Distance from the origin and scheduled time at the station from the real timetable (core/network.py)
  • Weather from the live Open-Meteo forecast at the station's coordinates, else its climatology
    (core/weather.py); station congestion from that station's own history (congestion_logs.csv)
  • An 80% prediction interval and a disruption probability from the model's residuals on history
  • Real SHAP explanations (minutes added / removed by each factor)
  • Proactive alerts and alternative trains that also halt at the same station

Request inputs use friendly 0–1 scales; they are converted to the units the model was trained on
(weather 0–10, congestion ≈10–160) before inference, and governance monitors drift in model units.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import settings
from core.network import network
from core.seasonality import season_factor
from core.weather import weather_for
from governance.context import report_inference
from governance.hooks import governance_hook
from governance.lifecycle_manager import lifecycle_manager
from modules.m3_delay_predictor.schema import (
    AlternativeTravelOption,
    DelayAlert,
    DelayPredictRequest,
    DelayPredictResponse,
)

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "weather_index": "Weather conditions",
    "congestion_at_station": "Station congestion",
    "distance_from_origin_km": "Distance travelled so far",
    "month": "Month of travel",
    "season_factor": "Festival / holiday season",
}
REFUND_DELAY_MIN = 180
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _display(feature: str, v: float, feats: Dict[str, float]) -> str:
    """Model-scale feature value → the units a passenger understands."""
    try:
        if feature == "weather_index":
            return f"{min(100.0, v * 10):.0f}% severity"
        if feature == "congestion_at_station":
            sf = float(feats.get("season_factor", 1.0)) or 1.0
            return f"{float(np.clip((v / sf - 10.0) / 0.8, 0, 100)):.0f}/100"
        if feature == "distance_from_origin_km":
            return f"{v:,.0f} km"
        if feature == "month":
            return _MONTHS[int(v) - 1]
        if feature == "season_factor":
            return f"×{v:.2f}" + (" peak" if v >= 1.15 else "")
    except Exception:
        pass
    return f"{v:.2f}"


def _congestion_to_model(index_0_100: float, sf: float) -> float:
    """Station congestion index (0–100) → the model's training scale (uniform 10–90 × season factor)."""
    return float((10.0 + 0.8 * float(np.clip(index_0_100, 0, 100))) * sf)


class DelayPredictorService:
    """Service to predict train delay minutes, confidence intervals, and rerouting alternatives."""

    def __init__(self) -> None:
        self._bundle: Optional[Dict[str, Any]] = None
        self._explainer: Optional[Any] = None
        self._trains_df: Optional[pd.DataFrame] = None
        self._congestion: Dict[Tuple, float] = {}
        self._residuals: Optional[np.ndarray] = None
        self._load_resources()

    # ── Resources ─────────────────────────────────────────────────────────────

    def _load_resources(self) -> None:
        art_dir = settings.models_artifacts_dir
        model_path = os.path.join(art_dir, "delay_predictor_v1.0.pkl")
        shap_path = os.path.join(art_dir, "shap_explainers", "shap_delay_predictor.pkl")
        raw = settings.data_raw_dir

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._bundle = pickle.load(f)
            except Exception as e:
                logger.error(f"[M3 Service] Failed to load delay_predictor model: {e}")
        if os.path.exists(shap_path):
            try:
                with open(shap_path, "rb") as f:
                    self._explainer = pickle.load(f)
            except Exception as e:
                logger.warning(f"[M3 Service] Could not load delay SHAP explainer: {e}")
        try:
            self._trains_df = pd.read_csv(os.path.join(raw, "trains.csv"))
        except Exception as e:
            logger.debug(f"[M3 Service] trains.csv: {e}")
        try:
            c = pd.read_csv(os.path.join(raw, "congestion_logs.csv"),
                            usecols=["station_code", "date", "hour", "congestion_index"])
            c["month"] = pd.to_datetime(c["date"]).dt.month
            for k, v in c.groupby(["station_code", "hour", "month"])["congestion_index"].median().items():
                self._congestion[("smh",) + tuple(k)] = float(v)
            for k, v in c.groupby(["station_code", "hour"])["congestion_index"].median().items():
                self._congestion[("sh",) + tuple(k)] = float(v)
            for k, v in c.groupby("station_code")["congestion_index"].median().items():
                self._congestion[("s", k)] = float(v)
            self._congestion[("all",)] = float(c["congestion_index"].median())
        except Exception as e:
            logger.debug(f"[M3 Service] congestion history: {e}")
        self._calibrate()

    def _calibrate(self) -> None:
        """Residuals of the model on a sample of history → honest prediction intervals."""
        if not self._bundle:
            return
        try:
            d = pd.read_csv(os.path.join(settings.data_raw_dir, "delay_logs.csv")).sample(6000, random_state=11)
            cols = self._bundle.get("feature_cols")
            pred = np.clip(self._bundle["model"].predict(d[cols]), 0, None)
            self._residuals = d["delay_minutes"].clip(0, 300).values - pred
        except Exception as e:
            logger.debug(f"[M3 Service] calibration skipped: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _train_row(self, train_number: str) -> Optional[pd.Series]:
        if self._trains_df is None:
            return None
        m = self._trains_df[self._trains_df["train_number"].astype(str) == str(train_number)]
        return m.iloc[0] if len(m) else None

    def station_congestion_index(self, station: str, hour: int, month: int) -> Tuple[float, str]:
        """Historical median congestion (0–100) for this station at this hour and month."""
        for key, label in ((("smh", station, hour, month), "station history (same hour & month)"),
                           (("sh", station, hour), "station history (same hour)"),
                           (("s", station), "station history"),
                           (("all",), "network average")):
            if key in self._congestion:
                return float(self._congestion[key]), label
        return 50.0, "default"

    @staticmethod
    def _predict(bundle: Dict[str, Any], feats: Dict[str, float]) -> float:
        cols = bundle.get("feature_cols", list(feats))
        x = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
        return float(np.clip(bundle["model"].predict(x)[0], 0.0, 360.0))

    def _interval(self, pred: float) -> Tuple[float, float, float]:
        """80% prediction interval and P(delay > 60 min) from residuals on historical data."""
        if self._residuals is None or not len(self._residuals):
            return max(0.0, pred - 9.0), pred + 13.0, float(pred > 60)
        lo, hi = np.percentile(self._residuals, [10, 90])
        p60 = float(np.mean(pred + self._residuals > 60))
        return max(0.0, pred + lo), pred + hi, p60

    def _shap(self, bundle: Dict[str, Any], feats: Dict[str, float]) -> List[Dict[str, Any]]:
        explainer = self._explainer if bundle is self._bundle else None
        if explainer is None:
            try:
                import shap
                explainer = shap.TreeExplainer(bundle["model"])
            except Exception:
                return []
        try:
            cols = bundle.get("feature_cols", list(feats))
            x = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
            vals = np.asarray(explainer.shap_values(x))[0]
            base = float(np.ravel(explainer.expected_value)[0])
            out = [{"factor": FEATURE_LABELS.get(c, c), "feature": c, "value": round(float(feats[c]), 2),
                    "display": _display(c, float(feats[c]), feats),
                    "impact_minutes": round(float(v), 1),
                    "severity": "High" if abs(v) >= 8 else "Moderate" if abs(v) >= 3 else "Low",
                    "direction": "adds delay" if v > 0 else "reduces delay"} for c, v in zip(cols, vals)]
            out.sort(key=lambda d: abs(d["impact_minutes"]), reverse=True)
            out.append({"factor": "Typical delay (model baseline)", "feature": "base_value", "value": None,
                        "impact_minutes": round(base, 1), "severity": "Info", "direction": "baseline"})
            return out
        except Exception as e:
            logger.debug(f"[M3 Service] SHAP failed: {e}")
            return []

    # ── Inference ─────────────────────────────────────────────────────────────

    @governance_hook("m3_delay_predictor", endpoint="/api/v1/predict/delay")
    def predict(self, request: DelayPredictRequest) -> DelayPredictResponse:
        """Forecast arrival / departure delay with interval, SHAP drivers, alerts and alternatives."""
        jd = request.journey_date
        sf = season_factor(jd)
        train = self._train_row(request.train_number)
        station = request.station_code.strip().upper()
        route_km = float(train["distance_km"]) if train is not None else 1200.0
        origin = str(train["source"]) if train is not None else None
        dest = str(train["destination"]) if train is not None else None
        halt = network.stop(request.train_number, station)

        if request.distance_from_origin_km is not None:
            dist_km, dist_src = float(request.distance_from_origin_km), "provided"
        elif halt is not None:
            dist_km, dist_src = float(halt["km_from_origin"]), "timetable"
        elif station == origin:
            dist_km, dist_src = 0.0, "timetable"
        elif station == dest:
            dist_km, dist_src = route_km, "timetable"
        else:
            dist_km, dist_src = route_km / 2.0, "assumed mid-route (train does not halt here)"

        scheduled = None
        if halt is not None:
            scheduled = str(halt.get("arrival") or "") or str(halt.get("departure") or "") or None
        elif train is not None:
            scheduled = str(train["arrival_time"] if station == dest else train["departure_time"])
        try:
            sched_hour = int(str(scheduled).split(":")[0]) if scheduled and scheduled != "nan" else 12
        except Exception:
            sched_hour = 12

        weather_conditions: Dict[str, Any] = {}
        if request.weather_index is not None:
            weather_model, weather_src = float(request.weather_index) * 10.0, "provided"
        else:
            lat, lon = network.coords(station)
            w = weather_for(lat, lon, jd)
            weather_model, weather_src, weather_conditions = float(w["index"]), w["source"], w["conditions"]
        if request.congestion_index is not None:
            cong_idx, cong_src = float(request.congestion_index) * 100.0, "provided"
        else:
            cong_idx, cong_src = self.station_congestion_index(station, sched_hour, jd.month)
        congestion_model = _congestion_to_model(cong_idx, sf)

        feats = {"weather_index": round(weather_model, 2), "congestion_at_station": round(congestion_model, 2),
                 "distance_from_origin_km": dist_km, "month": float(jd.month), "season_factor": float(sf)}

        bundle, version = lifecycle_manager.serving_bundle("m3_delay_predictor", self._bundle)
        if bundle and "model" in bundle:
            predicted_delay = round(self._predict(bundle, feats), 1)
            origin_feats = dict(feats, distance_from_origin_km=0.0)
            if origin and origin != station:
                o_idx, _ = self.station_congestion_index(origin, sched_hour, jd.month)
                origin_feats["congestion_at_station"] = round(_congestion_to_model(o_idx, sf), 2)
            departure_delay = round(self._predict(bundle, origin_feats), 1)
        else:
            predicted_delay = round(float(np.clip(dist_km / 120 + 0.45 * weather_model ** 1.6
                                                  + 0.22 * max(0, congestion_model - 30), 0, 240)), 1)
            departure_delay = round(float(np.clip(0.45 * weather_model ** 1.6, 0, 120)), 1)
            version = "heuristic"

        lower, upper, p_over_60 = self._interval(predicted_delay)
        lower, upper = round(lower, 1), round(upper, 1)
        report_inference(features=feats, prediction=predicted_delay, version=version)

        congestion_level = "Low" if cong_idx < 40 else "Moderate" if cong_idx < 70 else "High"
        severity = "Low" if predicted_delay < 15 else "Moderate" if predicted_delay <= 60 else "Severe"

        alerts: List[DelayAlert] = []
        if severity == "Low":
            alert = (f"Train {request.train_number} is expected to be largely on time at {station} "
                     f"(~{predicted_delay:.0f} min, 80% range {lower:.0f}–{upper:.0f} min).")
        elif severity == "Moderate":
            alert = (f"Moderate delay expected: about {predicted_delay:.0f} min at {station} "
                     f"(80% range {lower:.0f}–{upper:.0f} min). Keep at least 45 min for onward connections.")
            alerts.append(DelayAlert(severity="WARNING", title="Connection risk",
                                     message="Allow a buffer of at least 45 minutes for any onward connection.",
                                     action="Pick a later connection or a train listed below."))
        else:
            alert = (f"Disruption likely: {predicted_delay:.0f} min delay forecast at {station} "
                     f"(80% range {lower:.0f}–{upper:.0f} min). Alternatives on this corridor are listed below.")
            alerts.append(DelayAlert(severity="CRITICAL", title="Severe delay forecast",
                                     message=f"{p_over_60:.0%} chance the delay exceeds 60 minutes.",
                                     action="Consider an alternative train below."))
        if weather_model >= 4.5:
            wc = weather_conditions
            what = ("Dense fog" if wc.get("min_visibility_m", 99999) < 500 else "Fog" if wc.get("min_visibility_m", 99999) < 1000
                    else "Thunderstorms" if wc.get("storm") else "Heavy rain" if wc.get("precip_mm", 0) >= 30
                    else "Adverse weather")
            detail = (f" (rain {wc['precip_mm']:.0f} mm, visibility {wc['min_visibility_m'] / 1000:.1f} km)"
                      if wc else "")
            alerts.append(DelayAlert(severity="WARNING", title=f"{what} at {station}",
                                     message=f"{what} expected on the travel day{detail} — trains slow down in these conditions.",
                                     action="Check the live running status (NTES) before leaving for the station."))
        if cong_idx >= 70:
            alerts.append(DelayAlert(severity="INFO", title=f"Busy station ({station})",
                                     message=f"Platform congestion index {cong_idx:.0f}/100 ({cong_src}).",
                                     action="Arrive 20–30 minutes earlier than usual."))
        if upper >= REFUND_DELAY_MIN:
            alerts.append(DelayAlert(severity="INFO", title="Refund eligibility",
                                     message="If the train runs more than 3 hours late you can cancel for a full refund (file a TDR).",
                                     action="Keep your PNR handy."))

        alternatives: List[AlternativeTravelOption] = []
        if predicted_delay > 45:
            # other trains that also halt at this station, preferring ones from the same starting city
            pool = network.segments(origin, station, nearby=True) if origin and origin != station else pd.DataFrame()
            if len(pool):
                pool = pool[pool["train_number"].astype(str) != str(request.train_number)]
                pool = pool.sort_values("punctuality_pct", ascending=False)
            for _, tr in pool.head(4).iterrows():
                if bundle and "model" in bundle:
                    alt_feats = dict(feats, distance_from_origin_km=float(tr["km_at_destination"]))
                    alt_delay = self._predict(bundle, alt_feats) * (1.0 - (float(tr["punctuality_pct"]) - 80) / 100.0)
                else:
                    alt_delay = predicted_delay * 0.6
                adv = round(predicted_delay - alt_delay, 1)
                if adv <= 5:
                    continue
                alternatives.append(AlternativeTravelOption(
                    train_number=str(tr["train_number"]), train_name=str(tr["train_name"]),
                    departure_time=str(tr["departure_time"]), arrival_time=str(tr["arrival_time"]),
                    delay_advantage_minutes=adv,
                    reason=(f"Leaves {tr['source']} {tr['departure_time']}, reaches {station} {tr['arrival_time']}; "
                            f"{int(tr['punctuality_pct'])}% on time (simulated history)."),
                ))

        drivers = self._shap(bundle, feats) if bundle and "model" in bundle else []

        return DelayPredictResponse(
            train_number=request.train_number,
            station_code=station,
            predicted_delay_minutes=predicted_delay,
            departure_delay_minutes=departure_delay,
            confidence_interval={"lower_bound": lower, "upper_bound": upper, "coverage": 0.8},
            congestion_level=congestion_level,
            station_congestion_index=round(cong_idx / 100.0, 3),
            congestion_source=cong_src,
            weather_source=weather_src,
            weather_index_used=round(weather_model, 2),
            weather_conditions=weather_conditions,
            distance_from_origin_km=round(dist_km, 1),
            distance_source=dist_src,
            scheduled_time=scheduled if scheduled and scheduled != "nan" else None,
            disruption_probability=round(p_over_60, 3),
            proactive_alert=alert,
            alerts=alerts,
            alternative_options=alternatives,
            delay_drivers=drivers,
            explanation_method="SHAP (TreeExplainer)" if drivers else "unavailable",
            model_version=version,
            confidence_score=0.8,
        )


delay_predictor_service = DelayPredictorService()
