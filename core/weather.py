"""
core/weather.py
===============
Weather for the delay model — the same 0–10 "adverse weather index" in training and at prediction time.

The index is computed from physical conditions on the travel day at the station:
    rain     daily precipitation (mm)          0 → 0,  10 mm → 1.7,  30 mm → 3.8,  60 mm → 5.2
    fog      lowest visibility (m)             < 2 km → 1.5,  < 1 km → 3,  < 500 m → 4.5,  < 200 m → 6
    storm    thunderstorm (WMO codes 95–99)    + 2
    wind     strongest gust-free wind (km/h)   ≥ 30 → + 0.5,  ≥ 50 → + 1.5

At prediction time the conditions come from the **Open-Meteo forecast API** (free, no key; forecasts up to
16 days ahead and the recent past) for the station's real coordinates. Outside that window, or when the API
cannot be reached, the index falls back to the station's climatology for that month — the same regional
climate model the training data was drawn from (south-west monsoon by region, north-east monsoon on the
Tamil Nadu / south Andhra coast, winter fog over the Indo-Gangetic plain, pre-monsoon thunderstorms in the east).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

FORECAST_DAYS_AHEAD = 15
PAST_DAYS = 60
_CACHE_TTL_S = 3 * 3600
_cache: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
_lock = threading.Lock()


# ── Physical conditions → index (shared by the data generator and the service) ──

def weather_index(precip_mm: float, min_visibility_m: float = 20000.0, storm: bool = False,
                  wind_kmh: float = 10.0) -> float:
    rain = 6.0 * (1.0 - math.exp(-max(0.0, float(precip_mm)) / 30.0))
    v = float(min_visibility_m)
    fog = 6.0 if v < 200 else 4.5 if v < 500 else 3.0 if v < 1000 else 1.5 if v < 2000 else 0.0
    w = float(wind_kmh)
    gust = 1.5 if w >= 50 else 0.5 if w >= 30 else 0.0
    return round(float(min(10.0, rain + fog + (2.0 if storm else 0.0) + gust)), 2)


# ── Regional climatology (used to simulate training data and as the fallback) ──

def _monsoon_mean_mm(lat: float, lon: float) -> float:
    if (lat < 13.0 and lon < 77.2) or (lat < 21.5 and lon < 75.0):
        return 22.0          # Konkan, Goa, coastal Karnataka, Kerala
    if lon > 88.5:
        return 15.0          # north-east
    if lon > 83.0 and lat > 19.0:
        return 10.0          # Odisha, Bengal, Bihar, Jharkhand
    if lon < 75.5 and lat > 23.0:
        return 3.0           # Rajasthan, Kutch, Gujarat plains
    if lat < 13.5 and lon > 77.5:
        return 2.5           # Tamil Nadu (rain shadow in the south-west monsoon)
    return 7.0               # central India, Deccan, Gangetic plain


def _fog_probability(lat: float, lon: float, d: date) -> float:
    if not (24.0 <= lat <= 32.0 and 73.5 <= lon <= 88.5):
        return 0.0
    if d.month == 12:
        return 0.15 if d.day < 15 else 0.35
    if d.month == 1:
        return 0.40
    if d.month == 2:
        return 0.10 if d.day < 15 else 0.02
    return 0.0


def sample_conditions(lat: float, lon: float, d: date, rng: np.random.Generator) -> Dict[str, Any]:
    """One plausible day of weather at (lat, lon) on date d, drawn from the regional climatology."""
    m = d.month
    if m in (6, 7, 8, 9):
        mean, p_rain, p_storm = _monsoon_mean_mm(lat, lon) * (0.6 if m in (6, 9) else 1.0), 0.6, 0.08
    elif m in (10, 11, 12) and lat < 16.0 and lon > 78.0:
        mean, p_rain, p_storm = {10: 6.0, 11: 9.0, 12: 5.0}[m], 0.45, 0.05    # north-east monsoon coast
    elif m in (4, 5) and lon > 83.0:
        mean, p_rain, p_storm = 2.0, 0.25, 0.10                               # nor'westers
    else:
        mean, p_rain, p_storm = 0.6, 0.08, 0.01
    precip = float(rng.exponential(mean / p_rain)) if rng.random() < p_rain else 0.0
    storm = bool(rng.random() < p_storm)
    wind = float(max(0.0, rng.normal(14, 6) + (22 if storm else 0)))
    if m in (10, 11, 5) and lon > 79.0 and lat < 22.5 and rng.random() < 0.015:   # Bay of Bengal cyclone
        precip, wind, storm = precip + float(rng.uniform(80, 200)), float(rng.uniform(60, 110)), True
    fog = rng.random() < _fog_probability(lat, lon, d)
    vis = float(rng.uniform(40, 900)) if fog else float(rng.uniform(4000, 24000))
    return {"precip_mm": round(precip, 1), "min_visibility_m": round(vis), "storm": storm, "wind_kmh": round(wind, 1)}


def climatology_index(lat: Optional[float], lon: Optional[float], d: date) -> float:
    """Expected index for this place and time of year (mean of 400 seeded draws)."""
    lat = 23.0 if lat is None or (isinstance(lat, float) and math.isnan(lat)) else float(lat)
    lon = 79.0 if lon is None or (isinstance(lon, float) and math.isnan(lon)) else float(lon)
    key = ("clim", round(lat, 0), round(lon, 0), d.month, d.day // 15)
    with _lock:
        if key in _cache:
            return _cache[key][0]
    rng = np.random.default_rng(int(key[1] + 90) * 100_003 + int(key[2] + 180) * 1_009 + key[3] * 31 + key[4])
    val = round(float(np.mean([weather_index(**sample_conditions(lat, lon, d, rng)) for _ in range(400)])), 2)
    with _lock:
        _cache[key] = (val, {})
    return val


# ── Live: Open-Meteo forecast ────────────────────────────────────────────────

def _fetch_open_meteo(lat: float, lon: float, d: date) -> Dict[str, Any]:
    import requests
    params = {"latitude": round(lat, 3), "longitude": round(lon, 3), "timezone": "Asia/Kolkata",
              "daily": "precipitation_sum,weather_code,wind_speed_10m_max", "hourly": "visibility",
              "start_date": d.isoformat(), "end_date": d.isoformat()}
    r = requests.get(settings.open_meteo_url, params=params, timeout=4)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily") or {}
    precip = float((daily.get("precipitation_sum") or [0.0])[0] or 0.0)
    code = int((daily.get("weather_code") or [0])[0] or 0)
    wind = float((daily.get("wind_speed_10m_max") or [10.0])[0] or 10.0)
    vis_list = [v for v in ((j.get("hourly") or {}).get("visibility") or []) if v is not None]
    vis = float(min(vis_list)) if vis_list else 20000.0
    if code in (45, 48):
        vis = min(vis, 800.0)
    return {"precip_mm": round(precip, 1), "min_visibility_m": round(vis), "storm": 95 <= code <= 99,
            "wind_kmh": round(wind, 1), "weather_code": code}


def weather_for(lat: Optional[float], lon: Optional[float], d: date) -> Dict[str, Any]:
    """{'index', 'source', 'conditions'} for a station and travel date — live when possible."""
    have_coords = lat is not None and lon is not None and not (isinstance(lat, float) and math.isnan(lat))
    in_window = (date.today() - timedelta(days=PAST_DAYS)) <= d <= (date.today() + timedelta(days=FORECAST_DAYS_AHEAD))
    if settings.weather_live and have_coords and in_window:
        key = ("live", round(float(lat), 2), round(float(lon), 2), d.isoformat())
        with _lock:
            hit = _cache.get(key)
        if hit and time.time() - hit[1].get("_at", 0) < _CACHE_TTL_S:
            return {"index": hit[0], "source": "Open-Meteo forecast", "conditions": hit[1]["conditions"]}
        try:
            cond = _fetch_open_meteo(float(lat), float(lon), d)
            idx = weather_index(cond["precip_mm"], cond["min_visibility_m"], cond["storm"], cond["wind_kmh"])
            with _lock:
                _cache[key] = (idx, {"conditions": cond, "_at": time.time()})
            return {"index": idx, "source": "Open-Meteo forecast", "conditions": cond}
        except Exception as e:
            logger.info(f"[weather] Open-Meteo unavailable ({type(e).__name__}); using climatology")
    reason = ("live weather off" if not settings.weather_live else
              "no station coordinates" if not have_coords else
              "date outside the forecast window" if not in_window else "forecast unavailable")
    return {"index": climatology_index(lat, lon, d), "source": f"station climatology ({reason})", "conditions": {}}
