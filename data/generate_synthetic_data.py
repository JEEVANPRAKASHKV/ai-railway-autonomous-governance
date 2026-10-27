"""
data/generate_synthetic_data.py
================================
Dataset builder — real network and official fares, simulated outcomes.

Real inputs (no download needed; see data/real/):
  - stations and trains: Indian Railways timetable, datameet/railways (CC0, 2016 snapshot)
  - every halt with its real distance and schedule: data/real/train_stops.csv.gz
  - fares: Railway Board TAG 2025 fare tables (core/fares.py)
  - weather: regional climatology here; live Open-Meteo forecasts at prediction time (core/weather.py)

Simulated (no public data exists for these): booking outcomes (WL/RAC/confirmed), how full trains are,
delays, station crowding, user personas, fraud sessions, punctuality and coach counts.
All random operations are seeded with RANDOM_SEED=42 for reproducibility.

Outputs (saved to data/raw/):
  - trains.csv           : real reservable trains with official fares
  - stations.csv         : real stations with coordinates
  - pnr_history.csv      : 50,000 simulated WL/RAC outcomes on real trains
  - pricing_history.csv  : 100,000 fares on real segments (official rules, simulated occupancy)
  - user_profiles.csv    : 5,000 user personas
  - congestion_logs.csv  : station congestion time-series (400 busiest real stations)
  - delay_logs.csv       : 200,000 simulated delays at real halts
  - fraud_events.csv     : 10,000 labeled booking sessions
  - irctc_policies.txt   : IRCTC policy document for RAG
  - railway_faq.txt      : 500 Q&A pairs for RAG
  - data/baseline_distributions.pkl : reference feature stats for drift detection

Run:
    python data/generate_synthetic_data.py
"""

import os
import sys
import pickle
import random
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Reference Data ─────────────────────────────────────────────────────────────

# ── Real network (data/real, built from the datameet timetable) ───────────────
sys.path.insert(0, ROOT)
from core.fares import fare_breakdown, payable_fare, tatkal_charge   # noqa: E402
from core.weather import sample_conditions, weather_index             # noqa: E402

REAL_DIR = os.path.join(ROOT, "data", "real")
RNG = np.random.default_rng(RANDOM_SEED)

CLASSES = ["SL", "3A", "2A", "1A", "CC", "EC", "2S"]
CLASS_POPULARITY = {"SL": 35, "3A": 25, "2A": 15, "1A": 5, "CC": 10, "EC": 3, "2S": 7}
QUOTAS = ["GN", "TQ", "LADIES", "DEFENCE", "SENIOR", "HO", "DP"]
BERTH_TYPES = ["Lower", "Middle", "Upper", "Side Lower", "Side Upper", "RAC"]
FINAL_STATUSES = ["Confirmed", "RAC", "WL"]
PERSONAS = ["budget", "premium", "student", "senior", "business", "family"]
REASON_CODES = ["HEAVY_TRAFFIC", "SIGNAL_FAILURE", "ENGINE_FAILURE", "WEATHER", "TRACK_MAINTENANCE",
                "CREW_DELAY", "GATE_OBSTRUCTION", "FLOOD", "FOG", "PASSENGER_EMERGENCY", "ON_TIME"]
FRAUD_LABELS = ["legitimate", "suspicious", "bot"]

# Major city terminals (the busiest platforms on the network)
METRO_TERMINALS = {"NDLS", "DLI", "NZM", "ANVT", "CSTM", "BCT", "LTT", "DR", "BDTS", "HWH", "SDAH", "KOAA",
                   "MAS", "MS", "SBC", "YPR", "SC", "HYB", "KCG", "ADI", "PUNE", "LKO", "CNB", "PNBE", "JP"}
TIER_CONGESTION = {"Metro": 70, "A1": 55, "A": 45, "B": 30, "C": 20}
TIER_PLATFORMS = {"Metro": 16, "A1": 10, "A": 7, "B": 4, "C": 2}
# Simulated on-time performance by service (no open punctuality data exists)
TYPE_PUNCTUALITY = {"Rajdhani": 86, "Shatabdi": 88, "Duronto": 82, "Jan Shatabdi": 80, "Garib Rath": 72,
                    "Superfast": 74, "Sampark Kranti": 72, "Mail": 70, "Express": 68}
FLEXI_TYPES = {"Rajdhani", "Duronto", "Shatabdi"}
FLEXI_CAP = {"3A": 1.4}                                      # others 1.5; 1A and EC are outside flexi fare


def _hhmm(t: str) -> int:
    h, m = str(t).split(":")[:2]
    return int(h) * 60 + int(m)


def season_factor(d: date) -> float:
    """Returns congestion/demand multiplier based on season."""
    month = d.month
    day = d.day
    # Diwali peak (Oct-Nov), Summer peak (May-Jun), New Year (Dec-Jan)
    if month in (10, 11) and day > 10:  # Diwali season
        return 1.8
    elif month in (5, 6):              # Summer vacation
        return 1.6
    elif month in (12,) and day > 20:  # Christmas/New Year
        return 1.5
    elif month == 1 and day < 10:      # New Year
        return 1.4
    elif month in (3, 4):              # Holi season
        return 1.3
    else:
        return 1.0 + 0.1 * np.random.randn()


def _tier(code: str, halting: int) -> str:
    if code in METRO_TERMINALS:
        return "Metro"
    return "A1" if halting >= 120 else "A" if halting >= 50 else "B" if halting >= 15 else "C"


def load_stops() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REAL_DIR, "train_stops.csv.gz"), dtype={"train_number": str})


# ── Generator Functions ────────────────────────────────────────────────────────

def generate_stations() -> pd.DataFrame:
    """Real stations (datameet) + derived tier; congestion baseline and platform count are estimates."""
    logger.info("Building stations.csv from the real network ...")
    st = pd.read_csv(os.path.join(REAL_DIR, "stations.csv"))
    st["tier"] = [_tier(c, int(n)) for c, n in zip(st["station_code"], st["trains_halting"])]
    st["baseline_congestion"] = [TIER_CONGESTION[t] + int(RNG.integers(-5, 5)) for t in st["tier"]]
    st["platforms"] = st["tier"].map(TIER_PLATFORMS)
    cols = ["station_code", "station_name", "zone", "state", "tier", "baseline_congestion", "platforms",
            "latitude", "longitude", "trains_halting"]
    df = st[cols].reset_index(drop=True)
    df.to_csv(os.path.join(RAW_DIR, "stations.csv"), index=False)
    logger.info(f"  → {len(df)} real stations saved.")
    return df


def generate_trains(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Real trains (datameet 2016 timetable) + official TAG 2025 fares; punctuality and seats are simulated."""
    logger.info("Building trains.csv from the real network ...")
    t = pd.read_csv(os.path.join(REAL_DIR, "trains.csv"), dtype={"train_number": str, "return_train": str})
    stops = load_stops()
    coords = stations_df.set_index("station_code")[["latitude", "longitude"]]
    fog_belt = set(coords[(coords.latitude.between(24, 32)) & (coords.longitude.between(73.5, 88.5))].index)
    fog_trains = set(stops[stops["station_code"].isin(fog_belt)]["train_number"])
    rows = []
    for r in t.itertuples(index=False):
        classes = [c for c in str(r.classes).split("|") if c in CLASSES]
        if not classes:
            continue
        dist, dur_h = float(r.distance_km), float(r.duration_minutes) / 60.0
        punct = TYPE_PUNCTUALITY.get(r.train_type, 70) - 4.0 * dist / 1000.0 - (6 if r.train_number in fog_trains else 0)
        row = {
            "train_number": r.train_number, "train_name": r.train_name,
            "source": r.source, "destination": r.destination,
            "distance_km": int(dist), "duration_hours": round(dur_h, 2),
            "avg_speed_kmph": round(dist / max(dur_h, 0.1), 1),
            "classes": "|".join(classes), "quota_types": "GN|TQ|LADIES|SENIOR|HO",
            "departure_time": r.departure_time, "arrival_time": r.arrival_time,
            "punctuality_pct": int(np.clip(round(punct + RNG.normal(0, 4)), 45, 97)),
            "zone": r.zone, "train_type": r.train_type, "halts": int(r.halts),
        }
        for cls in CLASSES:
            f = payable_fare(r.train_type, cls, dist, r.train_number) if cls in classes else None
            row[f"base_fare_{cls.lower()}"] = int(f) if f else 0
        sl_coaches = int(RNG.integers(8, 13)) if "SL" in classes else 0
        ac3_coaches = (int(RNG.integers(8, 12)) if r.train_type in ("Rajdhani", "Duronto") else
                       12 if r.train_type == "Garib Rath" else int(RNG.integers(2, 7))) if "3A" in classes else 0
        row["total_seats_sl"] = sl_coaches * 72
        row["total_seats_3a"] = ac3_coaches * (74 if r.train_type == "Garib Rath" else 64)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "trains.csv"), index=False)
    logger.info(f"  → {len(df)} real trains saved (fares: TAG 2025).")
    return df


def _pick_class(classes: List[str]) -> str:
    return random.choices(classes, weights=[CLASS_POPULARITY.get(c, 5) for c in classes])[0]


def generate_pnr_history(trains_df: pd.DataFrame) -> pd.DataFrame:
    """Simulated booking outcomes on real trains (no public PNR-outcome data exists)."""
    logger.info("Generating pnr_history.csv (50,000 records) ...")
    rows = []
    trains = trains_df[["train_number", "classes"]].to_dict("records")
    start_date = date(2023, 1, 1)
    end_date = date(2026, 10, 1)
    total_days = (end_date - start_date).days

    for i in tqdm(range(50000), desc="PNR records"):
        journey_date = start_date + timedelta(days=random.randint(0, total_days))
        # Advance reservation period: 120 days until 31 Oct 2024, 60 days from 1 Nov 2024
        arp = 120 if journey_date < date(2024, 11, 1) + timedelta(days=60) else 60
        booking_days_before = random.randint(1, arp)
        booking_date = journey_date - timedelta(days=booking_days_before)
        if booking_date < start_date:
            booking_date = start_date

        t = random.choice(trains)
        train = t["train_number"]
        cls = _pick_class(str(t["classes"]).split("|"))
        quota = random.choices(QUOTAS, weights=[70, 10, 5, 5, 5, 3, 2])[0]
        initial_wl = random.randint(1, 150)

        clearance_probs = {"SL": 0.65, "3A": 0.50, "2A": 0.42, "1A": 0.38, "CC": 0.55, "EC": 0.45, "2S": 0.70}
        base_prob = clearance_probs.get(cls, 0.50)
        quota_adj = {"TQ": 0.85, "GN": 1.0, "LADIES": 1.1, "DEFENCE": 1.05, "SENIOR": 1.0}
        base_prob *= quota_adj.get(quota, 1.0)
        if initial_wl <= 10:
            base_prob = min(0.95, base_prob * 1.5)
        elif initial_wl <= 30:
            base_prob = min(0.90, base_prob * 1.2)
        elif initial_wl <= 60:
            pass
        elif initial_wl <= 100:
            base_prob *= 0.7
        else:
            base_prob *= 0.3
        sf = season_factor(journey_date)
        base_prob /= sf
        base_prob = max(0.05, min(0.98, base_prob))

        rand = random.random()
        if rand < base_prob:
            final_status = "Confirmed"
            berth = random.choices(BERTH_TYPES[:-1], weights=[30, 25, 25, 10, 10])[0]
        elif rand < base_prob + 0.15:
            final_status = "RAC"
            berth = "RAC"
        else:
            final_status = "WL"
            berth = "WL"

        rows.append({
            "pnr": f"PNR{i+1:07d}",
            "train_number": train,
            "journey_class": cls,
            "quota": quota,
            "journey_date": journey_date.isoformat(),
            "booking_date": booking_date.isoformat(),
            "days_to_departure": booking_days_before,
            "initial_wl_number": initial_wl,
            "final_status": final_status,
            "berth_type": berth,
            "seniority_flag": int(random.random() < 0.08),
            "disability_flag": int(random.random() < 0.03),
            "season_factor": round(sf, 2),
            "cancellation_date": (journey_date - timedelta(days=random.randint(1, 10))).isoformat()
                if final_status == "WL" and random.random() < 0.4 else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "pnr_history.csv"), index=False)
    logger.info(f"  → {len(df)} PNR records saved.")
    return df


def generate_pricing_history(trains_df: pd.DataFrame) -> pd.DataFrame:
    """Fares paid on real trains and real journey segments, priced by the official rules:
    TAG 2025 fare for the distance; flexi fare (+10% per 10% of berths sold, up to 1.5×, 1.4× in 3A, none in
    1A/EC) on Rajdhani, Duronto and Shatabdi; Tatkal charge in the Tatkal window. How full the train is
    (demand, route popularity) is simulated."""
    logger.info("Generating pricing_history.csv (100,000 records) ...")
    rows = []
    stops = load_stops()
    km_by_train = {k: g["km_from_origin"].values for k, g in stops.groupby("train_number")}
    trains = trains_df[["train_number", "classes", "train_type", "distance_km"]].to_dict("records")
    start_date = date(2024, 1, 1)
    end_date = date(2026, 10, 1)
    total_days = (end_date - start_date).days

    for i in tqdm(range(100000), desc="Pricing records"):
        t = random.choice(trains)
        train, ttype = t["train_number"], t["train_type"]
        journey_date = start_date + timedelta(days=random.randint(0, total_days))
        cls = _pick_class(str(t["classes"]).split("|"))
        kms = km_by_train.get(train)
        if kms is not None and len(kms) > 2 and random.random() < 0.5:     # an intermediate segment
            a, b = sorted(random.sample(range(len(kms)), 2))
            km = max(1.0, float(kms[b] - kms[a]))
        else:
            km = float(t["distance_km"])
        base_fare = payable_fare(ttype, cls, km, train) or 0.0
        if base_fare <= 0:
            continue

        arp = 120 if journey_date < date(2024, 11, 1) + timedelta(days=60) else 60
        days_to_dep = random.randint(0, arp)
        sf = season_factor(journey_date)
        popularity = round(random.uniform(0.3, 1.0), 2)
        demand_index = round(float(np.clip(1.0 + 0.5 * (1 - days_to_dep / arp) * sf + 0.2 * np.random.randn(), 0.5, 3.0)), 3)

        flexi = ttype in FLEXI_TYPES and cls not in ("1A", "EC")
        multiplier = 1.0
        if flexi:
            sold = float(np.clip((demand_index - 0.5) / 1.6 * (0.6 + 0.6 * popularity), 0.0, 1.0))
            multiplier = min(FLEXI_CAP.get(cls, 1.5), 1.0 + 0.1 * int(max(0.0, sold - 0.1) / 0.1))
        tatkal_flag = days_to_dep <= 1 and random.random() < 0.6
        tq = 0.0
        if tatkal_flag:
            b = fare_breakdown(ttype, cls, km, train)
            tq = tatkal_charge(cls, b["base_fare"]) if b else 0.0
        dynamic_fare = round(base_fare * multiplier + tq, 2)

        rows.append({
            "record_id": i + 1,
            "train_number": train,
            "journey_class": cls,
            "journey_date": journey_date.isoformat(),
            "booking_date": (journey_date - timedelta(days=days_to_dep)).isoformat(),
            "days_to_departure": days_to_dep,
            "distance_km": round(km, 1),
            "base_fare": round(base_fare, 2),
            "dynamic_fare": dynamic_fare,
            "demand_index": demand_index,
            "flexi_flag": int(flexi),
            "tatkal_flag": int(tatkal_flag),
            "tatkal_surcharge": tq,
            "season_factor": round(sf, 2),
            "route_popularity": popularity,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "pricing_history.csv"), index=False)
    logger.info(f"  → {len(df)} pricing records saved.")
    return df


def generate_user_profiles() -> pd.DataFrame:
    logger.info("Generating user_profiles.csv (5,000 records) ...")
    rows = []
    persona_weights = [0.30, 0.15, 0.25, 0.10, 0.12, 0.08]
    stops = load_stops()
    major = set(pd.read_csv(os.path.join(REAL_DIR, "stations.csv")).nlargest(150, "trains_halting")["station_code"])
    seqs = [g["station_code"].tolist() for _, g in stops[stops["station_code"].isin(major)].groupby("train_number")]
    seqs = [s for s in seqs if len(s) >= 2]

    for i in range(5000):
        persona = random.choices(PERSONAS, weights=persona_weights)[0]
        class_pref = {
            "budget": random.choices(CLASSES, weights=[50, 20, 5, 0, 15, 0, 10])[0],
            "premium": random.choices(CLASSES, weights=[5, 25, 40, 20, 10, 0, 0])[0],
            "student": random.choices(CLASSES, weights=[55, 20, 5, 0, 15, 0, 5])[0],
            "senior": random.choices(CLASSES, weights=[30, 35, 20, 5, 10, 0, 0])[0],
            "business": random.choices(CLASSES, weights=[0, 15, 30, 40, 15, 0, 0])[0],
            "family": random.choices(CLASSES, weights=[35, 30, 20, 5, 10, 0, 0])[0],
        }.get(persona, "SL")

        budget_range = {
            "budget": (200, 800), "premium": (1500, 5000), "student": (150, 600),
            "senior": (400, 1500), "business": (2000, 8000), "family": (600, 2500),
        }.get(persona, (300, 1200))

        frequent_routes = []
        for _ in range(random.randint(1, 4)):        # real station pairs served by a direct train
            s = random.choice(seqs)
            a, b = sorted(random.sample(range(len(s)), 2))
            frequent_routes.append(f"{s[a]}-{s[b]}")

        rows.append({
            "user_id": f"USR{i+1:05d}",
            "persona": persona,
            "preferred_class": class_pref,
            "budget_min": budget_range[0],
            "budget_max": budget_range[1],
            "frequent_routes": "|".join(frequent_routes),
            "senior_flag": int(persona == "senior" or random.random() < 0.08),
            "disability_flag": int(random.random() < 0.03),
            "preferred_quota": random.choices(QUOTAS[:5], weights=[70, 10, 5, 5, 10])[0],
            "booking_frequency_per_month": round(random.uniform(0.5, 8.0), 1),
            "avg_advance_booking_days": random.randint(5, 60),
            "loyalty_score": round(random.uniform(0, 100), 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "user_profiles.csv"), index=False)
    logger.info(f"  → {len(df)} user profiles saved.")
    return df


def generate_congestion_logs(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Simulated platform-crowding history for the 400 busiest real stations."""
    logger.info("Generating congestion_logs.csv ...")
    rows = []
    station_sample = stations_df.nlargest(400, "trains_halting")
    start_date = date(2025, 1, 1)
    end_date = date(2026, 10, 1)
    current = start_date

    while current <= end_date:
        sf = season_factor(current)
        hour = random.choice([7, 10, 13, 16, 19, 22])
        for _, row in station_sample.sample(60, random_state=random.randint(0, 999)).iterrows():
            baseline = row["baseline_congestion"]
            hour_factor = {7: 1.3, 10: 1.5, 13: 1.1, 16: 1.4, 19: 1.6, 22: 0.7}.get(hour, 1.0)
            congestion = np.clip(baseline * sf * hour_factor + np.random.normal(0, 8), 0, 100)
            reason = "NORMAL"
            if congestion > 85:
                reason = random.choice(["FESTIVAL", "HOLIDAY_RUSH", "SPECIAL_TRAIN", "CANCELLATION_SURGE"])
            elif congestion > 70:
                reason = random.choice(["PEAK_HOUR", "WEEKEND", "EXAM_SEASON"])
            rows.append({
                "station_code": row["station_code"],
                "date": current.isoformat(),
                "hour": hour,
                "congestion_index": round(float(congestion), 2),
                "reason_tag": reason,
                "season_factor": round(sf, 2),
            })
        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "congestion_logs.csv"), index=False)
    logger.info(f"  → {len(df)} congestion records saved.")
    return df


def generate_delay_logs(trains_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Simulated arrival delays at real halts of real trains. Each record uses the halt's real distance from the
    origin and scheduled time, and a day of weather drawn from that station's regional climatology (the same
    climate model and weather index used for live Open-Meteo data at prediction time)."""
    logger.info("Generating delay_logs.csv (200,000 records) ...")
    rows = []
    stops = load_stops()
    stops = stops[stops["seq"] > 0]
    stops = stops[stops["train_number"].isin(set(trains_df["train_number"]))].reset_index(drop=True)
    st = stations_df.set_index("station_code")
    lat = st["latitude"].to_dict()
    lon = st["longitude"].to_dict()
    base_cong = st["baseline_congestion"].to_dict()
    start_date = date(2024, 1, 1)
    end_date = date(2026, 10, 1)
    total_days = (end_date - start_date).days
    picks = RNG.integers(0, len(stops), size=200000)

    for i in tqdm(range(200000), desc="Delay records"):
        s = stops.iloc[int(picks[i])]
        code = s["station_code"]
        journey_date = start_date + timedelta(days=random.randint(0, total_days))
        sf = season_factor(journey_date)
        cond = sample_conditions(lat.get(code, 23.0) if pd.notna(lat.get(code)) else 23.0,
                                 lon.get(code, 79.0) if pd.notna(lon.get(code)) else 79.0, journey_date, RNG)
        weather = weather_index(**cond)

        # station crowding on the day (0–100) → the model's congestion scale (same mapping as the service)
        cong_idx = float(np.clip(base_cong.get(code, 30) + np.random.normal(0, 18), 0, 100))
        congestion = (10.0 + 0.8 * cong_idx) * sf
        distance = float(s["km_from_origin"])

        if cond["min_visibility_m"] < 1000:
            reason = "FOG"
        elif cond["precip_mm"] >= 60:
            reason = "FLOOD"
        elif cond["storm"] or cond["precip_mm"] >= 15:
            reason = "WEATHER"
        else:
            reason = random.choices(REASON_CODES, weights=[15, 8, 4, 0, 10, 8, 8, 0, 0, 4, 34])[0]

        base_delay = np.random.exponential(7)
        weather_effect = 1.2 * weather ** 1.5            # 0 → ~38 min at 10/10
        congestion_effect = 0.22 * max(0.0, congestion - 30)
        distance_effect = distance / 100.0               # delay accumulates along the route
        delay = base_delay * sf + weather_effect + congestion_effect + distance_effect + np.random.normal(0, 5)
        if reason == "ON_TIME" and weather < 2 and congestion < 60:
            delay = max(0, np.random.normal(2, 3))
        delay = max(0.0, delay)

        sched = str(s["arrival"]) if isinstance(s["arrival"], str) and s["arrival"] else "12:00"
        scheduled_arr = datetime.combine(journey_date, datetime.min.time()) + timedelta(minutes=_hhmm(sched))
        actual_arr = scheduled_arr + timedelta(minutes=delay)

        rows.append({
            "log_id": i + 1,
            "train_number": s["train_number"],
            "station_code": code,
            "journey_date": journey_date.isoformat(),
            "scheduled_arrival": scheduled_arr.strftime("%H:%M"),
            "actual_arrival": actual_arr.strftime("%H:%M"),
            "delay_minutes": round(delay, 1),
            "reason_code": reason,
            "weather_index": round(weather, 2),
            "precip_mm": cond["precip_mm"],
            "min_visibility_m": cond["min_visibility_m"],
            "congestion_at_station": round(congestion, 1),
            "distance_from_origin_km": round(distance, 1),
            "month": journey_date.month,
            "season_factor": round(sf, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "delay_logs.csv"), index=False)
    logger.info(f"  → {len(df)} delay records saved.")
    return df


def generate_fraud_events() -> pd.DataFrame:
    logger.info("Generating fraud_events.csv (10,000 records) ...")
    rows = []

    for i in range(10000):
        label_rand = random.random()
        if label_rand < 0.80:
            label = "legitimate"
        elif label_rand < 0.85:
            label = "suspicious"
        else:
            label = "bot"

        if label == "bot":
            rpm = round(random.uniform(50, 300), 1)
            pnr_count = random.randint(10, 80)
            tatkal = True
            account_age = random.randint(0, 30)
            pay_attempts = random.randint(5, 20)
            captcha_fail = random.randint(3, 15)
            booking_hour = random.choice([10, 11])
            bulk = True
            ip_risk = round(random.uniform(0.7, 1.0), 2)
        elif label == "suspicious":
            rpm = round(random.uniform(10, 50), 1)
            pnr_count = random.randint(3, 15)
            tatkal = random.random() > 0.4
            account_age = random.randint(1, 180)
            pay_attempts = random.randint(2, 8)
            captcha_fail = random.randint(1, 5)
            booking_hour = random.randint(9, 14)
            bulk = random.random() > 0.5
            ip_risk = round(random.uniform(0.3, 0.7), 2)
        else:
            rpm = round(random.uniform(0.1, 5.0), 1)
            pnr_count = random.randint(1, 4)
            tatkal = random.random() > 0.7
            account_age = random.randint(30, 3650)
            pay_attempts = random.randint(1, 3)
            captcha_fail = random.randint(0, 1)
            booking_hour = random.randint(6, 22)
            bulk = False
            ip_risk = round(random.uniform(0.0, 0.2), 2)

        rows.append({
            "session_id": f"SES{i+1:06d}",
            "ip_hash": f"IP{random.randint(100000, 999999):06d}",
            "requests_per_minute": rpm,
            "unique_pnr_attempts": pnr_count,
            "tatkal_flag": int(tatkal),
            "account_age_days": account_age,
            "payment_attempts": pay_attempts,
            "failed_captcha_count": captcha_fail,
            "booking_time_hour": booking_hour,
            "is_bulk_booking": int(bulk),
            "ip_risk_score": ip_risk,
            "label": label,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "fraud_events.csv"), index=False)
    logger.info(f"  → {len(df)} fraud events saved. Label dist: {df['label'].value_counts().to_dict()}")
    return df


def generate_irctc_policies() -> None:
    logger.info("Generating irctc_policies.txt ...")
    content = """IRCTC RAILWAY BOOKING & TRAVEL POLICIES — Synthetic Reference Document
=======================================================================
Version: 2026.1 | Last Updated: October 2026

SECTION 1: CANCELLATION POLICY
-------------------------------
1.1 General Cancellation Rules
All cancellations must be done before the charting of the train. Once charting is done, online cancellation is not permitted.

1.2 Cancellation Charges (Non-Tatkal Tickets)
- More than 48 hours before departure:
  * AC First Class (1A) / Executive Class (EC): ₹240 flat fee
  * AC 2-Tier (2A) / AC Chair Car (CC): ₹200 flat fee
  * AC 3-Tier (3A): ₹180 flat fee
  * Sleeper Class (SL): ₹120 flat fee
  * Second Class (2S): ₹60 flat fee
  Minimum cancellation charge: 25% of fare

- 12 to 48 hours before departure:
  * 25% of total fare (subject to minimum flat fee above)

- Less than 12 hours and up to 4 hours before departure:
  * 50% of total fare

- Less than 4 hours before departure or after departure:
  * No refund permitted

1.3 RAC Ticket Cancellation
RAC tickets cancelled before charting: Full refund minus ₹60 cancellation fee.

1.4 Waitlist (WL) Ticket Cancellation
If WL ticket not confirmed at time of charting, full refund is automatically processed. No action needed.
If WL ticket confirmed and then cancelled, normal cancellation charges apply.

SECTION 2: TATKAL BOOKING
--------------------------
2.1 Tatkal Booking Windows
- AC Classes (1A, 2A, 3A, CC, EC): Opens at 10:00 AM, one day before journey date
- Non-AC Classes (SL, 2S): Opens at 11:00 AM, one day before journey date

2.2 Tatkal Charges (Additional to Base Fare)
- AC First Class (1A): ₹400 per ticket
- AC 2-Tier (2A): ₹300 per ticket  
- AC 3-Tier (3A): ₹200 per ticket
- AC Chair Car (CC): ₹125 per ticket
- Sleeper Class (SL): ₹100 per ticket
- Second Class (2S): ₹10 per ticket

2.3 Tatkal Cancellation Policy
Tatkal tickets are generally non-refundable. No refund for cancellations.
Exception: If train is cancelled or runs late by more than 3 hours, full refund is provided.

2.4 Tatkal Quota
Tatkal quota is approximately 30% of total coach capacity reserved separately.
Booking restricted to 2 passengers per PNR under Tatkal quota.

SECTION 3: WAITLIST (WL) AND RAC POLICY
-----------------------------------------
3.1 Waitlist Categories
- WL (General Waitlist): Standard waitlist
- RLWL (Remote Location Waitlist): For stations not covered by General Quota
- PQWL (Pooled Quota Waitlist): For certain intermediate stations
- GNWL (General Quota Waitlist): Most common type

3.2 WL Confirmation Process
Waitlist tickets get confirmed as confirmed ticket holders cancel their bookings.
Confirmation depends on: initial WL number, class, quota, season, route popularity.
Low WL numbers (1-15) have highest confirmation probability (70-90%).
High WL numbers (>60) have low probability (<30%).

3.3 RAC (Reservation Against Cancellation)
RAC passengers get a guaranteed seat (side berth shared between 2 passengers).
RAC confirms to berth if one of the confirmed passengers cancels.

SECTION 4: REFUND POLICY
-------------------------
4.1 Automatic Refunds
- Cancelled trains: Full refund automatically credited to original payment method within 7 days
- WL tickets not confirmed at charting: Full refund within 5 business days
- Technical failures: Full refund within 7 days

4.2 Manual Refund Claims
For partial journey not performed due to train failure: Submit TDR (Ticket Deposit Receipt) within 72 hours.
TDR can be filed on IRCTC website under "My Bookings" section.

4.3 Refund Timeline
- Credit/Debit Card: 5-7 business days
- Net Banking: 3-5 business days
- IRCTC Wallet: Instant
- UPI: 1-3 business days

SECTION 5: SENIOR CITIZEN CONCESSION
--------------------------------------
5.1 Eligibility
- Male passengers aged 60 years and above
- Female passengers aged 58 years and above

5.2 Discount
- 50% discount on Sleeper Class, AC 3-Tier, and AC 2-Tier base fare
- 40% discount on AC First Class
- Note: Tatkal surcharge is not discounted

5.3 Berth Preference
Senior citizens are entitled to Lower Berth allocation preferentially.
System automatically allocates lower berths when available.

SECTION 6: PERSONS WITH DISABILITY (PWD) CONCESSION
-----------------------------------------------------
6.1 Eligibility and Discount
- 75% discount on all classes for persons with specified disabilities
- Escort accompanying PWD passenger also gets concession

6.2 Required Documentation
Valid disability certificate from Government-approved medical authority required at time of travel.

SECTION 7: PNR STATUS AND TRAIN TRACKING
-----------------------------------------
7.1 Checking PNR Status
- IRCTC website: www.irctc.co.in → "PNR Status" section
- NTES website: www.enquiry.indianrail.gov.in
- SMS: Send PNR number to 139
- Helpline: Call 14646 (24x7)

7.2 Train Running Status
- Real-time train tracking available on NTES
- "Where is my Train" feature on IRCTC Rail Connect app
- Status updates every 15 minutes at major stations

SECTION 8: DYNAMIC PRICING
---------------------------
8.1 Flexi Fare System
Premium trains (Rajdhani, Shatabdi, Duronto, Humsafar, Tejas, Gatimaan, Vande Bharat) follow dynamic pricing.
Base fare charged for initial 10% of seats. Fare increases by 10% for every 10% of seats sold.
Maximum fare: 1.5× base fare for these trains.

8.2 Demand-Based Pricing Factors
- Days to departure (nearer = higher demand)
- Season (peak season = higher prices)
- Route popularity
- Class-specific demand

SECTION 9: BOOKING RULES
--------------------------
9.1 Maximum Tickets Per User Per Transaction
- Normal quota: 6 passengers per PNR
- Tatkal quota: 2 passengers per PNR

9.2 Advance Reservation Period
- Trains can be booked up to 120 days in advance (excluding day of departure)
- Tatkal: 1 day before journey

9.3 ID Proof Requirements
At least one of the following for adults:
Aadhaar Card, PAN Card, Voter ID, Driving License, Passport
"""
    with open(os.path.join(RAW_DIR, "irctc_policies.txt"), "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("  → irctc_policies.txt saved.")


def generate_faq() -> None:
    logger.info("Generating railway_faq.txt (500 Q&A pairs) ...")
    qa_pairs = [
        ("How do I check my PNR status?", "You can check PNR status on the IRCTC website under 'PNR Status', by SMS to 139 with your PNR number, or by calling the helpline 14646."),
        ("What is Tatkal booking?", "Tatkal is a special quota for urgent bookings. AC class opens at 10:00 AM and SL class at 11:00 AM, one day before the journey. Additional Tatkal charges apply."),
        ("Can I cancel a Tatkal ticket?", "Tatkal tickets are generally non-refundable. However, if the train is cancelled or delayed by more than 3 hours, a full refund is provided."),
        ("What does WL mean on my ticket?", "WL stands for Waitlist. Your berth is not confirmed yet. If passengers ahead of you cancel, your WL number drops and you may get confirmed."),
        ("What is RAC status?", "RAC (Reservation Against Cancellation) means you have a guaranteed seat but may share a side berth with another passenger. You will get a full berth if a confirmed passenger cancels."),
        ("How long does a refund take?", "Refunds typically take 5-7 days for credit cards, 3-5 days for net banking, instant for IRCTC wallet, and 1-3 days for UPI."),
        ("What is the senior citizen discount?", "Male passengers 60+ and female passengers 58+ get 50% discount on SL, 3A, and 2A class tickets. 40% on 1A."),
        ("How much advance can I book a ticket?", "You can book up to 120 days in advance excluding the day of departure."),
        ("What is dynamic pricing on trains?", "Premium trains like Rajdhani, Shatabdi use flexi fare where prices increase by 10% for every 10% of seats sold. Maximum is 1.5x base fare."),
        ("How many passengers can I book per PNR?", "Maximum 6 passengers per PNR for normal quota. Maximum 2 passengers per PNR for Tatkal quota."),
        ("What is the cancellation fee for 3A class?", "For cancellations more than 48 hours before departure, the cancellation fee for 3A is ₹180. Between 12-48 hours, it is 25% of fare."),
        ("Can I get a refund for WL ticket?", "If your WL ticket is not confirmed at the time of charting, you automatically get a full refund. No action needed."),
        ("What documents do I need for travel?", "You need a valid government ID: Aadhaar Card, PAN Card, Voter ID, Driving License, or Passport."),
        ("How do I file a TDR?", "Go to IRCTC website, My Bookings section, and file a Ticket Deposit Receipt (TDR) within 72 hours of the journey for partial journey refunds."),
        ("What is the helpline number for IRCTC?", "IRCTC 24x7 helpline: 14646. Railway enquiry: 139."),
        ("What is GNWL?", "GNWL (General Quota Waitlist) is the most common waitlist type, covering all passengers booking from the train's origin station."),
        ("What is RLWL?", "RLWL (Remote Location Waitlist) is for passengers booking from intermediate stations not covered by the General Quota."),
        ("My train was cancelled. Will I get a refund?", "Yes, if your train is cancelled by Railways, you get a full refund automatically without any cancellation charges."),
        ("What is the Garib Rath Express?", "Garib Rath Express is an all-AC sleeper train offering affordable AC 3-Tier travel for budget-conscious passengers."),
        ("Can I book tickets without internet?", "Yes, you can book at any PRS (Passenger Reservation System) counter at railway stations. IRCTC agents also offer booking services."),
        ("What is the maximum fine for travelling without ticket?", "The penalty for travelling without a valid ticket is ₹250 plus the applicable fare for the journey undertaken."),
        ("How do I upgrade my class?", "Class upgrades are available at the station before departure, subject to availability. Contact the Station Manager or Ticket Examiner."),
    ]

    # Expand to 500 Q&A pairs by generating variations
    topic_templates = [
        ("How long is the journey from {src} to {dst}?",
         "The journey from {src} to {dst} typically takes {hrs} hours by major express trains. Duration varies by train type."),
        ("What is the fare from {src} to {dst} in Sleeper class?",
         "The approximate Sleeper class fare from {src} to {dst} is ₹{fare}. Final fare may vary based on dynamic pricing and quotas."),
        ("Is {train} running on time today?",
         "For real-time running status of {train}, please check NTES website or the IRCTC Rail Connect app's 'Where is my Train' feature."),
        ("What classes are available on {train}?",
         "{train} typically offers SL (Sleeper), 3A (AC 3-Tier), 2A (AC 2-Tier), and in some trains 1A (AC First Class). Please check availability on IRCTC."),
        ("Can I book a lower berth?",
         "You can select berth preference at the time of booking, but it is subject to availability. Senior citizens and pregnant women are given priority for lower berths."),
    ]

    route_pairs = [
        ("Mumbai", "Delhi", 16, 485), ("Chennai", "Kolkata", 27, 620), ("Delhi", "Bangalore", 35, 730),
        ("Pune", "Hyderabad", 10, 320), ("Mumbai", "Goa", 8, 280), ("Delhi", "Jaipur", 5, 195),
        ("Chennai", "Hyderabad", 12, 380), ("Kolkata", "Guwahati", 16, 450),
        ("Delhi", "Lucknow", 7, 240), ("Mumbai", "Ahmedabad", 8, 250),
    ]

    train_names = ["Rajdhani Express", "Shatabdi Express", "Duronto", "Mail Express", "Garib Rath",
                   "Vande Bharat", "Tejas Express", "Humsafar Express"]

    while len(qa_pairs) < 500:
        template = random.choice(topic_templates)
        q_tmpl, a_tmpl = template
        src, dst, hrs, fare = random.choice(route_pairs)
        train = random.choice(train_names)
        q = q_tmpl.format(src=src, dst=dst, hrs=hrs, fare=fare, train=train)
        a = a_tmpl.format(src=src, dst=dst, hrs=hrs, fare=fare, train=train)
        qa_pairs.append((q, a))

    with open(os.path.join(RAW_DIR, "railway_faq.txt"), "w", encoding="utf-8") as f:
        for q, a in qa_pairs[:500]:
            f.write(f"Q: {q}\nA: {a}\n\n")
    logger.info(f"  → railway_faq.txt saved with {len(qa_pairs[:500])} Q&A pairs.")


def save_baseline_distributions(
    pnr_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    fraud_df: pd.DataFrame,
) -> None:
    logger.info("Computing and saving baseline_distributions.pkl ...")
    baseline: Dict[str, Dict] = {}

    feature_sets = {
        "m1_ticket_predictor": (pnr_df, ["days_to_departure", "initial_wl_number", "season_factor"]),
        "m3_delay_predictor":  (delay_df, ["congestion_at_station", "weather_index", "distance_from_origin_km"]),
        "m5_fare_predictor":   (pricing_df, ["days_to_departure", "base_fare", "dynamic_fare", "demand_index"]),
        "m7_fraud_detector":   (fraud_df, ["requests_per_minute", "unique_pnr_attempts", "account_age_days", "payment_attempts", "failed_captcha_count"]),
    }

    for module, (df, cols) in feature_sets.items():
        for col in cols:
            if col in df.columns:
                vals = df[col].dropna().values
                key = f"{module}.{col}"
                baseline[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "hist_counts": np.histogram(vals, bins=20)[0].tolist(),
                    "hist_edges": np.histogram(vals, bins=20)[1].tolist(),
                }

    bl_path = os.path.join(ROOT, "data", "baseline_distributions.pkl")
    with open(bl_path, "wb") as f:
        pickle.dump(baseline, f)
    logger.info(f"  → baseline_distributions.pkl saved ({len(baseline)} feature stats).")


def main():
    logger.info("=" * 60)
    logger.info(" AI Railway Governance Platform — Synthetic Data Generator")
    logger.info(f" RANDOM_SEED={RANDOM_SEED} | Output: {RAW_DIR}")
    logger.info("=" * 60)

    stations_df = generate_stations()
    trains_df = generate_trains(stations_df)
    pnr_df = generate_pnr_history(trains_df)
    pricing_df = generate_pricing_history(trains_df)
    generate_user_profiles()
    generate_congestion_logs(stations_df)
    delay_df = generate_delay_logs(trains_df, stations_df)
    fraud_df = generate_fraud_events()
    generate_irctc_policies()
    generate_faq()
    save_baseline_distributions(pnr_df, pricing_df, delay_df, fraud_df)

    logger.info("")
    logger.info("=" * 60)
    logger.info(" ✅ All synthetic data generated successfully!")
    logger.info(" Files in data/raw/:")
    for f in sorted(os.listdir(RAW_DIR)):
        size = os.path.getsize(os.path.join(RAW_DIR, f))
        logger.info(f"   {f:<35} {size/1024:.1f} KB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
