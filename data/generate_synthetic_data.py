"""
data/generate_synthetic_data.py
================================
Master Synthetic Data Generator — Golden Rule 1: 100% Self-Contained.

Generates all datasets needed for the platform to run out-of-the-box.
All random operations are seeded with RANDOM_SEED=42 for reproducibility.

Outputs (saved to data/raw/):
  - trains.csv           : 500 Indian railway routes
  - stations.csv         : 200 major Indian stations
  - pnr_history.csv      : 50,000 historical PNR/WL/RAC records
  - pricing_history.csv  : 100,000 dynamic pricing records
  - user_profiles.csv    : 5,000 user personas
  - congestion_logs.csv  : 73,000 station congestion time-series
  - delay_logs.csv       : 200,000 train delay records
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

STATION_NAMES = [
    ("NDLS", "New Delhi", "NR", "Delhi", "Metro"),
    ("CSTM", "Chhatrapati Shivaji Terminus Mumbai", "CR", "Maharashtra", "Metro"),
    ("HWH", "Howrah Junction", "ER", "West Bengal", "Metro"),
    ("MAS", "Chennai Central", "SR", "Tamil Nadu", "Metro"),
    ("SBC", "Bengaluru City Junction", "SWR", "Karnataka", "Metro"),
    ("SC", "Secunderabad Junction", "SCR", "Telangana", "Metro"),
    ("PUNE", "Pune Junction", "CR", "Maharashtra", "A1"),
    ("ADI", "Ahmedabad Junction", "WR", "Gujarat", "A1"),
    ("JP", "Jaipur Junction", "NWR", "Rajasthan", "A1"),
    ("LKO", "Lucknow Charbagh NR", "NR", "Uttar Pradesh", "A1"),
    ("CNB", "Kanpur Central", "NCR", "Uttar Pradesh", "A1"),
    ("PRYJ", "Prayagraj Junction", "NCR", "Uttar Pradesh", "A"),
    ("BSB", "Varanasi Junction", "NER", "Uttar Pradesh", "A"),
    ("PATNA", "Patna Junction", "ECR", "Bihar", "A1"),
    ("GHY", "Guwahati", "NFR", "Assam", "A"),
    ("BBS", "Bhubaneswar", "ECOR", "Odisha", "A"),
    ("NGP", "Nagpur Junction", "CR", "Maharashtra", "A"),
    ("INDB", "Indore Junction BG", "WR", "Madhya Pradesh", "A"),
    ("BPL", "Bhopal Junction", "WCR", "Madhya Pradesh", "A"),
    ("JAT", "Jammu Tawi", "NR", "J&K", "A"),
    ("AMRITSAR", "Amritsar Junction", "NR", "Punjab", "A"),
    ("LDH", "Ludhiana Junction", "NR", "Punjab", "A"),
    ("ASR", "Amritsar Junction", "NR", "Punjab", "A"),
    ("UDZ", "Udaipur City", "NWR", "Rajasthan", "B"),
    ("JU", "Jodhpur Junction", "NWR", "Rajasthan", "A"),
    ("AII", "Ajmer Junction", "NWR", "Rajasthan", "A"),
    ("KOTA", "Kota Junction", "WCR", "Rajasthan", "A"),
    ("RTM", "Ratlam Junction", "WR", "Madhya Pradesh", "A"),
    ("BCT", "Mumbai Central", "WR", "Maharashtra", "Metro"),
    ("VSKP", "Visakhapatnam Junction", "ECOR", "Andhra Pradesh", "A"),
    ("GNT", "Guntur Junction", "SCR", "Andhra Pradesh", "B"),
    ("TVM", "Thiruvananthapuram Central", "SR", "Kerala", "A"),
    ("ERS", "Ernakulam Junction", "SR", "Kerala", "A"),
    ("CBE", "Coimbatore Junction", "SR", "Tamil Nadu", "A"),
    ("MDU", "Madurai Junction", "SR", "Tamil Nadu", "A"),
    ("TPJ", "Tiruchirappalli Junction", "SR", "Tamil Nadu", "A"),
    ("SA", "Salem Junction", "SR", "Tamil Nadu", "B"),
    ("RPM", "Perambur", "SR", "Tamil Nadu", "B"),
    ("MS", "Chennai Egmore", "SR", "Tamil Nadu", "A"),
    ("DLI", "Delhi Junction", "NR", "Delhi", "A"),
    ("GZB", "Ghaziabad Junction", "NR", "Uttar Pradesh", "B"),
    ("AGC", "Agra Cantt", "NCR", "Uttar Pradesh", "A"),
    ("MTJ", "Mathura Junction", "NCR", "Uttar Pradesh", "A"),
    ("AF", "Agra Fort", "NCR", "Uttar Pradesh", "B"),
    ("GWL", "Gwalior Junction", "NCR", "Madhya Pradesh", "A"),
    ("JHS", "Jhansi Junction", "NCR", "Uttar Pradesh", "A"),
    ("STA", "Satna", "WCR", "Madhya Pradesh", "B"),
    ("KGKP", "Kathgodam", "NER", "Uttarakhand", "B"),
    ("DDN", "Dehradun", "NR", "Uttarakhand", "A"),
    ("HW", "Haridwar Junction", "NR", "Uttarakhand", "A"),
    ("RKSH", "Rishikesh", "NR", "Uttarakhand", "C"),
    ("UHL", "Una Himachal", "NR", "Himachal Pradesh", "C"),
    ("SHM", "Shimla", "NR", "Himachal Pradesh", "B"),
    ("KLK", "Kalka", "NR", "Haryana", "B"),
    ("UMB", "Ambala Cant Junction", "NR", "Haryana", "A"),
    ("ROK", "Rohtak Junction", "NR", "Haryana", "B"),
    ("RE", "Rewari Junction", "NR", "Haryana", "B"),
    ("GGN", "Gurgaon", "NR", "Haryana", "B"),
    ("FDKD", "Faridabad", "NR", "Haryana", "B"),
    ("PANIPAT", "Panipat Junction", "NR", "Haryana", "B"),
    ("KRK", "Kurukshetra Junction", "NR", "Haryana", "B"),
    ("AMBALA", "Ambala City", "NR", "Haryana", "B"),
    ("BRC", "Vadodara Junction", "WR", "Gujarat", "A"),
    ("ST", "Surat", "WR", "Gujarat", "A"),
    ("ANND", "Anand Junction", "WR", "Gujarat", "B"),
    ("NKDA", "Navsari", "WR", "Gujarat", "C"),
    ("JT", "Jetpur Junction", "WR", "Gujarat", "B"),
    ("RJT", "Rajkot Junction", "WR", "Gujarat", "A"),
    ("BVP", "Bhavnagar Terminus", "WR", "Gujarat", "B"),
    ("JAM", "Jamnagar", "WR", "Gujarat", "B"),
    ("OKHA", "Okha", "WR", "Gujarat", "C"),
    ("VAPI", "Vapi", "WR", "Gujarat", "B"),
    ("DDAR", "Dahanu Road", "WR", "Maharashtra", "C"),
    ("KYN", "Kalyan Junction", "CR", "Maharashtra", "A"),
    ("LNL", "Lonavala", "CR", "Maharashtra", "B"),
    ("DD", "Dadar Central", "CR", "Maharashtra", "A"),
    ("DR", "Mumbai Dadar WR", "WR", "Maharashtra", "A"),
    ("BSL", "Bhusaval Junction", "CR", "Maharashtra", "A"),
    ("AWB", "Aurangabad", "CR", "Maharashtra", "A"),
    ("SOT", "Sholapur Junction", "CR", "Maharashtra", "A"),
    ("WD", "Wadi Junction", "SCR", "Karnataka", "B"),
    ("UBL", "Hubli Junction", "SWR", "Karnataka", "A"),
    ("MYS", "Mysuru Junction", "SWR", "Karnataka", "A"),
    ("BANG", "Yesvantpur Junction", "SWR", "Karnataka", "A"),
    ("DWR", "Dharwad", "SWR", "Karnataka", "B"),
    ("GDG", "Gadag Junction", "SWR", "Karnataka", "B"),
    ("BJP", "Vijayapura", "SWR", "Karnataka", "B"),
    ("GR", "Gulbarga", "SCR", "Karnataka", "B"),
    ("BIDAR", "Bidar", "SCR", "Karnataka", "C"),
    ("NED", "Nanded", "SCR", "Maharashtra", "B"),
    ("PBN", "Parbhani Junction", "SCR", "Maharashtra", "B"),
    ("MUE", "Mudkhed Junction", "SCR", "Maharashtra", "C"),
    ("HYB", "Hyderabad Deccan", "SCR", "Telangana", "A"),
    ("KCG", "Kacheguda", "SCR", "Telangana", "A"),
    ("FM", "Faizabad Junction", "NER", "Uttar Pradesh", "B"),
    ("GKP", "Gorakhpur Junction", "NER", "Uttar Pradesh", "A"),
    ("NJP", "New Jalpaiguri", "NFR", "West Bengal", "A"),
    ("MLDT", "Malda Town", "ER", "West Bengal", "B"),
    ("BWN", "Bardhaman Junction", "ER", "West Bengal", "B"),
    ("SDAH", "Sealdah", "ER", "West Bengal", "Metro"),
    ("BGP", "Bhagalpur Junction", "ECR", "Bihar", "A"),
    ("RJPB", "Rajendra Nagar Bihar", "ECR", "Bihar", "B"),
    ("DNKD", "Danapur", "ECR", "Bihar", "A"),
    ("MFP", "Muzaffarpur Junction", "ECR", "Bihar", "A"),
    ("SPJ", "Samastipur Junction", "ECR", "Bihar", "B"),
    ("RXL", "Raxaul Junction", "ECR", "Bihar", "B"),
    ("DBG", "Darbhanga Junction", "ECR", "Bihar", "B"),
    ("JYG", "Jaynagar", "ECR", "Bihar", "C"),
    ("SHC", "Sitamarhi", "ECR", "Bihar", "C"),
    ("SGR", "Siliguri Junction", "NFR", "West Bengal", "B"),
    ("AGTL", "Agartala", "NFR", "Tripura", "B"),
    ("LMG", "Lumding Junction", "NFR", "Assam", "B"),
    ("DBRG", "Dibrugarh", "NFR", "Assam", "A"),
    ("TGM", "Tangla", "NFR", "Assam", "C"),
    ("RNG", "Rangiya Junction", "NFR", "Assam", "B"),
    ("GHY", "Guwahati", "NFR", "Assam", "A"),
    ("JTTN", "Jiribam", "NFR", "Manipur", "C"),
    ("TPTY", "Tirupati", "SCR", "Andhra Pradesh", "A"),
    ("GTL", "Guntakal Junction", "SCR", "Andhra Pradesh", "A"),
    ("RU", "Renigunta Junction", "SCR", "Andhra Pradesh", "B"),
    ("QLN", "Kollam Junction", "SR", "Kerala", "A"),
    ("ALLP", "Alappuzha", "SR", "Kerala", "B"),
    ("TCR", "Thrissur", "SR", "Kerala", "A"),
    ("CLT", "Kozhikode", "SR", "Kerala", "A"),
    ("SRR", "Shoranur Junction", "SR", "Kerala", "B"),
    ("CAN", "Kannur", "SR", "Kerala", "B"),
    ("MAJN", "Mangaluru Junction", "SR", "Karnataka", "A"),
    ("UDZ", "Udupi", "SR", "Karnataka", "B"),
    ("MAQ", "Mangaluru Central", "SR", "Karnataka", "A"),
    ("KUDA", "Kundapura", "SR", "Karnataka", "C"),
    ("BKLN", "Bhatkal", "SR", "Karnataka", "C"),
    ("GOA", "Madgaon Junction", "KR", "Goa", "A"),
    ("PANAJI", "Karmali", "KR", "Goa", "B"),
    ("THVM", "Thivim", "KR", "Goa", "B"),
    ("RN", "Ratnagiri", "KR", "Maharashtra", "B"),
    ("CHI", "Chiplun", "KR", "Maharashtra", "B"),
    ("KED", "Khed", "KR", "Maharashtra", "C"),
    ("RWP", "Roha", "KR", "Maharashtra", "C"),
    ("PEN", "Pen", "CR", "Maharashtra", "C"),
    ("PANVEL", "Panvel Junction", "CR", "Maharashtra", "A"),
    ("VR", "Vasai Road", "WR", "Maharashtra", "B"),
    ("BVI", "Borivali", "WR", "Maharashtra", "B"),
    ("ANDHERI", "Andheri", "WR", "Maharashtra", "B"),
    ("PNVL", "Panvel", "CR", "Maharashtra", "A"),
    ("TL", "Titwala", "CR", "Maharashtra", "C"),
    ("AMBV", "Ambivli", "CR", "Maharashtra", "C"),
    ("BR", "Badlapur", "CR", "Maharashtra", "C"),
    ("VB", "Vangani", "CR", "Maharashtra", "C"),
    ("SRD", "Shahapur", "CR", "Maharashtra", "C"),
    ("KJT", "Khopoli", "CR", "Maharashtra", "C"),
    ("KYN", "Kalyan Junction", "CR", "Maharashtra", "A"),
    ("THANE", "Thane", "CR", "Maharashtra", "A"),
    ("CSMT", "Mumbai CST", "CR", "Maharashtra", "Metro"),
    ("PUNE", "Pune Junction", "CR", "Maharashtra", "A1"),
    ("KOLHAPUR", "Kolhapur", "CR", "Maharashtra", "B"),
    ("SNG", "Sangli", "CR", "Maharashtra", "B"),
    ("MRJ", "Miraj Junction", "CR", "Maharashtra", "B"),
    ("KOP", "Kopargaon", "CR", "Maharashtra", "C"),
    ("DD", "Dhond Junction", "CR", "Maharashtra", "B"),
    ("DD", "Daund Junction", "CR", "Maharashtra", "B"),
    ("KWV", "Kurduvadi Junction", "CR", "Maharashtra", "B"),
    ("WD", "Wadi Junction", "SCR", "Karnataka", "B"),
    ("GTL", "Guntakal Junction", "SCR", "Andhra Pradesh", "A"),
    ("VSKP", "Visakhapatnam", "ECOR", "Andhra Pradesh", "A"),
    ("SHM", "Shalimar", "ER", "West Bengal", "A"),
    ("KHURDA", "Khurda Road", "ECOR", "Odisha", "B"),
    ("CTC", "Cuttack Junction", "ECOR", "Odisha", "A"),
    ("RGDA", "Rayagada", "ECOR", "Odisha", "B"),
    ("TIG", "Titlagarh Junction", "ECOR", "Odisha", "B"),
    ("ROU", "Rourkela Junction", "SER", "Odisha", "A"),
    ("JSG", "Jharsuguda Junction", "SER", "Odisha", "B"),
    ("IB", "Ib", "SER", "Odisha", "C"),
    ("BSPB", "Bilaspur Junction", "SECR", "Chhattisgarh", "A"),
    ("R", "Raipur Junction", "SECR", "Chhattisgarh", "A"),
    ("DURG", "Durg Junction", "SECR", "Chhattisgarh", "A"),
    ("RIG", "Rajnandgaon", "SECR", "Chhattisgarh", "B"),
    ("GGP", "Gondia Junction", "SECR", "Maharashtra", "B"),
    ("WADI", "Wadi", "SCR", "Karnataka", "B"),
    ("RGA", "Raichur", "SCR", "Karnataka", "B"),
    ("DMM", "Dharmavaram Junction", "SCR", "Andhra Pradesh", "B"),
    ("AK", "Akola Junction", "CR", "Maharashtra", "A"),
    ("WR", "Wardha Junction", "CR", "Maharashtra", "B"),
    ("SWV", "Sawantwadi Road", "KR", "Maharashtra", "C"),
    ("KUDAL", "Kudal", "KR", "Maharashtra", "C"),
    ("KADAM", "Kadamba", "KR", "Maharashtra", "C"),
][:200]

TRAIN_TYPES = ["Rajdhani Express", "Shatabdi Express", "Duronto Express", "Superfast Express",
               "Express", "Mail", "Garib Rath", "Jan Shatabdi", "Humsafar Express",
               "Tejas Express", "Vande Bharat Express", "Intercity Express", "Passenger"]

TRAIN_ZONES = ["NR", "CR", "WR", "SR", "ER", "NER", "NFR", "SCR", "SER", "SWR", "ECR", "ECOR", "NWR", "WCR", "NCR", "SECR"]

CLASSES = ["SL", "3A", "2A", "1A", "CC", "EC", "2S"]
QUOTAS = ["GN", "TQ", "LADIES", "DEFENCE", "SENIOR", "HO", "DP"]
BERTH_TYPES = ["Lower", "Middle", "Upper", "Side Lower", "Side Upper", "RAC"]
FINAL_STATUSES = ["Confirmed", "RAC", "WL"]
PERSONAS = ["budget", "premium", "student", "senior", "business", "family"]
REASON_CODES = ["HEAVY_TRAFFIC", "SIGNAL_FAILURE", "ENGINE_FAILURE", "WEATHER", "TRACK_MAINTENANCE",
                "CREW_DELAY", "GATE_OBSTRUCTION", "FLOOD", "FOG", "PASSENGER_EMERGENCY", "ON_TIME"]
FRAUD_LABELS = ["legitimate", "suspicious", "bot"]


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


# ── Generator Functions ────────────────────────────────────────────────────────

def generate_stations() -> pd.DataFrame:
    logger.info("Generating stations.csv ...")
    rows = []
    for i, (code, name, zone, state, tier) in enumerate(STATION_NAMES[:200]):
        baseline_congestion = {"Metro": 70, "A1": 55, "A": 45, "B": 30, "C": 20}.get(tier, 35)
        rows.append({
            "station_code": code,
            "station_name": name,
            "zone": zone,
            "state": state,
            "tier": tier,
            "baseline_congestion": baseline_congestion + np.random.randint(-5, 5),
            "platforms": {"Metro": 18, "A1": 12, "A": 8, "B": 5, "C": 3}.get(tier, 4),
            "latitude": round(random.uniform(8.5, 37.0), 4),
            "longitude": round(random.uniform(68.0, 97.0), 4),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["station_code"]).reset_index(drop=True)
    df.to_csv(os.path.join(RAW_DIR, "stations.csv"), index=False)
    logger.info(f"  → {len(df)} stations saved.")
    return df


def generate_trains(stations_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating trains.csv ...")
    rows = []
    station_codes = stations_df["station_code"].tolist()

    major_routes = [
        ("NDLS", "CSTM", "12951", "Mumbai Rajdhani Express", 1385, 16.2, ["1A","2A","3A"], 88),
        ("NDLS", "MAS", "12433", "Chennai Rajdhani Express", 2180, 28.5, ["1A","2A","3A"], 84),
        ("NDLS", "HWH", "12301", "Howrah Rajdhani Express", 1441, 17.5, ["1A","2A","3A"], 86),
        ("NDLS", "SBC", "12627", "Karnataka Express", 3100, 44.0, ["SL","3A","2A","1A"], 72),
        ("NDLS", "JP", "12015", "Ajmer Shatabdi Express", 441, 5.0, ["CC","EC"], 91),
        ("NDLS", "LKO", "12229", "Lucknow Mail", 497, 6.5, ["SL","3A","2A","1A"], 78),
        ("NDLS", "PATNA", "12561", "Patna Sampark Kranti", 992, 12.0, ["SL","3A","2A"], 75),
        ("CSTM", "HWH", "12809", "Mumbai-Howrah Mail", 2013, 31.5, ["SL","3A","2A","1A"], 70),
        ("CSTM", "MAS", "11041", "Mumbai-Chennai Express", 1279, 20.5, ["SL","3A","2A"], 74),
        ("CSTM", "SBC", "11301", "Udyan Express", 1160, 19.0, ["SL","3A","2A"], 68),
        ("MAS", "HWH", "12841", "Coromandel Express", 1662, 26.5, ["SL","3A","2A","1A"], 82),
        ("MAS", "SBC", "12007", "Chennai-Mysuru Shatabdi", 497, 6.0, ["CC","EC"], 89),
        ("HWH", "NJP", "12343", "Darjeeling Mail", 600, 9.0, ["SL","3A","2A","1A"], 76),
        ("NDLS", "ADI", "12957", "Ahmedabad Rajdhani", 942, 13.0, ["1A","2A","3A"], 85),
        ("NDLS", "GHY", "12423", "Dibrugarh Rajdhani", 2423, 35.0, ["1A","2A","3A"], 80),
        ("NDLS", "TVM", "12625", "Kerala Express", 3054, 47.0, ["SL","3A","2A","1A"], 69),
        ("NDLS", "BSB", "13005", "Amritsar-Howrah Mail", 900, 14.0, ["SL","3A","2A"], 71),
        ("PUNE", "NDLS", "12123", "Deccan Queen Express", 1545, 25.0, ["SL","3A","2A","CC"], 77),
        ("ADI", "MAS", "12163", "Dadar-Chennai Express", 1856, 29.0, ["SL","3A","2A"], 73),
        ("SBC", "MAS", "12658", "Bengaluru-Chennai Express", 362, 5.5, ["SL","3A","CC"], 83),
    ]

    for src, dst, tno, tname, dist, dur, classes, punct in major_routes:
        base_sl = max(200, int(dist * 0.35 + random.randint(-20, 20)))
        base_3a = max(600, int(dist * 0.70 + random.randint(-50, 50)))
        base_2a = max(900, int(dist * 1.0 + random.randint(-80, 80)))
        base_1a = max(1500, int(dist * 1.6 + random.randint(-100, 100)))
        base_cc = max(400, int(dist * 0.45 + random.randint(-30, 30)))

        rows.append({
            "train_number": tno,
            "train_name": tname,
            "source": src,
            "destination": dst,
            "distance_km": dist,
            "duration_hours": dur,
            "avg_speed_kmph": round(dist / dur, 1),
            "classes": "|".join(classes),
            "quota_types": "GN|TQ|LADIES|DEFENCE|SENIOR",
            "departure_time": f"{random.randint(0,23):02d}:{random.choice(['00','15','30','45'])}",
            "arrival_time": f"{random.randint(0,23):02d}:{random.choice(['00','15','30','45'])}",
            "punctuality_pct": punct + random.randint(-5, 5),
            "base_fare_sl": base_sl if "SL" in classes else 0,
            "base_fare_3a": base_3a if "3A" in classes else 0,
            "base_fare_2a": base_2a if "2A" in classes else 0,
            "base_fare_1a": base_1a if "1A" in classes else 0,
            "base_fare_cc": base_cc if "CC" in classes else 0,
            "total_seats_sl": random.randint(400, 1200) if "SL" in classes else 0,
            "total_seats_3a": random.randint(100, 400) if "3A" in classes else 0,
            "zone": random.choice(TRAIN_ZONES),
            "train_type": tname.split()[-2] + " " + tname.split()[-1] if len(tname.split()) > 2 else "Express",
        })

    # Generate remaining random trains
    while len(rows) < 500:
        src_idx = random.randint(0, min(50, len(station_codes)-1))
        dst_idx = random.randint(0, min(50, len(station_codes)-1))
        if src_idx == dst_idx:
            continue
        src = station_codes[src_idx]
        dst = station_codes[dst_idx]
        dist = random.randint(100, 3500)
        speed = random.uniform(50, 100)
        dur = round(dist / speed, 1)
        classes = random.sample(CLASSES, random.randint(2, 4))
        base_sl = max(150, int(dist * 0.35))
        ttype = random.choice(TRAIN_TYPES)
        punct = random.randint(60, 95)

        rows.append({
            "train_number": str(10000 + len(rows)),
            "train_name": f"{src}-{dst} {ttype}",
            "source": src,
            "destination": dst,
            "distance_km": dist,
            "duration_hours": dur,
            "avg_speed_kmph": round(speed, 1),
            "classes": "|".join(classes),
            "quota_types": "GN|TQ" + ("|LADIES" if random.random() > 0.5 else ""),
            "departure_time": f"{random.randint(0,23):02d}:{random.choice(['00','15','30','45'])}",
            "arrival_time": f"{random.randint(0,23):02d}:{random.choice(['00','15','30','45'])}",
            "punctuality_pct": punct,
            "base_fare_sl": base_sl if "SL" in classes else 0,
            "base_fare_3a": max(500, int(base_sl * 2.0)) if "3A" in classes else 0,
            "base_fare_2a": max(800, int(base_sl * 3.0)) if "2A" in classes else 0,
            "base_fare_1a": max(1200, int(base_sl * 5.0)) if "1A" in classes else 0,
            "base_fare_cc": max(300, int(base_sl * 1.2)) if "CC" in classes else 0,
            "total_seats_sl": random.randint(200, 1200) if "SL" in classes else 0,
            "total_seats_3a": random.randint(60, 400) if "3A" in classes else 0,
            "zone": random.choice(TRAIN_ZONES),
            "train_type": ttype,
        })

    df = pd.DataFrame(rows[:500])
    df.to_csv(os.path.join(RAW_DIR, "trains.csv"), index=False)
    logger.info(f"  → {len(df)} trains saved.")
    return df


def generate_pnr_history(trains_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating pnr_history.csv (50,000 records) ...")
    rows = []
    train_numbers = trains_df["train_number"].tolist()
    start_date = date(2023, 1, 1)
    end_date = date(2026, 10, 1)
    total_days = (end_date - start_date).days

    for i in tqdm(range(50000), desc="PNR records"):
        journey_date = start_date + timedelta(days=random.randint(0, total_days))
        booking_days_before = random.randint(1, 120)
        booking_date = journey_date - timedelta(days=booking_days_before)
        if booking_date < start_date:
            booking_date = start_date

        train = random.choice(train_numbers)
        cls = random.choices(CLASSES, weights=[35, 25, 15, 5, 10, 3, 7])[0]
        quota = random.choices(QUOTAS, weights=[70, 10, 5, 5, 5, 3, 2])[0]
        initial_wl = random.randint(1, 150)

        # WL clearance probability: depends on class, quota, days before
        # SL clears more (65%), 3A (50%), 2A (42%), 1A (38%)
        clearance_probs = {"SL": 0.65, "3A": 0.50, "2A": 0.42, "1A": 0.38, "CC": 0.55, "EC": 0.45, "2S": 0.70}
        base_prob = clearance_probs.get(cls, 0.50)

        # Quota adjustments
        quota_adj = {"TQ": 0.85, "GN": 1.0, "LADIES": 1.1, "DEFENCE": 1.05, "SENIOR": 1.0}
        base_prob *= quota_adj.get(quota, 1.0)

        # WL number adjustments
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

        # Season adjustment
        sf = season_factor(journey_date)
        base_prob /= sf  # Higher demand → lower clearance

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

        seniority_flag = random.random() < 0.08
        disability_flag = random.random() < 0.03

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
            "seniority_flag": int(seniority_flag),
            "disability_flag": int(disability_flag),
            "season_factor": round(sf, 2),
            "cancellation_date": (journey_date - timedelta(days=random.randint(1, 10))).isoformat()
                if final_status == "WL" and random.random() < 0.4 else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "pnr_history.csv"), index=False)
    logger.info(f"  → {len(df)} PNR records saved.")
    return df


def generate_pricing_history(trains_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating pricing_history.csv (100,000 records) ...")
    rows = []
    train_numbers = trains_df["train_number"].tolist()
    start_date = date(2024, 1, 1)
    end_date = date(2026, 10, 1)
    total_days = (end_date - start_date).days
    trains_lookup = trains_df.set_index("train_number").to_dict("index")

    for i in tqdm(range(100000), desc="Pricing records"):
        train = random.choice(train_numbers)
        t_info = trains_lookup.get(train, {})
        journey_date = start_date + timedelta(days=random.randint(0, total_days))

        # Select class
        cls_list = [c for c in (t_info.get("classes", "SL|3A") or "SL|3A").split("|") if c]
        cls = random.choice(cls_list) if cls_list else "SL"

        fare_col = f"base_fare_{cls.lower().replace('a','a')}"
        base_fare = t_info.get(fare_col, 400)
        if not base_fare or base_fare == 0:
            base_fare = {"SL": 350, "3A": 950, "2A": 1400, "1A": 2200, "CC": 500, "EC": 1800, "2S": 150}.get(cls, 400)

        days_to_dep = random.randint(0, 120)
        sf = season_factor(journey_date)

        # Demand: higher demand nearer departure for confirmed trains
        demand_index = round(np.clip(
            1.0 + (0.5 * (1 - days_to_dep / 120)) * sf + 0.2 * np.random.randn(),
            0.5, 3.0
        ), 3)

        # Tatkal surcharge window
        tatkal_flag = days_to_dep <= 1
        tatkal_surcharge = 0
        if tatkal_flag and cls in ("SL", "3A", "2A", "1A"):
            tatkal_surcharge = {"SL": 100, "3A": 200, "2A": 300, "1A": 400}.get(cls, 100)

        dynamic_fare = round(base_fare * demand_index + tatkal_surcharge, 2)
        dynamic_fare = max(base_fare * 0.9, min(base_fare * 3.5, dynamic_fare))

        rows.append({
            "record_id": i + 1,
            "train_number": train,
            "journey_class": cls,
            "journey_date": journey_date.isoformat(),
            "booking_date": (journey_date - timedelta(days=days_to_dep)).isoformat(),
            "days_to_departure": days_to_dep,
            "base_fare": base_fare,
            "dynamic_fare": dynamic_fare,
            "demand_index": demand_index,
            "tatkal_flag": int(tatkal_flag),
            "tatkal_surcharge": tatkal_surcharge,
            "season_factor": round(sf, 2),
            "route_popularity": round(random.uniform(0.3, 1.0), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "pricing_history.csv"), index=False)
    logger.info(f"  → {len(df)} pricing records saved.")
    return df


def generate_user_profiles() -> pd.DataFrame:
    logger.info("Generating user_profiles.csv (5,000 records) ...")
    rows = []
    persona_weights = [0.30, 0.15, 0.25, 0.10, 0.12, 0.08]
    station_codes = [s[0] for s in STATION_NAMES[:50]]

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

        n_routes = random.randint(1, 4)
        frequent_routes = [f"{random.choice(station_codes)}-{random.choice(station_codes)}" for _ in range(n_routes)]

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
            "avg_advance_booking_days": random.randint(5, 90),
            "loyalty_score": round(random.uniform(0, 100), 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "user_profiles.csv"), index=False)
    logger.info(f"  → {len(df)} user profiles saved.")
    return df


def generate_congestion_logs(stations_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating congestion_logs.csv (~73,000 records) ...")
    rows = []
    station_sample = stations_df.sample(min(200, len(stations_df)), random_state=RANDOM_SEED)
    start_date = date(2025, 1, 1)
    end_date = date(2026, 10, 1)
    current = start_date

    while current <= end_date:
        sf = season_factor(current)
        hour = random.choice([7, 10, 13, 16, 19, 22])  # Sample 1 record per station per day at key hours
        for _, row in station_sample.sample(min(50, len(station_sample)), random_state=random.randint(0,999)).iterrows():
            baseline = row["baseline_congestion"]
            hour_factor = {7: 1.3, 10: 1.5, 13: 1.1, 16: 1.4, 19: 1.6, 22: 0.7}.get(hour, 1.0)
            congestion = np.clip(
                baseline * sf * hour_factor + np.random.normal(0, 8),
                0, 100
            )
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
    logger.info("Generating delay_logs.csv (200,000 records) ...")
    rows = []
    train_numbers = trains_df["train_number"].tolist()
    station_codes = stations_df["station_code"].tolist()
    start_date = date(2024, 1, 1)
    end_date = date(2026, 10, 1)
    total_days = (end_date - start_date).days

    for i in tqdm(range(200000), desc="Delay records"):
        journey_date = start_date + timedelta(days=random.randint(0, total_days))
        sf = season_factor(journey_date)
        month = journey_date.month

        # Causal structure (so the model can actually learn it):
        #   season base delay  +  weather effect  +  station congestion effect  +  distance effect  + noise
        if month in (6, 7, 8, 9):  # Monsoon: rain / flooding
            weather = float(np.clip(np.random.normal(6.0, 2.2), 0, 10))
            base_delay = np.random.exponential(12)
            reason = random.choices(REASON_CODES, weights=[20, 5, 3, 30, 5, 5, 5, 20, 2, 3, 2])[0]
        elif month in (12, 1):  # Winter fog
            weather = float(np.clip(np.random.normal(5.5, 2.5), 0, 10))
            base_delay = np.random.exponential(10)
            reason = random.choices(REASON_CODES, weights=[15, 5, 3, 10, 5, 5, 3, 2, 40, 3, 9])[0]
        else:
            weather = float(np.clip(np.random.normal(2.5, 2.0), 0, 10))
            base_delay = np.random.exponential(6)
            reason = random.choices(REASON_CODES, weights=[15, 8, 4, 5, 10, 8, 8, 2, 2, 4, 34])[0]

        congestion = float(np.random.uniform(10, 90) * sf)
        distance = random.randint(0, 2000)
        weather_effect = 0.45 * weather ** 1.6           # 0 → ~18 min at 10/10
        congestion_effect = 0.22 * max(0.0, congestion - 30)   # crowded platforms hold trains
        distance_effect = distance / 120.0               # delay accumulates along the route
        delay = base_delay * sf + weather_effect + congestion_effect + distance_effect + np.random.normal(0, 4)
        if reason == "ON_TIME" and weather < 4 and congestion < 60:
            delay = max(0, np.random.normal(2, 3))
        delay = max(0.0, delay)

        scheduled_hour = random.randint(0, 23)
        scheduled_min = random.choice([0, 15, 30, 45])
        scheduled_arr = datetime.combine(journey_date, __import__("datetime").time(scheduled_hour, scheduled_min))
        actual_arr = scheduled_arr + timedelta(minutes=delay)

        rows.append({
            "log_id": i + 1,
            "train_number": random.choice(train_numbers),
            "station_code": random.choice(station_codes),
            "journey_date": journey_date.isoformat(),
            "scheduled_arrival": scheduled_arr.strftime("%H:%M"),
            "actual_arrival": actual_arr.strftime("%H:%M"),
            "delay_minutes": round(delay, 1),
            "reason_code": reason,
            "weather_index": round(weather, 1),
            "congestion_at_station": round(congestion, 1),
            "distance_from_origin_km": distance,
            "month": month,
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
        ("मेरा PNR स्टेटस कैसे चेक करें?", "आप IRCTC वेबसाइट पर 'PNR Status' में, SMS से 139 पर PNR नंबर भेजकर, या 14646 हेल्पलाइन पर कॉल करके PNR स्टेटस चेक कर सकते हैं।"),
        ("Tatkal ticket book karna kab shuru hota hai?", "AC classes ke liye Tatkal booking subah 10 baje se shuru hoti hai. SL aur 2S classes ke liye 11 baje se. Yeh journey se ek din pehle hota hai."),
        ("WL ticket cancel karne par refund milega?", "Agar aapka WL ticket charting tak confirm nahi hota, to pura refund automatic mil jata hai. Agar ticket confirm tha aur aapne cancel kiya, to normal cancellation charges lagte hain."),
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
