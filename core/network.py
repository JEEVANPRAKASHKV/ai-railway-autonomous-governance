"""
core/network.py
===============
The real Indian Railways network for search: which trains halt where, in what order, and how far apart.

Built on data/raw/trains.csv + stations.csv (real trains and stations, generated from data/real/) and
data/real/train_stops.csv.gz (every halt of every train with real distance and timings). Any two halts of the
same train form a bookable journey, so a search for Kota → Surat finds the Rajdhani that runs Mumbai → Delhi.

`segments(src, dst)` returns one row per train serving the pair, shaped like a trains.csv row but for that
journey: departure/arrival at the two stations, the segment's duration and distance, and official TAG 2025
fares for the segment's distance (core/fares.py).

Limitations: the timetable is a 2016 snapshot and has no running days, so every train is treated as daily.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import threading
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from core.config import settings
from core.fares import payable_fare

logger = logging.getLogger(__name__)

FARE_CLASSES = ["SL", "3A", "2A", "1A", "CC", "EC", "2S"]
TIMETABLE_SOURCE = "Indian Railways timetable (datameet/railways, CC0) — 2016 snapshot"

# Stations that serve the same city: a search from one also looks at the others
CITY_GROUPS: Dict[str, Set[str]] = {
    "Delhi": {"NDLS", "DLI", "NZM", "DEE", "ANVT", "DEC"},
    "Mumbai": {"CSTM", "BCT", "LTT", "DR", "BDTS", "TNA"},
    "Kolkata": {"HWH", "SDAH", "KOAA", "SHM"},
    "Chennai": {"MAS", "MS", "TBM", "MSB"},
    "Bengaluru": {"SBC", "YPR", "BNC", "KJM", "BNCE"},
    "Hyderabad": {"SC", "HYB", "KCG"},
    "Pune": {"PUNE"},
    "Ahmedabad": {"ADI", "SBI"},
    "Lucknow": {"LKO", "LJN"},
    "Patna": {"PNBE", "RJPB", "PNC"},
    "Jaipur": {"JP", "GADJ"},
    "Kanpur": {"CNB"},
    "Guwahati": {"GHY", "KYQ"},
    "Thiruvananthapuram": {"TVC", "KCVL"},
    "Varanasi": {"BSB", "BSBS", "MUV"},
}
_CITY_OF = {code: city for city, codes in CITY_GROUPS.items() for code in codes}
# City and older / common names → a main station code
CITY_ALIASES: Dict[str, str] = {
    "delhi": "NDLS", "new delhi": "NDLS", "mumbai": "CSTM", "bombay": "CSTM", "kolkata": "HWH", "calcutta": "HWH",
    "chennai": "MAS", "madras": "MAS", "bangalore": "SBC", "bengaluru": "SBC", "hyderabad": "SC",
    "secunderabad": "SC", "pune": "PUNE", "poona": "PUNE", "ahmedabad": "ADI", "jaipur": "JP", "lucknow": "LKO",
    "patna": "PNBE", "kanpur": "CNB", "varanasi": "BSB", "benares": "BSB", "banaras": "BSB", "prayagraj": "ALD",
    "allahabad": "ALD", "agra": "AGC", "bhopal": "BPL", "indore": "INDB", "nagpur": "NGP", "surat": "ST",
    "vadodara": "BRC", "baroda": "BRC", "rajkot": "RJT", "amritsar": "ASR", "chandigarh": "CDG", "ludhiana": "LDH",
    "jammu": "JAT", "guwahati": "GHY", "bhubaneswar": "BBS", "puri": "PURI", "ranchi": "RNC", "jamshedpur": "TATA",
    "visakhapatnam": "VSKP", "vizag": "VSKP", "vijayawada": "BZA", "tirupati": "TPTY", "cochin": "ERS", "kochi": "ERS",
    "ernakulam": "ERS", "calicut": "CLT", "kozhikode": "CLT", "trivandrum": "TVC", "thiruvananthapuram": "TVC",
    "kannur": "CAN", "cannanore": "CAN", "trichur": "TCR", "thrissur": "TCR", "palakkad": "PGT", "palghat": "PGT",
    "kollam": "QLN", "quilon": "QLN", "mangalore": "MAQ", "mangaluru": "MAQ", "goa": "MAO", "madgaon": "MAO",
    "margao": "MAO", "mysore": "MYS", "mysuru": "MYS", "hubli": "UBL", "hubballi": "UBL", "belgaum": "BGM",
    "belagavi": "BGM", "coimbatore": "CBE", "madurai": "MDU", "trichy": "TPJ", "tiruchirappalli": "TPJ",
    "salem": "SA", "erode": "ED", "tirunelveli": "TEN", "kanyakumari": "CAPE", "rameswaram": "RMM",
    "pondicherry": "PDY", "puducherry": "PDY", "raipur": "R", "bilaspur": "BSP", "jabalpur": "JBP", "gwalior": "GWL",
    "jhansi": "JHS", "udaipur": "UDZ", "jodhpur": "JU", "ajmer": "AII", "kota": "KOTA", "bikaner": "BKN",
    "jaisalmer": "JSM", "dehradun": "DDN", "haridwar": "HW", "gorakhpur": "GKP", "darbhanga": "DBG",
    "gaya": "GAYA", "dhanbad": "DHN", "asansol": "ASN", "siliguri": "NJP", "new jalpaiguri": "NJP",
    "dibrugarh": "DBRG", "gurgaon": "GGN", "gurugram": "GGN", "warangal": "WL", "cuttack": "CTC",
    "alappuzha": "ALLP", "pondy": "PDY", "nizamuddin": "NZM", "hazrat nizamuddin": "NZM", "anand vihar": "ANVT",
    "kurla": "LTT", "lokmanya tilak": "LTT", "ltt": "LTT", "bandra": "BDTS", "bandra terminus": "BDTS",
    "dadar": "DR", "sealdah": "SDAH", "shalimar": "SHM", "chitpur": "KOAA", "yesvantpur": "YPR", "yeshwantpur": "YPR",
    "kacheguda": "KCG", "egmore": "MS", "chennai egmore": "MS", "csmt": "CSTM", "cst": "CSTM", "vt": "CSTM",
}
_SUFFIX = {"jn", "jct", "junction", "central", "cent", "city", "cantt", "cantonment", "terminus", "terminal",
           "t", "road", "rd", "halt", "main", "town"}
HUB_MIN_TRAINS = 60          # stations with at least this many trains halting are interchange candidates


def _hhmm(t) -> int:
    try:
        h, m = str(t).split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return -1


class RailNetwork:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self.trains = pd.DataFrame()
        self.stations = pd.DataFrame()
        self.stops = pd.DataFrame()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            raw = settings.data_raw_dir
            try:
                self.trains = pd.read_csv(os.path.join(raw, "trains.csv"), dtype={"train_number": str})
                self.stations = pd.read_csv(os.path.join(raw, "stations.csv"))
                self.stops = pd.read_csv(os.path.join(settings.project_root, "data", "real", "train_stops.csv.gz"),
                                         dtype={"train_number": str})
                self.stops = self.stops[self.stops["train_number"].isin(set(self.trains["train_number"]))]
            except Exception as e:
                logger.error(f"[network] could not load the real network: {e}")
            self._by_station = {k: g for k, g in self.stops.groupby("station_code")} if len(self.stops) else {}
            self._by_train = {k: g.sort_values("seq") for k, g in self.stops.groupby("train_number")} if len(self.stops) else {}
            self._train = self.trains.set_index("train_number") if len(self.trains) else pd.DataFrame()
            self._station = self.stations.set_index("station_code") if len(self.stations) else pd.DataFrame()
            self._loaded = True
            logger.info(f"[network] {len(self.trains)} trains · {len(self.stations)} stations · {len(self.stops)} halts")

    def reload(self) -> None:
        self._loaded = False
        self._load()

    # ── Lookups ───────────────────────────────────────────────────────────────

    def train(self, number: str) -> Optional[pd.Series]:
        self._load()
        n = str(number).strip()
        return self._train.loc[n] if n in self._train.index else None

    def station(self, code: str) -> Optional[pd.Series]:
        self._load()
        c = str(code).strip().upper()
        return self._station.loc[c] if c in self._station.index else None

    def station_name(self, code: str) -> str:
        s = self.station(code)
        return str(s["station_name"]) if s is not None else str(code)

    def coords(self, code: str) -> Tuple[Optional[float], Optional[float]]:
        s = self.station(code)
        if s is None or pd.isna(s.get("latitude")):
            return None, None
        return float(s["latitude"]), float(s["longitude"])

    def stops_of(self, number: str) -> pd.DataFrame:
        self._load()
        return self._by_train.get(str(number).strip(), pd.DataFrame())

    def stop(self, number: str, code: str) -> Optional[dict]:
        st = self.stops_of(number)
        if not len(st):
            return None
        m = st[st["station_code"] == str(code).strip().upper()]
        return m.iloc[0].to_dict() if len(m) else None

    # ── Station names → codes ────────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(text).lower())).strip()

    def _name_index(self) -> List[Tuple[str, str]]:
        """(normalised name, code) for every station — full names and names without 'Jn' / 'Central' / … —
        busiest stations first."""
        self._load()
        if getattr(self, "_names", None) is None:
            names: List[Tuple[str, str]] = []
            for r in self.stations.sort_values("trains_halting", ascending=False).itertuples():
                full = self._norm(r.station_name)
                base = " ".join(w for w in full.split() if w not in _SUFFIX) or full
                names += [(full, r.station_code), (base, r.station_code)]
            self._names = names
        return self._names

    def resolve(self, text: str) -> Optional[str]:
        """A station code for a station code, station name, city or common old name — tolerant of 'Jn',
        'Central' and small spelling slips ('kasargod' → KGQ). None when nothing is close."""
        self._load()
        t = self._norm(text)
        if not t:
            return None
        if t in CITY_ALIASES:
            return CITY_ALIASES[t]
        if t.upper() in self._station.index:
            return t.upper()
        names = self._name_index()
        tb = " ".join(w for w in t.split() if w not in _SUFFIX) or t
        for cand in (t, tb):
            for name, code in names:
                if name == cand:
                    return code
        if len(tb) >= 4:
            for name, code in names:
                if name.startswith(tb + " ") or (len(tb) >= 5 and name.startswith(tb)):
                    return code
            for name, code in names:                  # 'Kolkata Sealdah', 'Mumbai Bandra Terminus'
                if name.endswith(" " + tb):
                    return code
        if len(tb) >= 5:
            close = difflib.get_close_matches(tb, [n for n, _ in names] + list(CITY_ALIASES), n=1, cutoff=0.82)
            if close:
                return CITY_ALIASES.get(close[0]) or next(c for n, c in names if n == close[0])
        return None

    def suggest(self, text: str, n: int = 3) -> List[str]:
        """'Kasaragod (KGQ)'-style suggestions for an unrecognised place name."""
        t = self._norm(text)
        names = self._name_index()
        out: List[str] = []
        for m in difflib.get_close_matches(t, [x for x, _ in names], n=n * 3, cutoff=0.6):
            code = next(c for x, c in names if x == m)
            label = f"{self.station_name(code)} ({code})"
            if label not in out:
                out.append(label)
            if len(out) >= n:
                break
        return out

    def city_stations(self, code: str) -> Set[str]:
        c = str(code).strip().upper()
        return set(CITY_GROUPS.get(_CITY_OF.get(c, ""), {c})) | {c}

    def is_hub(self, code: str) -> bool:
        s = self.station(code)
        return s is not None and int(s.get("trains_halting", 0)) >= HUB_MIN_TRAINS

    # ── Journeys between two stations ─────────────────────────────────────────

    def segments(self, src: str, dst: str, nearby: bool = False) -> pd.DataFrame:
        """Every train that halts at src and later at dst (nearby=True: any station in the same city)."""
        self._load()
        srcs = self.city_stations(src) if nearby else {str(src).strip().upper()}
        dsts = self.city_stations(dst) if nearby else {str(dst).strip().upper()}
        a_parts = [self._by_station[x] for x in srcs if x in self._by_station]
        b_parts = [self._by_station[x] for x in dsts if x in self._by_station]
        a = pd.concat(a_parts, ignore_index=True) if a_parts else None
        b = pd.concat(b_parts, ignore_index=True) if b_parts else None
        if a is None or b is None or not len(a) or not len(b):
            return pd.DataFrame(columns=list(self.trains.columns) + ["origin", "terminus"])
        m = a.merge(b, on="train_number", suffixes=("_a", "_b"))
        m = m[m["seq_a"] < m["seq_b"]]
        if not len(m):
            return pd.DataFrame(columns=list(self.trains.columns) + ["origin", "terminus"])
        # one row per train: the pair with the shortest ride (matters only with nearby stations)
        m = m.assign(_ride=m["minutes_from_origin_b"] - m["minutes_from_origin_a"]).sort_values("_ride")
        m = m.drop_duplicates("train_number")
        rows = []
        for r in m.itertuples(index=False):
            t = self._train.loc[r.train_number]
            km = float(max(1, round(float(r.km_from_origin_b) - float(r.km_from_origin_a))))
            ride = float(r.minutes_from_origin_b) - float(r.minutes_from_origin_a + r.halt_minutes_a)
            row = t.to_dict()
            row.update({
                "train_number": r.train_number, "origin": t["source"], "terminus": t["destination"],
                "source": r.station_code_a, "destination": r.station_code_b,
                "departure_time": r.departure_a if isinstance(r.departure_a, str) and r.departure_a else t["departure_time"],
                "arrival_time": r.arrival_b if isinstance(r.arrival_b, str) and r.arrival_b else t["arrival_time"],
                "duration_hours": round(max(ride, 1.0) / 60.0, 2), "distance_km": round(km, 0),
                "avg_speed_kmph": round(km / max(ride / 60.0, 0.1), 1),
                "km_at_source": float(r.km_from_origin_a), "km_at_destination": float(r.km_from_origin_b),
                "day_at_source": int(r.day_a), "halts_between": int(r.seq_b - r.seq_a - 1),
            })
            offered = str(t["classes"]).split("|")
            for cls in FARE_CLASSES:
                f = payable_fare(t["train_type"], cls, km, r.train_number) if cls in offered else None
                row[f"base_fare_{cls.lower()}"] = int(f) if f else 0
            rows.append(row)
        out = pd.DataFrame(rows)
        return out.sort_values(["departure_time"]).reset_index(drop=True)

    def connections(self, src: str, dst: str, min_layover: int = 45, max_layover: int = 480,
                    nearby: bool = False, limit: int = 5) -> List[Tuple[pd.Series, pd.Series, int, float]]:
        """One-change journeys src → hub → dst: (first leg, second leg, layover minutes, door-to-door hours)."""
        self._load()
        srcs = self.city_stations(src) if nearby else {str(src).strip().upper()}
        dsts = self.city_stations(dst) if nearby else {str(dst).strip().upper()}
        to_hub: Dict[str, float] = {}               # fastest ride src → station on any train
        for s in srcs:
            for r in self._by_station.get(s, pd.DataFrame()).itertuples(index=False):
                st = self._by_train[r.train_number]
                for x in st[st["seq"] > r.seq].itertuples(index=False):
                    v = x.minutes_from_origin - r.minutes_from_origin
                    if v < to_hub.get(x.station_code, 1e9):
                        to_hub[x.station_code] = v
        from_hub: Dict[str, float] = {}             # fastest ride station → dst on any train
        for d in dsts:
            for r in self._by_station.get(d, pd.DataFrame()).itertuples(index=False):
                st = self._by_train[r.train_number]
                for x in st[st["seq"] < r.seq].itertuples(index=False):
                    v = r.minutes_from_origin - x.minutes_from_origin
                    if v < from_hub.get(x.station_code, 1e9):
                        from_hub[x.station_code] = v
        common = [h for h in (set(to_hub) & set(from_hub)) - srcs - dsts if self.is_hub(h)]
        hubs = sorted(common, key=lambda h: to_hub[h] + from_hub[h])[:10]     # the 10 most direct interchanges
        out = []
        for h in hubs:
            first = [x for x in (self.segments(s, h) for s in srcs) if len(x)]
            second = [x for x in (self.segments(h, d) for d in dsts) if len(x)]
            if not first or not second:
                continue
            first = pd.concat(first, ignore_index=True)
            second = pd.concat(second, ignore_index=True)
            first = first.nsmallest(6, "duration_hours")
            second = second.nsmallest(6, "duration_hours")
            for _, a in first.iterrows():
                arr_a = _hhmm(a["departure_time"]) + float(a["duration_hours"]) * 60
                for _, b in second.iterrows():
                    if b["train_number"] == a["train_number"]:
                        continue
                    dep_b = _hhmm(b["departure_time"])
                    while dep_b < arr_a + min_layover:
                        dep_b += 1440
                    lay = dep_b - arr_a
                    if lay > max_layover:
                        continue
                    total_h = (dep_b + float(b["duration_hours"]) * 60 - _hhmm(a["departure_time"])) / 60
                    out.append((a, b, int(round(lay)), round(total_h, 1)))
        out.sort(key=lambda c: c[3])
        seen, best = set(), []
        for c in out:                                   # one option per train pair
            k = (c[0]["train_number"], c[1]["train_number"])
            if k not in seen:
                seen.add(k)
                best.append(c)
            if len(best) >= limit:
                break
        return best


network = RailNetwork()
