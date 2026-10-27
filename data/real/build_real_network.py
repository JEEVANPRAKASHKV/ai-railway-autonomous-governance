"""
data/real/build_real_network.py
===============================
Builds the real Indian Railways network used by the platform from the open **datameet/railways** dataset
(https://github.com/datameet/railways, CC0 public domain; timetable snapshot of 2016).

Outputs (committed, so the app needs no download):
  data/real/stations.csv          every station a kept train halts at: code, name, zone, state, latitude, longitude,
                                  number of trains halting there (state/zone missing in the source are taken from
                                  the nearest station that has them)
  data/real/trains.csv            reservable long-distance trains: number, name, type, zone, origin, terminus,
                                  departure, arrival, journey minutes, distance (km), classes, halts
  data/real/train_stops.csv.gz    every halt of every kept train in order: arrival, departure, day, minutes and
                                  kilometres from the origin

Kept trains: Mail/Express, Superfast, Rajdhani, Duronto, Shatabdi, Jan Shatabdi, Garib Rath and Sampark Kranti
services with at least one reservable class. Suburban, passenger, MEMU/DEMU, toy trains and specials (0xxxx) are
dropped, as are slip-coach entries (e.g. 12307-Slip). A halt is an origin, a terminus, or a station where arrival and departure differ (pass-through points in
the source timetable carry identical times).

Kilometres per halt are not in the source: they are measured along the train's path through every timetable point
(halts and pass-through stations) from station coordinates, then scaled so the last halt equals the train's
published distance.

Run:
    python data/real/build_real_network.py            # downloads the three source files from GitHub
    python data/real/build_real_network.py --src DIR  # or uses a local clone
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_URL = "https://raw.githubusercontent.com/datameet/railways/master/{}"
KEEP_TYPES = {"Exp": "Express", "SF": "Superfast", "Mail": "Mail", "Raj": "Rajdhani", "Drnt": "Duronto",
              "Shtb": "Shatabdi", "JShtb": "Jan Shatabdi", "GR": "Garib Rath", "SKr": "Sampark Kranti"}
CLASS_FLAGS = [("first_ac", "1A"), ("second_ac", "2A"), ("third_ac", "3A"), ("first_class", "FC"),
               ("chair_car", "CC"), ("sleeper", "SL")]


def _load(src: Optional[str], name: str):
    if src:
        with open(os.path.join(src, name), encoding="utf-8") as f:
            return json.load(f)
    print(f"downloading {name} ...")
    with urllib.request.urlopen(RAW_URL.format(name), timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _hav(a, b) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _mins(t: str) -> Optional[int]:
    if not t or t == "None":
        return None
    h, m = t.split(":")[:2]
    return int(h) * 60 + int(m)


def _name(s: str) -> str:
    s = " ".join(str(s or "").split())
    return s.title() if s.isupper() else s


def _train_name(s: str) -> str:
    """'CHENNAI CENT - MYSORE Shatabdi Express' → 'Chennai Cent - Mysore Shatabdi Express' (short codes kept)."""
    return " ".join(w.title() if w.isupper() and len(w) > 3 else w for w in str(s).split())


def build(src: Optional[str]) -> None:
    stations = _load(src, "stations.json")["features"]
    trains = _load(src, "trains.json")["features"]
    schedules = _load(src, "schedules.json")

    st_meta: Dict[str, dict] = {}
    for f in stations:
        p, g = f["properties"], f.get("geometry")
        st_meta[p["code"]] = {"name": _name(p.get("name")), "zone": p.get("zone") or "", "state": p.get("state") or "",
                              "coords": tuple(g["coordinates"]) if g and g.get("coordinates") else None}

    by_train: Dict[str, List[dict]] = defaultdict(list)
    for r in schedules:
        by_train[str(r["train_number"])].append(r)

    train_rows, stop_rows = [], []
    halts_at: Dict[str, set] = defaultdict(set)
    for f in trains:
        p = f["properties"]
        num, ttype = str(p["number"]), p.get("type") or ""
        if ttype not in KEEP_TYPES or num.startswith("0") or not (num.isdigit() and len(num) == 5):   # specials, slip coaches
            continue
        classes = [c for flag, c in CLASS_FLAGS if p.get(flag)]
        if ttype == "Shtb":                 # Shatabdi: the source's first-AC flag is the Executive chair car
            classes = ["CC", "EC"]
        if ttype == "JShtb":
            classes = classes + ["2S"]
        if not classes or not any(c in classes for c in ("1A", "2A", "3A", "SL", "CC")):
            continue
        pts = sorted(by_train.get(num, []), key=lambda r: r["id"])
        if len(pts) < 2:
            continue

        # minutes from the origin's departure, kept monotonic
        t0 = _mins(pts[0]["departure"]) or _mins(pts[0]["arrival"])
        if t0 is None:
            continue
        mins, prev = [], 0
        for i, r in enumerate(pts):
            a = _mins(r["arrival"]) if i else None
            d = _mins(r["departure"]) if i < len(pts) - 1 else None
            ref = a if a is not None else d
            if ref is None:                 # untimed pass-through point: filled in below
                mins.append(None)
                continue
            m = (int(r.get("day") or 1) - 1) * 1440 + ref - t0
            while m < prev:
                m += 1440
            mins.append((m, (d - a) % 1440 if (a is not None and d is not None) else 0))
            prev = m + mins[-1][1]
        if mins[-1] is None:
            continue
        timed = [i for i, x in enumerate(mins) if x is not None]
        for i, x in enumerate(mins):        # untimed points: interpolate between the neighbouring timed ones
            if x is None:
                lo = max(k for k in timed if k < i)
                hi = min(k for k in timed if k > i)
                mins[i] = (round(mins[lo][0] + (mins[hi][0] - mins[lo][0]) * (i - lo) / (hi - lo)), 0)

        # path length through every timetable point that has coordinates
        coords = [st_meta.get(r["station_code"], {}).get("coords") for r in pts]
        cum: List[Optional[float]] = [None] * len(pts)
        last, acc = None, 0.0
        for i, c in enumerate(coords):
            if c is None:
                continue
            if last is not None:
                acc += _hav(coords[last], c)
            cum[i] = acc
            last = i
        known = [i for i, v in enumerate(cum) if v is not None]
        if len(known) < 2:
            continue
        for i in range(len(pts)):           # stations without coordinates: interpolate by running time
            if cum[i] is None:
                lo = max([k for k in known if k < i], default=known[0])
                hi = min([k for k in known if k > i], default=known[-1])
                if lo == hi:
                    cum[i] = cum[lo]
                else:
                    span = max(1, mins[hi][0] - mins[lo][0])
                    cum[i] = cum[lo] + (cum[hi] - cum[lo]) * (mins[i][0] - mins[lo][0]) / span
        published = float(p.get("distance") or 0)
        measured = cum[-1] - cum[0]
        scale = published / measured if measured > 0 and published > 0 and 0.8 <= published / measured <= 2.0 else 1.15
        km = [(v - cum[0]) * scale for v in cum]

        halts = [i for i, r in enumerate(pts) if i in (0, len(pts) - 1) or ("None" not in (r["arrival"], r["departure"])
                                                                         and r["arrival"] != r["departure"])]
        seen, kept = set(), []
        for i in halts:                     # a station listed twice (reversal) keeps its first halt
            code = pts[i]["station_code"]
            if code not in seen:
                seen.add(code)
                kept.append(i)
        if len(kept) < 2:
            continue
        total_min = mins[kept[-1]][0]
        dist = round(published if published > 0 else km[kept[-1]])
        if dist < 50 or total_min <= 0:
            continue
        for seq, i in enumerate(kept):
            r = pts[i]
            code = r["station_code"]
            stop_rows.append({
                "train_number": num, "seq": seq, "station_code": code,
                "arrival": "" if seq == 0 else (r["arrival"] or "")[:5],
                "departure": "" if seq == len(kept) - 1 else (r["departure"] or "")[:5],
                "day": int(r.get("day") or 1), "minutes_from_origin": mins[i][0],
                "halt_minutes": mins[i][1], "km_from_origin": round(km[i], 1),
            })
            halts_at[code].add(num)
        o, t = pts[kept[0]], pts[kept[-1]]
        train_rows.append({
            "train_number": num, "train_name": _train_name(p["name"]),
            "train_type": KEEP_TYPES[ttype], "zone": p.get("zone") or "",
            "source": o["station_code"], "destination": t["station_code"],
            "departure_time": (o["departure"] or "")[:5], "arrival_time": (t["arrival"] or "")[:5],
            "duration_minutes": int(total_min), "distance_km": int(dist),
            "classes": "|".join(classes), "halts": len(kept), "return_train": str(p.get("return_train") or ""),
        })

    # about half the source stations carry no state / zone: take them from the nearest station that does
    ref = [(m["coords"], m["state"], m["zone"]) for m in st_meta.values() if m["coords"] and m["state"] and m["zone"]]
    for code in halts_at:
        m = st_meta.get(code)
        if m and m["coords"] and not (m["state"] and m["zone"]):
            near = min(ref, key=lambda r: (r[0][0] - m["coords"][0]) ** 2 + (r[0][1] - m["coords"][1]) ** 2)
            m["state"], m["zone"] = m["state"] or near[1], m["zone"] or near[2]

    trains_df = pd.DataFrame(train_rows).sort_values("train_number")
    stops_df = pd.DataFrame(stop_rows)
    stops_df = stops_df[stops_df["train_number"].isin(trains_df["train_number"])]
    st_rows = []
    for code, nums in halts_at.items():
        m = st_meta.get(code, {})
        c = m.get("coords")
        st_rows.append({"station_code": code, "station_name": m.get("name") or code, "zone": m.get("zone", ""),
                        "state": m.get("state", ""), "latitude": round(c[1], 5) if c else None,
                        "longitude": round(c[0], 5) if c else None, "trains_halting": len(nums)})
    stations_df = pd.DataFrame(st_rows).sort_values(["trains_halting", "station_code"], ascending=[False, True])

    trains_df.to_csv(os.path.join(HERE, "trains.csv"), index=False)
    stations_df.to_csv(os.path.join(HERE, "stations.csv"), index=False)
    stops_df.to_csv(os.path.join(HERE, "train_stops.csv.gz"), index=False, compression="gzip")
    print(f"trains {len(trains_df)} · stations {len(stations_df)} · halts {len(stops_df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="local folder with stations.json, trains.json, schedules.json")
    build(ap.parse_args().src)
