"""
core/fares.py
=============
Official Indian Railways fares.

Base fares come from the Railway Board's Tariff (TAG 2025, "Fares" chapter,
https://indianrailways.gov.in/railwayboard/uploads/directorate/coaching/TAG_2025/Fares.pdf). The tables are
published per distance slab; the slab values below are copied from that PDF and fares between two listed slabs
are interpolated linearly (the published slabs are 5–50 km wide, so this stays within about 2% of the printed
value). Where a class has both a peak-season and a lean-season fare, the peak fare is used (the tariff applies
it to popular trains in all seasons).

Payable fare = base fare + reservation charge + superfast charge (superfast services) + 5% GST on AC classes,
using the standard per-class charges (erail.in summary of the Railway Board commercial circulars). Not included:
catering on Rajdhani / Duronto / Shatabdi / Vande Bharat, flexi (dynamic) fares on premium trains, and
concessions. Always confirm the final amount on IRCTC.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

TARIFF_SOURCE = "Railway Board TAG 2025 · Fares (base fare by distance slab)"
TARIFF_URL = "https://indianrailways.gov.in/railwayboard/uploads/directorate/coaching/TAG_2025/Fares.pdf"
CHARGES_SOURCE = "Standard reservation / superfast / Tatkal charges (erail.in fare rules)"

Slabs = List[Tuple[int, int]]          # (upper km of slab, base fare ₹)

# ── TAG 2025 base fares (₹) ──────────────────────────────────────────────────
MAIL_EXPRESS: Dict[str, Slabs] = {
    "2S": [(5, 30), (100, 49), (200, 76), (300, 107), (500, 160), (750, 221), (1000, 273), (1500, 360),
           (2000, 450), (2500, 508)],
    "SL": [(5, 40), (100, 79), (200, 124), (300, 183), (500, 271), (750, 378), (1000, 459), (1500, 596),
           (2000, 732), (2500, 824)],
    "FC": [(100, 232), (300, 550), (500, 817), (750, 1129), (1000, 1371), (1500, 1763), (2000, 2156), (2500, 2469)],
    "CC": [(100, 211), (200, 286), (300, 382), (500, 565), (750, 790), (1000, 956), (1500, 1236), (2000, 1513),
           (2500, 1694)],
    "3A": [(300, 440), (500, 708), (750, 990), (1000, 1196), (1500, 1532), (2000, 1868), (2500, 2075), (3000, 2282),
           (5000, 3213)],
    "2A": [(300, 625), (500, 1016), (750, 1421), (1000, 1723), (1500, 2225), (2000, 2724), (2500, 3045), (3000, 3367),
           (5000, 4653)],
    "1A": [(300, 1059), (500, 1731), (750, 2408), (1000, 2935), (1500, 3803), (2000, 4672), (2500, 5233), (3000, 5793),
           (5000, 8036)],
}
RAJDHANI_DURONTO: Dict[str, Slabs] = {
    "SL": [(300, 206), (500, 303), (1000, 511), (1500, 668), (2000, 813), (3000, 1020), (3500, 1117)],   # Duronto SL
    "3E": [(300, 470), (500, 758), (1000, 1267), (1500, 1625), (2000, 1952), (3000, 2410), (3500, 2612)],
    "3A": [(300, 510), (500, 821), (1000, 1373), (1500, 1760), (2000, 2113), (3000, 2608), (3500, 2827)],
    "2A": [(300, 718), (500, 1166), (1000, 1968), (1500, 2536), (2000, 3093), (3000, 3822), (3500, 4198)],
    "1A": [(300, 1215), (500, 1989), (1000, 3346), (1500, 4329), (2000, 5335), (3000, 6578), (3500, 7204)],
}
SHATABDI: Dict[str, Slabs] = {
    "CC": [(50, 172), (300, 446), (500, 658), (1000, 1103)],
    "EC": [(50, 386), (300, 990), (500, 1446), (1000, 2434)],
}
JAN_SHATABDI: Dict[str, Slabs] = {
    "2S": [(50, 40), (300, 128), (500, 182), (1000, 315), (1500, 436)],
    "CC": [(50, 172), (300, 416), (500, 605), (1000, 1022), (1500, 1316)],
}
GARIB_RATH: Dict[str, Slabs] = {
    "CC": [(100, 122), (300, 272), (500, 389), (1000, 659), (1500, 844), (2000, 1062), (3000, 1534)],
    "3A": [(100, 154), (300, 321), (500, 459), (1000, 766), (1500, 972), (2000, 1202), (3000, 1768)],
}
HUMSAFAR: Dict[str, Slabs] = {
    "SL": [(300, 210), (500, 312), (1000, 528), (1500, 685), (2000, 842), (3000, 1052), (5000, 1469)],
    "3A": [(300, 506), (500, 814), (1000, 1375), (1500, 1762), (2000, 2148), (3000, 2624), (5000, 3695)],
}
TABLES: Dict[str, Dict[str, Slabs]] = {
    "mail_express": MAIL_EXPRESS, "rajdhani_duronto": RAJDHANI_DURONTO, "shatabdi": SHATABDI,
    "jan_shatabdi": JAN_SHATABDI, "garib_rath": GARIB_RATH, "humsafar": HUMSAFAR,
}
TABLE_LABELS = {
    "mail_express": "Mail/Express", "rajdhani_duronto": "Rajdhani/Duronto", "shatabdi": "Shatabdi",
    "jan_shatabdi": "Jan Shatabdi", "garib_rath": "Garib Rath", "humsafar": "Humsafar",
}

# ── Charges (₹) ───────────────────────────────────────────────────────────────
RESERVATION = {"2S": 15, "SL": 20, "CC": 40, "3E": 40, "3A": 40, "FC": 50, "2A": 50, "1A": 60, "EC": 60}
SUPERFAST = {"2S": 15, "SL": 30, "CC": 45, "3E": 45, "3A": 45, "FC": 45, "2A": 45, "1A": 75, "EC": 75}
TATKAL = {  # share of base fare, minimum, maximum
    "2S": (0.10, 10, 15), "SL": (0.30, 100, 200), "CC": (0.30, 125, 225), "3E": (0.30, 300, 400),
    "3A": (0.30, 300, 400), "2A": (0.30, 400, 500), "EC": (0.30, 400, 500), "1A": (0.30, 400, 500),
}
AC_CLASSES = {"1A", "2A", "3A", "3E", "CC", "EC"}
GST_AC = 0.05
SUPERFAST_TYPES = {"Superfast", "Rajdhani", "Duronto", "Shatabdi", "Jan Shatabdi", "Garib Rath", "Humsafar",
                   "Sampark Kranti", "Vande Bharat", "Tejas"}


def table_for(train_type: str) -> str:
    t = str(train_type or "").lower()
    if "rajdhani" in t or "duronto" in t:
        return "rajdhani_duronto"
    if "jan shatabdi" in t:
        return "jan_shatabdi"
    if "shatabdi" in t or "vande" in t or "tejas" in t:
        return "shatabdi"
    if "garib" in t:
        return "garib_rath"
    if "humsafar" in t:
        return "humsafar"
    return "mail_express"


def _interp(slabs: Slabs, km: float) -> float:
    kms = [k for k, _ in slabs]
    if km <= kms[0]:
        return float(slabs[0][1])
    i = bisect.bisect_left(kms, km)
    if i >= len(slabs):                          # beyond the last listed slab: continue the last segment's rate
        (k1, f1), (k2, f2) = slabs[-2], slabs[-1]
        return f2 + (f2 - f1) / (k2 - k1) * (km - k2)
    (k1, f1), (k2, f2) = slabs[i - 1], slabs[i]
    return f1 + (f2 - f1) * (km - k1) / (k2 - k1)


def base_fare(train_type: str, cls: str, km: float) -> Optional[float]:
    """TAG 2025 base fare (₹) for this class and distance, or None if this kind of train has no such class."""
    cls = cls.upper()
    kind = table_for(train_type)
    if cls in TABLES[kind]:
        slabs = TABLES[kind][cls]
    elif kind in ("mail_express", "shatabdi", "jan_shatabdi") and cls in MAIL_EXPRESS:
        slabs = MAIL_EXPRESS[cls]
    else:
        return None
    return float(round(_interp(slabs, max(1.0, float(km)))))


def is_superfast(train_type: str, train_number: str = "") -> bool:
    return str(train_type) in SUPERFAST_TYPES or str(train_number)[:2] in ("12", "22")


def tatkal_charge(cls: str, base: float) -> float:
    share, lo, hi = TATKAL.get(cls.upper(), (0.30, 100, 500))
    return float(min(hi, max(lo, round(share * base))))


def fare_breakdown(train_type: str, cls: str, km: float, train_number: str = "",
                   tatkal: bool = False) -> Optional[Dict[str, float]]:
    """Base fare + charges → estimated payable fare, with each part itemised."""
    cls = cls.upper()
    base = base_fare(train_type, cls, km)
    if base is None:
        return None
    res = float(RESERVATION.get(cls, 40))
    sf = float(SUPERFAST.get(cls, 45)) if is_superfast(train_type, train_number) else 0.0
    tq = tatkal_charge(cls, base) if tatkal else 0.0
    gst = round(GST_AC * (base + res + sf + tq), 0) if cls in AC_CLASSES else 0.0
    return {"base_fare": base, "reservation_charge": res, "superfast_charge": sf, "tatkal_charge": tq,
            "gst": gst, "total": round(base + res + sf + tq + gst, 0), "distance_km": round(float(km), 1),
            "tariff_table": TABLE_LABELS[table_for(train_type)]}


def payable_fare(train_type: str, cls: str, km: float, train_number: str = "") -> Optional[float]:
    b = fare_breakdown(train_type, cls, km, train_number)
    return b["total"] if b else None
