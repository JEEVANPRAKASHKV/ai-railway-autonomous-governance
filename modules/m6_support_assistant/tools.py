"""
modules/m6_support_assistant/tools.py
=====================================
Tools the support assistant calls before falling back to policy retrieval:

  • PNR status lookup and refund tracking against booking records (pnr_history.csv, synthetic)
  • Train suggestions for a route (calls the governed train recommender, M2)
  • Journey-planning hand-off to the AI Travel Planner (M4)

Each tool returns a plain-text "record" that becomes the grounding context, so the hallucination
monitor scores the answer against the record it came from.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.config import settings

logger = logging.getLogger(__name__)

PNR_RE = re.compile(r"\bPNR\s*[-:#]?\s*(\d{7})\b", re.IGNORECASE)
ROUTE_RE = re.compile(r"\b(?:trains?|gaadi|gadi)\b.*?\bfrom\s+([a-z .]+?)\s+to\s+([a-z .]+?)(?:\s+(?:on|tomorrow|today|next|this|for|under|in)\b|[?.!,]|$)",
                      re.IGNORECASE)
PLAN_RE = re.compile(r"\b(plan|itinerary|best way|how (?:do|can) i (?:go|reach|travel))\b", re.IGNORECASE)

CANCEL_FEES = {"1A": 240, "EC": 240, "2A": 200, "CC": 200, "3A": 180, "SL": 120, "2S": 60}


class SupportTools:
    def __init__(self) -> None:
        self._pnr: Optional[pd.DataFrame] = None
        self._trains: Optional[pd.DataFrame] = None
        self._stations: Dict[str, str] = {}
        try:
            self._pnr = pd.read_csv(os.path.join(settings.data_raw_dir, "pnr_history.csv")).set_index("pnr")
        except Exception as e:
            logger.debug(f"[M6 tools] pnr history: {e}")
        try:
            self._trains = pd.read_csv(os.path.join(settings.data_raw_dir, "trains.csv"))
        except Exception as e:
            logger.debug(f"[M6 tools] trains: {e}")
        try:
            st = pd.read_csv(os.path.join(settings.data_raw_dir, "stations.csv"))
            for _, r in st.iterrows():
                self._stations[str(r["station_code"]).lower()] = str(r["station_code"])
                self._stations[str(r["station_name"]).lower()] = str(r["station_code"])
        except Exception as e:
            logger.debug(f"[M6 tools] stations: {e}")
        self._stations.update({"delhi": "NDLS", "new delhi": "NDLS", "mumbai": "CSTM", "bombay": "CSTM",
                               "kolkata": "HWH", "howrah": "HWH", "chennai": "MAS", "bangalore": "SBC",
                               "bengaluru": "SBC", "hyderabad": "SC", "pune": "PUNE", "jaipur": "JP",
                               "ahmedabad": "ADI"})

    def station(self, text: str) -> Optional[str]:
        t = text.strip().lower().rstrip(".")
        if t in self._stations:
            return self._stations[t]
        for name, code in self._stations.items():
            if len(name) > 3 and (t == name or t in name.split()):
                return code
        return None

    # ── PNR status & refund tracking ──────────────────────────────────────────

    def pnr_record(self, query: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        m = PNR_RE.search(query)
        if not m:
            return None
        pnr = f"PNR{m.group(1)}"
        if self._pnr is None or pnr not in self._pnr.index:
            text = f"No booking with {pnr} exists in the records."
            return text, text, {"tool": "pnr_lookup", "pnr": pnr, "found": False}
        r = self._pnr.loc[pnr]
        train_name = ""
        if self._trains is not None:
            t = self._trains[self._trains["train_number"].astype(str) == str(r["train_number"])]
            train_name = f" ({t.iloc[0]['train_name']})" if len(t) else ""
        jd = date.fromisoformat(str(r["journey_date"]))
        status = str(r["final_status"])
        berth = str(r["berth_type"])
        cancelled = isinstance(r["cancellation_date"], str) and r["cancellation_date"]
        lines = [f"{pnr}: train {r['train_number']}{train_name}, class {r['journey_class']}, quota {r['quota']}, "
                 f"journey date {jd.strftime('%d %b %Y')}.",
                 f"Booked at waitlist position WL {int(r['initial_wl_number'])}; status at chart preparation: {status}"
                 + (f", berth {berth}." if status == "Confirmed" else ".")]
        refund_line = self._refund(r, jd, status, cancelled)
        lines.append(refund_line)
        record = " ".join(lines)
        return record, record, {"tool": "pnr_lookup", "pnr": pnr, "found": True, "status": status,
                                "cancelled": bool(cancelled)}

    @staticmethod
    def _refund(r, jd: date, status: str, cancelled) -> str:
        fee = CANCEL_FEES.get(str(r["journey_class"]), 180)
        if cancelled:
            cd = date.fromisoformat(str(r["cancellation_date"]))
            credit = cd + timedelta(days=7)
            hours = (jd - cd).days * 24
            if status == "WL":
                return (f"Refund: cancelled on {cd.strftime('%d %b %Y')} while waitlisted — full fare refunded minus "
                        f"clerkage; expected in your original payment method by {credit.strftime('%d %b %Y')}.")
            rule = (f"flat fee ₹{fee}" if hours > 48 else "25% of the fare" if hours > 12 else
                    "50% of the fare" if hours > 4 else "no refund (cancelled within 4 hours)")
            return (f"Refund: cancelled on {cd.strftime('%d %b %Y')} ({hours} hours before departure) — deduction "
                    f"{rule}; balance expected by {credit.strftime('%d %b %Y')}.")
        if status == "WL" and jd <= date.today():
            return ("Refund: the ticket stayed waitlisted at chart preparation, so it was cancelled automatically "
                    "and the full fare is refunded within 5 business days.")
        return "Refund: none — the ticket has not been cancelled."

    # ── Train suggestions & planning hand-off ─────────────────────────────────

    def route(self, query: str) -> Optional[Tuple[str, str]]:
        m = ROUTE_RE.search(query)
        if not m:
            return None
        a, b = self.station(m.group(1)), self.station(m.group(2))
        return (a, b) if a and b and a != b else None

    def train_suggestions(self, src: str, dst: str) -> Tuple[str, str, Dict[str, Any]]:
        from modules.m2_train_recommender.schema import RecommendRequest
        from modules.m2_train_recommender.service import train_recommender_service
        res = train_recommender_service.recommend(RecommendRequest(
            source_station=src, destination_station=dst, journey_date=date.today() + timedelta(days=7)))
        if not res.recommendations:
            text = res.message or f"No direct trains found from {src} to {dst}."
            return text, text, {"tool": "train_recommender", "route": f"{src}-{dst}", "count": 0}
        lines = [f"Trains from {src} to {dst}:"]
        for r in res.recommendations[:3]:
            lines.append(f"{r.rank}. {r.train_name} ({r.train_number}) departs {r.departure_time}, arrives "
                         f"{r.arrival_time}, {r.duration_hours} h, about ₹{r.fare_estimate_inr:.0f}, "
                         f"{int(r.punctuality_score * 100)}% on time.")
        lines.append("Open Train Recommender for more options, or the AI Travel Planner to plan and book.")
        text = "\n".join(lines)
        return text, text, {"tool": "train_recommender", "route": f"{src}-{dst}", "count": len(res.recommendations)}

    @staticmethod
    def wants_planning(query: str) -> bool:
        return bool(PLAN_RE.search(query))


support_tools = SupportTools()
