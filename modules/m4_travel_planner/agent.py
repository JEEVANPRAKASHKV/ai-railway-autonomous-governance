"""
modules/m4_travel_planner/agent.py
==================================
Multi-agent travel planner with a mandatory Human-in-the-Loop (HITL) approval gate.

A coordinator runs specialised agents, each a small, auditable unit with its own trace entry:

  SecurityAgent   prompt-injection / instruction-override screening (blocks, never "sanitises")
  IntentAgent     natural language → constraints (rules first; the configured LLM (Sarvam AI / Gemini) fills gaps;
                  every value is validated against the station list / calendar)
  RouteAgent      direct trains + one-change connections, with real date-times (multi-day aware)
  PricingAgent    dynamic fare per leg from the governed Fare model (M5)
  BookingOddsAgent confirmation odds per leg from the governed Ticket model (M1)
  DelayAgent      expected arrival delay per leg from the governed Delay model (M3)
  OptimizerAgent  hard constraints (budget, arrival deadline incl. predicted delay) then persona ranking
  ValidatorAgent  independently re-checks the recommendation against the parsed constraints

Governance: every step is timed and logged; the AgentMonitor tracks tool calls, latency, validation
failures, injection attempts and rejection streaks and publishes AGENT_ANOMALY when something is off.
Nothing is booked until the passenger approves.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.config import settings
from core.network import network
from governance.context import report_inference
from governance.hooks import governance_hook
from governance.telemetry_bus import telemetry_bus
from modules.m4_travel_planner.hitl_gate import hitl_gate
from modules.m4_travel_planner.schema import (
    ApprovalRequest,
    JourneyLeg,
    PlannedItineraryOption,
    PlanRequest,
)

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 40          # tool-call budget per plan (loop / runaway guard)
MAX_PLAN_SECONDS = 10.0
MIN_LAYOVER_MIN = 45
MAX_LAYOVER_MIN = 8 * 60

CITY_SYNONYMS = {
    "delhi": "NDLS", "new delhi": "NDLS", "mumbai": "CSTM", "bombay": "CSTM", "kolkata": "HWH",
    "calcutta": "HWH", "howrah": "HWH", "chennai": "MAS", "madras": "MAS", "bangalore": "SBC",
    "bengaluru": "SBC", "hyderabad": "SC", "secunderabad": "SC", "jaipur": "JP", "pune": "PUNE",
    "ahmedabad": "ADI", "lucknow": "LKO", "patna": "PNBE", "bhopal": "BPL", "agra": "AGC",
    "varanasi": "BSB", "kanpur": "CNB", "nagpur": "NGP", "surat": "ST", "vadodara": "BRC",
    "amritsar": "ASR", "chandigarh": "CDG", "guwahati": "GHY", "bhubaneswar": "BBS", "kochi": "ERS",
    "ernakulam": "ERS", "coimbatore": "CBE", "madurai": "MDU", "visakhapatnam": "VSKP", "vizag": "VSKP",
    "indore": "INDB", "jodhpur": "JU", "udaipur": "UDZ", "kota": "KOTA", "ludhiana": "LDH",
    "jammu": "JAT", "goa": "MAO", "madgaon": "MAO", "mysore": "MYS", "mysuru": "MYS",
    "trivandrum": "TVC", "thiruvananthapuram": "TVC", "ranchi": "RNC", "raipur": "R", "gwalior": "GWL",
}

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass|skip|disable|turn off)\b.{0,40}\b(previous|prior|earlier|above|all|your|the|any)?\s*(instructions?|rules?|guard ?rails?|polic(y|ies)|approval|confirmation|gate|safety|checks?|system prompt)\b",
    r"\b(book|confirm|pay|purchase)\b.{0,30}\b(without|no need for|skip(ping)?)\b.{0,20}\b(approval|confirmation|asking|consent|me)\b",
    r"\bsystem\s+(override|prompt)\b",
    r"\b(developer|admin|god|jailbreak|dan)\s+mode\b",
    r"\bauto[- ]?(confirm|approve|book)\b",
    r"\byou are now\b",
    r"\b(reveal|print|show)\b.{0,20}\b(system prompt|api key|secret|password)\b",
]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _t(s: str) -> Tuple[int, int]:
    h, m = str(s).split(":")[:2]
    return int(h), int(m)


class _Trace:
    """Execution trace + tool-call accounting for one plan."""

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.tool_calls = 0
        self.t0 = time.perf_counter()

    def step(self, agent: str, step: str, status: str = "DONE", **detail: Any) -> None:
        self.steps.append({"agent": agent, "step": step, "status": status,
                           "t_ms": round((time.perf_counter() - self.t0) * 1000, 1), **detail})

    def tool(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > MAX_TOOL_CALLS:
            raise RuntimeError(f"tool-call budget exceeded ({MAX_TOOL_CALLS})")


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

class SecurityAgent:
    def scan(self, query: str) -> Tuple[bool, List[str]]:
        low = query.lower()
        hits = [p for p in INJECTION_PATTERNS if re.search(p, low)]
        return bool(hits), hits


class IntentAgent:
    def __init__(self, stations: Dict[str, str]) -> None:
        self.stations = stations   # lower-case name / code → code

    def _station(self, text: str) -> Optional[str]:
        t = re.sub(r"[^a-z .]", "", text.lower()).strip()
        if not t:
            return None
        if t in CITY_SYNONYMS:
            return CITY_SYNONYMS[t]
        if t in self.stations:
            return self.stations[t]
        hit = network.resolve(t)          # any real station name, 'Jn'/'Central' optional, small typos allowed
        if hit:
            return hit
        return None

    def _date(self, low: str, today: date) -> Tuple[Optional[date], str]:
        if "day after tomorrow" in low:
            return today + timedelta(days=2), "day after tomorrow"
        if re.search(r"\btomorrow\b|\bkal\b", low):
            return today + timedelta(days=1), "tomorrow"
        if re.search(r"\btoday\b|\baaj\b|\btonight\b", low):
            return today, "today"
        m = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", low)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(0)
            except ValueError:
                pass
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", low)
        if m:
            y = int(m.group(3)) if m.group(3) else today.year
            y = y + 2000 if y < 100 else y
            try:
                d = date(y, int(m.group(2)), int(m.group(1)))
                return (d if d >= today else date(y + 1, d.month, d.day)) if not m.group(3) else d, m.group(0)
            except ValueError:
                pass
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", low) or \
            re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", low)
        if m:
            a, b = m.group(1), m.group(2)
            day, mon = (int(a), MONTHS[b[:3]]) if a.isdigit() else (int(b), MONTHS[a[:3]])
            try:
                d = date(today.year, mon, day)
                return (d if d >= today else date(today.year + 1, mon, day)), m.group(0)
            except ValueError:
                pass
        if "weekend" in low:
            delta = (5 - today.weekday()) % 7 or 7
            return today + timedelta(days=delta), "this weekend (Saturday)"
        for i, wd in enumerate(WEEKDAYS):
            if re.search(rf"\b{wd}\b", low):
                delta = (i - today.weekday()) % 7 or 7   # the coming <weekday>
                return today + timedelta(days=delta), wd
        if "next week" in low:
            return today + timedelta(days=7), "next week"
        return None, ""

    def parse(self, query: str, trace: _Trace) -> Dict[str, Any]:
        low = " " + query.lower().replace("₹", " rs ") + " "
        today = date.today()
        c: Dict[str, Any] = {"source": None, "destination": None, "travel_date": None, "date_phrase": "",
                             "budget": None, "preferred_class": None, "arrival_before": None,
                             "method": "rules", "unresolved": []}

        m = re.search(r"\bfrom\s+([a-z .]+?)\s+(?:to|for|till)\s+([a-z .]+?)(?=\s+(?:on|by|before|under|below|within|tomorrow|today|next|this|in|with|and|at|reach|arriv|day|for|budget|rs|inr|\d)|[,.?!]|\s*$)", low) \
            or re.search(r"\b([a-z .]+?)\s+to\s+([a-z .]+?)(?=\s+(?:on|by|before|under|below|within|tomorrow|today|next|this|in|with|and|at|reach|arriv|day|for|budget|rs|inr|\d)|[,.?!]|\s*$)", low)
        if m:
            a = m.group(1).split()[-3:]
            for k in range(len(a)):
                cand = self._station(" ".join(a[k:]))
                if cand:
                    c["source"] = cand
                    break
            c["destination"] = self._station(m.group(2)) or self._station(m.group(2).split()[0] if m.group(2).split() else "")
            c["_raw_route"] = (m.group(1).strip(), m.group(2).strip())

        d, phrase = self._date(low, today)
        c["travel_date"], c["date_phrase"] = (d or today + timedelta(days=1)), (phrase or "not stated (assumed tomorrow)")

        bm = re.search(r"(?:budget|under|below|max(?:imum)?|within|upto|up to|less than)\s*(?:of\s*)?(?:rs\.?|inr)?\s*(\d{3,6})", low) \
            or re.search(r"(?:rs\.?|inr)\s*(\d{3,6})", low)
        if bm:
            c["budget"] = float(bm.group(1))
        tm = re.search(r"(?:before|by|reach by|arrive by|latest)\s*(\d{1,2})(?::|\.)?(\d{2})?\s*(am|pm)?", low)
        if tm:
            hr, mn, ap = int(tm.group(1)), int(tm.group(2) or 0), tm.group(3)
            if ap == "pm" and hr < 12:
                hr += 12
            if ap == "am" and hr == 12:
                hr = 0
            if hr <= 23:
                c["arrival_before"] = f"{hr:02d}:{mn:02d}"
        elif re.search(r"\bmorning\b", low):
            c["arrival_before"] = "11:00"
        for cls in ["1a", "2a", "3a", "sl", "cc", "ec", "2s"]:
            if re.search(rf"\b{cls}\b", low):
                c["preferred_class"] = cls.upper()
                break
        if not c["preferred_class"] and re.search(r"\bsleeper\b", low):
            c["preferred_class"] = "SL"
        if not c["preferred_class"] and re.search(r"\bac\b", low):
            c["preferred_class"] = "3A"

        # LLM assist: only for what the rules could not resolve, and every value is validated.
        if (not c["source"] or not c["destination"]):
            self._llm_fill(query, c, trace)

        if not c["source"]:
            c["unresolved"].append("origin")
        if not c["destination"]:
            c["unresolved"].append("destination")
        trace.step("IntentAgent", "Understand request", method=c["method"],
                   source=c["source"], destination=c["destination"], travel_date=str(c["travel_date"]),
                   date_phrase=c["date_phrase"], budget=c["budget"], arrival_before=c["arrival_before"],
                   preferred_class=c["preferred_class"], unresolved=c["unresolved"])
        return c

    def _llm_fill(self, query: str, c: Dict[str, Any], trace: _Trace) -> None:
        try:
            from core.llm_router import llm_router
            if not llm_router.has_llm:
                return
            trace.tool()
            resp = llm_router.complete(
                prompt=f"Request: {query}\nReturn JSON with keys origin_city and destination_city (city names in English, or null).",
                system_prompt="You extract travel details. Reply with a single JSON object and nothing else.",
                temperature=0.0, max_tokens=120)
            if resp.mode == "fallback":
                return
            via = {"sarvam": "Sarvam AI", "gemini": "Gemini", "azure_openai": "Azure OpenAI"}.get(resp.mode, resp.mode)
            txt = re.search(r"\{.*\}", resp.content, re.S)
            data = json.loads(txt.group(0)) if txt else {}
            src = self._station(str(data.get("origin_city") or ""))
            dst = self._station(str(data.get("destination_city") or ""))
            if src and not c["source"]:
                c["source"], c["method"] = src, f"rules + {via}"
            if dst and not c["destination"]:
                c["destination"], c["method"] = dst, f"rules + {via}"
        except Exception as e:
            logger.debug(f"[IntentAgent] LLM assist skipped: {e}")


class RouteAgent:
    def __init__(self, trains: pd.DataFrame) -> None:
        self.trains = trains

    def _leg(self, row: pd.Series, day: date) -> Dict[str, Any]:
        h, m = _t(row["departure_time"])
        dep = datetime.combine(day, datetime.min.time()).replace(hour=h, minute=m)
        arr = dep + timedelta(hours=float(row["duration_hours"]))
        return {"row": row, "dep": dep, "arr": arr}

    def search(self, src: str, dst: str, day: date, trace: _Trace) -> List[List[Dict[str, Any]]]:
        """Direct trains that halt at both stations (any station in the same city), then one-change
        connections through an interchange hub, on the real timetable."""
        journeys: List[List[Dict[str, Any]]] = []
        for _, r in network.segments(src, dst, nearby=True).iterrows():
            journeys.append([self._leg(r, day)])
        direct = len(journeys)
        for a, b, _lay, _h in network.connections(src, dst, MIN_LAYOVER_MIN, MAX_LAYOVER_MIN, nearby=True, limit=6):
            la = self._leg(a, day)
            for dd in (la["arr"].date(), la["arr"].date() + timedelta(days=1)):
                lb = self._leg(b, dd)
                lay = (lb["dep"] - la["arr"]).total_seconds() / 60
                if MIN_LAYOVER_MIN <= lay <= MAX_LAYOVER_MIN:
                    journeys.append([la, lb])
                    break
        trace.step("RouteAgent", "Search routes", direct_trains=direct, connections=len(journeys) - direct,
                   corridor=f"{src} → {dst}", date=str(day))
        return journeys


class CoordinatorTools:
    """Calls into the other governed AI services (M1 ticket odds, M3 delay, M5 fare)."""

    def __init__(self, history: Optional[pd.DataFrame]) -> None:
        self.history = history

    def typical_wl(self, train_number: str, cls: str) -> int:
        h = self.history
        if h is None:
            return 20
        sub = h[(h["train_number"].astype(str) == str(train_number)) & (h["journey_class"] == cls)]
        return int(sub["initial_wl_number"].median()) if len(sub) >= 5 else 20

    def fare(self, row: pd.Series, cls: str, day: date, trace: _Trace) -> float:
        trace.tool()
        from modules.m5_fare_predictor.schema import FarePredictRequest
        from modules.m5_fare_predictor.service import fare_predictor_service
        col = f"base_fare_{cls.lower()}"
        base = float(row.get(col, 0) or 0) or None
        try:
            r = fare_predictor_service.predict(FarePredictRequest(train_number=str(row["train_number"]),
                                                                  journey_class=cls, journey_date=day, base_fare=base))
            return float(r.current_predicted_fare_inr)
        except Exception:
            return float(base or row.get("base_fare_sl", 400) or 400)

    def odds(self, row: pd.Series, cls: str, day: date, trace: _Trace) -> float:
        trace.tool()
        from modules.m1_ticket_predictor.schema import TicketPredictRequest
        from modules.m1_ticket_predictor.service import ticket_predictor_service
        try:
            r = ticket_predictor_service.predict(TicketPredictRequest(
                train_number=str(row["train_number"]), journey_class=cls, quota="GN",
                journey_date=max(day, date.today() + timedelta(days=1)),
                current_wl_number=max(1, self.typical_wl(str(row["train_number"]), cls))))
            return float(r.confirmation_probability)
        except Exception:
            return 0.5

    def delay(self, row: pd.Series, day: date, trace: _Trace) -> float:
        trace.tool()
        from modules.m3_delay_predictor.schema import DelayPredictRequest
        from modules.m3_delay_predictor.service import delay_predictor_service
        try:
            r = delay_predictor_service.predict(DelayPredictRequest(
                train_number=str(row["train_number"]), station_code=str(row["destination"]), journey_date=day))
            return float(r.predicted_delay_minutes)
        except Exception:
            return 15.0


class ValidatorAgent:
    """Independent re-check: the recommendation must match what the passenger asked for."""

    def validate(self, opt: Optional[PlannedItineraryOption], c: Dict[str, Any]) -> Dict[str, Any]:
        if opt is None:
            return {"valid": False, "violations": ["no itinerary produced"], "checks": {}}
        checks = {
            # a station in the same city counts (Mumbai CSMT / Mumbai Central / Bandra Terminus …)
            "starts_at_requested_origin": opt.source in network.city_stations(c["source"]),
            "ends_at_requested_destination": opt.destination in network.city_stations(c["destination"]),
            "departs_on_requested_date": bool(opt.departure_datetime and opt.departure_datetime[:10] == str(c["travel_date"])),
            "legs_connect": all(opt.legs[i].destination == opt.legs[i + 1].source for i in range(len(opt.legs) - 1)),
            "fare_is_sum_of_legs": abs(opt.estimated_fare_inr - sum(l.fare_inr for l in opt.legs)) < 1.0,
        }
        if c.get("budget"):
            checks["within_budget"] = opt.estimated_fare_inr <= c["budget"]
        if c.get("arrival_before"):
            checks["arrives_before_deadline"] = bool(opt.constraint_checks.get("arrives_before_deadline"))
        violations = [k for k, v in checks.items() if not v]
        hard = {"starts_at_requested_origin", "ends_at_requested_destination", "departs_on_requested_date",
                "legs_connect", "fare_is_sum_of_legs"}
        return {"valid": not (set(violations) & hard), "violations": violations, "checks": checks,
                "all_constraints_met": not violations}


# ──────────────────────────────────────────────────────────────────────────────
# Agent behaviour monitor (governance)
# ──────────────────────────────────────────────────────────────────────────────

class AgentMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.plans = 0
        self.blocked = 0
        self.clarifications = 0
        self.validation_failures = 0
        self.constraint_misses = 0
        self.latencies: deque = deque(maxlen=300)
        self.tool_calls: deque = deque(maxlen=300)
        self.rejections: Dict[str, int] = defaultdict(int)
        self.decisions = {"approved": 0, "rejected": 0}
        self.anomalies: deque = deque(maxlen=50)

    def _anomaly(self, kind: str, detail: Dict[str, Any]) -> None:
        rec = {"kind": kind, **detail, "at": datetime.utcnow().isoformat() + "Z"}
        with self._lock:
            self.anomalies.append(rec)
        telemetry_bus.publish(module_id="m4_travel_planner", event_type="AGENT_ANOMALY", payload=rec,
                              severity="WARNING")

    def record_plan(self, trace: _Trace, status: str, validation: Dict[str, Any], user: str) -> None:
        lat = (time.perf_counter() - trace.t0)
        with self._lock:
            self.plans += 1
            self.latencies.append(lat * 1000)
            self.tool_calls.append(trace.tool_calls)
            if status == "BLOCKED":
                self.blocked += 1
            if status == "NEEDS_CLARIFICATION":
                self.clarifications += 1
            if validation and not validation.get("valid", True):
                self.validation_failures += 1
            if validation and not validation.get("all_constraints_met", True):
                self.constraint_misses += 1
        if status == "BLOCKED":
            self._anomaly("prompt_injection_attempt", {"user": user[:20]})
        if lat > MAX_PLAN_SECONDS:
            self._anomaly("slow_plan", {"seconds": round(lat, 1)})
        if trace.tool_calls > MAX_TOOL_CALLS * 0.8:
            self._anomaly("excessive_tool_calls", {"tool_calls": trace.tool_calls})
        if validation and not validation.get("valid", True) and status == "PENDING_APPROVAL":
            self._anomaly("invalid_recommendation", {"violations": validation.get("violations")})

    def record_decision(self, user: str, approved: bool) -> None:
        with self._lock:
            self.decisions["approved" if approved else "rejected"] += 1
            self.rejections[user] = 0 if approved else self.rejections[user] + 1
            streak = self.rejections[user]
        if streak >= 3:
            self._anomaly("repeated_rejections", {"user": user[:20], "streak": streak})

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            lat = sorted(self.latencies)
            d = self.decisions["approved"] + self.decisions["rejected"]
            return {
                "plans": self.plans, "blocked_injections": self.blocked, "clarifications_requested": self.clarifications,
                "validation_failures": self.validation_failures, "plans_missing_some_constraint": self.constraint_misses,
                "p95_plan_ms": round(lat[int(len(lat) * 0.95) - 1], 1) if lat else None,
                "avg_tool_calls": round(sum(self.tool_calls) / len(self.tool_calls), 1) if self.tool_calls else None,
                "approval_rate": round(self.decisions["approved"] / d, 3) if d else None,
                "decisions": dict(self.decisions), "recent_anomalies": list(self.anomalies)[-10:],
            }


agent_monitor = AgentMonitor()


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator
# ──────────────────────────────────────────────────────────────────────────────

class TravelPlannerAgent:
    """Coordinator: runs the agents in order and stops at the human approval gate."""

    def __init__(self) -> None:
        raw = settings.data_raw_dir
        self.trains = pd.read_csv(os.path.join(raw, "trains.csv"))
        stations: Dict[str, str] = {}
        try:
            st = pd.read_csv(os.path.join(raw, "stations.csv"))
            for _, r in st.iterrows():
                stations[str(r["station_code"]).lower()] = str(r["station_code"])
                stations[str(r["station_name"]).lower()] = str(r["station_code"])
        except Exception:
            pass
        for code in set(self.trains["source"]) | set(self.trains["destination"]):
            stations.setdefault(str(code).lower(), str(code))
        try:
            hist = pd.read_csv(os.path.join(raw, "pnr_history.csv"), usecols=["train_number", "journey_class", "initial_wl_number"])
        except Exception:
            hist = None
        self.security = SecurityAgent()
        self.intent = IntentAgent(stations)
        self.routes = RouteAgent(self.trains)
        self.tools = CoordinatorTools(hist)
        self.validator = ValidatorAgent()

    # Kept for backwards compatibility with earlier tests / callers
    def _check_prompt_injection(self, query: str) -> bool:
        return self.security.scan(query)[0]

    def _response(self, req_id: str, request: PlanRequest, status: str, trace: _Trace, summary: str,
                  constraints: Dict[str, Any], recommended=None, alternatives=None, validation=None,
                  question: Optional[str] = None) -> ApprovalRequest:
        validation = validation or {}
        agent_monitor.record_plan(trace, status, validation, request.user_id)
        clean = {k: (str(v) if isinstance(v, date) else v) for k, v in constraints.items() if not k.startswith("_")}
        out = ApprovalRequest(
            request_id=req_id, user_id=request.user_id, status=status,
            recommended_itinerary=recommended, alternative_itineraries=alternatives or [],
            total_estimated_cost_inr=recommended.estimated_fare_inr if recommended else 0.0,
            agent_reasoning_summary=summary, execution_trace=trace.steps, constraints=clean,
            validation=validation, clarification_question=question,
            agent_metrics={"tool_calls": trace.tool_calls, "plan_ms": round((time.perf_counter() - trace.t0) * 1000, 1)},
            expires_at=(datetime.utcnow() + timedelta(minutes=30)).isoformat() + "Z" if status == "PENDING_APPROVAL" else None,
        )
        if status == "PENDING_APPROVAL":
            hitl_gate.register_approval_request(out)
        report_inference(prediction="plan", version="multi-agent-v2")
        return out

    def _build_option(self, i: int, legs: List[Dict[str, Any]], cls: str, c: Dict[str, Any], trace: _Trace) -> PlannedItineraryOption:
        jl: List[JourneyLeg] = []
        for lg in legs:
            row = lg["row"]
            leg_cls = cls if cls in str(row["classes"]).split("|") else str(row["classes"]).split("|")[0]
            jl.append(JourneyLeg(
                train_number=str(row["train_number"]), train_name=str(row["train_name"]),
                source=str(row["source"]), destination=str(row["destination"]),
                departure=lg["dep"].isoformat(timespec="minutes"), arrival=lg["arr"].isoformat(timespec="minutes"),
                duration_hours=float(row["duration_hours"]),
                fare_inr=round(self.tools.fare(row, leg_cls, lg["dep"].date(), trace), 0),
                confirmation_probability=round(self.tools.odds(row, leg_cls, lg["dep"].date(), trace), 3),
                predicted_delay_minutes=round(self.tools.delay(row, lg["dep"].date(), trace), 1)))
        dep, arr = legs[0]["dep"], legs[-1]["arr"]
        delay = jl[-1].predicted_delay_minutes
        exp_arr = arr + timedelta(minutes=delay)
        fare = round(sum(l.fare_inr for l in jl), 0)
        conf = 1.0
        for l in jl:
            conf *= l.confirmation_probability
        checks: Dict[str, bool] = {}
        pros, cons = [], []
        if c.get("budget"):
            checks["within_budget"] = fare <= c["budget"]
            (pros if checks["within_budget"] else cons).append(
                f"{'Within' if checks['within_budget'] else 'Over'} budget: ₹{fare:,.0f} vs ₹{c['budget']:,.0f}")
        if c.get("arrival_before"):
            h, m = _t(c["arrival_before"])
            # "Reach before 10:00" = the first 10:00 after setting off (not 10:00 on whatever day the train arrives).
            deadline = datetime.combine(dep.date(), datetime.min.time()).replace(hour=h, minute=m)
            if deadline <= dep:
                deadline += timedelta(days=1)
            ok = exp_arr <= deadline
            checks["arrives_before_deadline"] = ok
            (pros if ok else cons).append(
                f"Expected arrival {exp_arr.strftime('%a %H:%M')} incl. ~{delay:.0f} min delay "
                f"{'is before' if ok else 'is after'} your target of {deadline.strftime('%a %H:%M')}")
        days = (exp_arr.date() - dep.date()).days
        if days >= 1:
            cons.append(f"Overnight journey — arrives {'next day' if days == 1 else f'{days} days later'}")
        layover = None
        if len(jl) > 1:
            layover = int((legs[1]["dep"] - legs[0]["arr"]).total_seconds() // 60)
            cons.append(f"One change at {jl[0].destination} ({layover // 60} h {layover % 60:02d} m layover)")
        (pros if conf >= 0.6 else cons).append(f"Confirmation odds {conf:.0%}")
        if delay < 20:
            pros.append(f"Low expected delay (~{delay:.0f} min)")
        name = jl[0].train_name if len(jl) == 1 else f"{jl[0].train_name} + {jl[1].train_name}"
        return PlannedItineraryOption(
            option_id=f"OPT-{i + 1}", train_number="/".join(l.train_number for l in jl), train_name=name,
            source=jl[0].source, destination=jl[-1].destination,
            departure_time=dep.strftime("%H:%M"), arrival_time=arr.strftime("%H:%M"),
            duration_hours=round((arr - dep).total_seconds() / 3600, 1), travel_class=cls,
            estimated_fare_inr=fare, confirmation_probability=round(conf, 3), predicted_delay_minutes=delay,
            confidence_score=round(min(0.95, 0.5 + conf / 2), 2), pros=pros, cons=cons, legs=jl,
            changes=len(jl) - 1, layover_minutes=layover,
            departure_datetime=dep.isoformat(timespec="minutes"), arrival_datetime=arr.isoformat(timespec="minutes"),
            expected_arrival_with_delay=exp_arr.isoformat(timespec="minutes"),
            constraint_checks=checks, meets_all_constraints=all(checks.values()) if checks else True,
        )

    @governance_hook("m4_travel_planner", endpoint="/api/v1/agent/plan")
    def plan(self, request: PlanRequest) -> ApprovalRequest:
        trace = _Trace()
        req_id = f"PLAN-{uuid.uuid4().hex[:8].upper()}"

        blocked, hits = self.security.scan(request.query)
        trace.step("SecurityAgent", "Prompt-injection screening", status="BLOCKED" if blocked else "PASSED",
                   patterns_matched=len(hits))
        if blocked:
            return self._response(req_id, request, "BLOCKED", trace,
                                  "This request tries to change the planner's rules (for example to skip your "
                                  "approval). It was blocked and logged. Please describe only your journey.", {})

        c = self.intent.parse(request.query, trace)
        if request.max_budget:
            c["budget"] = float(request.max_budget)
        if c["unresolved"]:
            raw = c.get("_raw_route")
            what = " and ".join(c["unresolved"])
            names = []
            if raw:
                if "origin" in c["unresolved"]:
                    names.append(raw[0])
                if "destination" in c["unresolved"]:
                    names.append(raw[1])
            hints = [h for nm in names for h in network.suggest(nm)]
            q = (f"I couldn't match the {what}" + (f" ('{', '.join(names)}')" if names else "") +
                 " to a station in the timetable. Which station or city do you mean?"
                 + (f" Did you mean {' or '.join(hints[:3])}?" if hints else ""))
            return self._response(req_id, request, "NEEDS_CLARIFICATION", trace, q, c, question=q)
        if c["source"] == c["destination"]:
            q = "Origin and destination are the same. Where would you like to go?"
            return self._response(req_id, request, "NEEDS_CLARIFICATION", trace, q, c, question=q)

        persona = (request.persona or "budget").lower()
        cls = c["preferred_class"] or ("SL" if persona in ("budget", "student") else "3A")
        journeys = self.routes.search(c["source"], c["destination"], c["travel_date"], trace)
        if not journeys:
            q = (f"No direct train or one-change connection runs from {c['source']} to {c['destination']} in the "
                 "timetable. Try a nearby major station?")
            return self._response(req_id, request, "NEEDS_CLARIFICATION", trace, q, c, question=q)

        # Cheap pre-ranking before calling the paid / slower model tools on the best few
        def pre(j):
            return (len(j) - 1, (j[-1]["arr"] - j[0]["dep"]).total_seconds())
        journeys = sorted(journeys, key=pre)[:5]
        try:
            options = [self._build_option(i, j, cls, c, trace) for i, j in enumerate(journeys)]
        except RuntimeError as e:   # tool-call budget exceeded
            trace.step("Coordinator", "Abort", status="ABORTED", reason=str(e))
            return self._response(req_id, request, "NEEDS_CLARIFICATION", trace,
                                  "Planning was stopped by the tool-call safety budget. Please narrow the request.", c)
        trace.step("PricingAgent/BookingOddsAgent/DelayAgent", "Score options with M5 fare, M1 odds, M3 delay models",
                   options=len(options), tool_calls=trace.tool_calls)

        def rank(o: PlannedItineraryOption):
            score = (0 if o.meets_all_constraints else 1, )
            if persona in ("budget", "student"):
                return score + (o.estimated_fare_inr, -o.confirmation_probability)
            if persona in ("business",):
                return score + (o.expected_arrival_with_delay or "", o.changes)
            return score + (o.changes, -o.confirmation_probability, o.estimated_fare_inr)
        options.sort(key=rank)
        for i, o in enumerate(options):
            o.option_id = f"OPT-{i + 1}"
        best, alts = options[0], options[1:4]
        trace.step("OptimizerAgent", "Rank by hard constraints, then persona", persona=persona,
                   meeting_all_constraints=sum(1 for o in options if o.meets_all_constraints))

        validation = self.validator.validate(best, c)
        trace.step("ValidatorAgent", "Independent check of the recommendation",
                   status="PASSED" if validation["valid"] else "FAILED", violations=validation["violations"])
        if not validation["valid"]:
            return self._response(req_id, request, "NEEDS_CLARIFICATION", trace,
                                  "The validator rejected the plan; nothing was proposed. Please try again.", c,
                                  validation=validation)

        trace.step("HITL Gate", "Stop for passenger approval", status="PENDING_APPROVAL", request_id=req_id)
        met = "meets all your constraints" if best.meets_all_constraints else \
            "is the closest match — no option meets every constraint (" + ", ".join(
                k.replace("_", " ") for k, v in best.constraint_checks.items() if not v) + ")"
        summary = (f"{best.train_name} from {best.source} to {best.destination} on "
                   f"{c['travel_date'].strftime('%a %d %b')} ({c['date_phrase']}) in {cls}: ₹{best.estimated_fare_inr:,.0f}, "
                   f"{best.confirmation_probability:.0%} confirmation odds, expected arrival "
                   f"{datetime.fromisoformat(best.expected_arrival_with_delay).strftime('%a %H:%M')}. It {met}. "
                   f"Compared {len(options)} option(s). Nothing is booked until you approve.")
        return self._response(req_id, request, "PENDING_APPROVAL", trace, summary, c, best, alts, validation)


travel_planner_agent = TravelPlannerAgent()
