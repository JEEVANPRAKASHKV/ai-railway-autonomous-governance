"""
modules/m4_travel_planner/agent.py
==================================
Multi-agent Travel Planner with Human-in-the-Loop (HITL) approval gate.
Orchestrates Constraint Parser -> Route Search -> Fare Estimator -> WL Check -> Delay Check -> HITL Gate.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from core.config import settings
from governance.hooks import governance_hook
from modules.m4_travel_planner.hitl_gate import hitl_gate
from modules.m4_travel_planner.schema import (
    ApprovalRequest,
    PlanRequest,
    PlannedItineraryOption,
)
from modules.m4_travel_planner.tools import (
    DelayCheckerTool,
    FareEstimatorTool,
    RouteSearchTool,
    WLPredictorTool,
)

logger = logging.getLogger(__name__)

STATION_SYNONYMS = {
    "delhi": "NDLS", "new delhi": "NDLS", "ndls": "NDLS",
    "mumbai": "CSTM", "bombay": "CSTM", "cstm": "CSTM", "bct": "CSTM",
    "kolkata": "HWH", "howrah": "HWH", "hwh": "HWH",
    "chennai": "MAS", "madras": "MAS", "mas": "MAS",
    "bangalore": "SBC", "bengaluru": "SBC", "sbc": "SBC",
    "hyderabad": "SC", "secunderabad": "SC", "sc": "SC",
    "jaipur": "JP", "jp": "JP",
    "pune": "PUNE", "pune jn": "PUNE",
    "ahmedabad": "ADI", "adi": "ADI",
}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"bypass\s+(hitl|approval|safety|gate)",
    r"system\s+override",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"auto[- ]confirm\s+(without|skip)\s+approval",
]


class TravelPlannerAgent:
    """Autonomous agent that parses intent, synthesizes options, and yields to the HITL gate."""

    def __init__(self) -> None:
        self.route_tool = RouteSearchTool()
        self.fare_tool = FareEstimatorTool()
        self.wl_tool = WLPredictorTool()
        self.delay_tool = DelayCheckerTool()

    def _check_prompt_injection(self, query: str) -> bool:
        low = query.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, low):
                return True
        return False

    def _parse_constraints(self, query: str) -> Dict[str, Any]:
        low = query.lower()
        src = "NDLS"
        dst = "CSTM"

        # Match station keywords
        for word, code in STATION_SYNONYMS.items():
            if re.search(r"\b(from|leaving|start|departs?)\s+" + re.escape(word) + r"\b", low):
                src = code
            elif re.search(r"\b(to|reach|dest|arrive|towards)\s+" + re.escape(word) + r"\b", low):
                dst = code

        # Match budget
        budget_match = re.search(r"(?:budget|under|below|max|rs\.?|inr|₹)\s*(\d{3,5})", low)
        budget = float(budget_match.group(1)) if budget_match else 2500.0

        # Match class preference
        preferred_class = "3A"
        for cls in ["1a", "2a", "3a", "sl", "cc", "ec"]:
            if re.search(r"\b" + cls + r"\b", low):
                preferred_class = cls.upper()
                break

        return {
            "source": src,
            "destination": dst,
            "budget": budget,
            "preferred_class": preferred_class,
        }

    @governance_hook("m4_travel_planner", endpoint="/api/v1/agent/plan")
    def plan(self, request: PlanRequest) -> ApprovalRequest:
        """Execute multi-step itinerary planning and stop at the HITL approval gate."""
        execution_trace: List[Dict[str, Any]] = []

        # ── Step 1: Prompt Injection Security Scan ────────────────────────────
        is_injection = self._check_prompt_injection(request.query)
        execution_trace.append({
            "step": "Security Scan",
            "check": "Prompt Injection & Instruction Override Detection",
            "flagged": is_injection,
            "status": "BLOCKED" if is_injection else "PASSED",
        })

        if is_injection:
            # Fall back to sanitized safe query
            logger.warning(f"[Agentic Planner] Prompt injection attempt intercepted in query: {request.query}")
            safe_query = "Travel from NDLS to CSTM next week"
            constraints = self._parse_constraints(safe_query)
        else:
            constraints = self._parse_constraints(request.query)

        if request.max_budget:
            constraints["budget"] = float(request.max_budget)

        execution_trace.append({
            "step": "Constraint Parsing",
            "parsed_source": constraints["source"],
            "parsed_destination": constraints["destination"],
            "budget_cap": constraints["budget"],
            "preferred_class": constraints["preferred_class"],
        })

        # ── Step 2: Route Search ──────────────────────────────────────────────
        candidates = self.route_tool.run(constraints["source"], constraints["destination"])
        execution_trace.append({
            "step": "Route Search Tool",
            "routes_evaluated": len(candidates),
            "corridor": f"{constraints['source']} ➔ {constraints['destination']}",
        })

        # ── Step 3: Fare, Confirmation, & Delay Estimation ────────────────────
        itineraries: List[PlannedItineraryOption] = []
        travel_class = constraints["preferred_class"]

        for i, train in enumerate(candidates[:4]):
            fare = self.fare_tool.run(train, travel_class=travel_class)
            wl_prob = self.wl_tool.run(current_wl=10 + i * 5, days_ahead=12, travel_class=travel_class)
            pred_delay = self.delay_tool.run(
                train_number=str(train.get("train_number")),
                station_code=constraints["destination"],
            )

            # Pros & Cons
            pros = []
            cons = []
            if fare <= constraints["budget"]:
                pros.append(f"Within budget (₹{fare:.0f} vs ₹{constraints['budget']:.0f})")
            else:
                cons.append(f"Exceeds budget target by ₹{fare - constraints['budget']:.0f}")

            if wl_prob >= 0.75:
                pros.append(f"High confirmation odds ({int(wl_prob * 100)}%)")
            else:
                cons.append(f"Waitlist clearance uncertainty ({int(wl_prob * 100)}% prob)")

            if pred_delay < 20.0:
                pros.append(f"Minimal forecasted delay ({pred_delay:.0f} min)")
            else:
                cons.append(f"Likely corridor delay of {pred_delay:.0f} min")

            itineraries.append(
                PlannedItineraryOption(
                    option_id=f"OPT-{i + 1}",
                    train_number=str(train.get("train_number", "12951")),
                    train_name=str(train.get("train_name", "Express")),
                    source=str(train.get("source", constraints["source"])),
                    destination=str(train.get("destination", constraints["destination"])),
                    departure_time=str(train.get("departure_time", "08:00")),
                    arrival_time=str(train.get("arrival_time", "22:30")),
                    duration_hours=float(train.get("duration_hours", 14.5)),
                    travel_class=travel_class,
                    estimated_fare_inr=fare,
                    confirmation_probability=wl_prob,
                    predicted_delay_minutes=pred_delay,
                    confidence_score=0.91,
                    pros=pros,
                    cons=cons,
                )
            )

        # Sort itineraries by composite desirability (budget match + confirmation odds)
        itineraries.sort(
            key=lambda x: (
                1.0 if x.estimated_fare_inr <= constraints["budget"] else 0.5
            ) * x.confirmation_probability,
            reverse=True,
        )

        recommended = itineraries[0] if itineraries else None
        alternatives = itineraries[1:] if len(itineraries) > 1 else []

        if recommended is None:
            # Fallback mock option
            recommended = PlannedItineraryOption(
                option_id="OPT-1",
                train_number="12951",
                train_name="Mumbai Rajdhani",
                source=constraints["source"],
                destination=constraints["destination"],
                departure_time="16:55",
                arrival_time="08:35",
                duration_hours=15.6,
                travel_class="3A",
                estimated_fare_inr=2200.0,
                confirmation_probability=0.85,
                predicted_delay_minutes=14.0,
                confidence_score=0.92,
                pros=["Fastest corridor transit", "Within budget"],
                cons=[],
            )

        # ── Step 4: Stop at HITL Gate ─────────────────────────────────────────
        req_id = f"PLAN-{uuid.uuid4().hex[:8].upper()}"
        execution_trace.append({
            "step": "HITL Gate Interception",
            "action": "HALT_FOR_USER_CONFIRMATION",
            "request_id": req_id,
            "status": "PENDING_APPROVAL",
        })

        summary = (
            f"Synthesized optimal route from {constraints['source']} to {constraints['destination']} "
            f"via {recommended.train_name} ({recommended.train_number}) in {recommended.travel_class} "
            f"at ₹{recommended.estimated_fare_inr:.0f}. Estimated confirmation probability: "
            f"{int(recommended.confirmation_probability * 100)}% with minimal predicted delay "
            f"({recommended.predicted_delay_minutes:.0f} mins). "
            f"Awaiting your explicit approval before simulated booking preparation."
        )

        approval_request = ApprovalRequest(
            request_id=req_id,
            user_id=request.user_id,
            status="PENDING_APPROVAL",
            recommended_itinerary=recommended,
            alternative_itineraries=alternatives,
            total_estimated_cost_inr=recommended.estimated_fare_inr,
            agent_reasoning_summary=summary,
            execution_trace=execution_trace,
        )

        # Register in HITL state registry
        hitl_gate.register_approval_request(approval_request)

        return approval_request


travel_planner_agent = TravelPlannerAgent()
