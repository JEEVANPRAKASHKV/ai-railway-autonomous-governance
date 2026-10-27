"""
core/fallback_engine.py
=======================
Intelligent rule-based fallback engine.
Activated automatically when Azure OpenAI credentials are absent or invalid.

Provides deterministic, data-driven responses for:
  - Travel Planner (regex constraint extraction + greedy schedule matching)
  - Customer Support (TF-IDF similarity over railway_faq.txt)
  - RCA Analysis (decision-tree diagnosis)
  - Supervisor Agent (threshold-based rule alerting)

Zero external API calls. No crash. Full functionality.
"""

from __future__ import annotations

import os
import re
import math
import time
import pickle
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from core.schemas import LLMResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded corpus for TF-IDF support assistant
# ---------------------------------------------------------------------------
_faq_questions: List[str] = []
_faq_answers: List[str] = []
_tfidf_matrix: Optional[Any] = None
_tfidf_vectorizer: Optional[Any] = None
_trains_df: Optional[pd.DataFrame] = None


def _load_faq() -> None:
    global _faq_questions, _faq_answers, _tfidf_matrix, _tfidf_vectorizer
    if _tfidf_vectorizer is not None:
        return
    faq_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw", "railway_faq.txt"
    )
    if not os.path.exists(faq_path):
        logger.warning("FAQ file not found – support fallback will use static responses.")
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        questions, answers = [], []
        with open(faq_path, encoding="utf-8") as f:
            content = f.read()

        # Parse Q/A pairs separated by blank lines
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.split("\n")
            q_lines = [l[2:].strip() for l in lines if l.startswith("Q:")]
            a_lines = [l[2:].strip() for l in lines if l.startswith("A:")]
            if q_lines and a_lines:
                questions.append(q_lines[0])
                answers.append(a_lines[0])

        if not questions:
            return

        _faq_questions = questions
        _faq_answers = answers
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        _tfidf_matrix = vec.fit_transform(questions)
        _tfidf_vectorizer = vec
        logger.info(f"[Fallback] Loaded {len(questions)} FAQ pairs for TF-IDF support.")
    except Exception as e:
        logger.error(f"[Fallback] Failed to load FAQ corpus: {e}")


def _load_trains() -> Optional[pd.DataFrame]:
    global _trains_df
    if _trains_df is not None:
        return _trains_df
    trains_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw", "trains.csv"
    )
    if os.path.exists(trains_path):
        _trains_df = pd.read_csv(trains_path)
        return _trains_df
    return None


# ---------------------------------------------------------------------------
# RCA Decision Tree
# ---------------------------------------------------------------------------
_RCA_MAP = {
    "m3_delay_predictor": {
        "congestion_spike": {
            "confidence": 0.87,
            "cause": "Station congestion data feed showing distribution shift (PSI elevated). Likely caused by seasonal peak traffic (festival season / holiday rush).",
            "remediations": [
                "Trigger retraining pipeline with last 14 days' congestion data.",
                "Apply exponential smoothing to congestion_index feature to dampen sudden spikes.",
                "Set temporary SLA tolerance to +20% for delay predictions.",
            ],
        },
        "weather_feed_lag": {
            "confidence": 0.65,
            "cause": "Weather data feed showing elevated null rate (>15%) in last 48 hours. Missing values propagating into delay model features.",
            "remediations": [
                "Apply Last Observation Carry-Forward (LOCF) imputation to weather features.",
                "Alert data engineering team to restore weather API connection.",
            ],
        },
    },
    "m1_ticket_predictor": {
        "quota_change": {
            "confidence": 0.80,
            "cause": "PNR history feed shows quota allocation patterns shifted (e.g., IRCTC modified Tatkal quota). Training distribution no longer matches production.",
            "remediations": [
                "Retrain with latest PNR records (last 30 days).",
                "Update quota_type feature encoding to include new quota codes.",
            ],
        },
        "seasonal_surge": {
            "confidence": 0.72,
            "cause": "High-season booking surge (summer vacation / Diwali). WL clearance rates deviating from historical averages.",
            "remediations": [
                "Apply seasonal bias correction factor to confirmation probabilities.",
                "Retrain with holiday-period data subset.",
            ],
        },
    },
    "m6_support_assistant": {
        "rag_index_stale": {
            "confidence": 0.75,
            "cause": "RAG knowledge base ChromaDB index may be stale or corrupted. Retrieval context quality degraded.",
            "remediations": [
                "Re-run rag_indexer.py to rebuild ChromaDB index from irctc_policies.txt.",
                "Clear chroma_db/ directory and regenerate embeddings.",
            ],
        },
        "hallucination_spike": {
            "confidence": 0.82,
            "cause": "Faithfulness score dropped below threshold for consecutive responses. LLM generating responses not grounded in retrieved context.",
            "remediations": [
                "Reduce LLM temperature to 0.1 for RAG responses.",
                "Increase minimum retrieved chunk relevance threshold from 0.5 to 0.7.",
                "Add explicit 'cite source' instruction to system prompt.",
            ],
        },
    },
    "m7_fraud_detector": {
        "bot_pattern_shift": {
            "confidence": 0.88,
            "cause": "New bot behavioral pattern detected: request velocity profile has shifted. Existing model thresholds may be under-detecting next-generation bots.",
            "remediations": [
                "Update rule thresholds for requests_per_minute and unique_pnr_attempts.",
                "Retrain fraud model with latest labeled fraud_events data.",
            ],
        },
    },
    "m4_travel_planner": {
        "tool_cascade_failure": {
            "confidence": 0.79,
            "cause": "Agent tool cascade failure: one or more upstream tools (M1/M3/M5) returning degraded outputs, causing planner to loop.",
            "remediations": [
                "Check health status of M1 (Ticket Predictor), M3 (Delay), and M5 (Fare).",
                "Implement tool-level circuit breaker: if tool fails >3 times, use cached result.",
            ],
        },
    },
    "default": {
        "unknown": {
            "confidence": 0.50,
            "cause": "Root cause could not be automatically determined. Manual inspection of telemetry logs recommended.",
            "remediations": [
                "Review last 100 telemetry events for the affected module.",
                "Check model artifact integrity (md5 checksum).",
                "Inspect input data quality gate violation logs.",
            ],
        }
    },
}

# ---------------------------------------------------------------------------
# Supervisor Priority Rules
# ---------------------------------------------------------------------------
_PRIORITY_RULES = [
    {"metric": "p95_latency_ms",        "op": "gt", "threshold": 5000,  "severity": "P0-Critical", "action": "notify_admin",       "desc": "P95 latency exceeded 5000ms SLA"},
    {"metric": "drift_psi",             "op": "gt", "threshold": 0.25,  "severity": "P0-Critical", "action": "trigger_retrain",     "desc": "Major data drift detected (PSI > 0.25)"},
    {"metric": "accuracy",              "op": "lt", "threshold": 0.78,  "severity": "P0-Critical", "action": "trigger_rollback",    "desc": "Model accuracy dropped below 0.78"},
    {"metric": "hallucination_rate",    "op": "gt", "threshold": 0.15,  "severity": "P1-Warning",  "action": "flag_review",         "desc": "Hallucination rate exceeded 15%"},
    {"metric": "fraud_fp_rate",         "op": "gt", "threshold": 0.12,  "severity": "P1-Warning",  "action": "notify_admin",        "desc": "Fraud false positive rate > 12%"},
    {"metric": "bias_dir",              "op": "lt", "threshold": 0.80,  "severity": "P1-Warning",  "action": "flag_bias_review",    "desc": "Disparate Impact Ratio below 0.80 — bias detected"},
    {"metric": "drift_psi",             "op": "gt", "threshold": 0.15,  "severity": "P1-Warning",  "action": "alert_drift",         "desc": "Moderate data drift detected (PSI > 0.15)"},
    {"metric": "schema_invalid_rate",   "op": "gt", "threshold": 0.05,  "severity": "P1-Warning",  "action": "alert_data_quality",  "desc": "Schema validation failure rate > 5%"},
    {"metric": "error_rate",            "op": "gt", "threshold": 0.02,  "severity": "P2-Info",     "action": "log_alert",           "desc": "Inference error rate > 2%"},
    {"metric": "p95_latency_ms",        "op": "gt", "threshold": 3000,  "severity": "P2-Info",     "action": "log_alert",           "desc": "P95 latency exceeded 3000ms soft threshold"},
]


class FallbackEngine:
    """
    Intelligent rule-based fallback engine.
    Activated when Azure OpenAI is unavailable.
    """

    # ── Customer Support ──────────────────────────────────────────────────────

    def support_query(self, query: str, language: str = "en") -> LLMResponse:
        """Best answer from the verified support knowledge base (data/knowledge/support_kb.json).
        The legacy FAQ / static answers below are only used if the knowledge base cannot be loaded."""
        t0 = time.perf_counter()
        try:
            from modules.m6_support_assistant.rag_indexer import rag_indexer
            hits = rag_indexer.retrieve(query, top_k=1)
            if hits and hits[0][1] >= 0.24:
                return LLMResponse(content=hits[0][0]["text"], mode="fallback", latency_ms=(time.perf_counter() - t0) * 1000)
            if rag_indexer.entries:
                return LLMResponse(content="I don't have a verified answer for that. Please call 139 (Rail Madad) or check irctc.co.in.",
                                   mode="fallback", latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception:
            pass
        _load_faq()

        curated = self._static_support_response(query)
        has_curated = not curated.startswith("Thank you for your query. For specific railway inquiries")
        if has_curated:
            answer = curated
        elif _tfidf_vectorizer is None or not _faq_questions:
            answer = curated
        else:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                query_vec = _tfidf_vectorizer.transform([query])
                sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                if best_score > 0.15:
                    answer = f"{_faq_answers[best_idx]}\n\n*(Source: IRCTC Railway FAQ — Rule-Based Mode)*"
                else:
                    answer = self._static_support_response(query)
            except Exception:
                answer = self._static_support_response(query)

        elapsed = (time.perf_counter() - t0) * 1000
        return LLMResponse(content=answer, mode="fallback", latency_ms=elapsed)

    def _static_support_response(self, query: str) -> str:
        q = query.lower()
        # Cancelling a CONFIRMED ticket → charges table (not the WL auto-refund rule)
        if ("cancel" in q or "charge" in q or "cancellation" in q) and not any(
            w in q for w in ("tatkal", "waitlist", "waiting list", "wl ", "rac", "not confirmed")
        ) and any(w in q for w in ("confirmed", "charge", "fee", "3a", "2a", "1a", "sl", "sleeper", "hours", "days", "before departure")):
            return (
                "Cancelling a confirmed (non-Tatkal) ticket (rules from April 2026): more than 72 hours before departure only a flat fee "
                "applies; 72 to 24 hours before, 25% of the fare is deducted; 24 to 8 hours before, 50%; and there is no refund "
                "within 8 hours of departure. "
                ""
            )
        # Train cancelled / badly delayed by Railways → automatic full refund
        if re.search(r"\btrain\b.{0,25}\bcancel|cancelled train|canceled train|train (is |was )?(very )?(late|delayed)", q) and re.search(r"refund|money|paisa|wapas", q):
            return (
                "If Railways cancels your train, you get a full refund automatically with no cancellation charges — it is credited "
                "to the original payment method within about 7 days. If the train runs more than 3 hours late you can also cancel "
                "for a full refund by filing a TDR on irctc.co.in before the train leaves your boarding station."
            )
        # Waitlisted ticket that did not confirm by chart preparation → automatic full refund
        if (re.search(r"\b(waitlist\w*|waiting list|wl)\b", q) or "not confirmed" in q) and re.search(r"refund|chart|cancel|money|paisa|wapas", q):
            return (
                "If your waitlisted (WL) ticket is not confirmed by the time the chart is prepared, the ticket is cancelled "
                "automatically and the full fare is refunded to your original payment method within about 5 business days — "
                "no action needed."
            )
        if "transfer" in q or "family" in q or "relative" in q or "father" in q or "mother" in q or "spouse" in q:
            return (
                "Confirmed ticket transfer to immediate family members (father, mother, brother, sister, "
                "spouse, son, daughter) is permitted by Indian Railways. You must submit a written request "
                "with government ID and relationship proof to the Chief Reservation Supervisor at least "
                "24 hours before train departure."
            )
        if "luggage" in q or "baggage" in q or "weight" in q or "excess" in q:
            return (
                "Free luggage allowances: 1A: 70kg (marginal 15kg), 2A: 50kg (marginal 10kg), 3A/CC: 40kg (marginal 10kg), "
                "SL: 40kg (marginal 10kg). Excess baggage exceeding allowance must be booked in advance at the parcel office."
            )
        if "child" in q or "kid" in q or "minor" in q or "children" in q:
            return (
                "Children under 5 years of age travel free without a berth. Children aged 5 to 11 can travel on half "
                "fare (without berth) or full adult fare (if separate berth is requested). Children 12+ pay full adult fare."
            )
        if "vande bharat" in q or "shatabdi" in q or "executive" in q or "ec class" in q:
            return (
                "Vande Bharat and Shatabdi trains feature AC Chair Car (CC, 3x2 seating) and Executive Chair Car "
                "(EC, 2x2 spacious 360-degree rotating seats). Onboard dining is optional at booking."
            )
        if "medical" in q or "doctor" in q or "emergency" in q or "sick" in q or "ambulance" in q:
            return (
                "In case of a medical emergency during travel, inform the TTE or dial Railway Helpline 139 immediately. "
                "All passenger trains carry first-aid boxes, and railway doctors can be arranged at the next scheduled halt."
            )
        if "pnr" in q:
            return "To check your PNR status, visit the IRCTC website or the National Train Enquiry System (NTES). The reservation chart is prepared by 8 PM the previous day for morning trains, otherwise at least 10 hours before departure."
        if "cancel" in q or "refund" in q:
            return "Cancellation refunds depend on how many hours before departure you cancel. AC classes attract a cancellation fee of ₹240 (1A), ₹200 (2A), ₹180 (3A). SL class: ₹120. Tatkal tickets are non-refundable."
        if "tatkal" in q:
            return "Tatkal booking opens at 10:00 AM for AC classes (one day before journey) and at 11:00 AM for SL/other classes. Tatkal charges apply on top of base fare."
        if "waitlist" in q or "wl" in q or "rac" in q:
            return "WL (Waitlist) tickets may get confirmed when other passengers cancel. RAC (Reservation Against Cancellation) ensures you get a seat but may share a berth. Confirmation probability depends on class, quota, and days to departure."
        if "fare" in q or "price" in q:
            return "Indian Railway fares vary by class, distance, train type, and season. Dynamic pricing applies to premium trains (Rajdhani, Shatabdi). Bookings open 60 days in advance."
        return "Thank you for your query. For specific railway inquiries, please consult the IRCTC portal at www.irctc.co.in or call the 24x7 RailMadad helpline at 139. I am currently answering using certified IRCTC policy guidelines."

    # ── Travel Planner ────────────────────────────────────────────────────────

    def plan_travel(self, query: str) -> Dict[str, Any]:
        """
        Regex-based constraint extraction + greedy schedule matching from trains.csv.
        Returns top 3 itinerary options.
        """
        constraints = self._extract_constraints(query)
        trains_df = _load_trains()

        itineraries = []
        if trains_df is not None and not trains_df.empty:
            # Filter by source and destination (case-insensitive partial match)
            src = constraints.get("source", "").lower()
            dst = constraints.get("destination", "").lower()
            filtered = pd.DataFrame()
            if src and dst:   # every train that halts at both stations, on the real timetable
                from core.network import network
                filtered = network.segments(src.upper(), dst.upper(), nearby=True)
            if not len(filtered):
                filtered = trains_df.copy()
                if src:
                    filtered = filtered[filtered["source"].str.lower().str.contains(src, na=False)]
                if dst:
                    filtered = filtered[filtered["destination"].str.lower().str.contains(dst, na=False)]

            budget = constraints.get("budget")
            if budget and "base_fare_sl" in filtered.columns:
                filtered = filtered[(filtered["base_fare_sl"] > 0) & (filtered["base_fare_sl"] <= budget)]

            # Sort by base_fare_sl (cheapest first), take top 3
            sort_col = "base_fare_sl" if "base_fare_sl" in filtered.columns else filtered.columns[0]
            filtered = filtered.sort_values(sort_col).head(3)

            for _, row in filtered.iterrows():
                itineraries.append({
                    "train_number": row.get("train_number", "N/A"),
                    "train_name": row.get("train_name", "N/A"),
                    "source": row.get("source", src),
                    "destination": row.get("destination", dst),
                    "departure": row.get("departure_time", "06:00"),
                    "arrival": row.get("arrival_time", "18:00"),
                    "duration_hrs": row.get("duration_hours", 12),
                    "base_fare_sl": row.get("base_fare_sl", 350),
                    "base_fare_3a": row.get("base_fare_3a", 950),
                    "punctuality_pct": row.get("punctuality_pct", 75),
                    "wl_confirmation_prob": 0.65,
                    "delay_risk": "Low",
                    "recommended": True,
                })
        else:
            # Static fallback itineraries
            itineraries = self._static_itineraries(constraints)

        return {
            "query": query,
            "extracted_constraints": constraints,
            "itineraries": itineraries,
            "mode": "fallback",
            "message": "Itineraries generated using rule-based engine (no LLM connection).",
        }

    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract travel constraints from natural language query using regex."""
        constraints: Dict[str, Any] = {}

        # Budget
        budget_match = re.search(r"(?:under|below|within|max|budget of?)\s*(?:rs\.?|₹)?\s*(\d+)", query, re.IGNORECASE)
        if budget_match:
            constraints["budget"] = int(budget_match.group(1))

        # Time constraint (arrive before HH:MM)
        time_match = re.search(r"(?:reach|arrive|before)\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?", query, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            if time_match.group(3) and time_match.group(3).lower() == "pm" and hour < 12:
                hour += 12
            constraints["arrive_before"] = f"{hour:02d}:{time_match.group(2) or '00'}"

        # Date (tomorrow, today, DD/MM, etc.)
        if re.search(r"\btomorrow\b", query, re.IGNORECASE):
            from datetime import timedelta
            constraints["journey_date"] = (date.today() + timedelta(days=1)).isoformat()
        elif re.search(r"\btoday\b", query, re.IGNORECASE):
            constraints["journey_date"] = date.today().isoformat()
        else:
            date_match = re.search(r"(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{2,4}))?", query)
            if date_match:
                d, m = int(date_match.group(1)), int(date_match.group(2))
                y = int(date_match.group(3)) if date_match.group(3) else date.today().year
                if y < 100:
                    y += 2000
                constraints["journey_date"] = f"{y}-{m:02d}-{d:02d}"

        # Class preference
        class_map = {"first class": "1A", "ac": "3A", "sleeper": "SL", "second class": "2S", "chair car": "CC"}
        for phrase, cls in class_map.items():
            if phrase in query.lower():
                constraints["travel_class"] = cls
                break

        # Source / Destination — look for "from X to Y" pattern
        route_match = re.search(
            r"(?:from|frm)\s+([A-Za-z\s]+?)\s+(?:to|2)\s+([A-Za-z\s]+?)(?:\s+(?:on|tomorrow|today|under|below|before|$))",
            query, re.IGNORECASE
        )
        if route_match:
            constraints["source"] = route_match.group(1).strip()
            constraints["destination"] = route_match.group(2).strip()

        return constraints

    def _static_itineraries(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        src = constraints.get("source", "Mumbai")
        dst = constraints.get("destination", "Delhi")
        return [
            {"train_number": "12951", "train_name": "Mumbai Rajdhani Express", "source": src, "destination": dst,
             "departure": "16:35", "arrival": "08:35", "duration_hrs": 16, "base_fare_sl": 0, "base_fare_3a": 1895,
             "punctuality_pct": 88, "wl_confirmation_prob": 0.72, "delay_risk": "Low", "recommended": True},
            {"train_number": "12137", "train_name": "Punjab Mail", "source": src, "destination": dst,
             "departure": "19:05", "arrival": "22:00", "duration_hrs": 27, "base_fare_sl": 485, "base_fare_3a": 1275,
             "punctuality_pct": 71, "wl_confirmation_prob": 0.58, "delay_risk": "Medium", "recommended": True},
            {"train_number": "12910", "train_name": "Garib Rath Express", "source": src, "destination": dst,
             "departure": "06:00", "arrival": "00:30", "duration_hrs": 18.5, "base_fare_sl": 0, "base_fare_3a": 695,
             "punctuality_pct": 76, "wl_confirmation_prob": 0.61, "delay_risk": "Low", "recommended": True},
        ]

    # ── RCA Analysis ──────────────────────────────────────────────────────────

    def analyze_rca(self, module: str, alert_description: str = "") -> Dict[str, Any]:
        """Decision-tree root cause analysis."""
        module_causes = _RCA_MAP.get(module, _RCA_MAP["default"])
        # Pick most likely cause based on keyword matching in alert description
        best_key = list(module_causes.keys())[0]
        best_conf = 0.0
        alert_lower = alert_description.lower()
        for key, info in module_causes.items():
            keywords = key.replace("_", " ").split()
            matches = sum(1 for kw in keywords if kw in alert_lower)
            if matches > 0 and info["confidence"] > best_conf:
                best_key = key
                best_conf = info["confidence"]

        selected = module_causes[best_key]
        return {
            "module": module,
            "alert": alert_description,
            "root_cause_id": best_key,
            "root_cause_description": selected["cause"],
            "confidence": selected["confidence"],
            "remediations": selected["remediations"],
            "mode": "fallback",
        }

    # ── Supervisor Agent ──────────────────────────────────────────────────────

    def supervisor_tick(self, metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Rule-based supervisor tick.
        metrics: {module_name: {metric_name: value, ...}}
        Returns list of alerts.
        """
        alerts = []
        import uuid

        for module, module_metrics in metrics.items():
            for rule in _PRIORITY_RULES:
                metric_val = module_metrics.get(rule["metric"])
                if metric_val is None:
                    continue
                triggered = False
                if rule["op"] == "gt" and metric_val > rule["threshold"]:
                    triggered = True
                elif rule["op"] == "lt" and metric_val < rule["threshold"]:
                    triggered = True

                if triggered:
                    alerts.append({
                        "alert_id": str(uuid.uuid4())[:8],
                        "priority": rule["severity"],
                        "module": module,
                        "description": f"{rule['desc']} — {module}: {rule['metric']}={metric_val:.3f} (threshold: {rule['threshold']})",
                        "metric_value": metric_val,
                        "threshold": rule["threshold"],
                        "action_taken": rule["action"],
                        "action_status": "Pending",
                        "mode": "fallback",
                    })
                    break  # One alert per module per tick (highest priority)

        return alerts

    # ── LLM-style wrapper for governance cockpit summaries ────────────────────

    def generate_governance_summary(self, telemetry_snapshot: Dict[str, Any]) -> LLMResponse:
        """Generate a plain-text governance summary from telemetry data."""
        lines = ["**Autonomous Governance Summary (Rule-Based Mode)**\n"]
        for module, metrics in telemetry_snapshot.items():
            p95 = metrics.get("p95_latency_ms", 0)
            accuracy = metrics.get("accuracy", 1.0)
            status = "✅ Healthy"
            if p95 > 5000 or accuracy < 0.78:
                status = "🔴 Critical"
            elif p95 > 3000 or accuracy < 0.82:
                status = "🟡 Degraded"
            lines.append(f"- **{module}**: {status} | P95={p95:.0f}ms | Accuracy={accuracy:.2%}")
        content = "\n".join(lines)
        return LLMResponse(content=content, mode="fallback")


# Singleton export
fallback_engine = FallbackEngine()
