"""
governance/report_generator.py
────────────────────────────────
Pillar 7 – Automated Governance Report Generator.

Generates comprehensive governance audit reports in:
  • JSON format (machine-readable, API-friendly)
  • Markdown format (human-readable, for Git/PR)
  • Summary dict (for Streamlit dashboard)

Report sections:
  1. Executive Summary
  2. Platform Health (SLA compliance, circuit breakers)
  3. Model Drift Analysis (per-model KS/PSI)
  4. Fairness & Bias Audit (DIR, DPD, EOD)
  5. LLM Faithfulness Summary
  6. Open Incidents (RCA engine)
  7. Model Lifecycle Status
  8. Recommendations & Action Items
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Report paths
# ──────────────────────────────────────────────────────────────────────────────

REPORTS_DIR = Path("governance/reports")


# ──────────────────────────────────────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Aggregates data from all governance pillars and produces structured
    audit reports. Reports are saved to disk and returned as dicts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._report_history: List[dict] = []
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        health_data: Optional[dict] = None,
        drift_data: Optional[List[dict]] = None,
        bias_data: Optional[List[dict]] = None,
        hallucination_data: Optional[dict] = None,
        incident_data: Optional[List[dict]] = None,
        lifecycle_data: Optional[dict] = None,
        telemetry_summary: Optional[List[dict]] = None,
    ) -> dict:
        """
        Generate a full governance report from provided pillar data.
        All arguments are optional — missing sections are marked 'N/A'.
        Returns the report dict and saves JSON + Markdown to disk.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        report_id = f"RPT-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # ── Executive Summary ─────────────────────────────────────────────────
        platform_status = "HEALTHY"
        critical_flags = []

        if health_data:
            if health_data.get("platform_status") == "DEGRADED":
                platform_status = "DEGRADED"
                critical_flags.append("One or more circuit breakers are OPEN.")

        if drift_data:
            drifting_models = [d["model_id"] for d in drift_data if d.get("overall_drift_detected")]
            if drifting_models:
                platform_status = "AT_RISK"
                critical_flags.append(f"Drift detected in models: {drifting_models}")

        if bias_data:
            biased_modules = [b["module_id"] for b in bias_data if not b.get("overall_fair", True)]
            if biased_modules:
                critical_flags.append(f"Bias detected in modules: {biased_modules}")

        if incident_data:
            open_critical = [i for i in incident_data if i.get("priority") == "CRITICAL" and not i.get("resolved")]
            if open_critical:
                platform_status = "CRITICAL"
                critical_flags.append(f"{len(open_critical)} CRITICAL incidents unresolved.")

        overall_risk = (
            "HIGH" if platform_status in ("CRITICAL", "DEGRADED")
            else "MEDIUM" if platform_status == "AT_RISK"
            else "LOW"
        )

        report = {
            "report_id": report_id,
            "generated_at": timestamp,
            "platform_status": platform_status,
            "overall_risk_level": overall_risk,
            "critical_flags": critical_flags,
            "sections": {
                "platform_health": health_data or {"status": "NOT_AVAILABLE"},
                "model_drift": {
                    "models_analyzed": len(drift_data) if drift_data else 0,
                    "models_drifting": len(drifting_models) if drift_data else 0,
                    "details": drift_data or [],
                },
                "fairness_audit": {
                    "modules_audited": len(bias_data) if bias_data else 0,
                    "modules_with_bias": len(biased_modules) if bias_data else 0,
                    "details": bias_data or [],
                },
                "llm_faithfulness": hallucination_data or {"status": "NOT_AVAILABLE"},
                "open_incidents": incident_data or [],
                "lifecycle": lifecycle_data or {"status": "NOT_AVAILABLE"},
                "recent_telemetry": telemetry_summary or [],
            },
            "recommendations": self._build_recommendations(
                critical_flags, drift_data, bias_data, incident_data
            ),
        }

        with self._lock:
            self._report_history.append({"report_id": report_id, "timestamp": timestamp, "risk": overall_risk})
            self._save_json(report, report_id)
            self._save_markdown(report, report_id)

        telemetry_bus.publish(
            module_id="report_generator",
            event_type="GOVERNANCE_REPORT_GENERATED",
            payload={"report_id": report_id, "platform_status": platform_status, "risk": overall_risk},
            severity="INFO",
        )

        return report

    # ── Recommendation Builder ────────────────────────────────────────────────

    @staticmethod
    def _build_recommendations(
        critical_flags: List[str],
        drift_data: Optional[List[dict]],
        bias_data: Optional[List[dict]],
        incident_data: Optional[List[dict]],
    ) -> List[dict]:
        recs = []
        priority_counter = 1

        for flag in critical_flags:
            recs.append({
                "priority": priority_counter,
                "action": flag,
                "urgency": "IMMEDIATE",
            })
            priority_counter += 1

        if drift_data:
            for d in drift_data:
                if d.get("overall_drift_detected"):
                    recs.append({
                        "priority": priority_counter,
                        "action": f"Retrain {d['model_id']} — drift detected in features: "
                                  f"{[f['feature'] for f in d.get('feature_drift', []) if f.get('drift_detected')][:3]}",
                        "urgency": "HIGH",
                    })
                    priority_counter += 1

        if bias_data:
            for b in bias_data:
                if not b.get("overall_fair", True):
                    recs.append({
                        "priority": priority_counter,
                        "action": f"Remediate bias in {b['module_id']}: {b.get('adverse_impact_flags', [])[:1]}",
                        "urgency": "HIGH",
                    })
                    priority_counter += 1

        if incident_data:
            for inc in incident_data:
                if not inc.get("resolved") and inc.get("priority") in ("CRITICAL", "HIGH"):
                    recs.append({
                        "priority": priority_counter,
                        "action": f"Resolve incident {inc['incident_id']}: {inc.get('primary_event_type')}",
                        "urgency": inc.get("priority", "MEDIUM"),
                    })
                    priority_counter += 1

        if not recs:
            recs.append({
                "priority": 1,
                "action": "All systems nominal — continue scheduled monitoring.",
                "urgency": "LOW",
            })

        return recs

    # ── File I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def _save_json(report: dict, report_id: str) -> None:
        try:
            path = REPORTS_DIR / f"{report_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass

    @staticmethod
    def _save_markdown(report: dict, report_id: str) -> None:
        try:
            path = REPORTS_DIR / f"{report_id}.md"
            lines = [
                f"# Governance Audit Report — {report_id}",
                f"**Generated:** {report['generated_at']}",
                f"**Platform Status:** `{report['platform_status']}`",
                f"**Overall Risk Level:** `{report['overall_risk_level']}`",
                "",
                "## Critical Flags",
            ]
            if report["critical_flags"]:
                for flag in report["critical_flags"]:
                    lines.append(f"- ⚠️ {flag}")
            else:
                lines.append("✅ No critical flags.")

            lines += ["", "## Recommendations"]
            for rec in report["recommendations"]:
                urgency_icon = "🔴" if rec["urgency"] == "IMMEDIATE" else "🟠" if rec["urgency"] == "HIGH" else "🟡"
                lines.append(f"**[P{rec['priority']}]** {urgency_icon} `{rec['urgency']}` — {rec['action']}")

            lines += ["", "## Platform Health"]
            health = report["sections"]["platform_health"]
            if isinstance(health, dict):
                lines.append(f"- Status: `{health.get('platform_status', 'N/A')}`")
                lines.append(f"- Total Requests: {health.get('total_requests', 'N/A')}")
                lines.append(f"- Error Rate: {health.get('overall_error_rate_pct', 'N/A')}%")
                open_cb = health.get("open_circuits", [])
                if open_cb:
                    lines.append(f"- ⚡ Open Circuit Breakers: {open_cb}")

            lines += ["", "## Model Drift"]
            drift_section = report["sections"]["model_drift"]
            lines.append(f"- Models Analyzed: {drift_section['models_analyzed']}")
            lines.append(f"- Models Drifting: {drift_section['models_drifting']}")

            lines += ["", "## Fairness Audit"]
            fair_section = report["sections"]["fairness_audit"]
            lines.append(f"- Modules Audited: {fair_section['modules_audited']}")
            lines.append(f"- Modules With Bias: {fair_section['modules_with_bias']}")

            lines += ["", "## Open Incidents"]
            incidents = report["sections"]["open_incidents"]
            if incidents:
                for inc in incidents[:5]:
                    resolved_str = "✅ Resolved" if inc.get("resolved") else "🔴 Open"
                    lines.append(
                        f"- `{inc.get('incident_id')}` [{inc.get('priority')}] "
                        f"{inc.get('primary_event_type')} — {resolved_str}"
                    )
            else:
                lines.append("✅ No open incidents.")

            lines += ["", "---", "*Auto-generated by AI Governance Control Tower*"]

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

    def get_report_history(self, n: int = 10) -> List[dict]:
        with self._lock:
            return self._report_history[-n:]

    def load_report(self, report_id: str) -> Optional[dict]:
        try:
            path = REPORTS_DIR / f"{report_id}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None


# ── Singleton export ──────────────────────────────────────────────────────────
report_generator = ReportGenerator()
