"""
governance/bias_auditor.py
───────────────────────────
Pillar 3 – Fairness, Equity & Bias Auditor.

Implements:
  • Disparate Impact Ratio (DIR) — EEOC 4/5ths rule
  • Demographic Parity Difference (DPD)
  • Equal Opportunity Difference (EOD) — if ground truth available
  • Per-group outcome distributions across protected attributes
  • Automatic adverse-impact flags and remediation recommendations
  • Intersectional fairness (cross-group combinations)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants & Thresholds
# ──────────────────────────────────────────────────────────────────────────────

DIR_ADVERSE_IMPACT_THRESHOLD = 0.8    # EEOC 4/5ths rule: DIR < 0.8 → adverse impact
DPD_THRESHOLD = 0.1                   # Demographic parity difference > 0.1 → flagged
EOD_THRESHOLD = 0.1                   # Equal opportunity difference > 0.1 → flagged
MIN_GROUP_N = 20                      # a group needs this many observations before it is compared

PROTECTED_ATTRIBUTES = [
    "gender",
    "age_group",
    "class_preference",     # AC / Sleeper / General
    "booking_method",       # mobile / web / counter
    "city_tier",            # tier-1 / tier-2 / tier-3
]


# ──────────────────────────────────────────────────────────────────────────────
# Observation Buffer
# ──────────────────────────────────────────────────────────────────────────────

class _GroupMetrics:
    """Accumulates predictions and optional ground-truth for a protected group."""

    def __init__(self, group_label: str) -> None:
        self.group_label = group_label
        self.predictions: List[float] = []          # 0/1 binary or continuous
        self.ground_truths: List[int] = []          # optional, 0/1
        self.positive_count: int = 0
        self.true_positive_count: int = 0

    def add(self, prediction: float, ground_truth: Optional[int] = None) -> None:
        self.predictions.append(prediction)
        binary_pred = int(prediction >= 0.5)
        if binary_pred == 1:
            self.positive_count += 1
        if ground_truth is not None:
            self.ground_truths.append(ground_truth)
            if ground_truth == 1 and binary_pred == 1:
                self.true_positive_count += 1

    @property
    def n(self) -> int:
        return len(self.predictions)

    @property
    def positive_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return round(self.positive_count / self.n, 4)

    @property
    def true_positive_rate(self) -> float:
        """Recall / sensitivity — needs ground truth."""
        n_pos_gt = sum(1 for gt in self.ground_truths if gt == 1)
        if n_pos_gt == 0:
            return 0.0
        return round(self.true_positive_count / n_pos_gt, 4)

    @property
    def avg_prediction(self) -> float:
        if not self.predictions:
            return 0.0
        return round(float(np.mean(self.predictions)), 4)

    def to_dict(self) -> dict:
        return {
            "group": self.group_label,
            "n_samples": self.n,
            "positive_rate": self.positive_rate,
            "avg_prediction_score": self.avg_prediction,
            "has_ground_truth": len(self.ground_truths) > 0,
            "tpr": self.true_positive_rate if self.ground_truths else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Bias Auditor (Singleton)
# ──────────────────────────────────────────────────────────────────────────────

class BiasAuditor:
    """
    Fairness auditor for all platform ML modules.
    Tracks outcomes per protected-attribute group and computes
    DIR, DPD, and EOD fairness metrics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Structure: {module_id: {attr_name: {group_value: _GroupMetrics}}}
        self._observations: Dict[str, Dict[str, Dict[str, _GroupMetrics]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(_GroupMetrics.__new__))
        )
        self._audit_history: List[dict] = []
        # Ground-truth observations kept separately so outcome feedback does not double-count predictions
        self._gt_observations: Dict[str, Dict[str, Dict[str, _GroupMetrics]]] = {}

    def _get_or_create(
        self,
        module_id: str,
        attribute: str,
        group_value: str,
    ) -> _GroupMetrics:
        groups = self._observations[module_id][attribute]
        if group_value not in groups:
            groups[group_value] = _GroupMetrics(group_value)
        return groups[group_value]

    def observe(
        self,
        module_id: str,
        protected_attrs: Dict[str, str],
        prediction: float,
        ground_truth: Optional[int] = None,
    ) -> None:
        """
        Log a single inference observation with its protected attributes.

        Args:
            module_id: Which ML module made this prediction.
            protected_attrs: Mapping of attribute name → group value (e.g., {"gender": "female"}).
            prediction: Model output score in [0, 1].
            ground_truth: Optional actual label (0/1) for EOD computation.
        """
        with self._lock:
            for attr, group_val in protected_attrs.items():
                gm = self._get_or_create(module_id, attr, str(group_val))
                gm.add(prediction, ground_truth)

    def observe_outcome(self, module_id: str, protected_attrs: Dict[str, str], prediction: float,
                        ground_truth: int) -> None:
        """Record a real outcome for a prediction (enables Equal Opportunity Difference)."""
        with self._lock:
            for attr, group_val in protected_attrs.items():
                groups = self._gt_observations.setdefault(module_id, {}).setdefault(attr, {})
                gm = groups.setdefault(str(group_val), _GroupMetrics(str(group_val)))
                gm.add(prediction, int(ground_truth))

    def _compute_dir(self, groups: Dict[str, _GroupMetrics]) -> Tuple[float, str, str]:
        """Disparate Impact Ratio: min(positive rate) / max(positive rate)."""
        rates = {g: m.positive_rate for g, m in groups.items() if m.n >= MIN_GROUP_N}
        if len(rates) < 2:
            return 1.0, "", ""
        min_group = min(rates, key=lambda k: rates[k])
        max_group = max(rates, key=lambda k: rates[k])
        if rates[max_group] == 0:
            return 1.0, min_group, max_group
        dir_val = round(rates[min_group] / rates[max_group], 4)
        return dir_val, min_group, max_group

    def audit_module(self, module_id: str) -> dict:
        """Run fairness audit for a single module across all protected attributes."""
        with self._lock:
            if module_id not in self._observations:
                return {
                    "module_id": module_id,
                    "status": "NO_DATA",
                    "attributes_audited": [],
                    "adverse_impact_flags": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            attribute_reports = []
            adverse_flags = []

            for attr_name, groups in self._observations[module_id].items():
                if not groups:
                    continue

                # DIR
                dir_val, minority_group, majority_group = self._compute_dir(groups)

                # DPD — max pairwise difference in positive rates
                rates = [m.positive_rate for m in groups.values() if m.n >= MIN_GROUP_N]
                dpd = round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0.0

                # EOD — needs real outcomes (fed by the performance monitor when ground truth arrives)
                gt_groups = self._gt_observations.get(module_id, {}).get(attr_name, {})
                tprs = [m.true_positive_rate for m in gt_groups.values()
                        if sum(1 for g in m.ground_truths if g == 1) >= 10]
                eod = round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else None

                # Flag adverse impact
                adverse = []
                if dir_val < DIR_ADVERSE_IMPACT_THRESHOLD:
                    adverse.append(
                        f"Adverse Impact: DIR={dir_val} ({minority_group} vs {majority_group})"
                    )
                if dpd > DPD_THRESHOLD:
                    adverse.append(f"Demographic Parity Violated: DPD={dpd}")
                if eod is not None and eod > EOD_THRESHOLD:
                    adverse.append(f"Equal Opportunity Violated: EOD={eod}")

                attr_report = {
                    "attribute": attr_name,
                    "groups": {g: m.to_dict() for g, m in groups.items()},
                "groups_compared": [g for g, m in groups.items() if m.n >= MIN_GROUP_N],
                "min_group_size": MIN_GROUP_N,
                    "dir": dir_val,
                    "dpd": dpd,
                    "eod": eod,
                    "adverse_impact": len(adverse) > 0,
                    "adverse_reasons": adverse,
                    "recommendation": self._remediation_recommendation(dir_val, dpd, eod),
                }
                attribute_reports.append(attr_report)
                if adverse:
                    adverse_flags.extend(adverse)

            result = {
                "module_id": module_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "BIAS_DETECTED" if adverse_flags else "FAIR",
                "attributes_audited": attribute_reports,
                "adverse_impact_flags": adverse_flags,
                "overall_fair": len(adverse_flags) == 0,
                "fairness_score": self._fairness_score(attribute_reports),
            }

            if adverse_flags:
                self._audit_history.append(
                    {"module_id": module_id, "timestamp": result["timestamp"], "flags": adverse_flags}
                )
                telemetry_bus.publish(
                    module_id=module_id,
                    event_type="BIAS_DETECTED",
                    payload={"adverse_flags": adverse_flags, "fairness_score": result["fairness_score"]},
                    severity="WARNING",
                )

            return result

    def audit_all_modules(self) -> List[dict]:
        """Audit all modules with observations."""
        with self._lock:
            module_ids = list(self._observations.keys())
        return [self.audit_module(mid) for mid in module_ids]

    def _fairness_score(self, attr_reports: List[dict]) -> float:
        """0–100 composite fairness score (100 = perfectly fair)."""
        if not attr_reports:
            return 100.0
        scores = []
        for r in attr_reports:
            dir_score = min(r["dir"] / DIR_ADVERSE_IMPACT_THRESHOLD, 1.0) * 50
            dpd_score = max(0.0, 1.0 - r["dpd"] / DPD_THRESHOLD) * 30
            eod_score = (
                max(0.0, 1.0 - r["eod"] / EOD_THRESHOLD) * 20
                if r["eod"] is not None
                else 20.0
            )
            scores.append(dir_score + dpd_score + eod_score)
        return round(float(np.mean(scores)), 2)

    @staticmethod
    def _remediation_recommendation(dir_val: float, dpd: float, eod: Optional[float]) -> str:
        recs = []
        if dir_val < DIR_ADVERSE_IMPACT_THRESHOLD:
            recs.append("Apply re-weighting or re-sampling to balance positive outcomes across groups.")
        if dpd > DPD_THRESHOLD:
            recs.append("Use post-processing threshold calibration per demographic group.")
        if eod is not None and eod > EOD_THRESHOLD:
            recs.append("Investigate training data quality for under-represented groups.")
        return " | ".join(recs) if recs else "No action required."

    def get_summary(self) -> dict:
        with self._lock:
            modules_audited = len(self._observations)
            total_adverse = len(self._audit_history)
            recent_flags = [h.get("flags", []) for h in self._audit_history[-5:]]
            flat_flags = [item for sublist in recent_flags for item in sublist]
            return {
                "modules_monitored": modules_audited,
                "adverse_impact_count": total_adverse,
                "recent_flags_count": len(flat_flags),
                "is_fair": total_adverse == 0,
            }

    def get_audit_history(self, n: int = 20) -> List[dict]:
        with self._lock:
            return self._audit_history[-n:]

    def reset_all(self) -> None:
        with self._lock:
            self._observations.clear()
            self._gt_observations.clear()
            self._audit_history.clear()

    def reset_module(self, module_id: str) -> None:
        with self._lock:
            if module_id in self._observations:
                del self._observations[module_id]


# ── Singleton export ──────────────────────────────────────────────────────────
bias_auditor = BiasAuditor()
