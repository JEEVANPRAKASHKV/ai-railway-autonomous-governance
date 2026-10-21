"""
governance/performance_monitor.py
──────────────────────────────────
Ground-truth performance monitoring — the part of governance that needs *actual outcomes*.

Input drift (drift_detector) tells you the data changed. Only outcomes tell you the model got
worse. Every governed prediction is registered here with the features the model saw; when the
real outcome arrives (actual delay, final PNR status, actual fare, analyst fraud verdict, the train
the passenger actually picked, a support satisfaction rating, a planner approval), it is joined to
the prediction and scored.

Per module this gives:
  • live quality metrics vs the training-time baseline (accuracy, MAE, MAPE, FPR/FNR, hit@3/MRR, CSAT)
  • concept-drift / quality-degradation alerts (PERFORMANCE_DEGRADED) that trigger retraining
  • labelled rows (features + outcome) that retraining adds to the training set
  • per-version metrics, used for champion-vs-canary comparison before promotion
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from governance.telemetry_bus import telemetry_bus

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry" / "model_registry.json"

WINDOW = 300            # outcomes kept per module for live metrics
MIN_OUTCOMES = 30       # below this, a metric is "insufficient data", never an alert
ALERT_COOLDOWN_S = 120

# module → how its outcomes are scored and when it counts as degraded
TASKS: Dict[str, Dict[str, Any]] = {
    "m1_ticket_predictor": {"kind": "classification", "metric": "accuracy", "higher": True,
                            "baseline": ("wl_predictor", "accuracy"), "max_drop": 0.08,
                            "label": "Final PNR status at chart preparation"},
    "m2_train_recommender": {"kind": "ranking", "metric": "hit_at_3", "higher": True,
                             "fixed_baseline": 0.70, "max_drop": 0.15,
                             "label": "Train the passenger actually chose"},
    "m3_delay_predictor": {"kind": "regression", "metric": "mae", "higher": False,
                           "baseline": ("delay_predictor", "mae_minutes"), "max_rel_rise": 0.30,
                           "label": "Actual arrival delay (minutes)"},
    "m4_travel_planner": {"kind": "approval", "metric": "approval_rate", "higher": True,
                          "fixed_baseline": 0.60, "max_drop": 0.25,
                          "label": "Passenger approved the proposed plan"},
    "m5_fare_predictor": {"kind": "regression_pct", "metric": "mape", "higher": False,
                          "baseline": ("fare_predictor", "mape_pct"), "abs_limit": 5.0,
                          "label": "Fare actually charged"},
    "m6_support_assistant": {"kind": "csat", "metric": "csat", "higher": True,
                             "fixed_baseline": 4.0, "max_drop": 0.8,
                             "label": "Passenger satisfaction rating (1–5)"},
    "m7_fraud_detector": {"kind": "binary", "metric": "fpr", "higher": False,
                          "fpr_limit": 0.12, "fnr_limit": 0.15,
                          "label": "Analyst verdict on the session"},
}
POSITIVE_FRAUD = {"bot", "suspicious"}


def _registry_baseline(key: str, metric: str) -> Optional[float]:
    try:
        reg = json.loads(REGISTRY.read_text()).get("models", {}).get(key, {})
        ver = reg.get("versions", {}).get(reg.get("current_version", "v1.0"), {})
        val = ver.get(metric)
        return float(val) if val is not None else None
    except Exception:
        return None


class PerformanceMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: "OrderedDict[str, dict]" = OrderedDict()   # audit_id → prediction record
        self._outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self._labelled: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self._last_alert: Dict[str, float] = {}
        self._listeners: Dict[str, List] = defaultdict(list)
        self._baselines: Dict[str, float] = {}
        for mid, t in TASKS.items():
            if "baseline" in t:
                b = _registry_baseline(*t["baseline"])
                if b is not None:
                    self._baselines[mid] = b
            elif "fixed_baseline" in t:
                self._baselines[mid] = t["fixed_baseline"]

    def add_listener(self, module_id: str, fn) -> None:
        """fn(prediction_record, actual) is called for every accepted outcome (e.g. online weight learning)."""
        self._listeners[module_id].append(fn)

    # ── Registration ──────────────────────────────────────────────────────────

    def register_prediction(self, audit_id: str, module_id: str, features: Optional[dict],
                            prediction: Any, version: Optional[str] = None, protected: Optional[dict] = None,
                            score: Optional[float] = None) -> None:
        if module_id not in TASKS or prediction is None:
            return
        with self._lock:
            self._pending[audit_id] = {"module_id": module_id, "features": features or {},
                                       "prediction": prediction, "version": version or "v1.0.0",
                                       "protected": protected or {}, "score": score, "ts": time.time()}
            while len(self._pending) > 50_000:
                self._pending.popitem(last=False)

    def get_prediction(self, audit_id: str) -> Optional[dict]:
        with self._lock:
            return self._pending.get(audit_id)

    def pending_ids(self, module_id: str, limit: int = 200) -> List[str]:
        with self._lock:
            ids = [a for a, r in reversed(self._pending.items())
                   if r["module_id"] == module_id and not r.get("scored")]
        return ids[:limit]

    # ── Outcomes ──────────────────────────────────────────────────────────────

    def record_outcome(self, audit_id: str, actual: Any, source: str = "passenger") -> dict:
        with self._lock:
            rec = self._pending.get(audit_id)
            if rec is None:
                return {"accepted": False, "error": "Unknown or expired prediction id."}
            if rec.get("scored"):
                return {"accepted": False, "error": "An outcome was already recorded for this prediction."}
            mid = rec["module_id"]
            sample = self._score(mid, rec["prediction"], actual)
            if sample is None:
                return {"accepted": False, "error": "Outcome is not valid for this module."}
            sample.update({"version": rec["version"], "source": source, "ts": time.time()})
            rec["scored"] = True
            self._outcomes[mid].append(sample)
            if rec["features"] and TASKS[mid]["kind"] in ("classification", "regression", "regression_pct", "binary"):
                self._labelled[mid].append((dict(rec["features"]), actual))
            protected, score = rec.get("protected") or {}, rec.get("score")
        for fn in self._listeners.get(mid, []):
            try:
                fn(rec, actual)
            except Exception:
                pass
        # Ground truth also feeds the fairness auditor (Equal Opportunity Difference)
        if protected and score is not None and TASKS[mid]["kind"] in ("classification", "binary"):
            try:
                from governance.bias_auditor import bias_auditor
                gt = int(str(actual) == "Confirmed") if mid == "m1_ticket_predictor" else int(str(actual).lower() in POSITIVE_FRAUD)
                bias_auditor.observe_outcome(mid, protected, float(score), gt)
            except Exception:
                pass
        return {"accepted": True, "module_id": mid, "sample": sample}

    @staticmethod
    def _score(mid: str, pred: Any, actual: Any) -> Optional[dict]:
        kind = TASKS[mid]["kind"]
        try:
            if kind == "classification":
                return {"correct": float(str(pred["label"]) == str(actual)),
                        "brier": (float(pred.get("p_confirm", 0.5)) - float(str(actual) == "Confirmed")) ** 2}
            if kind == "regression":
                return {"abs_err": abs(float(pred) - float(actual)), "err": float(pred) - float(actual)}
            if kind == "regression_pct":
                a = float(actual)
                return {"ape": abs(float(pred) - a) / max(a, 1.0) * 100.0}
            if kind == "binary":
                p_pos = str(pred).lower() in POSITIVE_FRAUD
                a_pos = str(actual).lower() in POSITIVE_FRAUD
                return {"tp": float(p_pos and a_pos), "fp": float(p_pos and not a_pos),
                        "tn": float(not p_pos and not a_pos), "fn": float(not p_pos and a_pos)}
            if kind == "ranking":
                shown = [str(x) for x in ((pred.get("shown") if isinstance(pred, dict) else pred) or [])]
                rank = shown.index(str(actual)) + 1 if str(actual) in shown else None
                return {"hit3": float(rank is not None and rank <= 3), "rr": (1.0 / rank) if rank else 0.0}
            if kind == "csat":
                r = float(actual)
                if not 1 <= r <= 5:
                    return None
                return {"rating": r}
            if kind == "approval":
                return {"approved": float(bool(actual))}
        except Exception:
            return None
        return None

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(mid: str, samples: List[dict]) -> Dict[str, Optional[float]]:
        n = len(samples)
        if not n:
            return {}
        kind = TASKS[mid]["kind"]
        mean = lambda k: sum(s[k] for s in samples) / n
        if kind == "classification":
            return {"accuracy": round(mean("correct"), 4), "brier": round(mean("brier"), 4)}
        if kind == "regression":
            return {"mae": round(mean("abs_err"), 2), "bias": round(mean("err"), 2)}
        if kind == "regression_pct":
            return {"mape": round(mean("ape"), 2)}
        if kind == "binary":
            tp, fp, tn, fn = (sum(s[k] for s in samples) for k in ("tp", "fp", "tn", "fn"))
            return {"fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
                    "fnr": round(fn / (fn + tp), 4) if fn + tp else 0.0,
                    "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                    "false_positives": int(fp), "false_negatives": int(fn)}
        if kind == "ranking":
            return {"hit_at_3": round(mean("hit3"), 4), "mrr": round(mean("rr"), 4)}
        if kind == "csat":
            return {"csat": round(mean("rating"), 2),
                    "satisfied_pct": round(sum(1 for s in samples if s["rating"] >= 4) / n * 100, 1)}
        if kind == "approval":
            return {"approval_rate": round(mean("approved"), 4)}
        return {}

    def _degraded(self, mid: str, metrics: dict, n: int) -> Tuple[bool, str]:
        if n < MIN_OUTCOMES:
            return False, f"insufficient data ({n}/{MIN_OUTCOMES} outcomes)"
        t = TASKS[mid]
        base = self._baselines.get(mid)
        if t["kind"] == "binary":
            fpr, fnr = metrics.get("fpr", 0), metrics.get("fnr", 0)
            if fpr > t["fpr_limit"]:
                return True, f"false-positive rate {fpr:.0%} > {t['fpr_limit']:.0%}"
            if fnr > t["fnr_limit"]:
                return True, f"false-negative rate {fnr:.0%} > {t['fnr_limit']:.0%}"
            return False, "within limits"
        val = metrics.get(t["metric"])
        if val is None or base is None:
            return False, "no baseline"
        if "max_drop" in t and val < base - t["max_drop"]:
            return True, f"{t['metric']} {val:.3g} vs baseline {base:.3g} (allowed drop {t['max_drop']})"
        if "max_rel_rise" in t and val > base * (1 + t["max_rel_rise"]):
            return True, f"{t['metric']} {val:.3g} vs baseline {base:.3g} (+{(val / base - 1):.0%})"
        if "abs_limit" in t and val > max(t["abs_limit"], base * 2):
            return True, f"{t['metric']} {val:.3g}% exceeds limit {max(t['abs_limit'], base * 2):.3g}%"
        return False, "within limits"

    def module_report(self, mid: str) -> dict:
        with self._lock:
            samples = list(self._outcomes.get(mid, []))
            pending = sum(1 for r in self._pending.values() if r["module_id"] == mid and not r.get("scored"))
            labelled = len(self._labelled.get(mid, []))
        metrics = self._aggregate(mid, samples)
        degraded, reason = self._degraded(mid, metrics, len(samples))
        by_version: Dict[str, dict] = {}
        for v in sorted({s["version"] for s in samples}):
            vs = [s for s in samples if s["version"] == v]
            by_version[v] = {"n": len(vs), **self._aggregate(mid, vs)}
        t = TASKS[mid]
        return {"module_id": mid, "outcome": t["label"], "primary_metric": t["metric"],
                "higher_is_better": t["higher"], "baseline": self._baselines.get(mid),
                "n_outcomes": len(samples), "awaiting_outcome": pending, "labelled_rows": labelled,
                "metrics": metrics, "degraded": degraded, "status_reason": reason,
                "by_version": by_version,
                "sources": sorted({s.get("source", "?") for s in samples})}

    def all_reports(self) -> List[dict]:
        return [self.module_report(mid) for mid in TASKS]

    def check_degradation(self) -> List[dict]:
        """Publish PERFORMANCE_DEGRADED (concept drift) for any module whose live quality fell."""
        out = []
        now = time.time()
        for rep in self.all_reports():
            if not rep["degraded"]:
                continue
            out.append(rep)
            mid = rep["module_id"]
            if now - self._last_alert.get(mid, 0) >= ALERT_COOLDOWN_S:
                self._last_alert[mid] = now
                telemetry_bus.publish(
                    module_id=mid, event_type="PERFORMANCE_DEGRADED",
                    payload={"reason": rep["status_reason"], "metrics": rep["metrics"],
                             "baseline": rep["baseline"], "n_outcomes": rep["n_outcomes"]},
                    severity="CRITICAL",
                )
        return out

    def compare_versions(self, mid: str, champion: str, challenger: str, min_n: int = 20) -> dict:
        """Canary analysis: is the challenger at least as good as the champion on live outcomes?"""
        rep = self.module_report(mid)
        t = TASKS[mid]
        key = t["metric"]
        c, h = rep["by_version"].get(champion, {}), rep["by_version"].get(challenger, {})
        if c.get("n", 0) < min_n or h.get("n", 0) < min_n:
            return {"decision": "insufficient_data", "champion": c, "challenger": h, "metric": key, "min_n": min_n}
        cv, hv = c.get(key), h.get(key)
        if cv is None or hv is None:
            return {"decision": "insufficient_data", "champion": c, "challenger": h, "metric": key}
        better = hv >= cv if t["higher"] else hv <= cv
        return {"decision": "challenger_ok" if better else "challenger_worse", "metric": key,
                "champion": c, "challenger": h}

    def labelled_rows(self, mid: str) -> List[Tuple[dict, Any]]:
        with self._lock:
            return list(self._labelled.get(mid, []))

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._outcomes.clear()
            self._labelled.clear()
            self._last_alert.clear()


performance_monitor = PerformanceMonitor()
