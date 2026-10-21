"""
governance/lifecycle_manager.py
─────────────────────────────────
Pillar 6 – Model Lifecycle Manager.

Implements:
  • Model version registry (every trained version with metadata, persisted)
  • Real retraining: the project's own training pipeline, a fixed hold-out set shared by every version,
    plus labelled production outcomes when concept drift was the trigger
  • Champion / challenger evaluation with a minimum-gain rule (no promotion on noise)
  • Canary deployment: the challenger really serves N% of live traffic, scored on live outcomes
  • Promotion only after human approval (HITL); stopping a canary or rolling back is immediate
  • Rollback that restores the previous version's artifact into the live service (hot swap)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import pickle
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = ROOT / "models" / "registry" / "model_registry.json"      # written by the training pipeline
LIFECYCLE_STATE_PATH = ROOT / "models" / "registry" / "lifecycle_state.json"    # written by this manager

# model_id -> (training function, primary metric, higher_is_better, registry key, service attr holding the bundle)
TRAINABLE = {
    "m1_ticket_predictor": ("train_wl_predictor", "accuracy", True, "wl_predictor", "_wl_bundle"),
    "m3_delay_predictor": ("train_delay_predictor", "mae_minutes", False, "delay_predictor", "_bundle"),
    "m5_fare_predictor": ("train_fare_predictor", "mape_pct", False, "fare_predictor", "_bundle"),
    "m7_fraud_detector": ("train_fraud_detector", "f1_weighted", True, "fraud_detector", "_bundle"),
}
SERVICE_SINGLETONS = {
    "m1_ticket_predictor": ("modules.m1_ticket_predictor.service", "ticket_predictor_service"),
    "m3_delay_predictor": ("modules.m3_delay_predictor.service", "delay_predictor_service"),
    "m5_fare_predictor": ("modules.m5_fare_predictor.service", "fare_predictor_service"),
    "m7_fraud_detector": ("modules.m7_fraud_detector.service", "fraud_detector_service"),
}
RULE_BASED = {
    "m2_train_recommender": "Persona-weighted multi-objective ranker (rule-based, fairness-audited)",
    "m4_travel_planner": "Agentic planner orchestrating M1/M3/M5 tools (no trained weights)",
    "m6_support_assistant": "TF-IDF RAG over IRCTC policy text (no trained weights)",
}
RETRAIN_COOLDOWN_S = 600
CANARY_DEFAULT_PCT = 10   # 10% of live traffic goes to the challenger
MIN_REL_GAIN = 0.01       # hold-out metric must improve by ≥1% (relative) to count as "better"
MAX_HOLDOUT_REGRESSION = 0.02   # with production outcomes, hold-out may not get worse by more than 2%
MIN_RECENT_GAIN = 0.02    # …and the challenger must beat the champion on recent outcomes by ≥2%
# …and the gain must also be material in absolute terms (relative gains on tiny errors are noise)
MIN_ABS_GAIN = {"accuracy": 0.01, "f1_weighted": 0.01, "mae_minutes": 0.5, "mape_pct": 0.2}
_TRAIN_LOCK = threading.Lock()  # the training module uses module-level settings: one training run at a time
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model Version Data Model
# ──────────────────────────────────────────────────────────────────────────────

class ModelVersion:
    def __init__(
        self,
        model_id: str,
        version: str,
        stage: str,
        artifact_path: str,
        metrics: Optional[dict] = None,
        created_by: str = "system",
    ) -> None:
        self.model_id = model_id
        self.version = version
        self.stage = stage          # "staging" | "production" | "archived" | "challenger"
        self.artifact_path = artifact_path
        self.metrics = metrics or {}
        self.created_by = created_by
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.promoted_at: Optional[str] = None
        self.rolled_back: bool = False
        self.canary_pct: int = 0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage,
            "artifact_path": self.artifact_path,
            "metrics": self.metrics,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "rolled_back": self.rolled_back,
            "canary_pct": self.canary_pct,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle Manager (Singleton)
# ──────────────────────────────────────────────────────────────────────────────

class LifecycleManager:
    """
    Manages the full lifecycle of all platform ML models:
    training → staging → canary → production → archived.
    Supports automatic rollback on governance signal.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {model_id: {version_id: ModelVersion}}
        self._registry: Dict[str, Dict[str, ModelVersion]] = {}
        # {model_id: version_id}  — current production version
        self._production: Dict[str, str] = {}
        # {model_id: version_id}  — current challenger (for canary/A-B)
        self._challenger: Dict[str, str] = {}
        self._retraining_queue: List[str] = []
        self._bundle_cache: Dict[str, object] = {}      # "model_id:version" → loaded challenger bundle
        self._last_retrain: Dict[str, float] = {}
        self._retraining_now: set = set()
        self._retrain_history: List[dict] = []
        self._promoted_at: Dict[str, float] = {}
        self._load_registry()
        self._restore_state()

    def _load_registry(self) -> None:
        """Bootstrap the registry from the REAL training registry (models/registry/model_registry.json)."""
        real: dict = {}
        try:
            with open(MODEL_REGISTRY_PATH) as f:
                real = json.load(f).get("models", {})
        except Exception:
            real = {}

        def _metrics(key: str, prefix: str = "") -> dict:
            entry = real.get(key, {})
            ver = entry.get("versions", {}).get(entry.get("current_version", "v1.0"), {})
            return {prefix + k: v for k, v in ver.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}

        for model_id, (_, _, _, key, _) in TRAINABLE.items():
            metrics = _metrics(key)
            if model_id == "m1_ticket_predictor":
                metrics.update({"berth_" + k: v for k, v in _metrics("berth_estimator").items() if k == "accuracy"})
            mv = ModelVersion(model_id, "v1.0.0", "production", f"models/artifacts/{key}_v1.0.pkl",
                              metrics=metrics, created_by="training_pipeline")
            mv.promoted_at = real.get(key, {}).get("versions", {}).get("v1.0", {}).get("train_date")
            mv.canary_pct = 100
            self._registry[model_id] = {"v1.0.0": mv}
            self._production[model_id] = "v1.0.0"

        for model_id, desc in RULE_BASED.items():
            mv = ModelVersion(model_id, "v1.0.0", "production", f"(rule-based) {desc}",
                              metrics={}, created_by="platform_release")
            mv.canary_pct = 100
            self._registry[model_id] = {"v1.0.0": mv}
            self._production[model_id] = "v1.0.0"

    def _restore_state(self) -> None:
        """Re-apply promotions / challengers persisted by a previous run (disaster recovery)."""
        try:
            data = json.loads(LIFECYCLE_STATE_PATH.read_text())
        except Exception:
            return
        for mid, entry in data.items():
            if mid not in self._registry:
                continue
            for vid, d in (entry.get("versions") or {}).items():
                if vid in self._registry[mid]:
                    continue
                path = ROOT / str(d.get("artifact_path", ""))
                if str(d.get("artifact_path", "")).endswith(".pkl") and not path.exists():
                    continue
                mv = ModelVersion(mid, vid, d.get("stage", "archived"), d.get("artifact_path", ""),
                                  metrics=d.get("metrics") or {}, created_by=d.get("created_by", "restored"))
                mv.created_at = d.get("created_at", mv.created_at)
                mv.promoted_at = d.get("promoted_at")
                mv.rolled_back = bool(d.get("rolled_back"))
                mv.canary_pct = int(d.get("canary_pct") or 0)
                self._registry[mid][vid] = mv
            prod = entry.get("production")
            if prod and prod in self._registry[mid] and prod != self._production.get(mid):
                for v in self._registry[mid].values():
                    if v.stage == "production" and v.version != prod:
                        v.stage = "archived"
                self._registry[mid][prod].stage = "production"
                self._production[mid] = prod
                self._serve_version(mid, prod)
            ch = entry.get("challenger")
            if ch and ch in self._registry[mid]:
                self._challenger[mid] = ch

    def _save_registry(self) -> None:
        """Persist lifecycle state (kept separate from the training pipeline's registry)."""
        try:
            LIFECYCLE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {mid: {"production": self._production.get(mid),
                          "challenger": self._challenger.get(mid),
                          "versions": {v: mv.to_dict() for v, mv in vers.items()}}
                    for mid, vers in self._registry.items()}
            with open(LIFECYCLE_STATE_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    # ── Serving hot-swap ──────────────────────────────────────────────────────

    def _serve_version(self, model_id: str, version_id: str) -> bool:
        """Load a version's artifact (and its SHAP explainer) into the live service. No restart needed."""
        spec = TRAINABLE.get(model_id)
        mv = self._registry.get(model_id, {}).get(version_id)
        if not spec or not mv or not str(mv.artifact_path).endswith(".pkl"):
            return False
        try:
            mod_name, attr = SERVICE_SINGLETONS[model_id]
            service = getattr(importlib.import_module(mod_name), attr)
            path = ROOT / mv.artifact_path
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            setattr(service, spec[4], bundle)
            shap_path = path.parent / "shap_explainers" / f"shap_{spec[3]}.pkl"
            if not shap_path.exists():
                shap_path = ROOT / "models" / "artifacts" / "shap_explainers" / f"shap_{spec[3]}.pkl"
            if shap_path.exists() and hasattr(service, "_explainer"):
                with open(shap_path, "rb") as f:
                    service._explainer = pickle.load(f)
            return True
        except Exception:
            return False

    def _load_bundle(self, model_id: str, version_id: str):
        key = f"{model_id}:{version_id}"
        if key in self._bundle_cache:
            return self._bundle_cache[key]
        mv = self._registry.get(model_id, {}).get(version_id)
        if not mv or not str(mv.artifact_path).endswith(".pkl"):
            return None
        try:
            with open(ROOT / mv.artifact_path, "rb") as f:
                bundle = pickle.load(f)
            self._bundle_cache[key] = bundle
            return bundle
        except Exception:
            return None

    def serving_bundle(self, model_id: str, champion_bundle):
        """
        Canary routing. Returns (bundle, version) for ONE request: the challenger for canary_pct% of
        traffic, otherwise the production champion. Every prediction is tagged with the version that
        served it, so the performance monitor can compare the two on real outcomes.
        """
        prod = self._production.get(model_id, "v1.0.0")
        ch_id = self._challenger.get(model_id)
        if ch_id:
            mv = self._registry.get(model_id, {}).get(ch_id)
            if mv and mv.canary_pct > 0 and random.random() * 100 < mv.canary_pct:
                b = self._load_bundle(model_id, ch_id)
                if b is not None:
                    return b, ch_id
        return champion_bundle, prod

    def production_version(self, model_id: str) -> str:
        return self._production.get(model_id, "v1.0.0")

    # ── Registration ──────────────────────────────────────────────────────────

    def register_version(
        self,
        model_id: str,
        artifact_path: str,
        metrics: Optional[dict] = None,
        stage: str = "staging",
        created_by: str = "training_pipeline",
    ) -> str:
        """Register a newly trained model version. Returns version_id."""
        with self._lock:
            existing = self._registry.get(model_id, {})
            version_num = len(existing) + 1
            version_id = f"v{version_num}.0.0"
            mv = ModelVersion(
                model_id=model_id,
                version=version_id,
                stage=stage,
                artifact_path=artifact_path,
                metrics=metrics,
                created_by=created_by,
            )
            self._registry.setdefault(model_id, {})[version_id] = mv
            telemetry_bus.publish(
                module_id=model_id,
                event_type="MODEL_VERSION_REGISTERED",
                payload={"version": version_id, "stage": stage, "artifact_path": artifact_path},
                severity="INFO",
            )
            return version_id

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote_to_canary(self, model_id: str, version_id: str, canary_pct: int = CANARY_DEFAULT_PCT) -> dict:
        """Promote a staging version to canary (challenger) with N% traffic split."""
        with self._lock:
            mv = self._registry.get(model_id, {}).get(version_id)
            if not mv:
                return {"success": False, "error": f"Version {version_id} not found."}
            mv.stage = "challenger"
            mv.canary_pct = canary_pct
            self._challenger[model_id] = version_id
            telemetry_bus.publish(
                module_id=model_id,
                event_type="MODEL_CANARY_STARTED",
                payload={"version": version_id, "canary_pct": canary_pct},
                severity="INFO",
            )
            return {"success": True, "model_id": model_id, "version": version_id, "canary_pct": canary_pct}

    def promote_to_production(self, model_id: str, version_id: str) -> dict:
        """Promote a canary/staging version to full production, retiring the current champion."""
        with self._lock:
            mv = self._registry.get(model_id, {}).get(version_id)
            if not mv:
                return {"success": False, "error": f"Version {version_id} not found."}

            # Archive current production
            old_prod_id = self._production.get(model_id)
            if old_prod_id and old_prod_id in self._registry.get(model_id, {}):
                self._registry[model_id][old_prod_id].stage = "archived"

            mv.stage = "production"
            mv.promoted_at = datetime.now(timezone.utc).isoformat()
            mv.canary_pct = 100
            self._production[model_id] = version_id
            self._challenger.pop(model_id, None)
            self._serve_version(model_id, version_id)
            self._promoted_at[model_id] = time.time()
            self._save_registry()
            telemetry_bus.publish(
                module_id=model_id,
                event_type="MODEL_PROMOTED",
                payload={"version": version_id, "previous_version": old_prod_id},
                severity="INFO",
            )
            return {
                "success": True,
                "model_id": model_id,
                "promoted_version": version_id,
                "archived_version": old_prod_id,
            }

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback(self, model_id: str, reason: str = "governance_signal") -> dict:
        """
        Roll back model_id to the previous stable production version.
        Called automatically when ROLLBACK_TRIGGER_EVENTS are received.
        """
        with self._lock:
            versions = self._registry.get(model_id, {})
            # Find the most recent archived version to restore
            archived = [
                mv for mv in versions.values()
                if mv.stage == "archived" and not mv.rolled_back
            ]
            archived.sort(key=lambda mv: mv.created_at, reverse=True)

            if not archived:
                return {"success": False, "error": "No archived version available for rollback."}

            # Archive current production
            curr_prod_id = self._production.get(model_id)
            if curr_prod_id and curr_prod_id in versions:
                versions[curr_prod_id].stage = "archived"
                versions[curr_prod_id].rolled_back = True

            # Restore previous
            restore_mv = archived[0]
            restore_mv.stage = "production"
            restore_mv.promoted_at = datetime.now(timezone.utc).isoformat()
            self._production[model_id] = restore_mv.version
            self._challenger.pop(model_id, None)
            self._serve_version(model_id, restore_mv.version)
            self._save_registry()

            telemetry_bus.publish(
                module_id=model_id,
                event_type="MODEL_ROLLED_BACK",
                payload={
                    "restored_version": restore_mv.version,
                    "reason": reason,
                    "rolled_back_version": curr_prod_id,
                },
                severity="WARNING",
            )
            return {
                "success": True,
                "model_id": model_id,
                "restored_version": restore_mv.version,
                "reason": reason,
            }

    def stop_canary(self, model_id: str, reason: str = "canary_underperformed") -> dict:
        """Withdraw the challenger from live traffic (risk-reducing, so no approval needed)."""
        with self._lock:
            vid = self._challenger.pop(model_id, None)
            if not vid:
                return {"success": False, "error": "No active canary."}
            mv = self._registry[model_id][vid]
            mv.stage, mv.canary_pct = "rejected", 0
            self._save_registry()
        telemetry_bus.publish(module_id=model_id, event_type="CANARY_STOPPED",
                              payload={"version": vid, "reason": reason}, severity="WARNING")
        return {"success": True, "model_id": model_id, "stopped_version": vid, "reason": reason}

    def recently_promoted(self, model_id: str, within_s: float = 3600) -> bool:
        return time.time() - self._promoted_at.get(model_id, 0) < within_s

    # ── Retraining Queue ──────────────────────────────────────────────────────

    def queue_retraining(self, model_id: str) -> None:
        """Queue a model for retraining (triggered by drift/SLA signal)."""
        with self._lock:
            if model_id not in self._retraining_queue:
                self._retraining_queue.append(model_id)
                telemetry_bus.publish(
                    module_id=model_id,
                    event_type="RETRAINING_QUEUED",
                    payload={"model_id": model_id},
                    severity="INFO",
                )

    def clear_retraining_queue(self) -> None:
        with self._lock:
            self._retraining_queue.clear()

    def get_retraining_queue(self) -> List[str]:
        with self._lock:
            return list(self._retraining_queue)

    def dequeue_retraining(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._retraining_queue:
                self._retraining_queue.remove(model_id)

    def execute_retraining(self, model_id: str, reason: str = "autonomous_drift_response") -> dict:
        """Autonomous retraining (called by the supervisor on drift). Runs in the background with a cooldown."""
        if model_id not in TRAINABLE:
            return {"success": False, "skipped": "not_trainable", "model_id": model_id}
        with self._lock:
            last = self._last_retrain.get(model_id, 0)
            if time.time() - last < RETRAIN_COOLDOWN_S or model_id in self._retraining_now:
                return {"success": False, "skipped": "cooldown_or_in_progress", "model_id": model_id}
            self._last_retrain[model_id] = time.time()
        threading.Thread(target=self.retrain, args=(model_id, reason),
                         daemon=True, name=f"retrain-{model_id}").start()
        return {"success": True, "started": True, "model_id": model_id,
                "note": "Retraining started in background; challenger will be evaluated against the champion."}

    # ── Evaluation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _score_rows(model_id: str, bundle, rows: List) -> Optional[float]:
        """Primary metric of `bundle` on labelled production rows [(features, actual), …]."""
        if not bundle or not rows:
            return None
        try:
            import numpy as np
            import pandas as pd
            cols = bundle.get("feature_cols")
            rows = [r for r in rows if all(c in r[0] for c in cols)]
            if not rows:
                return None
            X = pd.DataFrame([[float(r[0][c]) for c in cols] for r in rows], columns=cols)
            model = bundle["model"]
            if model_id == "m1_ticket_predictor":
                pred = bundle["target_encoder"].inverse_transform(model.predict(X).astype(int))
                return float(np.mean([str(p) == str(r[1]) for p, r in zip(pred, rows)]))
            if model_id == "m7_fraud_detector":
                pred = bundle["label_encoder"].inverse_transform(model.predict(X).astype(int))
                return float(np.mean([str(p) == str(r[1]) for p, r in zip(pred, rows)]))
            pred = model.predict(X)
            y = np.array([float(r[1]) for r in rows])
            if model_id == "m3_delay_predictor":
                return float(np.mean(np.abs(np.clip(pred, 0, None) - y)))
            if model_id == "m5_fare_predictor":
                return float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0)) * 100)
        except Exception:
            return None
        return None

    def retrain(self, model_id: str, reason: str = "operator_request") -> dict:
        """
        REAL retraining with the project's training pipeline.

        • The hold-out split is fixed (SPLIT_SEED), so the challenger is scored on exactly the same test
          set as the champion — an apples-to-apples comparison.
        • If labelled production outcomes exist (concept drift), 70% of them are added to training and
          30% are kept aside; both models are scored on that recent slice too.
        • The challenger is accepted only on a meaningful gain (MIN_REL_GAIN on hold-out, or MIN_RECENT_GAIN
          on recent outcomes without regressing on hold-out). Accepted challengers start a 10% canary;
          promotion to 100% needs human approval.
        """
        spec = TRAINABLE.get(model_id)
        if not spec:
            return {"success": True, "model_id": model_id, "outcome": "not_applicable",
                    "detail": RULE_BASED.get(model_id, "This module has no trained weights; it is governed through "
                                                        "parameter, policy and prompt review.")}
        with self._lock:
            if model_id in self._retraining_now:
                return {"success": False, "error": "Retraining already in progress for this model."}
            self._retraining_now.add(model_id)
        fn_name, metric, higher, key, _ = spec
        try:
            from governance.performance_monitor import performance_monitor
            labelled = performance_monitor.labelled_rows(model_id)
            random.Random(7).shuffle(labelled)
            n_eval = int(len(labelled) * 0.3) if len(labelled) >= 20 else 0
            eval_rows, train_rows = labelled[:n_eval], labelled[n_eval:]

            with _TRAIN_LOCK:   # serialises concurrent retrains (fixes the shared-global race)
                trainer_path = ROOT / "models" / "train_all_models.py"
                if getattr(self, "_trainer", None) is None:
                    sp = importlib.util.spec_from_file_location("rb_train_all_models", trainer_path)
                    self._trainer = importlib.util.module_from_spec(sp)
                    sp.loader.exec_module(self._trainer)
                T = self._trainer
                ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                staging = ROOT / "models" / "artifacts" / "challengers" / f"{model_id}_{ts}"
                (staging / "shap_explainers").mkdir(parents=True, exist_ok=True)
                old_dir, old_seed = T.ARTIFACTS_DIR, T.RANDOM_SEED
                seed = 1000 + (int(time.time()) % 9973)
                t0 = time.time()
                try:
                    T.ARTIFACTS_DIR, T.RANDOM_SEED = str(staging), seed
                    T.EXTRA_TRAINING_ROWS[key] = train_rows
                    new_metrics = getattr(T, fn_name)()
                    used_rows = T.LAST_EXTRA_ROWS_USED.get(key, 0)
                finally:
                    T.ARTIFACTS_DIR, T.RANDOM_SEED = old_dir, old_seed
                    T.EXTRA_TRAINING_ROWS.pop(key, None)
                train_s = round(time.time() - t0, 1)

            pkl = next((p.name for p in staging.glob("*.pkl")), None)
            if not pkl:
                raise RuntimeError("training produced no artifact")
            with open(staging / pkl, "rb") as f:
                challenger_bundle = pickle.load(f)

            with self._lock:
                versions = self._registry[model_id]
                champ_id = self._production.get(model_id)
                champ_mv = versions[champ_id]
                champ = champ_mv.metrics.get(metric)
            new_val = float(new_metrics.get(metric))
            champ_bundle = self._load_bundle(model_id, champ_id)
            if champ_bundle is None:
                try:
                    mod_name, attr = SERVICE_SINGLETONS[model_id]
                    champ_bundle = getattr(getattr(importlib.import_module(mod_name), attr), spec[4], None)
                except Exception:
                    champ_bundle = None

            def rel_gain(new, old):
                if old in (None, 0):
                    return 0.0
                return (new - old) / abs(old) if higher else (old - new) / abs(old)

            holdout_gain = rel_gain(new_val, champ) if champ is not None else 0.0
            recent = None
            if eval_rows:
                c_recent = self._score_rows(model_id, champ_bundle, eval_rows)
                h_recent = self._score_rows(model_id, challenger_bundle, eval_rows)
                recent_metric_higher = model_id in ("m1_ticket_predictor", "m7_fraud_detector")
                if c_recent is not None and h_recent is not None:
                    g = ((h_recent - c_recent) / max(abs(c_recent), 1e-9)) if recent_metric_higher \
                        else ((c_recent - h_recent) / max(abs(c_recent), 1e-9))
                    recent = {"n": len(eval_rows), "metric": "accuracy" if recent_metric_higher else metric,
                              "champion": round(c_recent, 4), "challenger": round(h_recent, 4), "rel_gain": round(g, 4)}

            abs_floor = MIN_ABS_GAIN.get(metric, 0.0)
            holdout_abs = abs(new_val - champ) if champ is not None else 0.0
            recent_abs = abs(recent["champion"] - recent["challenger"]) if recent else 0.0
            if holdout_gain >= MIN_REL_GAIN and holdout_abs >= abs_floor:
                improved, why = True, f"hold-out {metric} improved {holdout_gain:+.1%} (≥ {MIN_REL_GAIN:.0%} required)"
            elif (recent and recent["rel_gain"] >= MIN_RECENT_GAIN and recent_abs >= abs_floor
                  and holdout_gain >= -MAX_HOLDOUT_REGRESSION):
                improved, why = True, (f"recent outcomes improved {recent['rel_gain']:+.1%} with hold-out change "
                                       f"{holdout_gain:+.1%} (within -{MAX_HOLDOUT_REGRESSION:.0%})")
            else:
                improved = False
                why = (f"no meaningful gain: hold-out {holdout_gain:+.2%}"
                       + (f", recent {recent['rel_gain']:+.1%}" if recent else ", no labelled production outcomes yet"))

            with self._lock:
                version_id = f"v{len(versions) + 1}.0.0"
                mv = ModelVersion(model_id, version_id, "challenger" if improved else "rejected",
                                  f"models/artifacts/challengers/{staging.name}/{pkl}",
                                  metrics={k: round(float(v), 4) for k, v in new_metrics.items()
                                           if isinstance(v, (int, float)) and not isinstance(v, bool)},
                                  created_by=f"retraining ({reason})")
                mv.metrics["production_rows_used"] = used_rows
                versions[version_id] = mv
                if improved:
                    mv.canary_pct = CANARY_DEFAULT_PCT
                    self._challenger[model_id] = version_id
                    self._bundle_cache[f"{model_id}:{version_id}"] = challenger_bundle
                if model_id in self._retraining_queue:
                    self._retraining_queue.remove(model_id)
                self._save_registry()
                record = {"model_id": model_id, "version": version_id, "reason": reason, "accepted": improved,
                          "why": why, "metric": metric, "champion": champ, "challenger": round(new_val, 4),
                          "recent": recent, "production_rows_used": used_rows, "train_seconds": train_s,
                          "at": datetime.now(timezone.utc).isoformat()}
                self._retrain_history.append(record)
                self._retrain_history = self._retrain_history[-50:]

            outcome = "challenger_better" if improved else "champion_retained"
            telemetry_bus.publish(
                module_id=model_id, event_type="MODEL_RETRAINED",
                payload={"new_version": version_id, "outcome": outcome, **{k: record[k] for k in ("why", "metric", "champion", "challenger", "production_rows_used")}},
                severity="INFO",
            )
            return {
                "success": True, "model_id": model_id, "new_version": version_id, "outcome": outcome,
                "decision": why,
                "comparison": {metric: f"champion {champ} → challenger {round(new_val, 4)} "
                                       f"({'higher' if higher else 'lower'} is better, same hold-out set)"},
                "recent_outcomes": recent, "production_rows_used": used_rows,
                "metrics": mv.metrics, "train_seconds": train_s,
                "next_step": (f"Challenger now serves {CANARY_DEFAULT_PCT}% of live traffic as a canary. Promotion to "
                              "100% needs a human approval in the Control Tower."
                              if improved else "Champion retained — challenger archived for audit."),
            }
        except Exception as exc:
            logger.exception(f"[Lifecycle] retraining {model_id} failed")
            telemetry_bus.publish(module_id=model_id, event_type="RETRAIN_FAILED",
                                  payload={"error": f"{type(exc).__name__}: {exc}", "reason": reason}, severity="WARNING")
            with self._lock:
                self._retrain_history.append({"model_id": model_id, "accepted": False, "failed": True,
                                              "why": f"{type(exc).__name__}: {exc}", "reason": reason,
                                              "at": datetime.now(timezone.utc).isoformat()})
            return {"success": False, "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            with self._lock:
                self._retraining_now.discard(model_id)

    def retrain_history(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._retrain_history)[-n:]

    def promote_challenger(self, model_id: str) -> dict:
        """Promote the registered challenger to production and hot-swap it into serving."""
        vid = self._challenger.get(model_id)
        if not vid:
            return {"success": False, "error": "No challenger registered for this model."}
        return self.promote_to_production(model_id, vid)

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_model_status(self, model_id: str) -> dict:
        with self._lock:
            versions = self._registry.get(model_id, {})
            prod_id = self._production.get(model_id)
            challenger_id = self._challenger.get(model_id)
            return {
                "model_id": model_id,
                "production_version": prod_id,
                "challenger_version": challenger_id,
                "total_versions": len(versions),
                "versions": [mv.to_dict() for mv in versions.values()],
            }

    def get_platform_status(self) -> dict:
        with self._lock:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "models_registered": list(self._registry.keys()),
                "production_versions": dict(self._production),
                "challenger_versions": dict(self._challenger),
                "retraining_queue": list(self._retraining_queue),
                "retraining_in_progress": sorted(self._retraining_now),
                "canary": {mid: {"version": vid, "traffic_pct": self._registry[mid][vid].canary_pct}
                           for mid, vid in self._challenger.items() if vid in self._registry.get(mid, {})},
                "total_model_versions": sum(len(v) for v in self._registry.values()),
            }


# ── Singleton export ──────────────────────────────────────────────────────────
lifecycle_manager = LifecycleManager()
