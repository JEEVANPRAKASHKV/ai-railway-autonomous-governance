"""
governance/lifecycle_manager.py
─────────────────────────────────
Pillar 6 – Model Lifecycle Manager.

Implements:
  • Model version registry (track all trained versions with metadata)
  • Canary deployment (route N% of traffic to new model)
  • Champion/Challenger A/B testing framework
  • Automatic rollback on performance degradation
  • Retraining trigger via drift/SLA signal subscription
  • Model promotion workflow (staging → production)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pickle
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
ROLLBACK_TRIGGER_EVENTS = {"SLA_BREACH", "MODEL_DRIFT_DETECTED", "PREDICTION_FAILURE"}
CANARY_DEFAULT_PCT = 10   # 10% traffic to challenger by default


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
        self._load_registry()

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

    def auto_rollback_on_event(self, event: dict) -> Optional[dict]:
        """
        Called by supervisor when a governance event is received.
        Triggers rollback if the event qualifies.
        """
        if event.get("event_type") in ROLLBACK_TRIGGER_EVENTS:
            model_id = event.get("module_id", "")
            if model_id in self._registry:
                return self.rollback(model_id, reason=event.get("event_type", "unknown"))
        return None

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

    def execute_retraining(self, model_id: str) -> dict:
        """Autonomous retraining (called by the supervisor on drift). Runs in the background with a cooldown."""
        with self._lock:
            last = getattr(self, "_last_retrain", {}).get(model_id, 0)
            if time.time() - last < RETRAIN_COOLDOWN_S or model_id in getattr(self, "_retraining_now", set()):
                return {"success": False, "skipped": "cooldown_or_in_progress", "model_id": model_id}
            if not hasattr(self, "_last_retrain"):
                self._last_retrain = {}
            self._last_retrain[model_id] = time.time()
        threading.Thread(target=self.retrain, args=(model_id, "autonomous_drift_response"),
                         daemon=True, name=f"retrain-{model_id}").start()
        return {"success": True, "started": True, "model_id": model_id,
                "note": "Retraining started in background; challenger will be evaluated against the champion."}

    def retrain(self, model_id: str, reason: str = "operator_request") -> dict:
        """
        REAL retraining: trains a challenger with the project's training pipeline (fresh split/seed) into a
        staging area, evaluates it on a held-out set, and only registers it as a canary challenger if its
        primary metric beats the production champion. Nothing is served until promoted.
        """
        spec = TRAINABLE.get(model_id)
        if not spec:
            return {"success": True, "model_id": model_id, "outcome": "not_applicable",
                    "detail": RULE_BASED.get(model_id, "This module has no trained weights; it is governed through "
                                                        "parameter, policy and prompt review.")}
        if not hasattr(self, "_retraining_now"):
            self._retraining_now = set()
        with self._lock:
            if model_id in self._retraining_now:
                return {"success": False, "error": "Retraining already in progress for this model."}
            self._retraining_now.add(model_id)
        fn_name, metric, higher, key, _ = spec
        try:
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
                new_metrics = getattr(T, fn_name)()
            finally:
                T.ARTIFACTS_DIR, T.RANDOM_SEED = old_dir, old_seed
            train_s = round(time.time() - t0, 1)

            with self._lock:
                versions = self._registry[model_id]
                champ_id = self._production.get(model_id)
                champ = versions[champ_id].metrics.get(metric)
                new_val = float(new_metrics.get(metric))
                improved = (new_val > champ) if higher else (new_val < champ)
                pkl = next((p.name for p in staging.glob("*.pkl")), None)
                version_id = f"v{len(versions) + 1}.0.0"
                mv = ModelVersion(model_id, version_id, "challenger" if improved else "rejected",
                                  f"models/artifacts/challengers/{staging.name}/{pkl}",
                                  metrics={k: round(float(v), 4) for k, v in new_metrics.items()
                                           if isinstance(v, (int, float))},
                                  created_by=f"autonomous_retraining ({reason})")
                versions[version_id] = mv
                if improved:
                    mv.canary_pct = 10
                    self._challenger[model_id] = version_id
                if model_id in self._retraining_queue:
                    self._retraining_queue.remove(model_id)
                self._save_registry()

            outcome = "challenger_better" if improved else "champion_retained"
            telemetry_bus.publish(
                module_id=model_id,
                event_type="MODEL_RETRAINED",
                payload={"new_version": version_id, "outcome": outcome, "metric": metric,
                         "champion": champ, "challenger": round(new_val, 4), "reason": reason, "train_seconds": train_s},
                severity="INFO",
            )
            return {
                "success": True, "model_id": model_id, "new_version": version_id, "outcome": outcome,
                "comparison": {metric: f"champion {champ} → challenger {round(new_val, 4)} "
                                       f"({'better' if improved else 'not better'}; {'higher' if higher else 'lower'} is better)"},
                "metrics": mv.metrics, "train_seconds": train_s,
                "next_step": ("Challenger registered as a 10% canary. Promote it to serve all traffic."
                              if improved else "Champion retained — challenger archived for audit."),
            }
        except Exception as exc:
            return {"success": False, "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            with self._lock:
                self._retraining_now.discard(model_id)

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
                "total_model_versions": sum(len(v) for v in self._registry.values()),
            }


# ── Singleton export ──────────────────────────────────────────────────────────
lifecycle_manager = LifecycleManager()
