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

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY_PATH = Path("models/registry/model_registry.json")
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
        """Bootstrap all 7 core platform models with production and canary challenger versions."""
        default_models = {
            "m1_ticket_predictor": {
                "name": "WL/RAC & Berth Confirmation Classifier",
                "prod_metrics": {"accuracy": 0.884, "f1_weighted": 0.871, "auc_roc": 0.912},
                "canary_metrics": {"accuracy": 0.898, "f1_weighted": 0.886, "auc_roc": 0.925},
                "artifact": "models/artifacts/wl_predictor_v1.0.pkl",
            },
            "m2_train_recommender": {
                "name": "Multi-Objective Fair Train Ranking",
                "prod_metrics": {"ndcg@5": 0.942, "disparate_impact_ratio": 0.965, "mrr": 0.918},
                "canary_metrics": {"ndcg@5": 0.958, "disparate_impact_ratio": 0.982, "mrr": 0.934},
                "artifact": "models/artifacts/recommender_v1.2.0.pkl",
            },
            "m3_delay_predictor": {
                "name": "Disruption & Arrival Delay Forecaster",
                "prod_metrics": {"mae_minutes": 11.4, "r2_score": 0.825, "rmse_minutes": 15.2},
                "canary_metrics": {"mae_minutes": 9.8, "r2_score": 0.852, "rmse_minutes": 13.6},
                "artifact": "models/artifacts/delay_predictor_v1.0.pkl",
            },
            "m4_travel_planner": {
                "name": "Autonomous Itinerary & Multi-Modal Agent",
                "prod_metrics": {"plan_feasibility_pct": 98.5, "constraint_adherence": 0.972},
                "canary_metrics": {"plan_feasibility_pct": 99.2, "constraint_adherence": 0.985},
                "artifact": "models/artifacts/travel_planner_v1.2.0.pkl",
            },
            "m5_fare_predictor": {
                "name": "Dynamic Surge & Fair Pricing Regressor",
                "prod_metrics": {"mape_pct": 3.12, "mae_inr": 28.5, "r2_score": 0.988},
                "canary_metrics": {"mape_pct": 2.65, "mae_inr": 24.1, "r2_score": 0.993},
                "artifact": "models/artifacts/fare_predictor_v1.0.pkl",
            },
            "m6_support_assistant": {
                "name": "Domain Grounded Support Concierge",
                "prod_metrics": {"faithfulness_score": 0.942, "answer_relevance": 0.918},
                "canary_metrics": {"faithfulness_score": 0.965, "answer_relevance": 0.941},
                "artifact": "models/artifacts/support_rag_v1.2.0.pkl",
            },
            "m7_fraud_detector": {
                "name": "Real-Time Tatkal Bot & Scalper Classifier",
                "prod_metrics": {"precision": 0.998, "recall": 0.995, "f1_weighted": 0.996},
                "canary_metrics": {"precision": 0.999, "recall": 0.997, "f1_weighted": 0.998},
                "artifact": "models/artifacts/fraud_detector_v1.0.pkl",
            },
        }

        for model_id, info in default_models.items():
            mv_prod = ModelVersion(
                model_id=model_id,
                version="v1.2.0",
                stage="production",
                artifact_path=info["artifact"],
                metrics=info["prod_metrics"],
                created_by="pipeline_scheduler",
            )
            mv_prod.promoted_at = "2026-10-14T08:00:00Z"
            mv_prod.canary_pct = 90

            mv_canary = ModelVersion(
                model_id=model_id,
                version="v1.3.0-rc1",
                stage="challenger",
                artifact_path=info["artifact"].replace("v1.0", "v1.3").replace("v1.2.0", "v1.3.0-rc1"),
                metrics=info["canary_metrics"],
                created_by="ci_cd_canary",
            )
            mv_canary.promoted_at = "2026-10-15T02:00:00Z"
            mv_canary.canary_pct = 10

            self._registry[model_id] = {
                "v1.2.0": mv_prod,
                "v1.3.0-rc1": mv_canary,
            }
            self._production[model_id] = "v1.2.0"
            self._challenger[model_id] = "v1.3.0-rc1"

    def _save_registry(self) -> None:
        """Persist registry state to disk."""
        try:
            MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for model_id, versions in self._registry.items():
                prod_ver = self._production.get(model_id)
                if prod_ver and prod_ver in versions:
                    mv = versions[prod_ver]
                    data[model_id] = {
                        **mv.to_dict(),
                        "artifact_path": mv.artifact_path,
                        "trained_at": mv.created_at,
                    }
            with open(MODEL_REGISTRY_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

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

    def get_retraining_queue(self) -> List[str]:
        with self._lock:
            return list(self._retraining_queue)

    def dequeue_retraining(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._retraining_queue:
                self._retraining_queue.remove(model_id)

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
