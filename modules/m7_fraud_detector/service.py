"""
modules/m7_fraud_detector/service.py
====================================
Inference service for Module 7: Fraud, Security & Bot Detection.
Classifies live session velocity, Tatkal surge scalping, and automated scripted booking patterns.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import threading
import time
from collections import defaultdict, deque

from core.config import settings
from governance.context import report_inference
from governance.hooks import governance_hook
from governance.lifecycle_manager import lifecycle_manager
from modules.m7_fraud_detector.schema import (
    FraudDetectRequest,
    FraudDetectResponse,
    FraudInvestigativeFeature,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = ["requests_per_minute", "unique_pnr_attempts", "tatkal_flag", "account_age_days",
                "payment_attempts", "failed_captcha_count", "booking_time_hour", "is_bulk_booking", "ip_risk_score"]
FRAUD_LABELS = {
    "requests_per_minute": "Request velocity", "unique_pnr_attempts": "Distinct PNR lookups",
    "tatkal_flag": "Tatkal window", "account_age_days": "Account age", "payment_attempts": "Payment attempts",
    "failed_captcha_count": "Failed CAPTCHAs", "booking_time_hour": "Hour of booking",
    "is_bulk_booking": "Bulk booking", "ip_risk_score": "Network reputation",
}


class FraudDetectorService:
    """Service to compute real-time fraud risk scores, classify threat levels, and prescribe automated countermeasures."""

    def __init__(self) -> None:
        self._bundle: Optional[Dict[str, Any]] = None
        self._explainer: Optional[Any] = None
        self._iforest = None
        self._iforest_cut: Optional[float] = None
        self._lock = threading.Lock()
        # Behaviour memory for account-misuse detection (hashed identifiers only)
        self._ip_sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._user_ips: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._alerts: deque = deque(maxlen=300)
        self._load_resources()
        self._fit_anomaly_detector()

    def _load_resources(self) -> None:
        art_dir = settings.models_artifacts_dir
        model_path = os.path.join(art_dir, "fraud_detector_v1.0.pkl")
        shap_path = os.path.join(art_dir, "shap_explainers", "shap_fraud_detector.pkl")

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._bundle = pickle.load(f)
            except Exception as e:
                logger.error(f"[M7 Service] Failed to load fraud detector bundle: {e}")

        if os.path.exists(shap_path):
            try:
                with open(shap_path, "rb") as f:
                    self._explainer = pickle.load(f)
            except Exception as e:
                logger.warning(f"[M7 Service] Could not load fraud SHAP explainer: {e}")

    def _fit_anomaly_detector(self) -> None:
        """Unsupervised IsolationForest on LEGITIMATE sessions → flags novel patterns the classifier never saw."""
        try:
            from sklearn.ensemble import IsolationForest
            df = pd.read_csv(os.path.join(settings.data_raw_dir, "fraud_events.csv"))
            cols = (self._bundle or {}).get("feature_cols") or FEATURE_COLS
            legit = df[df["label"] == "legitimate"][cols].fillna(0)
            self._iforest = IsolationForest(n_estimators=150, contamination=0.01, random_state=42).fit(legit)
            self._iforest_cut = float(np.percentile(self._iforest.score_samples(legit), 1))
        except Exception as e:
            logger.debug(f"[M7 Service] anomaly detector unavailable: {e}")

    def _misuse_signals(self, req: FraudDetectRequest) -> List[str]:
        """Account misuse: many sessions behind one IP, or one account hopping across many IPs (10-min window)."""
        now = time.time()
        signals: List[str] = []
        with self._lock:
            q = self._ip_sessions[req.ip_hash]
            q.append((now, req.session_id, req.user_id or ""))
            recent = [x for x in q if now - x[0] <= 600]
            sessions = {x[1] for x in recent}
            accounts = {x[2] for x in recent if x[2]}
            if len(sessions) >= 8:
                signals.append(f"{len(sessions)} sessions from the same network in 10 min")
            if len(accounts) >= 4:
                signals.append(f"{len(accounts)} different accounts from the same network in 10 min")
            if req.user_id:
                u = self._user_ips[req.user_id]
                u.append((now, req.ip_hash))
                ips = {x[1] for x in u if now - x[0] <= 600}
                if len(ips) >= 4:
                    signals.append(f"account used from {len(ips)} different networks in 10 min")
        return signals

    def _shap_drivers(self, bundle: Dict[str, Any], input_df: pd.DataFrame, cls_idx: int, cls_name: str) -> List[Dict[str, Any]]:
        explainer = self._explainer if bundle is self._bundle else None
        try:
            if explainer is None:
                import shap
                explainer = shap.TreeExplainer(bundle["model"])
            sv = explainer.shap_values(input_df)
            arr = np.asarray(sv)
            if isinstance(sv, list):
                vals = np.asarray(sv[cls_idx])[0]
            elif arr.ndim == 3:
                vals = arr[0, :, cls_idx]
            else:
                vals = arr[0]
            out = [{"feature": FRAUD_LABELS.get(c, c), "raw_feature": c, "value": float(input_df.iloc[0][c]),
                    "impact": round(float(v), 3),
                    "direction": f"towards '{cls_name}'" if v > 0 else f"away from '{cls_name}'"}
                   for c, v in zip(input_df.columns, vals)]
            out.sort(key=lambda d: abs(d["impact"]), reverse=True)
            return out[:6]
        except Exception as e:
            logger.debug(f"[M7 Service] SHAP failed: {e}")
            return []

    def recent_alerts(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-n:][::-1]

    @governance_hook("m7_fraud_detector", endpoint="/api/v1/security/score")
    def score_session(self, request: FraudDetectRequest) -> FraudDetectResponse:
        """Run ML inference for threat scoring, investigative feature breakdown, and countermeasure prescription."""
        ip_risk = request.ip_risk_score if request.ip_risk_score is not None else 0.08

        feat_vector = {
            "requests_per_minute": float(request.requests_per_minute),
            "unique_pnr_attempts": float(request.unique_pnr_attempts),
            "tatkal_flag": float(int(request.tatkal_flag)),
            "account_age_days": float(request.account_age_days),
            "payment_attempts": float(request.payment_attempts),
            "failed_captcha_count": float(request.failed_captcha_count),
            "booking_time_hour": float(request.booking_time_hour),
            "is_bulk_booking": float(int(request.is_bulk_booking)),
            "ip_risk_score": float(ip_risk),
        }

        # ── Model Inference ───────────────────────────────────────────────────
        prob_bot = 0.01
        prob_suspicious = 0.05
        prob_legit = 0.94

        bundle, version = lifecycle_manager.serving_bundle("m7_fraud_detector", self._bundle)
        input_df = None
        classes = ["bot", "legitimate", "suspicious"]
        probs = None
        if bundle and "model" in bundle:
            model = bundle["model"]
            label_enc = bundle.get("label_encoder")
            cols = bundle.get("feature_cols", list(feat_vector.keys()))

            input_df = pd.DataFrame([[feat_vector[c] for c in cols]], columns=cols)
            probs = model.predict_proba(input_df)[0]
            classes = list(label_enc.classes_) if label_enc else ["bot", "legitimate", "suspicious"]

            prob_map = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
            prob_bot = prob_map.get("bot", 0.0)
            prob_suspicious = prob_map.get("suspicious", 0.0)
            prob_legit = prob_map.get("legitimate", 0.95)
        else:
            # Deterministic heuristic fallback
            if request.requests_per_minute > 60 or request.failed_captcha_count > 3:
                prob_bot = 0.95
                prob_legit = 0.02
                prob_suspicious = 0.03
            elif request.requests_per_minute > 25 or request.unique_pnr_attempts > 6:
                prob_suspicious = 0.65
                prob_legit = 0.30
                prob_bot = 0.05

        # Risk score (0.0 to 100.0)
        risk_score = round(float(np.clip((prob_bot * 100.0) + (prob_suspicious * 50.0), 0.0, 100.0)), 1)
        predicted_label = max({"bot": prob_bot, "suspicious": prob_suspicious, "legitimate": prob_legit}.items(),
                              key=lambda kv: kv[1])[0]

        # Novel-pattern detection and account-misuse signals (not covered by the supervised classifier)
        anomaly_score = None
        is_anomalous = False
        if self._iforest is not None:
            try:
                cols_if = (self._bundle or {}).get("feature_cols") or FEATURE_COLS
                x_if = pd.DataFrame([[feat_vector[c] for c in cols_if]], columns=cols_if)
                sc = float(self._iforest.score_samples(x_if)[0])
                anomaly_score = round(sc, 4)
                is_anomalous = bool(self._iforest_cut is not None and sc < self._iforest_cut)
            except Exception:
                pass
        misuse = self._misuse_signals(request)
        if (is_anomalous or misuse) and risk_score < 25.0:
            risk_score = 25.0  # escalate to at least a step-up challenge; the reasons are listed below
        report_inference(features=feat_vector, prediction=predicted_label, version=version if bundle else "heuristic")

        # Category and Action
        if risk_score >= 80.0:
            category = "Bot Detected"
            action = "BLOCK_AND_REPORT"
        elif risk_score >= 50.0:
            category = "High Risk"
            action = "RATE_LIMIT"
        elif risk_score >= 25.0:
            category = "Suspicious"
            action = "STEP_UP_CHALLENGE"
        else:
            category = "Safe"
            action = "ALLOW"

        # ── Feature Investigation Breakdown ───────────────────────────────────
        investigation: List[FraudInvestigativeFeature] = []

        # 1. Request velocity
        rpm_sev = "High Anomaly" if request.requests_per_minute > 50 else "Suspicious" if request.requests_per_minute > 20 else "Normal"
        investigation.append(
            FraudInvestigativeFeature(
                feature="requests_per_minute",
                value=request.requests_per_minute,
                severity=rpm_sev,
                description=f"Submission velocity: {request.requests_per_minute:.1f} req/min (Normal human threshold < 20)",
            )
        )

        # 2. Captcha failures
        cap_sev = "High Anomaly" if request.failed_captcha_count >= 3 else "Suspicious" if request.failed_captcha_count >= 1 else "Normal"
        investigation.append(
            FraudInvestigativeFeature(
                feature="failed_captcha_count",
                value=request.failed_captcha_count,
                severity=cap_sev,
                description=f"Failed captcha challenges: {request.failed_captcha_count}",
            )
        )

        # 3. Account age
        age_sev = "Suspicious" if request.account_age_days < 7 else "Normal"
        investigation.append(
            FraudInvestigativeFeature(
                feature="account_age_days",
                value=request.account_age_days,
                severity=age_sev,
                description=f"IRCTC account tenure: {request.account_age_days} days (New accounts under 7 days flagged during Tatkal)",
            )
        )

        # 4. IP Risk
        ip_sev = "High Anomaly" if ip_risk > 0.70 else "Suspicious" if ip_risk > 0.35 else "Normal"
        investigation.append(
            FraudInvestigativeFeature(
                feature="ip_risk_score",
                value=ip_risk,
                severity=ip_sev,
                description=f"Subnet threat reputation: {ip_risk:.2f} (Hosting/VPN proxy indicators)",
            )
        )

        # ── Incident Dossier ──────────────────────────────────────────────────
        dossier = {
            "session_id": request.session_id,
            "ip_hash": request.ip_hash,
            "risk_score": risk_score,
            "threat_verdict": category,
            "tatkal_surge_active": bool(request.tatkal_flag),
            "bot_probability": round(prob_bot, 4),
            "suspicious_probability": round(prob_suspicious, 4),
            "mitigation_instructions": (
                "Issue 429 Rate Limit and enforce mandatory biometric / Aadhaar OTP step-up challenge."
                if action in ["RATE_LIMIT", "BLOCK_AND_REPORT"]
                else "Session authorized for regular booking gateway."
            ),
        }

        # ── SHAP drivers for the predicted class ──────────────────────────────
        shap_drivers: List[Dict[str, Any]] = []
        if input_df is not None:
            shap_drivers = self._shap_drivers(bundle, input_df, classes.index(predicted_label), predicted_label)

        for m in misuse:
            investigation.append(FraudInvestigativeFeature(feature="account_misuse", value=1.0,
                                                           severity="Suspicious", description=m))
        if is_anomalous:
            investigation.append(FraudInvestigativeFeature(
                feature="behaviour_anomaly", value=float(anomaly_score or 0), severity="Suspicious",
                description="Session pattern is unlike 99% of legitimate sessions (IsolationForest)."))
        dossier.update({"predicted_label": predicted_label, "anomaly_score": anomaly_score,
                        "novel_pattern": is_anomalous, "account_misuse_signals": misuse,
                        "model_version": version if bundle else "heuristic"})

        if action != "ALLOW" or is_anomalous or misuse:
            with self._lock:
                self._alerts.append({
                    "session_id": request.session_id, "ip_hash": request.ip_hash[:6] + "…",
                    "risk_score": risk_score, "category": category, "action": action,
                    "predicted_label": predicted_label,
                    "why": [d["feature"] + (" ↑" if d["impact"] > 0 else " ↓") for d in shap_drivers[:3]] + misuse
                           + (["novel behaviour pattern"] if is_anomalous else []),
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })

        return FraudDetectResponse(
            session_id=request.session_id,
            ip_hash=request.ip_hash,
            fraud_risk_score=risk_score,
            risk_category=category,
            recommended_action=action,
            investigation_features=investigation,
            incident_dossier=dossier,
            shap_risk_drivers=shap_drivers,
            confidence_score=round(float(max(prob_bot, prob_legit, prob_suspicious)), 2),
        )


fraud_detector_service = FraudDetectorService()
