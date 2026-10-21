"""
models/train_all_models.py
===========================
Master Model Training Script — Golden Rule 1: Self-Contained.

Trains all 5 ML models from synthetic data and saves artifacts.
All models saved to models/artifacts/.
SHAP explainers, feature schemas, and model registry also saved here.

Run:
    python models/train_all_models.py
"""

import os
import sys
import json
import pickle
import logging
import warnings
from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, r2_score, mean_absolute_percentage_error,
    precision_score, recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
ARTIFACTS_DIR = os.path.join(ROOT, "models", "artifacts")
REGISTRY_DIR = os.path.join(ROOT, "models", "registry")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "shap_explainers"), exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)

RANDOM_SEED = 42          # model seed (varied by autonomous retraining)
SPLIT_SEED = 42           # hold-out split seed — FIXED so every model version is scored on the same test set
np.random.seed(RANDOM_SEED)

# Labelled production outcomes (features exactly as the model saw them + actual outcome), injected by the
# lifecycle manager when it retrains after concept drift. They are added to the TRAINING split only, so
# the fixed hold-out set stays comparable across versions.
EXTRA_TRAINING_ROWS: Dict[str, List[Tuple[dict, Any]]] = {}
LAST_EXTRA_ROWS_USED: Dict[str, int] = {}


def _append_extra(model_key: str, X_train: pd.DataFrame, y_train, encoder=None):
    rows = EXTRA_TRAINING_ROWS.get(model_key) or []
    LAST_EXTRA_ROWS_USED[model_key] = 0
    if not rows:
        return X_train, y_train
    feats, ys = [], []
    for f, y in rows:
        if all(c in f for c in X_train.columns):
            try:
                ys.append(int(encoder.transform([str(y)])[0]) if encoder is not None else float(y))
                feats.append([float(f[c]) for c in X_train.columns])
            except Exception:
                continue
    if not feats:
        return X_train, y_train
    X_extra = pd.DataFrame(feats, columns=X_train.columns)
    # Up-weight recent reality a little by repeating it (production outcomes are scarce vs history).
    reps = max(1, min(5, len(X_train) // (len(feats) * 50) or 1))
    X_new = pd.concat([X_train] + [X_extra] * reps, ignore_index=True)
    y_arr = np.concatenate([np.asarray(y_train)] + [np.asarray(ys)] * reps)
    LAST_EXTRA_ROWS_USED[model_key] = len(feats)
    logger.info(f"  + {len(feats)} labelled production outcomes added to training (x{reps})")
    return X_new, y_arr

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please run: python data/generate_synthetic_data.py first."
        )
    df = pd.read_csv(path)
    logger.info(f"  Loaded {filename}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def save_artifact(obj: Any, name: str) -> str:
    path = os.path.join(ARTIFACTS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    logger.info(f"  ✓ Saved {name} ({size_kb:.1f} KB)")
    return path


def save_shap_explainer(explainer: Any, model_key: str) -> str:
    path = os.path.join(ARTIFACTS_DIR, "shap_explainers", f"shap_{model_key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(explainer, f)
    logger.info(f"  ✓ Saved SHAP explainer for {model_key}")
    return path


# ── Feature Columns Registry ───────────────────────────────────────────────────

feature_columns_registry: Dict[str, List[str]] = {}


def save_feature_columns(model_key: str, columns: List[str]) -> None:
    feature_columns_registry[model_key] = columns


def write_feature_columns() -> None:
    path = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
    with open(path, "w") as f:
        json.dump(feature_columns_registry, f, indent=2)
    logger.info(f"  ✓ Saved feature_columns.json")


# ── Model Registry ─────────────────────────────────────────────────────────────

def build_registry(metrics: Dict[str, Dict]) -> Dict:
    today = date.today().isoformat()
    registry = {"models": {}}

    model_names = {
        "wl_predictor":     ("WL/RAC Ticket Predictor", "m1_ticket_predictor"),
        "berth_estimator":  ("Berth Type Estimator", "m1_ticket_predictor"),
        "delay_predictor":  ("Train Delay Predictor", "m3_delay_predictor"),
        "fare_predictor":   ("Dynamic Fare Predictor", "m5_fare_predictor"),
        "fraud_detector":   ("Fraud & Bot Detector", "m7_fraud_detector"),
    }

    for key, (name, module) in model_names.items():
        m = metrics.get(key, {})
        registry["models"][key] = {
            "name": name,
            "module": module,
            "current_version": "v1.0",
            "versions": {
                "v1.0": {
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                    "train_date": today,
                    "status": "production",
                    "artifact_path": f"models/artifacts/{key}_v1.0.pkl",
                }
            }
        }

    path = os.path.join(REGISTRY_DIR, "model_registry.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"  ✓ Saved model_registry.json")
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: WL/RAC Ticket Confirmation Predictor
# ─────────────────────────────────────────────────────────────────────────────

def train_wl_predictor() -> Dict:
    logger.info("\n" + "─" * 50)
    logger.info("Training Model 1: WL/RAC Ticket Confirmation Predictor")
    logger.info("─" * 50)

    df = load_csv("pnr_history.csv")
    df["journey_date"] = pd.to_datetime(df["journey_date"])
    df["month"] = df["journey_date"].dt.month
    df["day_of_week"] = df["journey_date"].dt.dayofweek
    df["is_peak_season"] = df["month"].isin([5, 6, 10, 11, 12]).astype(int)

    # Encode categoricals
    class_enc = LabelEncoder()
    quota_enc = LabelEncoder()
    df["class_enc"] = class_enc.fit_transform(df["journey_class"].fillna("SL"))
    df["quota_enc"] = quota_enc.fit_transform(df["quota"].fillna("GN"))

    feature_cols = [
        "initial_wl_number", "class_enc", "quota_enc",
        "days_to_departure", "month", "day_of_week",
        "is_peak_season", "season_factor",
        "seniority_flag", "disability_flag",
    ]
    save_feature_columns("wl_predictor", feature_cols)

    target_enc = LabelEncoder()
    X = df[feature_cols].fillna(0)
    y = target_enc.fit_transform(df["final_status"].fillna("WL"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
    )
    X_train, y_train = _append_extra("wl_predictor", X_train, y_train, target_enc)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"  WL Predictor — Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Save model + encoders as a bundle
    bundle = {
        "model": model,
        "class_encoder": class_enc,
        "quota_encoder": quota_enc,
        "target_encoder": target_enc,
        "feature_cols": feature_cols,
    }
    save_artifact(bundle, "wl_predictor_v1.0.pkl")

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    save_shap_explainer(explainer, "wl_predictor")

    return {"accuracy": acc, "f1_weighted": f1, "n_classes": len(target_enc.classes_)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Berth Type Estimator
# ─────────────────────────────────────────────────────────────────────────────

def train_berth_estimator() -> Dict:
    logger.info("\n" + "─" * 50)
    logger.info("Training Model 2: Berth Type Estimator")
    logger.info("─" * 50)

    df = load_csv("pnr_history.csv")
    df = df[df["final_status"] == "Confirmed"].copy()
    df["journey_date"] = pd.to_datetime(df["journey_date"])
    df["month"] = df["journey_date"].dt.month
    df["is_peak_season"] = df["month"].isin([5, 6, 10, 11, 12]).astype(int)

    class_enc = LabelEncoder()
    quota_enc = LabelEncoder()
    df["class_enc"] = class_enc.fit_transform(df["journey_class"].fillna("SL"))
    df["quota_enc"] = quota_enc.fit_transform(df["quota"].fillna("GN"))

    feature_cols = [
        "initial_wl_number", "class_enc", "quota_enc",
        "days_to_departure", "month", "is_peak_season",
        "seniority_flag", "disability_flag",
    ]
    save_feature_columns("berth_estimator", feature_cols)

    # Filter to only berths that make sense
    df = df[df["berth_type"].isin(["Lower", "Middle", "Upper", "Side Lower", "Side Upper", "RAC"])]
    berth_enc = LabelEncoder()
    X = df[feature_cols].fillna(0)
    y = berth_enc.fit_transform(df["berth_type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"  Berth Estimator — Accuracy: {acc:.4f} | F1: {f1:.4f}")

    bundle = {
        "model": model,
        "class_encoder": class_enc,
        "quota_encoder": quota_enc,
        "berth_encoder": berth_enc,
        "feature_cols": feature_cols,
    }
    save_artifact(bundle, "berth_estimator_v1.0.pkl")

    explainer = shap.TreeExplainer(model)
    save_shap_explainer(explainer, "berth_estimator")

    return {"accuracy": acc, "f1_weighted": f1}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Train Delay Predictor
# ─────────────────────────────────────────────────────────────────────────────

def train_delay_predictor() -> Dict:
    logger.info("\n" + "─" * 50)
    logger.info("Training Model 3: Train Delay Predictor")
    logger.info("─" * 50)

    df = load_csv("delay_logs.csv")

    feature_cols = [
        "weather_index", "congestion_at_station",
        "distance_from_origin_km", "month", "season_factor",
    ]
    save_feature_columns("delay_predictor", feature_cols)

    X = df[feature_cols].fillna(0)
    y = df["delay_minutes"].fillna(0).clip(0, 300)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED
    )
    X_train, y_train = _append_extra("delay_predictor", X_train, y_train)

    model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test).clip(0)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"  Delay Predictor — MAE: {mae:.2f} min | R²: {r2:.4f}")

    bundle = {"model": model, "feature_cols": feature_cols}
    save_artifact(bundle, "delay_predictor_v1.0.pkl")

    explainer = shap.TreeExplainer(model)
    save_shap_explainer(explainer, "delay_predictor")

    return {"mae_minutes": mae, "r2_score": r2}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: Dynamic Fare Predictor
# ─────────────────────────────────────────────────────────────────────────────

def train_fare_predictor() -> Dict:
    logger.info("\n" + "─" * 50)
    logger.info("Training Model 4: Dynamic Fare Predictor")
    logger.info("─" * 50)

    df = load_csv("pricing_history.csv")
    df["journey_date"] = pd.to_datetime(df["journey_date"])
    df["month"] = df["journey_date"].dt.month
    df["day_of_week"] = df["journey_date"].dt.dayofweek
    df["is_peak_season"] = df["month"].isin([5, 6, 10, 11, 12]).astype(int)

    class_enc = LabelEncoder()
    df["class_enc"] = class_enc.fit_transform(df["journey_class"].fillna("SL"))

    feature_cols = [
        "base_fare", "class_enc", "days_to_departure",
        "demand_index", "tatkal_flag", "season_factor",
        "route_popularity", "month", "day_of_week", "is_peak_season",
    ]
    save_feature_columns("fare_predictor", feature_cols)

    X = df[feature_cols].fillna(0)
    y = df["dynamic_fare"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED
    )
    X_train, y_train = _append_extra("fare_predictor", X_train, y_train)

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test).clip(0)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE (avoid division by zero)
    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))) * 100 if mask.sum() > 0 else 0

    logger.info(f"  Fare Predictor — MAE: ₹{mae:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

    bundle = {"model": model, "class_encoder": class_enc, "feature_cols": feature_cols}
    save_artifact(bundle, "fare_predictor_v1.0.pkl")

    explainer = shap.TreeExplainer(model)
    save_shap_explainer(explainer, "fare_predictor")

    return {"mae_inr": mae, "mape_pct": mape, "r2_score": r2}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5: Fraud & Bot Detection
# ─────────────────────────────────────────────────────────────────────────────

def train_fraud_detector() -> Dict:
    logger.info("\n" + "─" * 50)
    logger.info("Training Model 5: Fraud & Bot Detector")
    logger.info("─" * 50)

    df = load_csv("fraud_events.csv")

    feature_cols = [
        "requests_per_minute", "unique_pnr_attempts",
        "tatkal_flag", "account_age_days",
        "payment_attempts", "failed_captcha_count",
        "booking_time_hour", "is_bulk_booking",
        "ip_risk_score",
    ]
    save_feature_columns("fraud_detector", feature_cols)

    label_enc = LabelEncoder()
    X = df[feature_cols].fillna(0)
    y = label_enc.fit_transform(df["label"].fillna("legitimate"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
    )
    X_train, y_train = _append_extra("fraud_detector", X_train, y_train, label_enc)

    # Scale class weights for imbalanced dataset
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"  Fraud Detector — Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=label_enc.classes_)}")

    bundle = {
        "model": model,
        "label_encoder": label_enc,
        "feature_cols": feature_cols,
    }
    save_artifact(bundle, "fraud_detector_v1.0.pkl")

    explainer = shap.TreeExplainer(model)
    save_shap_explainer(explainer, "fraud_detector")

    return {
        "accuracy": acc, "precision": precision,
        "recall": recall, "f1_weighted": f1,
        "classes": list(label_enc.classes_),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info(" AI Railway Governance Platform — Model Training")
    logger.info(f" Output: {ARTIFACTS_DIR}")
    logger.info("=" * 60)

    all_metrics: Dict[str, Dict] = {}

    try:
        all_metrics["wl_predictor"] = train_wl_predictor()
    except Exception as e:
        logger.error(f"WL Predictor training failed: {e}")
        all_metrics["wl_predictor"] = {"error": str(e)}

    try:
        all_metrics["berth_estimator"] = train_berth_estimator()
    except Exception as e:
        logger.error(f"Berth Estimator training failed: {e}")
        all_metrics["berth_estimator"] = {"error": str(e)}

    try:
        all_metrics["delay_predictor"] = train_delay_predictor()
    except Exception as e:
        logger.error(f"Delay Predictor training failed: {e}")
        all_metrics["delay_predictor"] = {"error": str(e)}

    try:
        all_metrics["fare_predictor"] = train_fare_predictor()
    except Exception as e:
        logger.error(f"Fare Predictor training failed: {e}")
        all_metrics["fare_predictor"] = {"error": str(e)}

    try:
        all_metrics["fraud_detector"] = train_fraud_detector()
    except Exception as e:
        logger.error(f"Fraud Detector training failed: {e}")
        all_metrics["fraud_detector"] = {"error": str(e)}

    # Save feature columns registry
    write_feature_columns()

    # Build and save model registry
    registry = build_registry(all_metrics)

    logger.info("\n" + "=" * 60)
    logger.info(" ✅ Model Training Complete! Summary:")
    for key, metrics in all_metrics.items():
        logger.info(f"   {key:<22}: {metrics}")

    logger.info("\n Artifacts in models/artifacts/:")
    for f in sorted(os.listdir(ARTIFACTS_DIR)):
        if not os.path.isdir(os.path.join(ARTIFACTS_DIR, f)):
            size = os.path.getsize(os.path.join(ARTIFACTS_DIR, f))
            logger.info(f"   {f:<40} {size/1024:.1f} KB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
