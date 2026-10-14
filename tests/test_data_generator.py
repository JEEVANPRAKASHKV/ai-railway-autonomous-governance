"""
tests/test_data_generator.py
============================
Validation tests for synthetic data generation (Golden Rule 1).
"""

import os
import pickle
import pandas as pd
import pytest

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASELINE_PKL = os.path.join(os.path.dirname(__file__), "..", "data", "baseline_distributions.pkl")

EXPECTED_FILES = [
    "trains.csv",
    "stations.csv",
    "pnr_history.csv",
    "pricing_history.csv",
    "user_profiles.csv",
    "congestion_logs.csv",
    "delay_logs.csv",
    "fraud_events.csv",
    "irctc_policies.txt",
    "railway_faq.txt",
]


def test_all_synthetic_files_exist():
    """Verify all 10 synthetic files are generated."""
    for filename in EXPECTED_FILES:
        filepath = os.path.join(DATA_RAW_DIR, filename)
        assert os.path.exists(filepath), f"Missing expected synthetic file: {filename}"
        assert os.path.getsize(filepath) > 0, f"Synthetic file is empty: {filename}"


def test_baseline_distributions_exist():
    """Verify baseline distributions pickle file exists and contains valid stats."""
    assert os.path.exists(BASELINE_PKL), "baseline_distributions.pkl does not exist"
    with open(BASELINE_PKL, "rb") as f:
        stats = pickle.load(f)
    assert isinstance(stats, dict)
    assert len(stats) >= 10, f"Expected >= 10 features in baseline distribution, got {len(stats)}"
    # Check structure of one entry
    first_key = list(stats.keys())[0]
    entry = stats[first_key]
    for required_key in ["mean", "std", "min", "max"]:
        assert required_key in entry, f"Missing {required_key} in feature stat for {first_key}"


def test_pnr_dataset_schema():
    """Verify pnr_history schema and non-emptiness."""
    df = pd.read_csv(os.path.join(DATA_RAW_DIR, "pnr_history.csv"), nrows=100)
    required_cols = ["pnr", "train_number", "initial_wl_number", "final_status", "journey_class"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col} in pnr_history.csv"


def test_fraud_dataset_schema():
    """Verify fraud_events schema and labels."""
    df = pd.read_csv(os.path.join(DATA_RAW_DIR, "fraud_events.csv"), nrows=100)
    assert "label" in df.columns
    assert "requests_per_minute" in df.columns
    assert "unique_pnr_attempts" in df.columns
    assert "ip_risk_score" in df.columns


def test_faq_file_contents():
    """Verify railway_faq.txt has Q&A format."""
    faq_path = os.path.join(DATA_RAW_DIR, "railway_faq.txt")
    with open(faq_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Q:" in content
    assert "A:" in content
