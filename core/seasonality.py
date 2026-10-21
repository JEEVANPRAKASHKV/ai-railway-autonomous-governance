"""
core/seasonality.py
===================
Single source of truth for calendar features, shared by training data and inference.

The synthetic generator derives `season_factor` from the travel date (festival / vacation peaks).
Services previously used a different rule (1.4 if peak month else 1.0), so the models saw inputs at
inference that never occurred in training. This module mirrors the generator's deterministic rule.
"""

from __future__ import annotations

from datetime import date

PEAK_MONTHS = (5, 6, 10, 11, 12)


def season_factor(d: date) -> float:
    """Demand / congestion multiplier for a travel date (same rule as data/generate_synthetic_data.py)."""
    m, day = d.month, d.day
    if m in (10, 11) and day > 10:   # Diwali / Chhath
        return 1.8
    if m in (5, 6):                  # summer vacation
        return 1.6
    if m == 12 and day > 20:         # Christmas / New Year
        return 1.5
    if m == 1 and day < 10:          # New Year
        return 1.4
    if m in (3, 4):                  # Holi
        return 1.3
    return 1.0                       # generator adds N(0, 0.1) noise here; its expectation is 1.0


def is_peak(d: date) -> int:
    return int(d.month in PEAK_MONTHS)


def climatology_weather(d: date) -> float:
    """Expected adverse-weather index (0–10) for the month: monsoon and winter fog are worse."""
    if d.month in (6, 7, 8, 9):
        return 6.0
    if d.month in (12, 1):
        return 5.5
    return 2.5
