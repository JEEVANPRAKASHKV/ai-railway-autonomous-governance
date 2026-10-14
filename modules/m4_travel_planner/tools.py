"""
modules/m4_travel_planner/tools.py
==================================
Modular tools used by the Agentic AI Travel Planner:
  - RouteSearchTool
  - FareEstimatorTool
  - WLPredictorTool
  - DelayCheckerTool
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import settings

logger = logging.getLogger(__name__)


class RouteSearchTool:
    """Tool to search candidate railway routes and schedules."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self) -> None:
        path = os.path.join(settings.data_raw_dir, "trains.csv")
        if os.path.exists(path):
            try:
                self._df = pd.read_csv(path)
            except Exception as e:
                logger.error(f"[RouteSearchTool] Load error: {e}")

    def run(self, source: str, destination: str) -> List[Dict[str, Any]]:
        src = source.strip().upper()
        dst = destination.strip().upper()
        if self._df is None or len(self._df) == 0:
            self._load()

        if self._df is not None:
            mask = (self._df["source"].str.upper() == src) & (self._df["destination"].str.upper() == dst)
            matches = self._df[mask]
            if len(matches) == 0:
                # Return representative corridor trains
                matches = self._df.head(4)
            return matches.to_dict(orient="records")
        return []


class FareEstimatorTool:
    """Tool to estimate base and dynamic fare for given class and distance."""

    def run(self, train_row: Dict[str, Any], travel_class: str = "3A") -> float:
        cls_key = f"base_fare_{travel_class.lower()}"
        base_fare = float(train_row.get(cls_key, 0.0))
        if base_fare <= 0:
            sl_fare = float(train_row.get("base_fare_sl", 350.0))
            multipliers = {"SL": 1.0, "3A": 2.4, "2A": 3.4, "1A": 5.5, "CC": 1.8}
            base_fare = sl_fare * multipliers.get(travel_class.upper(), 2.0)
        return round(base_fare, 2)


class WLPredictorTool:
    """Tool to check ticket confirmation odds."""

    def run(self, current_wl: int = 15, days_ahead: int = 14, travel_class: str = "3A") -> float:
        from modules.m1_ticket_predictor.service import ticket_predictor_service
        from modules.m1_ticket_predictor.schema import TicketPredictRequest
        from datetime import timedelta

        try:
            req = TicketPredictRequest(
                train_number="12301",
                journey_class=travel_class,
                quota="GN",
                journey_date=date.today() + timedelta(days=max(1, days_ahead)),
                current_wl_number=max(1, current_wl),
            )
            res = ticket_predictor_service.predict(req)
            return res.confirmation_probability
        except Exception:
            # Deterministic heuristic fallback
            prob = max(0.1, 0.95 - (current_wl * 0.018) + (days_ahead * 0.005))
            return round(min(0.98, prob), 3)


class DelayCheckerTool:
    """Tool to check historical on-time performance and delay probability."""

    def run(self, train_number: str, station_code: str = "NDLS") -> float:
        from modules.m3_delay_predictor.service import delay_predictor_service
        from modules.m3_delay_predictor.schema import DelayPredictRequest

        try:
            req = DelayPredictRequest(
                train_number=str(train_number),
                station_code=station_code,
                journey_date=date.today(),
            )
            res = delay_predictor_service.predict(req)
            return res.predicted_delay_minutes
        except Exception:
            return 12.0
