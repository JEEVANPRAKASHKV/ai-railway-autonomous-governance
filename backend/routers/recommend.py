"""
backend/routers/recommend.py
============================
FastAPI router for Module 2: Smart Train Recommender.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_recommender_service
from modules.m2_train_recommender.schema import RecommendRequest, RecommendResponse
from modules.m2_train_recommender.service import TrainRecommenderService

from governance.hooks import CircuitOpenError, DataQualityError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["Module 2: Train Recommender"])


@router.post(
    "/trains",
    response_model=RecommendResponse,
    summary="Recommend trains based on persona multi-factor scoring and fairness equity audit",
)
def recommend_trains(
    request: RecommendRequest,
    service: TrainRecommenderService = Depends(get_recommender_service),
) -> RecommendResponse:
    try:
        return service.recommend(request)
    except (CircuitOpenError, DataQualityError, HTTPException):
        raise
    except Exception:
        logger.exception("Train recommendation failed")
        raise HTTPException(status_code=500, detail="Train recommendation failed. The error has been logged.")
