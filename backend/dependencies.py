"""
backend/dependencies.py
=======================
FastAPI dependency injection providers.
"""

from __future__ import annotations

from core.config import PlatformSettings, settings
from governance.telemetry_bus import TelemetryBus, telemetry_bus
from modules.m1_ticket_predictor.service import TicketPredictorService, ticket_predictor_service
from modules.m2_train_recommender.service import TrainRecommenderService, train_recommender_service
from modules.m3_delay_predictor.service import DelayPredictorService, delay_predictor_service
from modules.m4_travel_planner.agent import TravelPlannerAgent, travel_planner_agent
from modules.m4_travel_planner.hitl_gate import HITLApprovalGate, hitl_gate
from modules.m5_fare_predictor.service import FarePredictorService, fare_predictor_service
from modules.m6_support_assistant.service import SupportAssistantService, support_assistant_service
from modules.m7_fraud_detector.service import FraudDetectorService, fraud_detector_service


def get_platform_settings() -> PlatformSettings:
    return settings


def get_telemetry_bus() -> TelemetryBus:
    return telemetry_bus


def get_ticket_service() -> TicketPredictorService:
    return ticket_predictor_service


def get_recommender_service() -> TrainRecommenderService:
    return train_recommender_service


def get_delay_service() -> DelayPredictorService:
    return delay_predictor_service


def get_planner_agent() -> TravelPlannerAgent:
    return travel_planner_agent


def get_hitl_gate() -> HITLApprovalGate:
    return hitl_gate


def get_fare_service() -> FarePredictorService:
    return fare_predictor_service


def get_support_service() -> SupportAssistantService:
    return support_assistant_service


def get_fraud_service() -> FraudDetectorService:
    return fraud_detector_service
