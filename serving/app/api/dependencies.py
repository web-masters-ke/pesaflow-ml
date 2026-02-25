"""FastAPI Dependency Injection — Manages singleton services and connections."""

import os
from functools import lru_cache
from typing import Any

import redis.asyncio as redis
from loguru import logger

from feature_engineering.aml_features import AMLFeatureExtractor
from feature_engineering.fraud_features import FraudFeatureExtractor
from feature_engineering.merchant_features import MerchantFeatureExtractor
from models.aml.aml_model import AMLRiskModel
from models.fraud.fraud_model import FraudDetectionModel
from models.merchant.merchant_model import MerchantRiskModel
from monitoring.metrics import PesaflowMetrics
from serving.app.services.aml_scoring_service import AMLScoringService
from serving.app.services.decision_engine import (
    AMLDecisionEngine,
    AMLThresholdConfig,
    FraudDecisionEngine,
    MerchantDecisionEngine,
    MerchantThresholdConfig,
    ThresholdConfig,
)
from serving.app.services.fraud_scoring_service import FraudScoringService
from serving.app.services.merchant_risk_service import MerchantRiskService
from serving.app.services.sanctions_service import SanctionsScreeningService
from serving.app.settings import Settings, get_settings


class ServiceContainer:
    """Singleton container for all services and connections."""

    def __init__(self) -> None:
        self._initialized = False
        self.settings: Settings | None = None
        self.redis_client: redis.Redis | None = None
        self.db_pool: Any = None

        # Models
        self.fraud_model: FraudDetectionModel | None = None
        self.aml_model: AMLRiskModel | None = None
        self.merchant_model: MerchantRiskModel | None = None

        # Services
        self.fraud_scoring_service: FraudScoringService | None = None
        self.aml_scoring_service: AMLScoringService | None = None
        self.merchant_risk_service: MerchantRiskService | None = None
        self.sanctions_service: SanctionsScreeningService | None = None
        self.metrics: PesaflowMetrics | None = None

    async def initialize(self) -> None:
        """Initialize all connections and services."""
        if self._initialized:
            return

        self.settings = get_settings()
        logger.info(f"Initializing services for {self.settings.ENVIRONMENT}")

        # Initialize Prometheus metrics
        self.metrics = PesaflowMetrics()

        # Initialize Redis
        self.redis_client = redis.from_url(
            self.settings.REDIS_URL,
            decode_responses=True,
            max_connections=50,
        )

        # Initialize DB pool (optional — graceful if unavailable)
        try:
            import asyncpg

            self.db_pool = await asyncpg.create_pool(
                self.settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"),
                min_size=5,
                max_size=self.settings.DB_POOL_SIZE,
                server_settings={"search_path": self.settings.DATABASE_SCHEMA + ",public"},
            )
            logger.info("Database pool initialized")
        except Exception as e:
            logger.warning(f"Database pool initialization failed (will use Redis-only mode): {e}")
            self.db_pool = None

        # Initialize models
        self.fraud_model = FraudDetectionModel(version=self.settings.FRAUD_MODEL_VERSION)
        self.aml_model = AMLRiskModel(version=self.settings.AML_MODEL_VERSION)

        # Try to load model artifacts
        fraud_path = os.path.join(self.settings.MODEL_STORAGE_PATH, "fraud", "model.txt")
        aml_path = os.path.join(self.settings.MODEL_STORAGE_PATH, "aml", "model.json")

        if os.path.exists(fraud_path):
            self.fraud_model.load(fraud_path)
        else:
            logger.warning(f"Fraud model artifact not found at {fraud_path} — run training first")

        if os.path.exists(aml_path):
            self.aml_model.load(aml_path)
        else:
            logger.warning(f"AML model artifact not found at {aml_path} — run training first")

        # Capture redis_client as a local variable for type narrowing
        redis_cl = self.redis_client
        if redis_cl is None:
            raise RuntimeError("Redis client must be initialized before services")

        # Initialize sanctions service
        self.sanctions_service = SanctionsScreeningService(
            redis_client=redis_cl,
            db_pool=self.db_pool,
            fuzzy_threshold=self.settings.SANCTIONS_FUZZY_THRESHOLD,
        )
        await self.sanctions_service.load_sanctions_data()

        # Initialize feature extractors
        fraud_features = FraudFeatureExtractor(redis_client=redis_cl, db_pool=self.db_pool)
        aml_features = AMLFeatureExtractor(redis_client=redis_cl, db_pool=self.db_pool)

        # Initialize decision engines
        fraud_decision = FraudDecisionEngine(
            ThresholdConfig(
                approve_below=self.settings.FRAUD_APPROVE_THRESHOLD,
                review_above=self.settings.FRAUD_REVIEW_THRESHOLD,
                block_above=self.settings.FRAUD_BLOCK_THRESHOLD,
            )
        )
        aml_decision = AMLDecisionEngine(
            AMLThresholdConfig(
                medium_above=self.settings.AML_MEDIUM_THRESHOLD,
                high_above=self.settings.AML_HIGH_THRESHOLD,
                critical_above=self.settings.AML_CRITICAL_THRESHOLD,
            )
        )

        # Initialize scoring services
        self.fraud_scoring_service = FraudScoringService(
            model=self.fraud_model,
            feature_extractor=fraud_features,
            decision_engine=fraud_decision,
            sanctions_service=self.sanctions_service,
            metrics=self.metrics,
            redis_client=redis_cl,
            db_pool=self.db_pool,
        )

        self.aml_scoring_service = AMLScoringService(
            model=self.aml_model,
            feature_extractor=aml_features,
            decision_engine=aml_decision,
            sanctions_service=self.sanctions_service,
            metrics=self.metrics,
            redis_client=redis_cl,
            db_pool=self.db_pool,
        )

        # Initialize merchant model
        self.merchant_model = MerchantRiskModel(version=self.settings.MERCHANT_MODEL_VERSION)
        merchant_path = os.path.join(self.settings.MODEL_STORAGE_PATH, "merchant", "model.txt")
        if os.path.exists(merchant_path):
            self.merchant_model.load(merchant_path)
        else:
            logger.warning(f"Merchant model artifact not found at {merchant_path} — run training first")

        # Initialize merchant features + decision engine
        merchant_features = MerchantFeatureExtractor(redis_client=redis_cl, db_pool=self.db_pool)
        merchant_decision = MerchantDecisionEngine(
            MerchantThresholdConfig(
                standard_below=self.settings.MERCHANT_STANDARD_THRESHOLD,
                enhanced_above=self.settings.MERCHANT_ENHANCED_THRESHOLD,
                restricted_above=self.settings.MERCHANT_RESTRICTED_THRESHOLD,
                blocked_above=self.settings.MERCHANT_BLOCKED_THRESHOLD,
            )
        )

        self.merchant_risk_service = MerchantRiskService(
            model=self.merchant_model,
            feature_extractor=merchant_features,
            decision_engine=merchant_decision,
            metrics=self.metrics,
            redis_client=redis_cl,
            db_pool=self.db_pool,
        )

        self._initialized = True
        logger.info("All services initialized successfully")

    async def shutdown(self) -> None:
        """Graceful shutdown of connections."""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
        logger.info("Services shut down")


# Singleton container
_container = ServiceContainer()


async def get_container() -> ServiceContainer:
    return _container


async def get_fraud_service() -> FraudScoringService:
    if not _container.fraud_scoring_service:
        raise RuntimeError("Fraud scoring service not initialized")
    return _container.fraud_scoring_service


async def get_aml_service() -> AMLScoringService:
    if not _container.aml_scoring_service:
        raise RuntimeError("AML scoring service not initialized")
    return _container.aml_scoring_service


async def get_merchant_service() -> MerchantRiskService:
    if not _container.merchant_risk_service:
        raise RuntimeError("Merchant risk service not initialized")
    return _container.merchant_risk_service


async def get_metrics() -> PesaflowMetrics:
    if not _container.metrics:
        raise RuntimeError("Metrics not initialized")
    return _container.metrics
