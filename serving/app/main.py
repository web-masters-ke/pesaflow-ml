"""Pesaflow ML — Real-Time Fraud Detection & AML Risk Scoring Engine."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.responses import JSONResponse

from serving.app.api.dependencies import _container
from serving.app.api.routes import (
    admin,
    aml_scoring,
    batch,
    case_management,
    fraud_scoring,
    health,
    labels,
    merchant_risk,
)
from serving.app.middleware.request_id import RequestIDMiddleware
from serving.app.middleware.security import SecurityHeadersMiddleware
from serving.app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} [{settings.ENVIRONMENT}]")

    # Initialize all services
    await _container.initialize()

    # Start Kafka consumer (if configured)
    try:
        from serving.app.kafka.consumer import start_kafka_consumer

        await start_kafka_consumer()
    except Exception as e:
        logger.warning(f"Kafka consumer not started: {e}")

    yield

    # Shutdown
    logger.info("Shutting down services...")
    try:
        from serving.app.kafka.consumer import stop_kafka_consumer

        await stop_kafka_consumer()
    except Exception:
        pass
    await _container.shutdown()


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title="Pesaflow ML — Fraud Detection & AML Engine",
        description=(
            "Real-time fraud detection and AML risk scoring engine for Pesaflow. "
            "Provides ML-powered transaction scoring, sanctions screening, "
            "SHAP explainability, and configurable decision rules."
        ),
        version=settings.SERVICE_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    )

    # === Middleware ===
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.ENVIRONMENT == "development" else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )

    # === Routes ===
    app.include_router(health.router)
    app.include_router(fraud_scoring.router, prefix="/api/v1/ml")
    app.include_router(aml_scoring.router, prefix="/api/v1")
    app.include_router(merchant_risk.router, prefix="/api/v1/ml")
    app.include_router(case_management.router, prefix="/api/v1")
    app.include_router(batch.router, prefix="/api/v1/ml")
    app.include_router(labels.router, prefix="/api/v1/ml")
    app.include_router(admin.router, prefix="/api/v1/ml")

    # === Global exception handler ===
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
