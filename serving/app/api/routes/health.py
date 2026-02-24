"""Health Check Routes."""

from datetime import datetime

from fastapi import APIRouter, Depends
from prometheus_client import generate_latest
from starlette.responses import Response

from serving.app.api.dependencies import ServiceContainer, get_container
from serving.app.schemas.common import HealthResponse
from serving.app.settings import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> dict:
    return {"status": "OK"}


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    settings = get_settings()
    fraud_loaded = container.fraud_model.is_loaded if container.fraud_model else False
    aml_loaded = container.aml_model.is_loaded if container.aml_model else False

    return HealthResponse(
        status="READY" if (fraud_loaded or aml_loaded) else "DEGRADED",
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        environment=settings.ENVIRONMENT,
        models_loaded={"fraud": fraud_loaded, "aml": aml_loaded},
        timestamp=datetime.utcnow(),
    )


@router.get("/metrics", summary="Prometheus metrics endpoint")
async def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain; charset=utf-8")
