"""Label Feedback & Active Learning API Routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from serving.app.api.dependencies import ServiceContainer, get_container
from serving.app.middleware.auth import require_role
from serving.app.schemas.labels import (
    ALConfigResponse,
    ALConfigUpdateRequest,
    ALCreateCasesRequest,
    ALCreateCasesResponse,
    ALMetricsResponse,
    ALQueueResponse,
    ALRefreshResponse,
    Domain,
    LabelBatchSubmitRequest,
    LabelBatchSubmitResponse,
    LabelPropagationResponse,
    LabelStatisticsResponse,
    LabelSubmitRequest,
    LabelSubmitResponse,
)
from serving.app.services.active_learning_service import ActiveLearningService
from serving.app.services.label_feedback_service import LabelFeedbackService

router = APIRouter(prefix="/labels", tags=["Label Feedback & Active Learning"])


def _get_label_service(container: ServiceContainer) -> LabelFeedbackService:
    from serving.app.kafka.producer import publish_label_updated

    if container.redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return LabelFeedbackService(
        db_pool=container.db_pool,
        redis_client=container.redis_client,
        kafka_publish=publish_label_updated,
    )


def _get_al_service(container: ServiceContainer) -> ActiveLearningService:
    if container.redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return ActiveLearningService(
        db_pool=container.db_pool,
        redis_client=container.redis_client,
    )


# ─── Label Submission ──────────────────────────────────────────────────


@router.post(
    "/submit",
    response_model=LabelSubmitResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a single label for a prediction",
)
async def submit_label(
    request: LabelSubmitRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR", "FRAUD_ANALYST")),
) -> LabelSubmitResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_label_service(container)
    try:
        return await svc.submit_label(
            prediction_id=str(request.prediction_id),
            domain=request.domain,
            label=request.label,
            label_source=request.label_source,
            labeled_by=str(request.labeled_by) if request.labeled_by else None,
            reason=request.reason,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Label submission failed: {e}")


@router.post(
    "/submit/batch",
    response_model=LabelBatchSubmitResponse,
    summary="Submit up to 500 labels in a batch",
)
async def submit_batch(
    request: LabelBatchSubmitRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR", "FRAUD_ANALYST")),
) -> LabelBatchSubmitResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_label_service(container)
    result = await svc.submit_batch([item.model_dump() for item in request.labels])
    return LabelBatchSubmitResponse(**result)


# ─── Label Statistics ──────────────────────────────────────────────────


@router.get(
    "/statistics",
    response_model=LabelStatisticsResponse,
    summary="Get per-domain label statistics",
)
async def get_statistics(
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR", "FRAUD_ANALYST")),
) -> LabelStatisticsResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_label_service(container)
    return await svc.get_label_statistics()


# ─── Label Propagation ─────────────────────────────────────────────────


@router.post(
    "/propagate",
    response_model=LabelPropagationResponse,
    summary="Trigger case-to-label propagation",
)
async def propagate_labels(
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> LabelPropagationResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_label_service(container)
    fraud_count = await svc.propagate_fraud_review_labels()
    aml_count = await svc.propagate_aml_case_labels()

    return LabelPropagationResponse(
        fraud_labels_propagated=fraud_count,
        aml_labels_propagated=aml_count,
        total_propagated=fraud_count + aml_count,
    )


# ─── Active Learning Queue ────────────────────────────────────────────


@router.get(
    "/active-learning/queue",
    response_model=ALQueueResponse,
    summary="Get prioritized review queue",
)
async def get_al_queue(
    domain: Domain = Query(..., description="Domain to get queue for"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR", "FRAUD_ANALYST")),
) -> ALQueueResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_al_service(container)
    return await svc.get_queue(domain, limit=limit, offset=offset)


@router.post(
    "/active-learning/refresh",
    response_model=ALRefreshResponse,
    summary="Trigger active learning queue refresh",
)
async def refresh_al_queues(
    domains: list[Domain] | None = None,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ALRefreshResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_al_service(container)
    target_domains = domains or [Domain.FRAUD, Domain.AML, Domain.MERCHANT]
    items_added = {}

    for d in target_domains:
        added = await svc.refresh_queue(d)
        items_added[d.value] = added

    expired = await svc.expire_stale_items()

    return ALRefreshResponse(
        domains_refreshed=[d.value for d in target_domains],
        items_added=items_added,
        items_expired={d.value: expired // len(target_domains) for d in target_domains},
    )


# ─── Active Learning Config ───────────────────────────────────────────


@router.get(
    "/active-learning/config/{domain}",
    response_model=ALConfigResponse,
    summary="Get AL config for a domain",
)
async def get_al_config(
    domain: Domain,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ALConfigResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_al_service(container)
    config = await svc.get_config(domain)
    if not config:
        raise HTTPException(status_code=404, detail=f"No config for domain {domain.value}")
    return config


@router.put(
    "/active-learning/config/{domain}",
    response_model=ALConfigResponse,
    summary="Update AL config for a domain",
)
async def update_al_config(
    domain: Domain,
    request: ALConfigUpdateRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ALConfigResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_al_service(container)
    try:
        return await svc.update_config(domain, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── Active Learning Metrics ──────────────────────────────────────────


@router.get(
    "/active-learning/metrics/{domain}",
    response_model=ALMetricsResponse,
    summary="Get AL effectiveness metrics",
)
async def get_al_metrics(
    domain: Domain,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ALMetricsResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_al_service(container)
    return await svc.get_metrics(domain)


# ─── Auto-Create Cases from Queue ─────────────────────────────────────


@router.post(
    "/active-learning/create-cases",
    response_model=ALCreateCasesResponse,
    summary="Auto-create review cases from top queue items",
)
async def create_cases_from_queue(
    request: ALCreateCasesRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ALCreateCasesResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    svc = _get_al_service(container)
    created_ids = await svc.create_review_cases(request.domain, request.count)

    return ALCreateCasesResponse(
        domain=request.domain,
        cases_created=len(created_ids),
        prediction_ids=[UUID(pid) for pid in created_ids],
    )
