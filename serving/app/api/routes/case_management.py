"""AML Case Management Routes — CRUD for AML investigation cases."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from serving.app.api.dependencies import ServiceContainer, get_container
from serving.app.middleware.auth import require_role
from serving.app.schemas.aml import (
    AMLCaseCreateRequest,
    AMLCaseDetailResponse,
    AMLCaseListResponse,
    AMLCaseResponse,
    AMLCaseUpdateRequest,
)
from serving.app.schemas.labels import Domain, LabelSource
from serving.app.services.label_feedback_service import LabelFeedbackService

router = APIRouter(prefix="/aml/cases", tags=["AML Case Management"])


@router.post("", response_model=AMLCaseResponse, status_code=status.HTTP_201_CREATED, summary="Create AML case")
async def create_case(
    request: AMLCaseCreateRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR")),
) -> AMLCaseResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    case_id = str(uuid.uuid4())
    now = datetime.utcnow()

    try:
        async with container.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO aml_cases (id, entity_type, entity_id, trigger_reason, risk_score, priority, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                case_id,
                request.entity_type,
                str(request.entity_id),
                request.trigger_reason,
                request.risk_score,
                request.priority,
                "OPEN",
                now,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create case: {e}")

    return AMLCaseResponse(
        case_id=uuid.UUID(case_id),
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        trigger_reason=request.trigger_reason,
        risk_score=request.risk_score,
        priority=request.priority,
        status="OPEN",
        created_at=now,
    )


@router.get("/{case_id}", response_model=AMLCaseDetailResponse, summary="Get AML case details")
async def get_case(
    case_id: str,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR", "FRAUD_ANALYST")),
) -> AMLCaseDetailResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    async with container.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM aml_cases WHERE id = $1",
            case_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    return AMLCaseDetailResponse(
        case_id=row["id"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        trigger_reason=row["trigger_reason"],
        risk_score=float(row["risk_score"]),
        priority=row["priority"],
        status=row["status"],
        assigned_to=row.get("assigned_to"),
        notes=row.get("notes"),
        resolution=row.get("resolution"),
        created_at=row["created_at"],
        updated_at=row.get("updated_at"),
    )


@router.put("/{case_id}", response_model=AMLCaseDetailResponse, summary="Update AML case")
async def update_case(
    case_id: str,
    request: AMLCaseUpdateRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR")),
) -> AMLCaseDetailResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    now = datetime.utcnow()

    async with container.db_pool.acquire() as conn:
        # Check case exists
        existing = await conn.fetchrow("SELECT * FROM aml_cases WHERE id = $1", case_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

        await conn.execute(
            """
            UPDATE aml_cases
            SET status = $2, assigned_to = $3, notes = $4, resolution = $5, updated_at = $6
            WHERE id = $1
            """,
            case_id,
            request.status,
            str(request.assigned_to) if request.assigned_to else existing.get("assigned_to"),
            request.notes or existing.get("notes"),
            request.resolution or existing.get("resolution"),
            now,
        )

        # Re-fetch updated record
        row = await conn.fetchrow("SELECT * FROM aml_cases WHERE id = $1", case_id)

    # Auto-propagate label on case closure
    if request.status in ("CLOSED_CONFIRMED", "CLOSED_FALSE_POSITIVE"):
        try:
            label = 1 if request.status == "CLOSED_CONFIRMED" else 0
            prediction_id = row.get("prediction_id")
            if prediction_id and container.redis_client:
                svc = LabelFeedbackService(
                    db_pool=container.db_pool,
                    redis_client=container.redis_client,
                )
                await svc.submit_label(
                    prediction_id=str(prediction_id),
                    domain=Domain.AML,
                    label=label,
                    label_source=LabelSource.SAR_CONFIRMED if label == 1 else LabelSource.MANUAL_REVIEW,
                    reason=f"Auto-propagated from case {case_id} closure ({request.status})",
                )
                # Mark case as label-propagated
                async with container.db_pool.acquire() as conn2:
                    await conn2.execute(
                        "UPDATE aml_cases SET label_propagated = TRUE WHERE id = $1",
                        case_id,
                    )
        except Exception as e:
            logger.warning(f"Label propagation from case closure failed: {e}")

    return AMLCaseDetailResponse(
        case_id=row["id"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        trigger_reason=row["trigger_reason"],
        risk_score=float(row["risk_score"]),
        priority=row["priority"],
        status=row["status"],
        assigned_to=row.get("assigned_to"),
        notes=row.get("notes"),
        resolution=row.get("resolution"),
        created_at=row["created_at"],
        updated_at=row.get("updated_at"),
    )


@router.get("", response_model=AMLCaseListResponse, summary="List AML cases")
async def list_cases(
    case_status: str | None = Query(None, alias="status"),
    priority: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR", "FRAUD_ANALYST")),
) -> AMLCaseListResponse:
    if not container.db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    # Build dynamic query
    conditions = []
    params = []
    param_idx = 1

    if case_status:
        conditions.append(f"status = ${param_idx}")
        params.append(case_status)
        param_idx += 1

    if priority:
        conditions.append(f"priority = ${param_idx}")
        params.append(priority)
        param_idx += 1

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    async with container.db_pool.acquire() as conn:
        # Get total count
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) as total FROM aml_cases {where_clause}",
            *params,
        )
        total = count_row["total"]

        # Get paginated results
        rows = await conn.fetch(
            f"""
            SELECT * FROM aml_cases {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """,
            *params,
            limit,
            offset,
        )

    cases = [
        AMLCaseDetailResponse(
            case_id=row["id"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            trigger_reason=row["trigger_reason"],
            risk_score=float(row["risk_score"]),
            priority=row["priority"],
            status=row["status"],
            assigned_to=row.get("assigned_to"),
            notes=row.get("notes"),
            resolution=row.get("resolution"),
            created_at=row["created_at"],
            updated_at=row.get("updated_at"),
        )
        for row in rows
    ]

    return AMLCaseListResponse(total=total, cases=cases, limit=limit, offset=offset)
