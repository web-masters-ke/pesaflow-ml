"""Batch Scoring & Regulatory Export Routes."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from serving.app.api.dependencies import ServiceContainer, get_container
from serving.app.middleware.auth import require_role
from serving.app.schemas.aml import AMLBatchScoreRequest, AMLBatchScoreResponse
from serving.app.schemas.fraud import FraudBatchScoreRequest, FraudBatchScoreResponse
from serving.app.schemas.merchant import MerchantBatchScoreRequest, MerchantBatchScoreResponse
from serving.app.services.batch_scoring_service import BatchScoringService
from serving.app.services.regulatory_export_service import RegulatoryExportService

router = APIRouter(prefix="/batch", tags=["Batch Scoring & Export"])


def _get_batch_service(container: ServiceContainer) -> BatchScoringService:
    return BatchScoringService(
        fraud_service=container.fraud_scoring_service,
        aml_service=container.aml_scoring_service,
        merchant_service=container.merchant_risk_service,
        db_pool=container.db_pool,
    )


def _get_export_service(container: ServiceContainer) -> RegulatoryExportService:
    return RegulatoryExportService(db_pool=container.db_pool)


@router.post("/fraud/rescore", summary="Batch rescore fraud transactions")
async def batch_fraud_rescore(
    request: FraudBatchScoreRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> dict:
    service = _get_batch_service(container)
    result = await service.run_batch_job("FRAUD_RESCORE", request.transactions)

    return {
        "job_id": result["job_id"],
        "total": result["total"],
        "processed": result["processed"],
        "failed": result["failed"],
        "duration_ms": result["duration_ms"],
    }


@router.post("/aml/rescore", summary="Batch rescore AML transactions")
async def batch_aml_rescore(
    request: AMLBatchScoreRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> dict:
    service = _get_batch_service(container)
    result = await service.run_batch_job("AML_RESCORE", request.transactions)

    return {
        "job_id": result["job_id"],
        "total": result["total"],
        "processed": result["processed"],
        "failed": result["failed"],
        "duration_ms": result["duration_ms"],
    }


@router.post("/merchant/rescore", summary="Batch rescore merchants")
async def batch_merchant_rescore(
    request: MerchantBatchScoreRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> dict:
    service = _get_batch_service(container)
    result = await service.run_batch_job("MERCHANT_RESCORE", request.merchants)

    return {
        "job_id": result["job_id"],
        "total": result["total"],
        "processed": result["processed"],
        "failed": result["failed"],
        "duration_ms": result["duration_ms"],
    }


@router.get("/jobs/{job_id}", summary="Get batch job status")
async def get_batch_job_status(
    job_id: str,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> dict:
    service = _get_batch_service(container)
    status = await service.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status


@router.post("/export/sar", summary="Export Suspicious Activity Report")
async def export_sar(
    start_date: datetime = Query(..., description="Start date (ISO 8601)"),
    end_date: datetime = Query(..., description="End date (ISO 8601)"),
    export_format: str = Query("CSV", pattern="^(CSV|JSON)$"),
    min_risk_score: float = Query(0.70, ge=0.0, le=1.0),
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR")),
) -> dict:
    service = _get_export_service(container)
    return await service.export_sar(start_date, end_date, export_format, min_risk_score)


@router.post("/export/str", summary="Export Suspicious Transaction Report")
async def export_str(
    start_date: datetime = Query(..., description="Start date (ISO 8601)"),
    end_date: datetime = Query(..., description="End date (ISO 8601)"),
    export_format: str = Query("CSV", pattern="^(CSV|JSON)$"),
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR")),
) -> dict:
    service = _get_export_service(container)
    return await service.export_str(start_date, end_date, export_format)


@router.get("/export/download/{filename}", summary="Download exported file")
async def download_export(
    filename: str,
    _auth: dict = Depends(require_role("ML_ADMIN", "AML_AUDITOR")),
) -> FileResponse:
    import os

    file_path = os.path.join("./exports", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Export file not found: {filename}")

    media_type = "text/csv" if filename.endswith(".csv") else "application/json"
    return FileResponse(file_path, media_type=media_type, filename=filename)
