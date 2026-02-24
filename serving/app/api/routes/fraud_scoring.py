"""Fraud Scoring API Routes."""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from serving.app.api.dependencies import get_fraud_service
from serving.app.middleware.auth import verify_jwt
from serving.app.schemas.fraud import (
    FraudBatchScoreRequest,
    FraudBatchScoreResponse,
    FraudExplanationResponse,
    FraudScoreRequest,
    FraudScoreResponse,
)
from serving.app.services.fraud_scoring_service import FraudScoringService

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])


@router.post(
    "/score",
    response_model=FraudScoreResponse,
    summary="Score transaction for fraud risk",
    description="Real-time fraud risk scoring. Returns risk score (0-1), risk level, and decision.",
)
async def score_transaction(
    request: FraudScoreRequest,
    service: FraudScoringService = Depends(get_fraud_service),
    _auth: dict = Depends(verify_jwt),
) -> FraudScoreResponse:
    return await service.score_transaction(request)


@router.post(
    "/score/batch",
    response_model=FraudBatchScoreResponse,
    summary="Batch score transactions for fraud risk",
)
async def batch_score_transactions(
    request: FraudBatchScoreRequest,
    service: FraudScoringService = Depends(get_fraud_service),
    _auth: dict = Depends(verify_jwt),
) -> FraudBatchScoreResponse:
    start = time.time()
    results = []
    failed = 0

    for tx in request.transactions:
        try:
            result = await service.score_transaction(tx)
            results.append(result)
        except Exception:
            failed += 1

    duration_ms = int((time.time() - start) * 1000)

    return FraudBatchScoreResponse(
        total=len(request.transactions),
        processed=len(results),
        failed=failed,
        results=results,
        model_version=results[0].model_version if results else "unknown",
        duration_ms=duration_ms,
    )


@router.get(
    "/explanation/{transaction_id}",
    response_model=FraudExplanationResponse,
    summary="Get fraud risk explanation for a transaction",
    description="Returns SHAP-based feature contributions explaining the fraud risk score.",
)
async def get_explanation(
    transaction_id: str,
    service: FraudScoringService = Depends(get_fraud_service),
    _auth: dict = Depends(verify_jwt),
) -> FraudExplanationResponse:
    result = await service.explain_transaction(transaction_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction prediction not found")
    return result
