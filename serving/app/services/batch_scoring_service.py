"""Batch Scoring Service — Re-scores historical transactions for fraud, AML, and merchant risk."""

import time
import uuid
from datetime import datetime
from typing import Any

from loguru import logger

from serving.app.services.aml_scoring_service import AMLScoringService
from serving.app.services.fraud_scoring_service import FraudScoringService
from serving.app.services.merchant_risk_service import MerchantRiskService


class BatchScoringService:
    """Batch re-scoring service for fraud, AML, and merchant models."""

    def __init__(
        self,
        fraud_service: FraudScoringService | None = None,
        aml_service: AMLScoringService | None = None,
        merchant_service: MerchantRiskService | None = None,
        db_pool: Any = None,
    ):
        self._fraud = fraud_service
        self._aml = aml_service
        self._merchant = merchant_service
        self._db = db_pool

    async def run_batch_job(
        self,
        job_type: str,
        requests: list,
        initiated_by: str | None = None,
    ) -> dict:
        """Execute a batch scoring job."""
        job_id = str(uuid.uuid4())
        start = time.time()

        # Record job start
        await self._record_job_start(job_id, job_type, len(requests), initiated_by)

        results = []
        failed = 0

        for req in requests:
            try:
                result: Any = None
                if job_type == "FRAUD_RESCORE" and self._fraud:
                    result = await self._fraud.score_transaction(req)
                elif job_type == "AML_RESCORE" and self._aml:
                    result = await self._aml.score_transaction(req)
                elif job_type == "MERCHANT_RESCORE" and self._merchant:
                    result = await self._merchant.score_merchant(req)
                else:
                    failed += 1
                    continue

                results.append(result)
            except Exception as e:
                failed += 1
                logger.error(f"Batch item failed: {e}")

        duration_ms = int((time.time() - start) * 1000)

        # Record job completion
        await self._record_job_complete(job_id, len(results), failed)

        return {
            "job_id": job_id,
            "job_type": job_type,
            "total": len(requests),
            "processed": len(results),
            "failed": failed,
            "duration_ms": duration_ms,
            "results": results,
        }

    async def get_job_status(self, job_id: str) -> dict | None:
        """Get batch job status."""
        if not self._db:
            return None

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM batch_scoring_jobs WHERE id = $1",
                    job_id,
                )
                if row:
                    return {
                        "job_id": str(row["id"]),
                        "job_type": row["job_type"],
                        "status": row["status"],
                        "total_records": row["total_records"],
                        "processed_records": row["processed_records"],
                        "failed_records": row["failed_records"],
                        "started_at": row.get("started_at"),
                        "completed_at": row.get("completed_at"),
                    }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
        return None

    async def _record_job_start(self, job_id: str, job_type: str, total: int, initiated_by: str | None) -> None:
        if not self._db:
            return
        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO batch_scoring_jobs
                    (id, job_type, status, total_records, started_at, initiated_by)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    job_id,
                    job_type,
                    "RUNNING",
                    total,
                    datetime.utcnow(),
                    initiated_by,
                )
        except Exception as e:
            logger.error(f"Failed to record job start: {e}")

    async def _record_job_complete(self, job_id: str, processed: int, failed: int) -> None:
        if not self._db:
            return
        try:
            async with self._db.acquire() as conn:
                status = "COMPLETED" if failed == 0 else "COMPLETED" if processed > 0 else "FAILED"
                await conn.execute(
                    """
                    UPDATE batch_scoring_jobs
                    SET status = $2, processed_records = $3, failed_records = $4, completed_at = $5
                    WHERE id = $1
                    """,
                    job_id,
                    status,
                    processed,
                    failed,
                    datetime.utcnow(),
                )
        except Exception as e:
            logger.error(f"Failed to record job completion: {e}")
