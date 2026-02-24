"""Regulatory Export Service — SAR/STR/CTR export for compliance reporting."""

import csv
import hashlib
import io
import json
import os
import uuid
from datetime import datetime
from typing import Any

from loguru import logger


class RegulatoryExportService:
    """Generates regulatory compliance reports (SAR, STR, CTR) in CSV/JSON format."""

    def __init__(self, db_pool: Any = None, export_dir: str = "./exports"):
        self._db = db_pool
        self._export_dir = export_dir

    async def export_sar(
        self,
        start_date: datetime,
        end_date: datetime,
        export_format: str = "CSV",
        min_risk_score: float = 0.70,
    ) -> dict:
        """Export Suspicious Activity Reports (SAR)."""
        records = await self._fetch_sar_records(start_date, end_date, min_risk_score)

        if export_format.upper() == "CSV":
            content, file_path = self._generate_csv(records, "SAR", start_date, end_date)
        else:
            content, file_path = self._generate_json(records, "SAR", start_date, end_date)

        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # Log export
        await self._log_export("SAR", export_format, len(records), start_date, end_date, file_path, file_hash)

        return {
            "export_type": "SAR",
            "format": export_format,
            "record_count": len(records),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "file_path": file_path,
            "file_hash": file_hash,
        }

    async def export_str(
        self,
        start_date: datetime,
        end_date: datetime,
        export_format: str = "CSV",
    ) -> dict:
        """Export Suspicious Transaction Reports (STR)."""
        records = await self._fetch_str_records(start_date, end_date)

        if export_format.upper() == "CSV":
            content, file_path = self._generate_csv(records, "STR", start_date, end_date)
        else:
            content, file_path = self._generate_json(records, "STR", start_date, end_date)

        file_hash = hashlib.sha256(content.encode()).hexdigest()
        await self._log_export("STR", export_format, len(records), start_date, end_date, file_path, file_hash)

        return {
            "export_type": "STR",
            "format": export_format,
            "record_count": len(records),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "file_path": file_path,
            "file_hash": file_hash,
        }

    async def _fetch_sar_records(self, start_date: datetime, end_date: datetime, min_score: float) -> list[dict]:
        """Fetch suspicious activity records from AML predictions and cases."""
        if not self._db:
            return []

        try:
            async with self._db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        ap.id as prediction_id,
                        ap.transaction_id,
                        ap.user_id,
                        ap.risk_score,
                        ap.risk_level,
                        ap.decision,
                        ap.top_risk_factors,
                        ap.created_at,
                        ac.id as case_id,
                        ac.status as case_status,
                        ac.trigger_reason
                    FROM aml_predictions ap
                    LEFT JOIN aml_cases ac ON ac.entity_id = ap.transaction_id
                    WHERE ap.risk_score >= $1
                      AND ap.created_at >= $2
                      AND ap.created_at <= $3
                    ORDER BY ap.risk_score DESC
                    """,
                    min_score,
                    start_date,
                    end_date,
                )

                return [
                    {
                        "prediction_id": str(row["prediction_id"]),
                        "transaction_id": str(row["transaction_id"]),
                        "user_id": str(row["user_id"]),
                        "risk_score": float(row["risk_score"]),
                        "risk_level": row["risk_level"],
                        "decision": row["decision"],
                        "risk_factors": row.get("top_risk_factors", []),
                        "case_id": str(row["case_id"]) if row["case_id"] else None,
                        "case_status": row.get("case_status"),
                        "trigger_reason": row.get("trigger_reason"),
                        "timestamp": row["created_at"].isoformat(),
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch SAR records: {e}")
            return []

    async def _fetch_str_records(self, start_date: datetime, end_date: datetime) -> list[dict]:
        """Fetch suspicious transaction records (BLOCK + REVIEW decisions)."""
        if not self._db:
            return []

        try:
            async with self._db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        id as prediction_id,
                        transaction_id,
                        user_id,
                        risk_score,
                        risk_level,
                        decision,
                        top_risk_factors,
                        created_at
                    FROM aml_predictions
                    WHERE decision IN ('BLOCK', 'REVIEW')
                      AND created_at >= $1
                      AND created_at <= $2
                    ORDER BY created_at DESC
                    """,
                    start_date,
                    end_date,
                )

                return [
                    {
                        "prediction_id": str(row["prediction_id"]),
                        "transaction_id": str(row["transaction_id"]),
                        "user_id": str(row["user_id"]),
                        "risk_score": float(row["risk_score"]),
                        "risk_level": row["risk_level"],
                        "decision": row["decision"],
                        "risk_factors": row.get("top_risk_factors", []),
                        "timestamp": row["created_at"].isoformat(),
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch STR records: {e}")
            return []

    def _generate_csv(self, records: list[dict], report_type: str, start: datetime, end: datetime) -> tuple[str, str]:
        """Generate CSV export file."""
        os.makedirs(self._export_dir, exist_ok=True)

        filename = f"{report_type}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        file_path = os.path.join(self._export_dir, filename)

        output = io.StringIO()
        if records:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            for record in records:
                # Convert lists to strings for CSV
                row = {k: json.dumps(v) if isinstance(v, list) else v for k, v in record.items()}
                writer.writerow(row)

        content = output.getvalue()

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Generated {report_type} CSV: {file_path} ({len(records)} records)")
        return content, file_path

    def _generate_json(self, records: list[dict], report_type: str, start: datetime, end: datetime) -> tuple[str, str]:
        """Generate JSON export file."""
        os.makedirs(self._export_dir, exist_ok=True)

        filename = f"{report_type}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
        file_path = os.path.join(self._export_dir, filename)

        export = {
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "record_count": len(records),
            "records": records,
        }

        content = json.dumps(export, indent=2, default=str)

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Generated {report_type} JSON: {file_path} ({len(records)} records)")
        return content, file_path

    async def _log_export(
        self,
        export_type: str,
        fmt: str,
        count: int,
        start: datetime,
        end: datetime,
        path: str,
        file_hash: str,
    ) -> None:
        """Log export to database for audit trail."""
        if not self._db:
            return

        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO regulatory_export_log
                    (id, export_type, format, record_count, date_range_start, date_range_end, file_path, file_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    str(uuid.uuid4()),
                    export_type,
                    fmt,
                    count,
                    start,
                    end,
                    path,
                    file_hash,
                )
        except Exception as e:
            logger.error(f"Failed to log export: {e}")
