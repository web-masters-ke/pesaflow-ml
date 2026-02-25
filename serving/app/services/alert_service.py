"""Alert Service — Multi-channel alerting (Slack, Email, PagerDuty, Kafka)."""

import json
import uuid
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from serving.app.kafka.producer import get_producer
from serving.app.settings import get_settings


class AlertService:
    """Multi-channel alert dispatcher for critical risk events."""

    def __init__(self, db_pool: Any = None):
        self._db = db_pool
        self._settings = get_settings()

    async def send_alert(
        self,
        alert_type: str,
        severity: str,
        subject: str,
        message: str,
        channels: list[str] | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> list[dict]:
        """Send alert to configured channels."""
        if channels is None:
            channels = self._get_default_channels(severity)

        results = []
        for channel in channels:
            delivered = False
            try:
                if channel == "SLACK":
                    delivered = await self._send_slack(subject, message, severity)
                elif channel == "EMAIL":
                    delivered = await self._send_email(subject, message, severity)
                elif channel == "PAGERDUTY":
                    delivered = await self._send_pagerduty(subject, message, severity)
                elif channel == "KAFKA":
                    delivered = await self._send_kafka(alert_type, subject, message, severity, entity_type, entity_id)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")

            alert_id = str(uuid.uuid4())
            await self._log_alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                channel=channel,
                subject=subject,
                message=message,
                entity_type=entity_type,
                entity_id=entity_id,
                delivered=delivered,
            )

            results.append({"channel": channel, "delivered": delivered, "alert_id": alert_id})

        return results

    async def _send_slack(self, subject: str, message: str, severity: str) -> bool:
        """Send alert to Slack via webhook."""
        webhook_url = self._settings.ALERT_SLACK_WEBHOOK_URL
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        emoji = {"CRITICAL": ":rotating_light:", "WARNING": ":warning:", "INFO": ":information_source:"}.get(
            severity, ":bell:"
        )

        payload = {
            "text": f"{emoji} *{subject}*",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} {subject}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message},
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:* {severity} | *Time:* {datetime.utcnow().isoformat()}Z",
                        },
                    ],
                },
            ],
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload, timeout=10)
            return resp.status_code == 200

    async def _send_email(self, subject: str, message: str, severity: str) -> bool:
        """Send alert via email (SMTP)."""
        smtp_host = self._settings.ALERT_EMAIL_SMTP_HOST
        email_to = self._settings.ALERT_EMAIL_TO
        if not smtp_host or not email_to:
            logger.warning("Email SMTP not configured")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(f"Severity: {severity}\n\n{message}")
            msg["Subject"] = f"[Pesaflow ML Alert] [{severity}] {subject}"
            msg["From"] = self._settings.ALERT_EMAIL_FROM
            msg["To"] = email_to

            with smtplib.SMTP(smtp_host, self._settings.ALERT_EMAIL_SMTP_PORT) as server:
                server.starttls()
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False

    async def _send_pagerduty(self, subject: str, message: str, severity: str) -> bool:
        """Send alert to PagerDuty."""
        api_key = self._settings.ALERT_PAGERDUTY_API_KEY
        service_id = self._settings.ALERT_PAGERDUTY_SERVICE_ID
        if not api_key or not service_id:
            logger.warning("PagerDuty not configured")
            return False

        pd_severity = {"CRITICAL": "critical", "WARNING": "warning", "INFO": "info"}.get(severity, "warning")

        payload = {
            "routing_key": api_key,
            "event_action": "trigger",
            "payload": {
                "summary": subject,
                "severity": pd_severity,
                "source": "pesaflow-ml",
                "custom_details": {"message": message},
            },
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10,
            )
            return resp.status_code in (200, 202)

    async def _send_kafka(
        self,
        alert_type: str,
        subject: str,
        message: str,
        severity: str,
        entity_type: str | None,
        entity_id: str | None,
    ) -> bool:
        """Publish alert event to Kafka."""
        try:
            producer = await get_producer()
            await producer.send_and_wait(
                "ml.alerts",
                value={
                    "alert_type": alert_type,
                    "severity": severity,
                    "subject": subject,
                    "message": message,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            return True
        except Exception as e:
            logger.error(f"Kafka alert failed: {e}")
            return False

    def _get_default_channels(self, severity: str) -> list[str]:
        """Get default alert channels based on severity."""
        if severity == "CRITICAL":
            return ["SLACK", "PAGERDUTY", "KAFKA"]
        elif severity == "WARNING":
            return ["SLACK", "KAFKA"]
        return ["KAFKA"]

    async def _log_alert(
        self,
        alert_id: str,
        alert_type: str,
        severity: str,
        channel: str,
        subject: str,
        message: str,
        entity_type: str | None,
        entity_id: str | None,
        delivered: bool,
    ) -> None:
        """Log alert to database."""
        if not self._db:
            return

        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO alert_log
                    (id, alert_type, severity, channel, subject, message,
                     entity_type, entity_id, delivered, delivered_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    alert_id,
                    alert_type,
                    severity,
                    channel,
                    subject,
                    message,
                    entity_type,
                    entity_id,
                    delivered,
                    datetime.utcnow() if delivered else None,
                )
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
