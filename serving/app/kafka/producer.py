"""Kafka Producer — Publishes fraud/AML decisions, alerts, and case events."""

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from loguru import logger

from serving.app.settings import get_settings

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        settings = get_settings()
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            client_id=f"{settings.KAFKA_CLIENT_ID}-producer",
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        await _producer.start()
        logger.info("Kafka producer started")
    return _producer


async def stop_producer() -> None:
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")


async def publish_fraud_decision(transaction_id: str, decision: dict) -> None:
    """Publish fraud decision to fraud.decisions topic."""
    producer = await get_producer()
    await producer.send_and_wait(
        "fraud.decisions",
        key=transaction_id,
        value=decision,
    )


async def publish_fraud_review(transaction_id: str, review_data: dict) -> None:
    """Publish to fraud.review.queue for manual review."""
    producer = await get_producer()
    await producer.send_and_wait(
        "fraud.review.queue",
        key=transaction_id,
        value=review_data,
    )


async def publish_fraud_alert(alert_data: dict) -> None:
    """Publish fraud alert (critical risk, drift, service failure)."""
    producer = await get_producer()
    await producer.send_and_wait(
        "fraud.alerts",
        value=alert_data,
    )


async def publish_aml_scored(transaction_id: str, result: dict) -> None:
    """Publish AML scoring result."""
    producer = await get_producer()
    await producer.send_and_wait(
        "aml.scored",
        key=transaction_id,
        value=result,
    )


async def publish_aml_blocked(transaction_id: str, block_data: dict) -> None:
    """Publish AML block decision."""
    producer = await get_producer()
    await producer.send_and_wait(
        "aml.blocked",
        key=transaction_id,
        value=block_data,
    )


async def publish_aml_case(case_data: dict) -> None:
    """Publish AML case creation event."""
    producer = await get_producer()
    await producer.send_and_wait(
        "aml.case.created",
        value=case_data,
    )


async def publish_merchant_risk_scored(merchant_id: str, result: dict) -> None:
    """Publish merchant risk scoring result."""
    producer = await get_producer()
    await producer.send_and_wait(
        "merchant.risk.scored",
        key=merchant_id,
        value=result,
    )


async def publish_merchant_tier_change(merchant_id: str, tier_data: dict) -> None:
    """Publish merchant tier change event."""
    producer = await get_producer()
    await producer.send_and_wait(
        "merchant.tier.changed",
        key=merchant_id,
        value=tier_data,
    )


async def publish_merchant_blocked(merchant_id: str, block_data: dict) -> None:
    """Publish merchant blocked event."""
    producer = await get_producer()
    await producer.send_and_wait(
        "merchant.risk.blocked",
        key=merchant_id,
        value=block_data,
    )
