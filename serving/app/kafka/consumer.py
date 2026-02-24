"""Kafka Consumer — Consumes transaction events for real-time scoring."""

import asyncio
import json

from aiokafka import AIOKafkaConsumer
from loguru import logger

from serving.app.api.dependencies import _container
from serving.app.kafka.producer import (
    publish_aml_blocked,
    publish_aml_scored,
    publish_fraud_decision,
    publish_fraud_review,
    publish_merchant_blocked,
    publish_merchant_risk_scored,
)
from serving.app.schemas.aml import AMLScoreRequest
from serving.app.schemas.fraud import FraudScoreRequest
from serving.app.schemas.merchant import MerchantScoreRequest
from serving.app.settings import get_settings

_consumer: AIOKafkaConsumer | None = None
_consumer_task: asyncio.Task | None = None


async def start_kafka_consumer() -> None:
    """Start Kafka consumer for transaction events."""
    global _consumer, _consumer_task

    settings = get_settings()
    _consumer = AIOKafkaConsumer(
        "wallet.transactions.created",
        "transactions.created",
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_GROUP_ID,
        client_id=f"{settings.KAFKA_CLIENT_ID}-consumer",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    await _consumer.start()
    logger.info("Kafka consumer started, listening on wallet.transactions.created, transactions.created")

    _consumer_task = asyncio.create_task(_consume_loop())


async def stop_kafka_consumer() -> None:
    """Stop Kafka consumer gracefully."""
    global _consumer, _consumer_task

    if _consumer_task:
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass

    if _consumer:
        await _consumer.stop()
        _consumer = None
        logger.info("Kafka consumer stopped")


async def _consume_loop() -> None:
    """Main consumer loop — processes transaction events."""
    while True:
        try:
            async for message in _consumer:
                try:
                    await _process_transaction_event(message.topic, message.value)
                except Exception as e:
                    logger.error(f"Error processing message from {message.topic}: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Kafka consumer loop error: {e}")
            await asyncio.sleep(5)  # Backoff before retry


async def _process_transaction_event(topic: str, data: dict) -> None:
    """Process a single transaction event — run fraud + AML scoring."""
    transaction_id = data.get("transaction_id") or data.get("id")
    if not transaction_id:
        logger.warning(f"Received message without transaction_id on {topic}")
        return

    logger.debug(f"Processing transaction {transaction_id} from {topic}")

    # === Fraud Scoring ===
    if _container.fraud_scoring_service and _container.fraud_model and _container.fraud_model.is_loaded:
        try:
            fraud_request = FraudScoreRequest(
                transaction_id=transaction_id,
                user_id=data.get("user_id", data.get("sender_id")),
                amount=float(data.get("amount", 0)),
                currency=data.get("currency", "KES"),
                transaction_type=data.get("transaction_type", data.get("type", "UNKNOWN")),
                device_fingerprint=data.get("device_fingerprint", data.get("device_id")),
                ip_address=data.get("ip_address"),
                timestamp=data.get("timestamp"),
            )

            fraud_result = await _container.fraud_scoring_service.score_transaction(fraud_request)

            # Publish fraud decision
            await publish_fraud_decision(
                str(transaction_id),
                fraud_result.model_dump(mode="json"),
            )

            # If REVIEW, send to review queue
            if fraud_result.decision.value == "REVIEW":
                await publish_fraud_review(
                    str(transaction_id),
                    fraud_result.model_dump(mode="json"),
                )

            logger.info(
                f"Fraud scored: tx={transaction_id} score={fraud_result.risk_score} decision={fraud_result.decision.value}"
            )

        except Exception as e:
            logger.error(f"Fraud scoring failed for {transaction_id}: {e}")

    # === AML Scoring ===
    if _container.aml_scoring_service and _container.aml_model and _container.aml_model.is_loaded:
        try:
            aml_request = AMLScoreRequest(
                transaction_id=transaction_id,
                user_id=data.get("user_id", data.get("sender_id")),
                sender_wallet_id=data.get("sender_wallet_id"),
                receiver_wallet_id=data.get("receiver_wallet_id"),
                amount=float(data.get("amount", 0)),
                currency=data.get("currency", "KES"),
                device_id=data.get("device_fingerprint", data.get("device_id")),
                ip_address=data.get("ip_address"),
                channel=data.get("channel"),
                timestamp=data.get("timestamp"),
            )

            aml_result = await _container.aml_scoring_service.score_transaction(aml_request)

            # Publish AML result
            await publish_aml_scored(
                str(transaction_id),
                aml_result.model_dump(mode="json"),
            )

            # If BLOCK, publish block event
            if aml_result.decision.value == "BLOCK":
                await publish_aml_blocked(
                    str(transaction_id),
                    aml_result.model_dump(mode="json"),
                )

            logger.info(
                f"AML scored: tx={transaction_id} score={aml_result.risk_score} decision={aml_result.decision.value}"
            )

        except Exception as e:
            logger.error(f"AML scoring failed for {transaction_id}: {e}")

    # === Merchant Risk Scoring ===
    merchant_id = data.get("merchant_id")
    if merchant_id and _container.merchant_risk_service and _container.merchant_model and _container.merchant_model.is_loaded:
        try:
            merchant_request = MerchantScoreRequest(
                merchant_id=merchant_id,
                transaction_id=transaction_id,
                amount=float(data.get("amount", 0)),
                currency=data.get("currency", "KES"),
                customer_id=data.get("user_id", data.get("sender_id")),
                timestamp=data.get("timestamp"),
            )

            merchant_result = await _container.merchant_risk_service.score_merchant(merchant_request)

            await publish_merchant_risk_scored(
                str(merchant_id),
                merchant_result.model_dump(mode="json"),
            )

            if merchant_result.decision.value == "BLOCK":
                await publish_merchant_blocked(
                    str(merchant_id),
                    merchant_result.model_dump(mode="json"),
                )

            logger.info(
                f"Merchant scored: merchant={merchant_id} score={merchant_result.risk_score} tier={merchant_result.merchant_tier}"
            )

        except Exception as e:
            logger.error(f"Merchant scoring failed for {merchant_id}: {e}")
