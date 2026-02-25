"""Airflow DAG — Label Feedback & Active Learning Pipeline.

Runs every 15 minutes to:
  1. Propagate labels from fraud review cases and AML cases
  2. Refresh active learning queues for all 3 domains
  3. Expire stale queue items
  4. Refresh the label_statistics materialized view
"""

import asyncio
import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

default_args = {
    "owner": "wasaa_pesaflow_ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}


def _get_connection_params():
    """Read DATABASE_URL and REDIS_URL from environment variables."""
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://pesaflow:pesaflow_secure_2024@localhost:5432/pesaflow",
    )
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Strip SQLAlchemy prefix — asyncpg needs raw postgresql:// URI
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    return database_url, redis_url


async def _create_db_pool(database_url: str):
    """Create an asyncpg connection pool."""
    import asyncpg

    return await asyncpg.create_pool(
        database_url,
        min_size=1,
        max_size=3,
        server_settings={"search_path": "ai_schema,public"},
    )


async def _create_redis_client(redis_url: str):
    """Create a redis.asyncio client."""
    import redis.asyncio as aioredis

    return aioredis.from_url(redis_url, decode_responses=True)


# ─── Task: Propagate Labels ───────────────────────────────────────────


async def _propagate_labels_async():
    """Propagate fraud review cases + AML cases into labels."""
    from loguru import logger

    from serving.app.services.label_feedback_service import LabelFeedbackService

    database_url, redis_url = _get_connection_params()
    pool = await _create_db_pool(database_url)
    redis_client = await _create_redis_client(redis_url)

    try:
        service = LabelFeedbackService(db_pool=pool, redis_client=redis_client)

        fraud_count = await service.propagate_fraud_review_labels()
        logger.info(f"Propagated {fraud_count} fraud review labels")

        aml_count = await service.propagate_aml_case_labels()
        logger.info(f"Propagated {aml_count} AML case labels")

        logger.info(f"Label propagation complete: fraud={fraud_count}, aml={aml_count}")
    finally:
        await pool.close()
        await redis_client.aclose()


def propagate_labels(**context):
    """Propagate fraud review cases + AML cases into labels."""
    asyncio.run(_propagate_labels_async())


# ─── Task: Refresh Active Learning Queues ──────────────────────────────


async def _refresh_al_queues_async():
    """Recompute active learning queues for all 3 domains."""
    from loguru import logger

    from serving.app.schemas.labels import Domain
    from serving.app.services.active_learning_service import ActiveLearningService

    database_url, redis_url = _get_connection_params()
    pool = await _create_db_pool(database_url)
    redis_client = await _create_redis_client(redis_url)

    try:
        service = ActiveLearningService(db_pool=pool, redis_client=redis_client)

        for domain in [Domain.FRAUD, Domain.AML, Domain.MERCHANT]:
            added = await service.refresh_queue(domain)
            logger.info(f"AL queue refresh [{domain.value}]: {added} items added")
    finally:
        await pool.close()
        await redis_client.aclose()


def refresh_al_queues(**context):
    """Recompute active learning queues for all 3 domains."""
    asyncio.run(_refresh_al_queues_async())


# ─── Task: Expire Stale Items ──────────────────────────────────────────


async def _expire_stale_items_async():
    """Clean up expired active learning queue items."""
    from loguru import logger

    from serving.app.services.active_learning_service import ActiveLearningService

    database_url, redis_url = _get_connection_params()
    pool = await _create_db_pool(database_url)
    redis_client = await _create_redis_client(redis_url)

    try:
        service = ActiveLearningService(db_pool=pool, redis_client=redis_client)
        expired = await service.expire_stale_items()
        logger.info(f"Expired {expired} stale AL queue items")
    finally:
        await pool.close()
        await redis_client.aclose()


def expire_stale_items(**context):
    """Clean up expired active learning queue items."""
    asyncio.run(_expire_stale_items_async())


# ─── Task: Refresh Label Statistics ────────────────────────────────────


async def _refresh_label_statistics_async():
    """Refresh the label_statistics materialized view."""
    from loguru import logger

    database_url, _ = _get_connection_params()
    pool = await _create_db_pool(database_url)

    try:
        async with pool.acquire() as conn:
            try:
                await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY label_statistics")
                logger.info("Refreshed label_statistics materialized view (CONCURRENTLY)")
            except Exception:
                await conn.execute("REFRESH MATERIALIZED VIEW label_statistics")
                logger.info("Refreshed label_statistics materialized view (non-concurrent)")
    finally:
        await pool.close()


def refresh_label_statistics(**context):
    """Refresh the label_statistics materialized view."""
    asyncio.run(_refresh_label_statistics_async())


# ─── DAG Definition ────────────────────────────────────────────────────

with DAG(
    dag_id="pesaflow_label_feedback",
    description="Label propagation, active learning queue refresh, and statistics (every 15 min)",
    schedule="*/15 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["pesaflow", "labels", "active-learning"],
) as dag:

    propagate_labels_task = PythonOperator(
        task_id="propagate_labels",
        python_callable=propagate_labels,
    )

    refresh_al_queues_task = PythonOperator(
        task_id="refresh_al_queues",
        python_callable=refresh_al_queues,
    )

    expire_stale_items_task = PythonOperator(
        task_id="expire_stale_items",
        python_callable=expire_stale_items,
    )

    refresh_label_statistics_task = PythonOperator(
        task_id="refresh_label_statistics",
        python_callable=refresh_label_statistics,
    )

    # Labels must be propagated first, then queues refreshed, then cleanup + stats
    propagate_labels_task >> refresh_al_queues_task >> [expire_stale_items_task, refresh_label_statistics_task]
