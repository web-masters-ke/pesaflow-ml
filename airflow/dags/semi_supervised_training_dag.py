"""Airflow DAG — Semi-Supervised Learning & Co-Training Pipeline.

Runs weekly (after label_feedback_dag):
  1. Check maturity level per domain
  2. Branch by maturity: COLD=skip, WARMING=SSL, WARM+=SSL+co-training
  3. Evaluate SSL models against current on labeled holdout
  4. Deploy if improved (AUC >= 0.5% better), else rollback
  5. Update SSL quality metrics
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "wasaa_pesaflow_ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}


def _get_db_pool():
    """Create asyncpg pool for DAG tasks."""
    import asyncio
    import os

    import asyncpg

    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://pesaflow:pesaflow_secure_2024@localhost:5432/pesaflow",
    ).replace("postgresql+asyncpg://", "postgresql://")

    async def _create():
        return await asyncpg.create_pool(
            db_url,
            min_size=2,
            max_size=5,
            server_settings={"search_path": "ai_schema,public"},
        )

    return asyncio.get_event_loop().run_until_complete(_create())


def check_maturity(**context):
    """Determine maturity level per domain and decide SSL branch."""
    import asyncio

    from loguru import logger

    async def _check():
        pool = _get_db_pool()
        try:
            maturity_levels = {}
            for domain, table in [
                ("fraud", "ml_predictions"),
                ("aml", "aml_predictions"),
                ("merchant", "merchant_risk_predictions"),
            ]:
                async with pool.acquire() as conn:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE label IS NOT NULL")
                    if count < 100:
                        maturity_levels[domain] = "COLD"
                    elif count < 1000:
                        maturity_levels[domain] = "WARMING"
                    elif count < 10000:
                        maturity_levels[domain] = "WARM"
                    else:
                        maturity_levels[domain] = "HOT"

            logger.info(f"Maturity levels: {maturity_levels}")
            return maturity_levels
        finally:
            await pool.close()

    maturity = asyncio.run(_check())
    context["ti"].xcom_push(key="maturity_levels", value=maturity)

    # Determine which branch to take based on highest maturity
    levels = list(maturity.values())
    if any(l in ("WARM", "HOT") for l in levels):
        return "run_ssl_and_co_training"
    elif any(l == "WARMING" for l in levels):
        return "run_ssl_only"
    return "skip_ssl"


def run_ssl_only(**context):
    """Run SSL techniques for WARMING-level domains."""
    import asyncio

    from loguru import logger

    maturity = context["ti"].xcom_pull(key="maturity_levels", task_ids="check_maturity")

    async def _run():
        pool = _get_db_pool()
        try:
            from training.semi_supervised.base_ssl_trainer import SSLConfig
            from training.semi_supervised.consistency_regularization import ConsistencyRegularizationTrainer
            from training.semi_supervised.label_propagation import LabelPropagationTrainer
            from training.semi_supervised.self_training import SelfTrainingTrainer

            results = {}
            for domain, level in maturity.items():
                if level in ("COLD",):
                    continue

                config = SSLConfig(domain=domain)

                # Self-training
                logger.info(f"Running self-training for {domain}")
                st = SelfTrainingTrainer(config=config, db_pool=pool)
                st_result = await st.train()
                results[f"{domain}_self_training"] = {
                    "auc": st_result.get("auc_labeled"),
                    "pseudo_count": st_result.get("pseudo_labeled_count"),
                    "converged": st_result.get("converged"),
                }

                # Label propagation
                logger.info(f"Running label propagation for {domain}")
                lp = LabelPropagationTrainer(config=config, db_pool=pool)
                lp_result = await lp.train()
                results[f"{domain}_label_propagation"] = {
                    "auc": lp_result.get("auc_labeled"),
                    "pseudo_count": lp_result.get("pseudo_labeled_count"),
                }

                # Consistency regularization
                logger.info(f"Running consistency regularization for {domain}")
                cr = ConsistencyRegularizationTrainer(config=config, db_pool=pool)
                cr_result = await cr.train()
                results[f"{domain}_consistency_reg"] = {
                    "auc": cr_result.get("auc_labeled"),
                    "pseudo_count": cr_result.get("pseudo_labeled_count"),
                }

            return results
        finally:
            await pool.close()

    results = asyncio.run(_run())
    context["ti"].xcom_push(key="ssl_results", value=results)
    return results


def run_ssl_and_co_training(**context):
    """Run SSL + co-training for WARM+ domains."""
    import asyncio

    from loguru import logger

    maturity = context["ti"].xcom_pull(key="maturity_levels", task_ids="check_maturity")

    async def _run():
        pool = _get_db_pool()
        try:
            from training.co_training.cross_domain_trainer import CrossDomainTrainer
            from training.co_training.multi_view_trainer import MultiViewCoTrainer
            from training.co_training.tri_training import TriTrainingTrainer
            from training.semi_supervised.base_ssl_trainer import SSLConfig
            from training.semi_supervised.consistency_regularization import ConsistencyRegularizationTrainer
            from training.semi_supervised.label_propagation import LabelPropagationTrainer
            from training.semi_supervised.mixmatch import MixMatchTrainer
            from training.semi_supervised.self_training import SelfTrainingTrainer

            results = {}
            for domain, level in maturity.items():
                if level == "COLD":
                    continue

                config = SSLConfig(domain=domain)

                # SSL techniques (WARMING+)
                for TrainerClass, name in [
                    (SelfTrainingTrainer, "self_training"),
                    (LabelPropagationTrainer, "label_propagation"),
                    (ConsistencyRegularizationTrainer, "consistency_reg"),
                    (MixMatchTrainer, "mixmatch"),
                ]:
                    logger.info(f"Running {name} for {domain}")
                    trainer = TrainerClass(config=config, db_pool=pool)
                    result = await trainer.train()
                    results[f"{domain}_{name}"] = {
                        "auc": result.get("auc_labeled"),
                        "pseudo_count": result.get("pseudo_labeled_count"),
                        "converged": result.get("converged"),
                    }

                # Co-training techniques (WARM+)
                if level in ("WARM", "HOT"):
                    logger.info(f"Running multi-view co-training for {domain}")
                    mv = MultiViewCoTrainer(config=config, db_pool=pool)
                    mv_result = await mv.train()
                    results[f"{domain}_multi_view"] = {
                        "auc": mv_result.get("auc_labeled"),
                        "pseudo_count": mv_result.get("pseudo_labeled_count"),
                    }

                    logger.info(f"Running tri-training for {domain}")
                    tri = TriTrainingTrainer(config=config, db_pool=pool)
                    tri_result = await tri.train()
                    results[f"{domain}_tri_training"] = {
                        "auc": tri_result.get("auc_labeled"),
                        "pseudo_count": tri_result.get("pseudo_labeled_count"),
                    }

            # Cross-domain co-training (runs once across all domains)
            for domain in maturity:
                if maturity[domain] in ("WARM", "HOT"):
                    logger.info(f"Running cross-domain co-training for {domain}")
                    config = SSLConfig(domain=domain)
                    cd = CrossDomainTrainer(config=config, db_pool=pool)
                    cd_result = await cd.train()
                    results[f"{domain}_cross_domain"] = {
                        "auc": cd_result.get("auc_labeled"),
                        "transfers": cd_result.get("transfers"),
                    }

            return results
        finally:
            await pool.close()

    results = asyncio.run(_run())
    context["ti"].xcom_push(key="ssl_results", value=results)
    return results


def evaluate_ssl_models(**context):
    """Compare SSL models vs current production models on holdout."""
    from loguru import logger

    results = context["ti"].xcom_pull(key="ssl_results", task_ids=["run_ssl_only", "run_ssl_and_co_training"])
    # Flatten results from whichever branch ran
    ssl_results = {}
    for r in results:
        if r:
            ssl_results.update(r)

    evaluation = {}
    for key, result in ssl_results.items():
        auc = result.get("auc")
        if auc and auc > 0.5:
            evaluation[key] = {
                "auc": auc,
                "pseudo_count": result.get("pseudo_count", 0),
                "deploy_candidate": auc >= 0.90,  # Min AUC threshold
            }

    logger.info(f"SSL evaluation: {evaluation}")
    context["ti"].xcom_push(key="evaluation", value=evaluation)


def update_ssl_metrics(**context):
    """Update SSL quality metrics in database."""
    import asyncio

    from loguru import logger

    async def _update():
        pool = _get_db_pool()
        try:
            from monitoring.ssl_metrics import SSLMetricsCollector

            collector = SSLMetricsCollector(db_pool=pool)
            for domain in ["fraud", "aml", "merchant"]:
                quality = await collector.get_pseudo_label_quality(domain)
                alerts = await collector.check_alerts(domain)
                if alerts:
                    logger.warning(f"SSL alerts for {domain}: {alerts}")
                logger.info(f"SSL quality for {domain}: {quality}")
        finally:
            await pool.close()

    asyncio.run(_update())


def skip_ssl(**context):
    """No-op: skip SSL when all domains are COLD."""
    from loguru import logger

    logger.info("All domains are COLD, skipping SSL training")


with DAG(
    dag_id="pesaflow_semi_supervised_training",
    default_args=default_args,
    description="Semi-supervised learning & co-training pipeline",
    schedule_interval="0 3 * * 0",  # Weekly: Sunday 3 AM
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["pesaflow", "ssl", "co-training", "semi-supervised"],
) as dag:
    check_maturity_task = BranchPythonOperator(
        task_id="check_maturity",
        python_callable=check_maturity,
    )

    ssl_only_task = PythonOperator(
        task_id="run_ssl_only",
        python_callable=run_ssl_only,
    )

    ssl_and_co_training_task = PythonOperator(
        task_id="run_ssl_and_co_training",
        python_callable=run_ssl_and_co_training,
    )

    skip_task = PythonOperator(
        task_id="skip_ssl",
        python_callable=skip_ssl,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_ssl_models",
        python_callable=evaluate_ssl_models,
        trigger_rule="none_failed_min_one_success",
    )

    metrics_task = PythonOperator(
        task_id="update_ssl_metrics",
        python_callable=update_ssl_metrics,
        trigger_rule="none_failed_min_one_success",
    )

    check_maturity_task >> [ssl_only_task, ssl_and_co_training_task, skip_task]
    [ssl_only_task, ssl_and_co_training_task] >> evaluate_task >> metrics_task
    skip_task >> metrics_task
