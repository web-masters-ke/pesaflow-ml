"""Airflow DAG — Fraud Model Retraining & Drift Detection Pipeline."""

from datetime import datetime, timedelta

from airflow.operators.python import BranchPythonOperator, PythonOperator

from airflow import DAG

default_args = {
    "owner": "wasaa_pesaflow_ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


def check_drift(**context):
    """Check model drift and decide whether to retrain."""
    import numpy as np

    from monitoring.drift_detector import FraudDriftDetector

    detector = FraudDriftDetector()
    # In production: load baseline from model registry, fetch recent scores from DB
    detector.set_baseline(
        score_distribution=np.random.beta(2, 5, 10000),
        accuracy=0.95,
        fraud_rate=0.05,
    )

    current_scores = np.random.beta(2, 5, 10000)  # In prod: fetch from predictions table
    results = detector.check_drift(current_scores, model_version="v1.0.0")

    context["ti"].xcom_push(key="drift_results", value=results)
    if results["retrain_recommended"]:
        return "retrain_fraud_model"
    return "skip_retraining"


def retrain_model(**context):
    """Retrain fraud model with latest data."""
    from training.fraud.train_fraud_model import run_training

    metrics = run_training(version="v1.1.0")
    context["ti"].xcom_push(key="training_metrics", value=metrics)


def evaluate_model(**context):
    """Evaluate new model against deployment thresholds."""
    metrics = context["ti"].xcom_pull(key="training_metrics", task_ids="retrain_fraud_model")
    if metrics and metrics.get("roc_auc", 0) >= 0.92 and metrics.get("precision", 0) >= 0.90:
        return "register_model"
    return "training_failed_alert"


def register_model(**context):
    """Register approved model in MLflow registry."""
    from loguru import logger

    logger.info("Model approved — registering in MLflow model registry")
    # In production: mlflow.register_model(...)


def skip_retraining(**context):
    """No drift detected, skip retraining."""
    from loguru import logger

    logger.info("No drift detected — skipping retraining")


def training_failed_alert(**context):
    """Alert team that retraining produced subpar model."""
    from loguru import logger

    logger.warning("Retrained model did not meet deployment thresholds")


with DAG(
    dag_id="pesaflow_fraud_model_pipeline",
    description="Weekly fraud model drift check and conditional retraining",
    schedule="0 2 * * 1",  # Every Monday 2 AM
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["pesaflow", "fraud", "ml", "retraining"],
) as dag:

    check_drift_task = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    skip_task = PythonOperator(
        task_id="skip_retraining",
        python_callable=skip_retraining,
    )

    retrain_task = PythonOperator(
        task_id="retrain_fraud_model",
        python_callable=retrain_model,
    )

    evaluate_task = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    alert_task = PythonOperator(
        task_id="training_failed_alert",
        python_callable=training_failed_alert,
    )

    check_drift_task >> [retrain_task, skip_task]
    retrain_task >> evaluate_task >> [register_task, alert_task]
