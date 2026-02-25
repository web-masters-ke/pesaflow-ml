"""Airflow DAG — AML Model Retraining & Drift Detection Pipeline."""

from datetime import datetime, timedelta

from airflow.operators.python import BranchPythonOperator, PythonOperator

from airflow import DAG

default_args = {
    "owner": "wasaa_pesaflow_ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


def check_aml_drift(**context):
    """Check AML model drift."""
    import numpy as np

    from monitoring.drift_detector import AMLDriftDetector

    detector = AMLDriftDetector()
    detector.set_baseline(
        score_distribution=np.random.beta(1.5, 8, 10000),
        accuracy=0.94,
        fraud_rate=0.03,
    )

    current_scores = np.random.beta(1.5, 8, 10000)
    results = detector.check_drift(current_scores, model_version="v1.0.0")

    context["ti"].xcom_push(key="drift_results", value=results)
    if results["retrain_recommended"]:
        return "retrain_aml_model"
    return "skip_aml_retraining"


def retrain_aml_model(**context):
    """Retrain AML model."""
    from training.aml.train_aml_model import run_training

    metrics = run_training(version="v1.1.0")
    context["ti"].xcom_push(key="training_metrics", value=metrics)


def evaluate_aml_model(**context):
    """Evaluate retrained AML model."""
    metrics = context["ti"].xcom_pull(key="training_metrics", task_ids="retrain_aml_model")
    if metrics and metrics.get("roc_auc", 0) >= 0.90 and metrics.get("precision", 0) >= 0.88:
        return "register_aml_model"
    return "aml_training_failed_alert"


def register_aml_model(**context):
    from loguru import logger

    logger.info("AML model approved — registering")


def skip_aml_retraining(**context):
    from loguru import logger

    logger.info("No AML drift — skipping")


def aml_training_failed_alert(**context):
    from loguru import logger

    logger.warning("AML retrained model below threshold")


with DAG(
    dag_id="pesaflow_aml_model_pipeline",
    description="Weekly AML model drift check and conditional retraining",
    schedule="0 3 * * 1",  # Every Monday 3 AM
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["pesaflow", "aml", "ml", "retraining"],
) as dag:

    check_task = BranchPythonOperator(
        task_id="check_aml_drift",
        python_callable=check_aml_drift,
    )

    skip_task = PythonOperator(
        task_id="skip_aml_retraining",
        python_callable=skip_aml_retraining,
    )

    retrain_task = PythonOperator(
        task_id="retrain_aml_model",
        python_callable=retrain_aml_model,
    )

    evaluate_task = BranchPythonOperator(
        task_id="evaluate_aml_model",
        python_callable=evaluate_aml_model,
    )

    register_task = PythonOperator(
        task_id="register_aml_model",
        python_callable=register_aml_model,
    )

    alert_task = PythonOperator(
        task_id="aml_training_failed_alert",
        python_callable=aml_training_failed_alert,
    )

    check_task >> [retrain_task, skip_task]
    retrain_task >> evaluate_task >> [register_task, alert_task]
