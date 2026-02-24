"""Synthetic Data Generator — Creates realistic fraud/AML training data for initial model bootstrap."""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger


def generate_fraud_dataset(n_samples: int = 50000, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic fraud detection training data.

    Fraud rate ~5% reflects real-world class imbalance.
    Features mimic Pesaflow transaction patterns.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    data = []

    # === Legitimate transactions ===
    for _ in range(n_legit):
        data.append(_generate_legit_transaction())

    # === Fraudulent transactions ===
    for _ in range(n_fraud):
        data.append(_generate_fraud_transaction())

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Generated fraud dataset: {len(df)} samples, {n_fraud} fraud ({fraud_rate*100:.1f}%)")
    return df


def _generate_legit_transaction() -> dict:
    return {
        "avg_transaction_amount_7d": np.random.lognormal(7.5, 1.0),
        "transaction_velocity_1h": np.random.poisson(2),
        "transaction_velocity_24h": np.random.poisson(8),
        "failed_login_attempts_24h": np.random.choice([0, 0, 0, 0, 1], p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        "account_age_days": np.random.randint(30, 2000),
        "historical_fraud_flag": 0,
        "device_risk_score": np.random.beta(2, 10),
        "device_fraud_count": 0,
        "distinct_user_count": np.random.choice([1, 1, 1, 2], p=[0.8, 0.1, 0.05, 0.05]),
        "amount": np.random.lognormal(7.5, 1.2),
        "geo_distance_from_last_tx": np.random.exponential(10),
        "time_of_day": np.random.uniform(0.25, 0.85),  # Mostly business hours
        "currency_risk": np.random.choice([0.1, 0.1, 0.1, 0.4], p=[0.7, 0.15, 0.1, 0.05]),
        "is_fraud": 0,
    }


def _generate_fraud_transaction() -> dict:
    fraud_type = random.choice(["high_amount", "velocity", "device", "ato", "new_account"])

    base = _generate_legit_transaction()
    base["is_fraud"] = 1

    if fraud_type == "high_amount":
        base["amount"] = np.random.lognormal(10, 0.8)
        base["geo_distance_from_last_tx"] = np.random.exponential(500)
        base["time_of_day"] = np.random.uniform(0.0, 0.25)  # Late night

    elif fraud_type == "velocity":
        base["transaction_velocity_1h"] = np.random.randint(8, 30)
        base["transaction_velocity_24h"] = np.random.randint(20, 80)
        base["amount"] = np.random.lognormal(8, 0.5)

    elif fraud_type == "device":
        base["device_risk_score"] = np.random.uniform(0.6, 1.0)
        base["device_fraud_count"] = np.random.randint(1, 10)
        base["distinct_user_count"] = np.random.randint(3, 15)

    elif fraud_type == "ato":
        base["failed_login_attempts_24h"] = np.random.randint(3, 15)
        base["geo_distance_from_last_tx"] = np.random.exponential(1000)
        base["amount"] = np.random.lognormal(9, 0.8)

    elif fraud_type == "new_account":
        base["account_age_days"] = np.random.randint(0, 5)
        base["amount"] = np.random.lognormal(9.5, 0.5)
        base["historical_fraud_flag"] = 0

    return base


def generate_aml_dataset(n_samples: int = 50000, suspicious_rate: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic AML training data.

    Suspicious rate ~3% — lower than fraud as AML patterns are rarer.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_sus = int(n_samples * suspicious_rate)
    n_clean = n_samples - n_sus

    data = []

    for _ in range(n_clean):
        data.append(_generate_clean_aml_transaction())

    for _ in range(n_sus):
        data.append(_generate_suspicious_aml_transaction())

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Generated AML dataset: {len(df)} samples, {n_sus} suspicious ({suspicious_rate*100:.1f}%)")
    return df


def _generate_clean_aml_transaction() -> dict:
    return {
        "amount": np.random.lognormal(7.5, 1.0),
        "velocity_1h": np.random.poisson(2),
        "velocity_24h": np.random.poisson(6),
        "total_volume_24h": np.random.lognormal(8.5, 1.0),
        "avg_amount_30d": np.random.lognormal(7.5, 0.8),
        "std_amount_30d": np.random.lognormal(6.0, 0.8),
        "time_of_day": np.random.uniform(0.25, 0.85),
        "is_cross_border": np.random.choice([0, 0, 0, 1], p=[0.85, 0.05, 0.05, 0.05]),
        "account_age_days": np.random.randint(60, 2000),
        "device_count_30d": np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1]),
        "ip_count_30d": np.random.choice([1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.1]),
        "new_device_flag": 0,
        "kyc_completeness_score": np.random.uniform(0.8, 1.0),
        "network_risk_score": np.random.beta(2, 10),
        "circular_transfer_flag": 0,
        "shared_device_cluster_size": np.random.choice([0, 0, 1], p=[0.8, 0.1, 0.1]),
        "high_risk_country_flag": 0,
        "sanctions_proximity_score": 0.0,
        "ip_country_mismatch": 0,
        "historical_structuring_flag": 0,
        "structuring_score_24h": 0.0,
        "rapid_drain_flag": 0,
        "is_suspicious": 0,
    }


def _generate_suspicious_aml_transaction() -> dict:
    pattern = random.choice(["structuring", "layering", "mule", "high_risk_geo", "rapid_drain"])

    base = _generate_clean_aml_transaction()
    base["is_suspicious"] = 1

    if pattern == "structuring":
        base["velocity_24h"] = np.random.randint(6, 20)
        base["amount"] = np.random.uniform(800000, 990000)  # Just under 1M KES threshold
        base["structuring_score_24h"] = np.random.uniform(0.5, 1.0)
        base["historical_structuring_flag"] = np.random.choice([0, 1], p=[0.3, 0.7])

    elif pattern == "layering":
        base["velocity_1h"] = np.random.randint(5, 15)
        base["network_risk_score"] = np.random.uniform(0.5, 0.9)
        base["circular_transfer_flag"] = 1
        base["shared_device_cluster_size"] = np.random.randint(3, 10)

    elif pattern == "mule":
        base["account_age_days"] = np.random.randint(1, 30)
        base["velocity_24h"] = np.random.randint(10, 40)
        base["total_volume_24h"] = np.random.lognormal(12, 0.5)
        base["rapid_drain_flag"] = 1
        base["kyc_completeness_score"] = np.random.uniform(0.3, 0.6)

    elif pattern == "high_risk_geo":
        base["high_risk_country_flag"] = 1
        base["sanctions_proximity_score"] = np.random.uniform(0.3, 1.0)
        base["ip_country_mismatch"] = np.random.choice([0, 1], p=[0.3, 0.7])
        base["is_cross_border"] = 1
        base["amount"] = np.random.lognormal(10, 0.8)

    elif pattern == "rapid_drain":
        base["rapid_drain_flag"] = 1
        base["amount"] = np.random.lognormal(11, 0.5)
        base["velocity_1h"] = np.random.randint(3, 10)
        base["new_device_flag"] = 1
        base["time_of_day"] = np.random.uniform(0.0, 0.2)

    return base


def generate_merchant_dataset(n_samples: int = 30000, risky_rate: float = 0.04, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic merchant risk training data.

    Risky rate ~4% — merchants flagged for high-risk behavior.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_risky = int(n_samples * risky_rate)
    n_clean = n_samples - n_risky

    data = []

    for _ in range(n_clean):
        data.append(_generate_clean_merchant())

    for _ in range(n_risky):
        data.append(_generate_risky_merchant())

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Generated merchant dataset: {len(df)} samples, {n_risky} risky ({risky_rate*100:.1f}%)")
    return df


def _generate_clean_merchant() -> dict:
    return {
        "transaction_count_1h": np.random.poisson(5),
        "transaction_count_24h": np.random.poisson(50),
        "transaction_volume_24h": np.random.lognormal(10, 1.0),
        "unique_customers_24h": np.random.poisson(30),
        "avg_transaction_amount_30d": np.random.lognormal(7.5, 0.8),
        "std_transaction_amount_30d": np.random.lognormal(6.0, 0.8),
        "chargeback_rate_90d": np.random.beta(1, 50),
        "refund_rate_90d": np.random.beta(2, 30),
        "account_age_days": np.random.randint(90, 2000),
        "fraud_transaction_rate": np.random.beta(1, 100),
        "high_risk_customer_ratio": np.random.beta(1, 20),
        "cross_border_ratio": np.random.beta(2, 20),
        "velocity_spike_flag": 0,
        "mcc_risk_score": np.random.beta(2, 10),
        "avg_customer_risk_score": np.random.beta(2, 10),
        "is_risky": 0,
    }


def _generate_risky_merchant() -> dict:
    pattern = random.choice(["chargeback", "fraud_funnel", "velocity_abuse", "mule_network", "mcc_fraud"])

    base = _generate_clean_merchant()
    base["is_risky"] = 1

    if pattern == "chargeback":
        base["chargeback_rate_90d"] = np.random.uniform(0.05, 0.25)
        base["refund_rate_90d"] = np.random.uniform(0.1, 0.4)
        base["fraud_transaction_rate"] = np.random.uniform(0.02, 0.1)

    elif pattern == "fraud_funnel":
        base["fraud_transaction_rate"] = np.random.uniform(0.05, 0.2)
        base["high_risk_customer_ratio"] = np.random.uniform(0.2, 0.6)
        base["avg_customer_risk_score"] = np.random.uniform(0.4, 0.8)

    elif pattern == "velocity_abuse":
        base["transaction_count_1h"] = np.random.randint(30, 100)
        base["transaction_count_24h"] = np.random.randint(200, 800)
        base["velocity_spike_flag"] = 1
        base["unique_customers_24h"] = np.random.poisson(5)

    elif pattern == "mule_network":
        base["account_age_days"] = np.random.randint(1, 30)
        base["cross_border_ratio"] = np.random.uniform(0.4, 0.9)
        base["high_risk_customer_ratio"] = np.random.uniform(0.3, 0.7)
        base["transaction_volume_24h"] = np.random.lognormal(13, 0.5)

    elif pattern == "mcc_fraud":
        base["mcc_risk_score"] = np.random.uniform(0.6, 1.0)
        base["fraud_transaction_rate"] = np.random.uniform(0.03, 0.15)
        base["chargeback_rate_90d"] = np.random.uniform(0.03, 0.15)
        base["avg_transaction_amount_30d"] = np.random.lognormal(10, 0.5)

    return base


def save_datasets(output_dir: str = "./training/data") -> tuple[str, str, str]:
    """Generate and save fraud, AML, and merchant datasets."""
    os.makedirs(output_dir, exist_ok=True)

    fraud_df = generate_fraud_dataset()
    aml_df = generate_aml_dataset()
    merchant_df = generate_merchant_dataset()

    fraud_path = os.path.join(output_dir, "fraud_training_data.parquet")
    aml_path = os.path.join(output_dir, "aml_training_data.parquet")
    merchant_path = os.path.join(output_dir, "merchant_training_data.parquet")

    fraud_df.to_parquet(fraud_path, index=False)
    aml_df.to_parquet(aml_path, index=False)
    merchant_df.to_parquet(merchant_path, index=False)

    logger.info(f"Saved fraud data: {fraud_path}")
    logger.info(f"Saved AML data: {aml_path}")
    logger.info(f"Saved merchant data: {merchant_path}")

    return fraud_path, aml_path, merchant_path


if __name__ == "__main__":
    save_datasets()
