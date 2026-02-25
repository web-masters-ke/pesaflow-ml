"""Feature view definitions for multi-view co-training.

Each domain's features are split into 2-3 independent views based on
the feature category (transaction, user, device, network, etc.).
The views should be as conditionally independent as possible given the label.
"""

# Feature indices for each view (matching FeatureVector.to_array() ordering)

FRAUD_VIEWS = {
    "transaction_view": {
        "indices": [9, 10, 11, 12],  # amount, geo_distance, time_of_day, currency_risk
        "names": ["amount", "geo_distance_from_last_tx", "time_of_day", "currency_risk"],
    },
    "user_view": {
        "indices": [0, 1, 2, 3, 4, 5],  # avg_7d, vel_1h, vel_24h, failed_logins, age, fraud_flag
        "names": [
            "avg_transaction_amount_7d",
            "transaction_velocity_1h",
            "transaction_velocity_24h",
            "failed_login_attempts_24h",
            "account_age_days",
            "historical_fraud_flag",
        ],
    },
    "device_view": {
        "indices": [6, 7, 8],  # device_risk, device_fraud_cnt, distinct_users
        "names": ["device_risk_score", "device_fraud_count", "distinct_user_count"],
    },
}

AML_VIEWS = {
    "transaction_view": {
        "indices": [0, 1, 2, 3, 4, 5, 6, 7],  # amount, velocities, volume, avg, std, time, cross_border
        "names": [
            "amount",
            "velocity_1h",
            "velocity_24h",
            "total_volume_24h",
            "avg_amount_30d",
            "std_amount_30d",
            "time_of_day",
            "is_cross_border",
        ],
    },
    "entity_view": {
        "indices": [8, 9, 10, 11, 12],  # account_age, device_count, ip_count, new_device, kyc
        "names": [
            "account_age_days",
            "device_count_30d",
            "ip_count_30d",
            "new_device_flag",
            "kyc_completeness_score",
        ],
    },
    "network_view": {
        "indices": [13, 14, 15, 16, 17, 18, 19, 20, 21],  # network + geographic + behavioral
        "names": [
            "network_risk_score",
            "circular_transfer_flag",
            "shared_device_cluster_size",
            "high_risk_country_flag",
            "sanctions_proximity_score",
            "ip_country_mismatch",
            "historical_structuring_flag",
            "structuring_score_24h",
            "rapid_drain_flag",
        ],
    },
}

MERCHANT_VIEWS = {
    "velocity_view": {
        "indices": [0, 1, 2, 3],  # tx_count_1h, tx_count_24h, volume_24h, unique_customers
        "names": [
            "transaction_count_1h",
            "transaction_count_24h",
            "transaction_volume_24h",
            "unique_customers_24h",
        ],
    },
    "aggregate_view": {
        "indices": [4, 5, 6, 7, 8],  # avg_amount, std_amount, chargeback, refund, age
        "names": [
            "avg_transaction_amount_30d",
            "std_transaction_amount_30d",
            "chargeback_rate_90d",
            "refund_rate_90d",
            "account_age_days",
        ],
    },
    "risk_view": {
        "indices": [
            9,
            10,
            11,
            12,
            13,
            14,
        ],  # fraud_rate, high_risk_ratio, cross_border, velocity_spike, mcc, customer_risk
        "names": [
            "fraud_transaction_rate",
            "high_risk_customer_ratio",
            "cross_border_ratio",
            "velocity_spike_flag",
            "mcc_risk_score",
            "avg_customer_risk_score",
        ],
    },
}

DOMAIN_VIEWS = {
    "fraud": FRAUD_VIEWS,
    "aml": AML_VIEWS,
    "merchant": MERCHANT_VIEWS,
}


def get_view_features(X, domain: str, view_name: str):
    """Extract feature subset for a specific view."""
    views = DOMAIN_VIEWS.get(domain, {})
    view = views.get(view_name)
    if not view:
        raise ValueError(f"Unknown view {view_name} for domain {domain}")
    return X[:, view["indices"]]


def get_view_names(domain: str) -> list[str]:
    """Get list of view names for a domain."""
    return list(DOMAIN_VIEWS.get(domain, {}).keys())
