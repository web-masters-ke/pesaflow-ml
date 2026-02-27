#!/usr/bin/env python3
"""
PesaFlow ML — Redis Online Feature Store Seed
================================================

Populates the real-time feature cache used during transaction scoring.
Mirrors the PostgreSQL offline seed data so that ML scoring sees
consistent features whether reading from Redis (hot path) or
PostgreSQL (fallback path).

Keys seeded:
  - User fraud features          user:{uid}:features
  - Device features              device:{fp}:features
  - User last-known geo          user:{uid}:last_geo
  - Fraud velocity counters      user:{uid}:txn_count:1h / 24h / volume:24h
  - AML velocity counters        aml:user:{uid}:txn_count:1h / 24h / volume:24h
  - AML aggregate features       aml:user:{uid}:aggregate_features
  - AML network features         aml:user:{uid}:network_features
  - AML known devices            aml:user:{uid}:known_devices  (SET)
  - AML avg velocity baseline    aml:user:{uid}:avg_velocity_1h
  - IP geolocation               ip_geo:{ip}
  - Merchant feature cache       merchant:features:{mid}
  - Merchant velocity counters   merchant:{mid}:txn_count:1h / 24h / txn_volume:24h
  - Merchant unique customers    merchant:{mid}:customers:24h  (SET)
  - Merchant risk signals        merchant:{mid}:fraud_txn_rate / etc.
  - Blacklists                   blacklist:users / devices / ips  (SET)
  - Population priors            pesaflow:priors:{domain}
  - Feature coverage stats       pesaflow:features:{domain}:coverage_stats
  - Maturity coverage            pesaflow:maturity:{domain}:feature_coverage

Usage:
  # Local
  REDIS_URL=redis://localhost:26379/0 python3 seeds/redis_online_seed.py

  # Via Docker (pipe into ML container which has redis-py)
  cat seeds/redis_online_seed.py | docker exec -i \\
    -e REDIS_URL=redis://pesaflow-redis:6379/2 pesaflow-ml python3 -

  # Or via docker exec if volume-mounted
  docker exec -e REDIS_URL=redis://pesaflow-redis:6379/2 \\
    pesaflow-ml python3 /app/seeds/redis_online_seed.py
"""

import json
import os
import sys
import time

try:
    import redis
except ImportError:
    print("ERROR: redis package required.  pip install redis")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/2")

# Longer TTLs for seed data (survive across test sessions)
TTL_FEATURES = 86400  # 24 h  (normal: 300 s)
TTL_VELOCITY_1H = 3600  # 1 h
TTL_VELOCITY_24H = 86400  # 24 h
TTL_DEVICES_SET = 2592000  # 30 days
TTL_PRIORS = 86400  # 24 h
TTL_GEO = 86400  # 24 h

NOW_TS = time.time()

# ---------------------------------------------------------------------------
# UUID helpers (must match SQL seed patterns)
# ---------------------------------------------------------------------------


def user_uuid(n: int) -> str:
    return f"30000001-5eed-4000-a000-{n:012d}"


def merchant_uuid(n: int) -> str:
    return f"20000001-5eed-4000-a000-{n:012d}"


def device_fp(n: int) -> str:
    return f"dev-seed-{n:02d}"


def device_num_for_user(n: int) -> int:
    """Map user number to device number (matches SQL metadata)."""
    if n <= 40:
        return 1 + ((n - 1) % 20)
    elif n <= 47:
        return 21 + ((n - 41) % 7)
    else:
        return 28 + ((n - 48) % 3)  # shared fraud devices


def user_ip(n: int) -> str:
    return f"41.89.{100 + n % 50}.1"


# ---------------------------------------------------------------------------
# 1. User fraud features  (50 users)
# ---------------------------------------------------------------------------


def seed_user_features(pipe):
    """Seed user:{uid}:features — fraud feature cache."""
    for n in range(1, 51):
        uid = user_uuid(n)
        if n <= 40:  # normal
            data = {
                "avg_transaction_amount_7d": 500.0 + n * 100,
                "transaction_velocity_1h": n % 3,
                "transaction_velocity_24h": 1 + n % 8,
                "failed_login_attempts_24h": 0,
                "account_age_days": 90 + n * 16,
                "historical_fraud_flag": 0,
            }
        elif n <= 47:  # suspicious
            data = {
                "avg_transaction_amount_7d": 5000.0 + (n - 40) * 5000,
                "transaction_velocity_1h": 5 + (n - 40) % 10,
                "transaction_velocity_24h": 15 + (n - 40) * 4,
                "failed_login_attempts_24h": 1 + (n - 41) % 3,
                "account_age_days": 14 + (n - 40) * 7,
                "historical_fraud_flag": 0,
            }
        else:  # fraudulent
            data = {
                "avg_transaction_amount_7d": 50000.0 + (n - 47) * 20000,
                "transaction_velocity_1h": 12 + (n - 47) * 5,
                "transaction_velocity_24h": 30 + (n - 47) * 10,
                "failed_login_attempts_24h": 5 + (n - 47) * 2,
                "account_age_days": 1 + (n - 47) * 3,
                "historical_fraud_flag": 1,
            }
        pipe.setex(f"user:{uid}:features", TTL_FEATURES, json.dumps(data))


# ---------------------------------------------------------------------------
# 2. Device features  (30 devices)
# ---------------------------------------------------------------------------


def seed_device_features(pipe):
    """Seed device:{fp}:features — device risk cache."""
    for n in range(1, 31):
        fp = device_fp(n)
        if n <= 20:  # clean
            data = {
                "device_risk_score": round(0.01 + n * 0.007, 4),
                "device_fraud_count": 0,
                "distinct_user_count": 1,
            }
        elif n <= 27:  # suspicious
            data = {
                "device_risk_score": round(0.20 + (n - 20) * 0.04, 4),
                "device_fraud_count": (n - 20) % 2,
                "distinct_user_count": 2 + (n - 21) % 2,
            }
        else:  # fraudulent — device sharing across users
            data = {
                "device_risk_score": round(0.60 + (n - 27) * 0.12, 4),
                "device_fraud_count": 2 + (n - 27),
                "distinct_user_count": 3 + (n - 27) * 2,
            }
        pipe.setex(f"device:{fp}:features", TTL_FEATURES, json.dumps(data))


# ---------------------------------------------------------------------------
# 3. User last-known geo  (50 users — all in Kenya)
# ---------------------------------------------------------------------------


def seed_user_geo(pipe):
    """Seed user:{uid}:last_geo — last transaction location."""
    for n in range(1, 51):
        uid = user_uuid(n)
        data = {
            "lat": round(-1.2864 + (n % 100) / 1000, 6),
            "lng": round(36.8172 + (n % 100) / 1000, 6),
        }
        pipe.setex(f"user:{uid}:last_geo", TTL_VELOCITY_24H, json.dumps(data))


# ---------------------------------------------------------------------------
# 4. Fraud velocity counters  (50 users)
# ---------------------------------------------------------------------------


def seed_fraud_velocity(pipe):
    """Seed user:{uid}:txn_count:1h / 24h / volume:24h."""
    for n in range(1, 51):
        uid = user_uuid(n)
        if n <= 40:
            v1h, v24h, vol = n % 3, 1 + n % 8, 500.0 + n * 100
        elif n <= 47:
            v1h = 5 + (n - 40) % 10
            v24h = 15 + (n - 40) * 4
            vol = 5000.0 + (n - 40) * 5000
        else:
            v1h = 12 + (n - 47) * 5
            v24h = 30 + (n - 47) * 10
            vol = 50000.0 + (n - 47) * 20000

        pipe.setex(f"user:{uid}:txn_count:1h", TTL_VELOCITY_1H, str(v1h))
        pipe.setex(f"user:{uid}:txn_count:24h", TTL_VELOCITY_24H, str(v24h))
        pipe.setex(f"user:{uid}:volume:24h", TTL_VELOCITY_24H, str(vol))


# ---------------------------------------------------------------------------
# 5. AML velocity counters  (50 users)
# ---------------------------------------------------------------------------


def seed_aml_velocity(pipe):
    """Seed aml:user:{uid}:txn_count:1h / 24h / volume:24h + avg baseline."""
    for n in range(1, 51):
        uid = user_uuid(n)
        if n <= 40:
            v1h, v24h = n % 3, 1 + n % 8
            vol = 5000.0 + n * 1000
            avg_v1h = 2.0
        elif n <= 47:
            v1h = 5 + (n - 40)
            v24h = 15 + (n - 40) * 5
            vol = 100000.0 + (n - 40) * 50000
            avg_v1h = 3.0
        else:
            v1h = 15 + (n - 47) * 3
            v24h = 40 + (n - 47) * 8
            vol = 500000.0 + (n - 47) * 200000
            avg_v1h = 4.0  # low baseline makes current spike obvious

        pipe.setex(f"aml:user:{uid}:txn_count:1h", TTL_VELOCITY_1H, str(v1h))
        pipe.setex(f"aml:user:{uid}:txn_count:24h", TTL_VELOCITY_24H, str(v24h))
        pipe.setex(f"aml:user:{uid}:volume:24h", TTL_VELOCITY_24H, str(vol))
        pipe.setex(f"aml:user:{uid}:avg_velocity_1h", TTL_VELOCITY_24H, str(avg_v1h))


# ---------------------------------------------------------------------------
# 6. AML aggregate features  (50 users)
# ---------------------------------------------------------------------------


def seed_aml_aggregate_features(pipe):
    """Seed aml:user:{uid}:aggregate_features — user-level AML stats."""
    for n in range(1, 51):
        uid = user_uuid(n)
        if n <= 40:
            data = {
                "avg_transaction_amount": 1000.0 + n * 500,
                "std_transaction_amount": 400.0 + n * 200,
                "device_count_30d": 1,
                "ip_count_30d": 1 + n % 3,
                "account_age_days": 90 + n * 16,
                "kyc_completeness_score": 1.0,
            }
        elif n <= 47:
            data = {
                "avg_transaction_amount": 50000.0 + (n - 40) * 20000,
                "std_transaction_amount": 20000.0 + (n - 40) * 8000,
                "device_count_30d": 2 + (n - 41) % 3,
                "ip_count_30d": 5 + (n - 40) * 2,
                "account_age_days": 14 + (n - 40) * 7,
                "kyc_completeness_score": 0.6 + (n - 40) * 0.05,
            }
        else:
            data = {
                "avg_transaction_amount": 200000.0 + (n - 47) * 50000,
                "std_transaction_amount": 80000.0 + (n - 47) * 20000,
                "device_count_30d": 5 + (n - 48) * 2,
                "ip_count_30d": 15 + (n - 47) * 5,
                "account_age_days": 1 + (n - 47) * 3,
                "kyc_completeness_score": 0.2 + (n - 47) * 0.1,
            }
        pipe.setex(
            f"aml:user:{uid}:aggregate_features",
            TTL_FEATURES,
            json.dumps(data),
        )


# ---------------------------------------------------------------------------
# 7. AML network features  (50 users)
# ---------------------------------------------------------------------------


def seed_aml_network_features(pipe):
    """Seed aml:user:{uid}:network_features — graph-based risk metrics."""
    for n in range(1, 51):
        uid = user_uuid(n)
        if n <= 40:
            data = {
                "network_risk_score": round(0.02 + n * 0.003, 4),
                "circular_transfer_flag": 0,
                "shared_device_cluster_size": 0,
            }
        elif n <= 47:
            data = {
                "network_risk_score": round(0.30 + (n - 40) * 0.07, 4),
                "circular_transfer_flag": 1 if n in (46, 47) else 0,
                "shared_device_cluster_size": 2 + (n - 41) % 3,
            }
        else:
            data = {
                "network_risk_score": round(0.70 + (n - 47) * 0.08, 4),
                "circular_transfer_flag": 1,
                "shared_device_cluster_size": 5 + (n - 47) * 2,
            }
        pipe.setex(
            f"aml:user:{uid}:network_features",
            TTL_FEATURES,
            json.dumps(data),
        )


# ---------------------------------------------------------------------------
# 8. AML known devices  (50 users — SET per user)
# ---------------------------------------------------------------------------


def seed_aml_known_devices(pipe):
    """Seed aml:user:{uid}:known_devices — device familiarity sets."""
    for n in range(1, 51):
        uid = user_uuid(n)
        key = f"aml:user:{uid}:known_devices"
        primary_dev = device_fp(device_num_for_user(n))
        pipe.sadd(key, primary_dev)
        # Suspicious/fraud users have extra devices
        if n > 40:
            pipe.sadd(key, f"dev-seed-extra-{n}")
        if n > 47:
            pipe.sadd(key, f"dev-seed-extra2-{n}")
            pipe.sadd(key, f"dev-seed-extra3-{n}")
        pipe.expire(key, TTL_DEVICES_SET)


# ---------------------------------------------------------------------------
# 9. IP geolocation mappings
# ---------------------------------------------------------------------------


def seed_ip_geo(pipe):
    """Seed ip_geo:{ip} — country code lookups for IP-country mismatch."""
    # All seed users are in Kenya
    for n in range(1, 51):
        ip = user_ip(n)
        if n <= 47:
            pipe.setex(f"ip_geo:{ip}", TTL_GEO, "KE")
        else:
            # Fraudulent users: IP geolocates to suspicious country
            pipe.setex(f"ip_geo:{ip}", TTL_GEO, "NG")

    # A few extra IPs for testing
    pipe.setex("ip_geo:41.89.200.1", TTL_GEO, "KE")
    pipe.setex("ip_geo:196.201.214.1", TTL_GEO, "KE")  # Safaricom
    pipe.setex("ip_geo:185.70.40.1", TTL_GEO, "IR")  # sanctioned
    pipe.setex("ip_geo:175.45.176.1", TTL_GEO, "KP")  # sanctioned


# ---------------------------------------------------------------------------
# 10. Merchant feature cache  (20 merchants)
# ---------------------------------------------------------------------------

# (num, mcc, features_dict)
MERCHANT_DATA = [
    # LOW risk — 12 merchants
    (
        1,
        "5411",
        {
            "transaction_count_1h": 15,
            "transaction_count_24h": 200,
            "transaction_volume_24h": 240000.0,
            "unique_customers_24h": 80,
            "avg_transaction_amount_30d": 1200.0,
            "std_transaction_amount_30d": 800.0,
            "chargeback_rate_90d": 0.0015,
            "refund_rate_90d": 0.008,
            "account_age_days": 380,
            "fraud_transaction_rate": 0.0005,
            "high_risk_customer_ratio": 0.02,
            "cross_border_ratio": 0.01,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.05,
            "avg_customer_risk_score": 0.08,
        },
    ),
    (
        2,
        "5812",
        {
            "transaction_count_1h": 10,
            "transaction_count_24h": 120,
            "transaction_volume_24h": 300000.0,
            "unique_customers_24h": 50,
            "avg_transaction_amount_30d": 2500.0,
            "std_transaction_amount_30d": 1500.0,
            "chargeback_rate_90d": 0.002,
            "refund_rate_90d": 0.01,
            "account_age_days": 360,
            "fraud_transaction_rate": 0.0008,
            "high_risk_customer_ratio": 0.03,
            "cross_border_ratio": 0.05,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.08,
            "avg_customer_risk_score": 0.07,
        },
    ),
    (
        3,
        "5732",
        {
            "transaction_count_1h": 3,
            "transaction_count_24h": 30,
            "transaction_volume_24h": 450000.0,
            "unique_customers_24h": 20,
            "avg_transaction_amount_30d": 15000.0,
            "std_transaction_amount_30d": 12000.0,
            "chargeback_rate_90d": 0.003,
            "refund_rate_90d": 0.015,
            "account_age_days": 350,
            "fraud_transaction_rate": 0.001,
            "high_risk_customer_ratio": 0.05,
            "cross_border_ratio": 0.08,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.12,
            "avg_customer_risk_score": 0.10,
        },
    ),
    (
        4,
        "7011",
        {
            "transaction_count_1h": 2,
            "transaction_count_24h": 25,
            "transaction_volume_24h": 200000.0,
            "unique_customers_24h": 15,
            "avg_transaction_amount_30d": 8000.0,
            "std_transaction_amount_30d": 5000.0,
            "chargeback_rate_90d": 0.001,
            "refund_rate_90d": 0.005,
            "account_age_days": 340,
            "fraud_transaction_rate": 0.0003,
            "high_risk_customer_ratio": 0.02,
            "cross_border_ratio": 0.15,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.06,
            "avg_customer_risk_score": 0.06,
        },
    ),
    (
        5,
        "4814",
        {
            "transaction_count_1h": 50,
            "transaction_count_24h": 500,
            "transaction_volume_24h": 250000.0,
            "unique_customers_24h": 200,
            "avg_transaction_amount_30d": 500.0,
            "std_transaction_amount_30d": 300.0,
            "chargeback_rate_90d": 0.0005,
            "refund_rate_90d": 0.002,
            "account_age_days": 330,
            "fraud_transaction_rate": 0.0002,
            "high_risk_customer_ratio": 0.01,
            "cross_border_ratio": 0.0,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.03,
            "avg_customer_risk_score": 0.05,
        },
    ),
    (
        7,
        "5999",
        {
            "transaction_count_1h": 8,
            "transaction_count_24h": 100,
            "transaction_volume_24h": 350000.0,
            "unique_customers_24h": 60,
            "avg_transaction_amount_30d": 3500.0,
            "std_transaction_amount_30d": 4000.0,
            "chargeback_rate_90d": 0.0035,
            "refund_rate_90d": 0.02,
            "account_age_days": 280,
            "fraud_transaction_rate": 0.0012,
            "high_risk_customer_ratio": 0.04,
            "cross_border_ratio": 0.10,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.10,
            "avg_customer_risk_score": 0.09,
        },
    ),
    (
        8,
        "5912",
        {
            "transaction_count_1h": 12,
            "transaction_count_24h": 130,
            "transaction_volume_24h": 104000.0,
            "unique_customers_24h": 90,
            "avg_transaction_amount_30d": 800.0,
            "std_transaction_amount_30d": 400.0,
            "chargeback_rate_90d": 0.0008,
            "refund_rate_90d": 0.003,
            "account_age_days": 270,
            "fraud_transaction_rate": 0.0001,
            "high_risk_customer_ratio": 0.01,
            "cross_border_ratio": 0.0,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.04,
            "avg_customer_risk_score": 0.04,
        },
    ),
    (
        10,
        "5511",
        {
            "transaction_count_1h": 0,
            "transaction_count_24h": 2,
            "transaction_volume_24h": 900000.0,
            "unique_customers_24h": 2,
            "avg_transaction_amount_30d": 450000.0,
            "std_transaction_amount_30d": 200000.0,
            "chargeback_rate_90d": 0.002,
            "refund_rate_90d": 0.01,
            "account_age_days": 240,
            "fraud_transaction_rate": 0.0005,
            "high_risk_customer_ratio": 0.03,
            "cross_border_ratio": 0.0,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.08,
            "avg_customer_risk_score": 0.07,
        },
    ),
    (
        11,
        "5411",
        {
            "transaction_count_1h": 20,
            "transaction_count_24h": 250,
            "transaction_volume_24h": 450000.0,
            "unique_customers_24h": 100,
            "avg_transaction_amount_30d": 1800.0,
            "std_transaction_amount_30d": 1000.0,
            "chargeback_rate_90d": 0.0012,
            "refund_rate_90d": 0.006,
            "account_age_days": 220,
            "fraud_transaction_rate": 0.0004,
            "high_risk_customer_ratio": 0.02,
            "cross_border_ratio": 0.01,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.05,
            "avg_customer_risk_score": 0.06,
        },
    ),
    (
        12,
        "5699",
        {
            "transaction_count_1h": 1,
            "transaction_count_24h": 15,
            "transaction_volume_24h": 90000.0,
            "unique_customers_24h": 12,
            "avg_transaction_amount_30d": 6000.0,
            "std_transaction_amount_30d": 3500.0,
            "chargeback_rate_90d": 0.0025,
            "refund_rate_90d": 0.012,
            "account_age_days": 200,
            "fraud_transaction_rate": 0.0006,
            "high_risk_customer_ratio": 0.03,
            "cross_border_ratio": 0.05,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.07,
            "avg_customer_risk_score": 0.08,
        },
    ),
    (
        17,
        "5541",
        {
            "transaction_count_1h": 25,
            "transaction_count_24h": 280,
            "transaction_volume_24h": 840000.0,
            "unique_customers_24h": 130,
            "avg_transaction_amount_30d": 3000.0,
            "std_transaction_amount_30d": 1500.0,
            "chargeback_rate_90d": 0.0008,
            "refund_rate_90d": 0.001,
            "account_age_days": 100,
            "fraud_transaction_rate": 0.0002,
            "high_risk_customer_ratio": 0.01,
            "cross_border_ratio": 0.0,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.05,
            "avg_customer_risk_score": 0.05,
        },
    ),
    (
        19,
        "5251",
        {
            "transaction_count_1h": 2,
            "transaction_count_24h": 28,
            "transaction_volume_24h": 140000.0,
            "unique_customers_24h": 18,
            "avg_transaction_amount_30d": 5000.0,
            "std_transaction_amount_30d": 3000.0,
            "chargeback_rate_90d": 0.0018,
            "refund_rate_90d": 0.008,
            "account_age_days": 60,
            "fraud_transaction_rate": 0.0005,
            "high_risk_customer_ratio": 0.02,
            "cross_border_ratio": 0.0,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.06,
            "avg_customer_risk_score": 0.07,
        },
    ),
    # MEDIUM risk — 5 merchants
    (
        6,
        "6012",
        {
            "transaction_count_1h": 5,
            "transaction_count_24h": 60,
            "transaction_volume_24h": 3000000.0,
            "unique_customers_24h": 30,
            "avg_transaction_amount_30d": 50000.0,
            "std_transaction_amount_30d": 40000.0,
            "chargeback_rate_90d": 0.006,
            "refund_rate_90d": 0.01,
            "account_age_days": 300,
            "fraud_transaction_rate": 0.002,
            "high_risk_customer_ratio": 0.08,
            "cross_border_ratio": 0.15,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.15,
            "avg_customer_risk_score": 0.12,
        },
    ),
    (
        9,
        "5944",
        {
            "transaction_count_1h": 1,
            "transaction_count_24h": 8,
            "transaction_volume_24h": 680000.0,
            "unique_customers_24h": 6,
            "avg_transaction_amount_30d": 85000.0,
            "std_transaction_amount_30d": 60000.0,
            "chargeback_rate_90d": 0.008,
            "refund_rate_90d": 0.03,
            "account_age_days": 260,
            "fraud_transaction_rate": 0.0025,
            "high_risk_customer_ratio": 0.10,
            "cross_border_ratio": 0.20,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.18,
            "avg_customer_risk_score": 0.14,
        },
    ),
    (
        16,
        "5131",
        {
            "transaction_count_1h": 6,
            "transaction_count_24h": 70,
            "transaction_volume_24h": 1750000.0,
            "unique_customers_24h": 25,
            "avg_transaction_amount_30d": 25000.0,
            "std_transaction_amount_30d": 20000.0,
            "chargeback_rate_90d": 0.005,
            "refund_rate_90d": 0.018,
            "account_age_days": 140,
            "fraud_transaction_rate": 0.0015,
            "high_risk_customer_ratio": 0.06,
            "cross_border_ratio": 0.08,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.12,
            "avg_customer_risk_score": 0.10,
        },
    ),
    (
        18,
        "5732",
        {
            "transaction_count_1h": 4,
            "transaction_count_24h": 35,
            "transaction_volume_24h": 700000.0,
            "unique_customers_24h": 15,
            "avg_transaction_amount_30d": 20000.0,
            "std_transaction_amount_30d": 18000.0,
            "chargeback_rate_90d": 0.007,
            "refund_rate_90d": 0.025,
            "account_age_days": 80,
            "fraud_transaction_rate": 0.003,
            "high_risk_customer_ratio": 0.12,
            "cross_border_ratio": 0.25,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.12,
            "avg_customer_risk_score": 0.15,
        },
    ),
    (
        20,
        "6211",
        {
            "transaction_count_1h": 2,
            "transaction_count_24h": 14,
            "transaction_volume_24h": 2100000.0,
            "unique_customers_24h": 8,
            "avg_transaction_amount_30d": 150000.0,
            "std_transaction_amount_30d": 120000.0,
            "chargeback_rate_90d": 0.009,
            "refund_rate_90d": 0.02,
            "account_age_days": 45,
            "fraud_transaction_rate": 0.0035,
            "high_risk_customer_ratio": 0.15,
            "cross_border_ratio": 0.40,
            "velocity_spike_flag": 0,
            "mcc_risk_score": 0.25,
            "avg_customer_risk_score": 0.18,
        },
    ),
    # HIGH risk — 3 merchants
    (
        13,
        "7995",
        {
            "transaction_count_1h": 35,
            "transaction_count_24h": 350,
            "transaction_volume_24h": 1750000.0,
            "unique_customers_24h": 100,
            "avg_transaction_amount_30d": 5000.0,
            "std_transaction_amount_30d": 8000.0,
            "chargeback_rate_90d": 0.035,
            "refund_rate_90d": 0.08,
            "account_age_days": 180,
            "fraud_transaction_rate": 0.02,
            "high_risk_customer_ratio": 0.25,
            "cross_border_ratio": 0.10,
            "velocity_spike_flag": 1,
            "mcc_risk_score": 0.45,
            "avg_customer_risk_score": 0.30,
        },
    ),
    (
        14,
        "7995",
        {
            "transaction_count_1h": 20,
            "transaction_count_24h": 220,
            "transaction_volume_24h": 2640000.0,
            "unique_customers_24h": 70,
            "avg_transaction_amount_30d": 12000.0,
            "std_transaction_amount_30d": 15000.0,
            "chargeback_rate_90d": 0.05,
            "refund_rate_90d": 0.12,
            "account_age_days": 160,
            "fraud_transaction_rate": 0.04,
            "high_risk_customer_ratio": 0.30,
            "cross_border_ratio": 0.15,
            "velocity_spike_flag": 1,
            "mcc_risk_score": 0.45,
            "avg_customer_risk_score": 0.35,
        },
    ),
    (
        15,
        "6051",
        {
            "transaction_count_1h": 8,
            "transaction_count_24h": 80,
            "transaction_volume_24h": 6400000.0,
            "unique_customers_24h": 35,
            "avg_transaction_amount_30d": 80000.0,
            "std_transaction_amount_30d": 70000.0,
            "chargeback_rate_90d": 0.025,
            "refund_rate_90d": 0.01,
            "account_age_days": 90,
            "fraud_transaction_rate": 0.015,
            "high_risk_customer_ratio": 0.20,
            "cross_border_ratio": 0.35,
            "velocity_spike_flag": 1,
            "mcc_risk_score": 0.35,
            "avg_customer_risk_score": 0.25,
        },
    ),
]


def seed_merchant_features(pipe):
    """Seed merchant:features:{mid} — full 15-feature merchant cache."""
    for num, _mcc, features in MERCHANT_DATA:
        mid = merchant_uuid(num)
        pipe.setex(f"merchant:features:{mid}", TTL_FEATURES, json.dumps(features))


# ---------------------------------------------------------------------------
# 11. Merchant velocity counters & risk signals  (20 merchants)
# ---------------------------------------------------------------------------


def seed_merchant_velocity_and_signals(pipe):
    """Seed merchant velocity counters, customer sets, and risk signals."""
    for num, _mcc, feat in MERCHANT_DATA:
        mid = merchant_uuid(num)

        # Velocity counters
        pipe.setex(f"merchant:{mid}:txn_count:1h", TTL_VELOCITY_1H, str(feat["transaction_count_1h"]))
        pipe.setex(f"merchant:{mid}:txn_count:24h", TTL_VELOCITY_24H, str(feat["transaction_count_24h"]))
        pipe.setex(f"merchant:{mid}:txn_volume:24h", TTL_VELOCITY_24H, str(feat["transaction_volume_24h"]))

        # Unique customers SET (add sample customer IDs)
        cust_key = f"merchant:{mid}:customers:24h"
        for c in range(1, feat["unique_customers_24h"] + 1):
            pipe.sadd(cust_key, user_uuid(1 + (c - 1) % 50))
        pipe.expire(cust_key, TTL_VELOCITY_24H)

        # Individual risk signals
        pipe.set(f"merchant:{mid}:fraud_txn_rate", str(feat["fraud_transaction_rate"]))
        pipe.set(f"merchant:{mid}:high_risk_customer_ratio", str(feat["high_risk_customer_ratio"]))
        pipe.set(f"merchant:{mid}:cross_border_ratio", str(feat["cross_border_ratio"]))
        pipe.set(f"merchant:{mid}:velocity_spike", str(feat["velocity_spike_flag"]))
        pipe.set(f"merchant:{mid}:mcc_risk_score", str(feat["mcc_risk_score"]))
        pipe.set(f"merchant:{mid}:avg_customer_risk", str(feat["avg_customer_risk_score"]))


# ---------------------------------------------------------------------------
# 12. Blacklists  (fraudulent entities)
# ---------------------------------------------------------------------------


def seed_blacklists(pipe):
    """Seed blacklist:users / devices / ips — fast override lookups."""
    # Blacklisted users (fraud users 48-50)
    for n in range(48, 51):
        pipe.sadd("blacklist:users", user_uuid(n))

    # Blacklisted devices (fraud devices 28-30)
    for n in range(28, 31):
        pipe.sadd("blacklist:devices", device_fp(n))

    # Blacklisted IPs (known fraud sources)
    pipe.sadd("blacklist:ips", "185.70.40.1")  # Iran range
    pipe.sadd("blacklist:ips", "175.45.176.1")  # DPRK range
    pipe.sadd("blacklist:ips", "41.89.148.99")  # seed fraud IP

    # No expiry on blacklists — persistent


# ---------------------------------------------------------------------------
# 13. Population priors  (all 3 domains)
# ---------------------------------------------------------------------------

FRAUD_PRIORS = {
    "avg_transaction_amount_7d": {
        "median": 2500.0,
        "mean": 3200.0,
        "mode": 1500.0,
        "std": 4500.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "transaction_velocity_1h": {
        "median": 2.0,
        "mean": 2.8,
        "mode": 1.0,
        "std": 3.5,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "transaction_velocity_24h": {
        "median": 8.0,
        "mean": 10.5,
        "mode": 5.0,
        "std": 12.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "failed_login_attempts_24h": {
        "median": 0.0,
        "mean": 0.3,
        "mode": 0.0,
        "std": 1.2,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "account_age_days": {
        "median": 180.0,
        "mean": 220.0,
        "mode": 90.0,
        "std": 180.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "historical_fraud_flag": {
        "median": 0.0,
        "mean": 0.06,
        "mode": 0.0,
        "std": 0.24,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "device_risk_score": {
        "median": 0.08,
        "mean": 0.12,
        "mode": 0.05,
        "std": 0.18,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "device_fraud_count": {
        "median": 0.0,
        "mean": 0.15,
        "mode": 0.0,
        "std": 0.7,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "distinct_user_count": {
        "median": 1.0,
        "mean": 1.4,
        "mode": 1.0,
        "std": 1.2,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "amount": {
        "median": 2000.0,
        "mean": 8500.0,
        "mode": 1000.0,
        "std": 25000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "geo_distance_from_last_tx": {
        "median": 5.0,
        "mean": 45.0,
        "mode": 0.0,
        "std": 180.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "time_of_day": {"median": 0.50, "mean": 0.48, "mode": 0.42, "std": 0.20, "sample_count": 500, "updated_at": NOW_TS},
    "currency_risk": {
        "median": 0.10,
        "mean": 0.12,
        "mode": 0.10,
        "std": 0.08,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
}

AML_PRIORS = {
    "amount": {
        "median": 5000.0,
        "mean": 15000.0,
        "mode": 1000.0,
        "std": 45000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "velocity_1h": {"median": 2.0, "mean": 3.5, "mode": 1.0, "std": 5.0, "sample_count": 500, "updated_at": NOW_TS},
    "velocity_24h": {"median": 8.0, "mean": 12.0, "mode": 5.0, "std": 15.0, "sample_count": 500, "updated_at": NOW_TS},
    "total_volume_24h": {
        "median": 20000.0,
        "mean": 55000.0,
        "mode": 5000.0,
        "std": 120000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "avg_amount_30d": {
        "median": 5000.0,
        "mean": 12000.0,
        "mode": 2000.0,
        "std": 30000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "std_amount_30d": {
        "median": 3000.0,
        "mean": 8000.0,
        "mode": 1000.0,
        "std": 20000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "time_of_day": {"median": 0.50, "mean": 0.48, "mode": 0.42, "std": 0.20, "sample_count": 500, "updated_at": NOW_TS},
    "is_cross_border": {
        "median": 0.0,
        "mean": 0.08,
        "mode": 0.0,
        "std": 0.27,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "account_age_days": {
        "median": 180.0,
        "mean": 200.0,
        "mode": 90.0,
        "std": 170.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "device_count_30d": {
        "median": 1.0,
        "mean": 1.6,
        "mode": 1.0,
        "std": 1.5,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "ip_count_30d": {"median": 2.0, "mean": 3.5, "mode": 1.0, "std": 5.0, "sample_count": 500, "updated_at": NOW_TS},
    "new_device_flag": {
        "median": 0.0,
        "mean": 0.10,
        "mode": 0.0,
        "std": 0.30,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "kyc_completeness_score": {
        "median": 1.0,
        "mean": 0.85,
        "mode": 1.0,
        "std": 0.25,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "network_risk_score": {
        "median": 0.08,
        "mean": 0.15,
        "mode": 0.02,
        "std": 0.22,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "circular_transfer_flag": {
        "median": 0.0,
        "mean": 0.04,
        "mode": 0.0,
        "std": 0.20,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "shared_device_cluster_size": {
        "median": 0.0,
        "mean": 0.8,
        "mode": 0.0,
        "std": 2.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "high_risk_country_flag": {
        "median": 0.0,
        "mean": 0.06,
        "mode": 0.0,
        "std": 0.24,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "sanctions_proximity_score": {
        "median": 0.0,
        "mean": 0.05,
        "mode": 0.0,
        "std": 0.18,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "ip_country_mismatch": {
        "median": 0.0,
        "mean": 0.06,
        "mode": 0.0,
        "std": 0.24,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "historical_structuring_flag": {
        "median": 0.0,
        "mean": 0.06,
        "mode": 0.0,
        "std": 0.24,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "structuring_score_24h": {
        "median": 0.0,
        "mean": 0.08,
        "mode": 0.0,
        "std": 0.20,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "rapid_drain_flag": {
        "median": 0.0,
        "mean": 0.06,
        "mode": 0.0,
        "std": 0.24,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
}

MERCHANT_PRIORS = {
    "transaction_count_1h": {
        "median": 8.0,
        "mean": 12.0,
        "mode": 3.0,
        "std": 15.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "transaction_count_24h": {
        "median": 80.0,
        "mean": 120.0,
        "mode": 30.0,
        "std": 140.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "transaction_volume_24h": {
        "median": 300000.0,
        "mean": 800000.0,
        "mode": 100000.0,
        "std": 1500000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "unique_customers_24h": {
        "median": 30.0,
        "mean": 50.0,
        "mode": 15.0,
        "std": 60.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "avg_transaction_amount_30d": {
        "median": 5000.0,
        "mean": 25000.0,
        "mode": 1500.0,
        "std": 80000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "std_transaction_amount_30d": {
        "median": 3000.0,
        "mean": 18000.0,
        "mode": 800.0,
        "std": 50000.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "chargeback_rate_90d": {
        "median": 0.002,
        "mean": 0.006,
        "mode": 0.001,
        "std": 0.012,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "refund_rate_90d": {
        "median": 0.008,
        "mean": 0.015,
        "mode": 0.005,
        "std": 0.025,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "account_age_days": {
        "median": 200.0,
        "mean": 210.0,
        "mode": 100.0,
        "std": 120.0,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "fraud_transaction_rate": {
        "median": 0.0005,
        "mean": 0.003,
        "mode": 0.0002,
        "std": 0.008,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "high_risk_customer_ratio": {
        "median": 0.03,
        "mean": 0.06,
        "mode": 0.02,
        "std": 0.08,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "cross_border_ratio": {
        "median": 0.05,
        "mean": 0.08,
        "mode": 0.0,
        "std": 0.12,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "velocity_spike_flag": {
        "median": 0.0,
        "mean": 0.15,
        "mode": 0.0,
        "std": 0.36,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "mcc_risk_score": {
        "median": 0.08,
        "mean": 0.12,
        "mode": 0.05,
        "std": 0.12,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
    "avg_customer_risk_score": {
        "median": 0.08,
        "mean": 0.10,
        "mode": 0.06,
        "std": 0.08,
        "sample_count": 500,
        "updated_at": NOW_TS,
    },
}


def seed_population_priors(pipe):
    """Seed pesaflow:priors:{domain} — population-level feature stats."""
    pipe.setex("pesaflow:priors:fraud", TTL_PRIORS, json.dumps(FRAUD_PRIORS))
    pipe.setex("pesaflow:priors:aml", TTL_PRIORS, json.dumps(AML_PRIORS))
    pipe.setex("pesaflow:priors:merchant", TTL_PRIORS, json.dumps(MERCHANT_PRIORS))


# ---------------------------------------------------------------------------
# 14. Feature coverage & maturity stats
# ---------------------------------------------------------------------------


def seed_coverage_stats(pipe):
    """Seed feature coverage stats used by maturity detection."""
    for domain, total in [("fraud", 13), ("aml", 22), ("merchant", 15)]:
        # In COLD mode: we have feature store data but no trained model
        # Coverage is high because feature stores are populated
        non_default = total - 1  # most features populated
        coverage = round(non_default / total, 4)

        pipe.setex(
            f"pesaflow:features:{domain}:coverage_stats",
            TTL_PRIORS,
            json.dumps({"total": total, "non_default": non_default}),
        )
        pipe.setex(
            f"pesaflow:maturity:{domain}:feature_coverage",
            TTL_PRIORS,
            str(coverage),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Connecting to Redis: {REDIS_URL}")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        r.ping()
    except redis.ConnectionError:
        print(f"ERROR: Cannot connect to Redis at {REDIS_URL}")
        sys.exit(1)

    pipe = r.pipeline()
    counts = {}

    def track(name, fn):
        before = len(pipe.command_stack)
        fn(pipe)
        counts[name] = len(pipe.command_stack) - before

    track("User fraud features (50)", seed_user_features)
    track("Device features (30)", seed_device_features)
    track("User last-geo (50)", seed_user_geo)
    track("Fraud velocity (50 users)", seed_fraud_velocity)
    track("AML velocity (50 users)", seed_aml_velocity)
    track("AML aggregate features (50)", seed_aml_aggregate_features)
    track("AML network features (50)", seed_aml_network_features)
    track("AML known devices (50)", seed_aml_known_devices)
    track("IP geolocation (54)", seed_ip_geo)
    track("Merchant features (20)", seed_merchant_features)
    track("Merchant velocity+signals (20)", seed_merchant_velocity_and_signals)
    track("Blacklists", seed_blacklists)
    track("Population priors (3 domains)", seed_population_priors)
    track("Coverage stats (3 domains)", seed_coverage_stats)

    results = pipe.execute()

    print(f"\nSeeded {len(results)} Redis commands successfully.\n")
    print("--- Breakdown ---")
    for name, count in counts.items():
        print(f"  {name:40s}  {count:>5d} ops")
    print(f"  {'TOTAL':40s}  {len(results):>5d} ops")

    # Quick verification
    print("\n--- Verification ---")
    sample_uid = user_uuid(1)
    sample_mid = merchant_uuid(1)
    print(f"  user:{sample_uid}:features → {r.get(f'user:{sample_uid}:features') is not None}")
    print(f"  device:dev-seed-01:features → {r.get('device:dev-seed-01:features') is not None}")
    print(f"  merchant:features:{sample_mid} → {r.get(f'merchant:features:{sample_mid}') is not None}")
    print(f"  blacklist:users count → {r.scard('blacklist:users')}")
    print(f"  blacklist:devices count → {r.scard('blacklist:devices')}")
    print(f"  pesaflow:priors:fraud → {r.get('pesaflow:priors:fraud') is not None}")
    print(f"  pesaflow:maturity:fraud:feature_coverage → {r.get('pesaflow:maturity:fraud:feature_coverage')}")
    print("\n--- Redis seed complete ---")


if __name__ == "__main__":
    main()
