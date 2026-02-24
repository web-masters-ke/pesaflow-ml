# Pesaflow ML — Real-Time Fraud Detection & AML Risk Scoring Engine

Production-grade ML system for real-time fraud detection, anti-money laundering (AML) risk scoring, and merchant risk assessment, purpose-built for African mobile money and fintech platforms.

## Architecture

```
                          +-----------+
                          |  FastAPI   |
                          |  Gateway   |
                          +-----+-----+
                                |
              +-----------------+------------------+
              |                 |                   |
        +-----v-----+    +-----v-----+    +--------v-------+
        |   Fraud    |    |    AML    |    |   Merchant     |
        |  Scoring   |    |  Scoring  |    |    Risk        |
        |  Service   |    |  Service  |    |   Service      |
        +-----+------+    +-----+-----+   +--------+-------+
              |                 |                    |
        +-----v------+   +-----v------+   +---------v------+
        |  Feature   |   |  Feature   |   |   Feature      |
        | Extractor  |   | Extractor  |   |  Extractor     |
        +-----+------+   +-----+------+   +---------+------+
              |                 |                    |
              +---------+-------+--------------------+
                        |
              +---------v---------+
              |   Redis (Cache)   |
              |   + PostgreSQL    |
              +-------------------+
                        |
              +---------v---------+
              |   Model Registry  |
              |     (MLflow)      |
              +-------------------+
```

**Key components:**
- **FastAPI Gateway** — API layer with auth, rate limiting, request validation
- **Scoring Services** — Domain-specific orchestrators (fraud, AML, merchant)
- **Feature Extractors** — Real-time feature computation from Redis + PostgreSQL
- **Decision Engines** — Threshold-based + rule-override decision logic
- **Data Maturity Detector** — Drives scoring mode (rules vs ML vs ensemble)
- **Drift Detector** — PSI, Wasserstein, per-feature drift, confidence tracking
- **Resilience Layer** — Retry with exponential backoff + circuit breakers

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd pesaflow-ml
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your Redis, PostgreSQL, and MLflow credentials

# Run the API server
uvicorn serving.app.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
```

### Docker

```bash
docker-compose up -d
```

## ML Models

| Domain | Algorithm | Features | Objective |
|--------|-----------|----------|-----------|
| Fraud Detection | LightGBM | 13 | Binary classification (AUC) |
| AML Risk | XGBoost | 22 | Binary classification (AUC) |
| Merchant Risk | LightGBM | 15 | Binary classification (AUC) |

All models use:
- Optuna hyperparameter optimization (50 trials per model)
- SHAP explainability (computed inline for MEDIUM+ risk decisions)
- Calibrated probabilities via isotonic regression
- Class imbalance handling via `scale_pos_weight`

## Feature Dictionary

### Fraud Features (13)

| Feature | Type | Description |
|---------|------|-------------|
| `avg_transaction_amount_7d` | float | User's 7-day rolling average transaction amount |
| `transaction_velocity_1h` | int | Number of transactions in last hour |
| `transaction_velocity_24h` | int | Number of transactions in last 24 hours |
| `failed_login_attempts_24h` | int | Failed login attempts in 24h window |
| `account_age_days` | int | Days since account creation |
| `historical_fraud_flag` | int | Whether user has prior fraud labels (0/1) |
| `device_risk_score` | float | Device fingerprint risk score (0-1) |
| `device_fraud_count` | int | Number of fraud events on this device |
| `distinct_user_count` | int | Unique users on this device |
| `amount` | float | Current transaction amount |
| `geo_distance_from_last_tx` | float | Haversine distance (km) from last transaction location |
| `time_of_day` | float | Hour normalized to 0-1 |
| `currency_risk` | float | Currency risk score (0.1=safe, 0.9=high-risk) |

### AML Features (22)

| Feature | Type | Description |
|---------|------|-------------|
| `amount` | float | Transaction amount |
| `velocity_1h` | int | Transaction count in last hour |
| `velocity_24h` | int | Transaction count in last 24 hours |
| `total_volume_24h` | float | Total transaction volume in 24h |
| `avg_amount_30d` | float | 30-day average transaction amount |
| `std_amount_30d` | float | 30-day standard deviation of amounts |
| `time_of_day` | float | Hour normalized to 0-1 |
| `is_cross_border` | int | Cross-border transaction flag (0/1) |
| `account_age_days` | int | Days since account creation |
| `device_count_30d` | int | Unique devices in 30 days |
| `ip_count_30d` | int | Unique IPs in 30 days |
| `new_device_flag` | int | First-seen device flag (0/1) |
| `kyc_completeness_score` | float | KYC verification completeness (0-1) |
| `network_risk_score` | float | Graph-based network risk score |
| `circular_transfer_flag` | int | Circular fund transfer detected (0/1) |
| `shared_device_cluster_size` | int | Size of shared-device user cluster |
| `high_risk_country_flag` | int | FATF grey/black list country (0/1) |
| `sanctions_proximity_score` | float | Proximity to sanctioned jurisdictions (0-1) |
| `ip_country_mismatch` | int | IP geolocation vs declared country mismatch (0/1) |
| `historical_structuring_flag` | int | Prior structuring behavior (0/1) |
| `structuring_score_24h` | float | Current structuring pattern score (0-1) |
| `rapid_drain_flag` | int | Rapid account drain detected (0/1) |

### Merchant Features (15)

| Feature | Type | Description |
|---------|------|-------------|
| `transaction_count_1h` | int | Transactions in last hour |
| `transaction_count_24h` | int | Transactions in last 24 hours |
| `transaction_volume_24h` | float | Total volume in 24h |
| `unique_customers_24h` | int | Unique customers in 24h |
| `avg_transaction_amount_30d` | float | 30-day average transaction |
| `std_transaction_amount_30d` | float | 30-day std dev of amounts |
| `chargeback_rate_90d` | float | 90-day chargeback rate |
| `refund_rate_90d` | float | 90-day refund rate |
| `account_age_days` | int | Days since merchant onboarding |
| `fraud_transaction_rate` | float | Proportion of fraudulent transactions |
| `high_risk_customer_ratio` | float | Ratio of high-risk customers |
| `cross_border_ratio` | float | Cross-border transaction ratio |
| `velocity_spike_flag` | int | Velocity anomaly flag (0/1) |
| `mcc_risk_score` | float | Merchant Category Code risk score |
| `avg_customer_risk_score` | float | Average customer risk score |

## API Endpoints

### Fraud Scoring
```
POST   /api/v1/fraud/score          — Score a single transaction
POST   /api/v1/fraud/batch          — Batch score up to 1000 transactions
GET    /api/v1/fraud/explain/{txn}   — Get SHAP explanation for a scored transaction
```

### AML Scoring
```
POST   /api/v1/aml/score            — Score a transaction for AML risk
POST   /api/v1/aml/batch            — Batch AML scoring
GET    /api/v1/aml/explain/{txn}     — Get SHAP explanation
```

### Merchant Risk
```
POST   /api/v1/merchant/score       — Score a merchant
GET    /api/v1/merchant/explain/{id} — Get SHAP explanation
```

### Case Management (AML)
```
POST   /api/v1/cases                — Create AML case
GET    /api/v1/cases                — List cases with filters
GET    /api/v1/cases/{id}           — Get case detail
PATCH  /api/v1/cases/{id}           — Update case (status, notes, resolution)
```

### Admin
```
PUT    /api/v1/admin/thresholds     — Update decision thresholds
GET    /api/v1/admin/thresholds     — Get threshold history
GET    /api/v1/health               — Service health check
```

## Data Maturity System

The system adapts its scoring strategy based on how much labeled data is available per domain. This prevents unreliable ML scores when training data is sparse.

| Level | Samples | Coverage | Behavior |
|-------|---------|----------|----------|
| **COLD** | <100 | <70% | Pure rules — ML model is not invoked |
| **WARMING** | 100-1000 | 70-90% | Hybrid — 30% ML + 70% anomaly detection |
| **WARM** | 1000-10000 | >90% | ML primary — 70% ML + 30% anomaly detection |
| **HOT** | >10000 | >90% + stable PSI | Full ensemble or pure ML |

Maturity is cached in Redis and recomputed every hour. The `DataMaturityDetector` monitors:
- Labeled sample count
- Positive rate (fraud/suspicious rate)
- Feature coverage (% of non-default feature values)
- Feature staleness (age of cached feature data)
- Distribution stability (PSI vs baseline)
- Label delay (time between transaction and label confirmation)

## Anomaly Model Blending Strategy

The system blends supervised ML predictions with unsupervised anomaly detection (Isolation Forest) based on data maturity. This provides robust scoring even when labeled data is limited.

### Blending Weights by Maturity

```
COLD:    [Rules: 100%] ←─── No ML, no anomaly
          Confidence: 0.1

WARMING: [ML: 30% | Anomaly: 70%] ←─── Anomaly-dominant
          Confidence: 0.3
          Rationale: Supervised signal is weak with <1000 labels.
          Isolation Forest provides base fraud probability.

WARM:    [ML: 70% | Anomaly: 30%] ←─── ML-dominant
          Confidence: varies (calibrated)
          Rationale: Supervised model has enough signal to lead.
          Anomaly score adds diversity and catches novel patterns.

HOT:     [Ensemble: 100%] or [ML: 100%]
          Confidence: ≥0.95
          Rationale: Fully trusted ensemble (LGB+XGB+RF stacked).
          Anomaly blending no longer needed.
```

### How Blending Works

In `_score_by_maturity()` for each scoring service:

1. The supervised model produces a calibrated probability (`ml_score`)
2. The Isolation Forest produces an anomaly score (`anomaly_score`, 0-1)
3. These are linearly combined: `blended = alpha * ml_score + (1 - alpha) * anomaly_score`
4. The `alpha` value comes from the maturity level (0.0 for COLD, 0.3 for WARMING, etc.)

The Decision Engine then applies confidence-aware thresholds — lower confidence results in stricter (lower) thresholds to catch more risky transactions.

## Decision Engine Logic

Each domain has a dedicated decision engine that converts ML risk scores into enforceable actions.

### Override Hierarchy (Hard Rules)

Hard rules always override ML scores, evaluated in priority order:

**Fraud:** Sanctioned country > Blacklisted device > Blacklisted IP > Blacklisted user > Velocity anomaly > New account + high amount

**AML:** Sanctions match > Watchlist hit > Blacklisted user > Blacklisted device > Structuring detected > Velocity spike

**Merchant:** Excessive chargebacks (>10%) > High fraud rate (>5%) > Velocity spike

### Score-Based Decisions

When no hard rules trigger, ML scores drive the decision:

| Score Range | Fraud Decision | AML Decision | Merchant Decision |
|------------|----------------|--------------|-------------------|
| Low | APPROVE | APPROVE | APPROVE (STANDARD tier) |
| Medium | APPROVE/MONITOR | MONITOR | MONITOR (ENHANCED tier) |
| High | REVIEW | REVIEW | REVIEW (RESTRICTED tier) |
| Critical | BLOCK | BLOCK | BLOCK (BLOCKED tier) |

### Fail-Closed Behavior

All three services fail closed on unhandled exceptions:
- Risk score set to 1.0
- Decision: BLOCK
- Risk level: CRITICAL
- Override reason: `FAIL_CLOSED` or `MODEL_FAILURE_FAIL_CLOSED`

## Monitoring & Drift Detection

### Drift Signals

| Signal | Method | Threshold | Action |
|--------|--------|-----------|--------|
| Score distribution shift | PSI | >0.2 | Retrain recommended |
| Score distribution shift | Wasserstein distance | >0.1 | Retrain recommended |
| Accuracy degradation | Accuracy drop from baseline | >5% | Retrain recommended |
| Fraud rate anomaly | Relative deviation | >15% | Alert + retrain |
| Per-feature drift | Feature-level PSI | >0.2 | Alert with top features |
| Confidence erosion | Mean confidence drop | >0.1 | Alert |
| Label feedback delay | Average label delay | >72h | Warning |

### Streaming Drift

Real-time drift detection via ADWIN (Adaptive Windowing) on Kafka streams with configurable window sizes.

## Configuration

### `config/model_config.yaml`

Controls model hyperparameters, training settings, Optuna optimization, SHAP explainability, drift detection thresholds, data maturity settings, and anomaly detection parameters.

Key settings:
- `optuna.enabled: true` — Automated hyperparameter tuning (50 trials per model)
- `explainability.compute_for: [MEDIUM, HIGH, CRITICAL]` — SHAP computed inline for non-LOW decisions
- `drift_detection.check_frequency: weekly` — Batch drift checks
- `data_maturity.check_interval_seconds: 3600` — Maturity reassessment every hour

### `config/service_config.yaml`

API settings, Redis/PostgreSQL connection strings, Kafka config, auth settings.

## Resilience

The system uses retry decorators and circuit breakers for all external service calls:

- **Retry**: 3 attempts with exponential backoff (0.1s base, 2s max) for Redis, PostgreSQL, and sanctions screening calls
- **Circuit Breaker**: Opens after 5 consecutive failures, 30-second cooldown before half-open recovery test
- Applied to: `_store_prediction()`, `_create_case()`, feature lookups, drift metric storage

## Deployment

### Docker Compose (Development)

```bash
docker-compose up -d
```

Starts: API server, Redis, PostgreSQL, Kafka, Prometheus, Grafana

### Kubernetes (Production)

```bash
kubectl apply -f k8s/deployment.yaml
```

### BentoML (Model Serving)

```bash
cd bentoml_service
bentoml build
bentoml serve service:svc
```

## Training Pipeline

Airflow DAGs orchestrate model retraining:
- `fraud_training_dag.py` — Fraud model retraining with data validation, Optuna optimization, champion/challenger evaluation
- `aml_training_dag.py` — AML model retraining with similar pipeline

Models are registered in MLflow with version tracking and A/B test support.

## Project Structure

```
pesaflow-ml/
├── config/                     # Model and service configuration
├── feature_engineering/        # Feature extractors (fraud, AML, merchant)
├── models/                     # Model wrappers (LightGBM, XGBoost, Isolation Forest, Ensemble)
├── monitoring/                 # Drift detection, data maturity, metrics
├── serving/app/                # FastAPI application
│   ├── api/routes/             # API endpoints
│   ├── schemas/                # Pydantic request/response models
│   ├── services/               # Scoring services, decision engines, resilience
│   ├── middleware/              # Auth, security, request ID
│   └── kafka/                  # Streaming consumer/producer
├── training/                   # Training scripts and Optuna optimization
├── governance/                 # Bias detection, model validation
├── airflow/                    # Training pipeline DAGs
├── migrations/                 # Database schema migrations
├── tests/                      # Unit and integration tests
├── docker/                     # Dockerfile
├── k8s/                        # Kubernetes manifests
└── bentoml_service/            # BentoML serving configuration
```
# pesaflow-ml
