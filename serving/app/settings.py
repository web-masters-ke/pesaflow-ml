from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    SERVICE_NAME: str = "wasaa-pesaflow-ml"
    SERVICE_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Database (shared with pesaflow-backend)
    DATABASE_URL: str = "postgresql+asyncpg://pesaflow:pesaflow_secure_2024@localhost:5432/pesaflow"
    DATABASE_SYNC_URL: str = "postgresql://pesaflow:pesaflow_secure_2024@localhost:5432/pesaflow"
    DATABASE_SCHEMA: str = "ai_schema"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Redis (ML-dedicated)
    REDIS_URL: str = "redis://localhost:26379/0"
    REDIS_FEATURE_TTL: int = 300
    REDIS_VELOCITY_TTL_1H: int = 3600
    REDIS_VELOCITY_TTL_24H: int = 86400

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:29094"
    KAFKA_GROUP_ID: str = "pesaflow-ml-group"
    KAFKA_CLIENT_ID: str = "pesaflow-ml"
    KAFKA_SECURITY_PROTOCOL: str = "PLAINTEXT"

    # Model Storage
    MODEL_STORAGE_PATH: str = "./model_artifacts"
    S3_BUCKET: str = "pesaflow-ml-models"
    AWS_REGION: str = "af-south-1"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:15937"
    MLFLOW_EXPERIMENT_FRAUD: str = "pesaflow-fraud-detection"
    MLFLOW_EXPERIMENT_AML: str = "pesaflow-aml-scoring"

    # JWT
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"

    # Fraud Thresholds
    FRAUD_MODEL_VERSION: str = "v1.0.0"
    FRAUD_APPROVE_THRESHOLD: float = 0.60
    FRAUD_REVIEW_THRESHOLD: float = 0.75
    FRAUD_BLOCK_THRESHOLD: float = 0.90

    # AML Thresholds
    AML_MODEL_VERSION: str = "v1.0.0"
    AML_MEDIUM_THRESHOLD: float = 0.50
    AML_HIGH_THRESHOLD: float = 0.70
    AML_CRITICAL_THRESHOLD: float = 0.85

    # Merchant Thresholds
    MERCHANT_MODEL_VERSION: str = "v1.0.0"
    MERCHANT_STANDARD_THRESHOLD: float = 0.40
    MERCHANT_ENHANCED_THRESHOLD: float = 0.40
    MERCHANT_RESTRICTED_THRESHOLD: float = 0.65
    MERCHANT_BLOCKED_THRESHOLD: float = 0.85

    # Alert Configuration
    ALERT_SLACK_WEBHOOK_URL: str = ""
    ALERT_EMAIL_SMTP_HOST: str = ""
    ALERT_EMAIL_SMTP_PORT: int = 587
    ALERT_EMAIL_FROM: str = "alerts@pesaflow.com"
    ALERT_EMAIL_TO: str = ""
    ALERT_PAGERDUTY_API_KEY: str = ""
    ALERT_PAGERDUTY_SERVICE_ID: str = ""

    # Sanctions
    SANCTIONS_FUZZY_THRESHOLD: float = 0.85


@lru_cache()
def get_settings() -> Settings:
    return Settings()
