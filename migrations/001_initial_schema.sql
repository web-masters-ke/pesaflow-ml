-- ============================================================
-- Pesaflow ML — Fraud Detection & AML Risk Scoring Engine
-- Initial Database Schema Migration
-- ============================================================
-- Runs inside the shared 'pesaflow' database using ai_schema
-- to coexist with the backend microservices.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS ai_schema;
SET search_path TO ai_schema;

-- === 1. Model Registry ===

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID NOT NULL,
    status VARCHAR(20) CHECK (status IN ('ACTIVE', 'INACTIVE')) NOT NULL
);
CREATE INDEX idx_ml_models_status ON ml_models(status);

CREATE TABLE IF NOT EXISTS ml_model_versions (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES ml_models(id),
    version VARCHAR(20) NOT NULL,
    training_dataset_hash VARCHAR(128) NOT NULL,
    hyperparameters JSONB NOT NULL,
    artifact_path TEXT NOT NULL,
    performance_metrics JSONB NOT NULL,
    deployed_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_model_versions_model_id ON ml_model_versions(model_id);
CREATE INDEX idx_model_versions_is_active ON ml_model_versions(is_active);
-- Only one active version per model
CREATE UNIQUE INDEX idx_model_versions_unique_active
    ON ml_model_versions(model_id) WHERE is_active = TRUE;


-- === 2. Threshold Configuration ===

CREATE TABLE IF NOT EXISTS ml_threshold_versions (
    id UUID PRIMARY KEY,
    model_version_id UUID REFERENCES ml_model_versions(id),
    approve_threshold NUMERIC(4,3) NOT NULL,
    review_threshold NUMERIC(4,3) NOT NULL,
    block_threshold NUMERIC(4,3) NOT NULL,
    environment VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_threshold_order CHECK (approve_threshold < review_threshold AND review_threshold < block_threshold)
);


-- === 3. Fraud Predictions (partitioned by month) ===

CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID NOT NULL,
    transaction_id UUID NOT NULL,
    user_id UUID NOT NULL,
    model_version_id VARCHAR(50) NOT NULL,
    threshold_version_id VARCHAR(50) NOT NULL,
    risk_score NUMERIC(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('APPROVE', 'REVIEW', 'MONITOR', 'BLOCK')),
    override_flag BOOLEAN DEFAULT FALSE,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partitions
CREATE TABLE ml_predictions_2026_01 PARTITION OF ml_predictions
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE ml_predictions_2026_02 PARTITION OF ml_predictions
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE ml_predictions_2026_03 PARTITION OF ml_predictions
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE ml_predictions_2026_04 PARTITION OF ml_predictions
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE ml_predictions_2026_05 PARTITION OF ml_predictions
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE ml_predictions_2026_06 PARTITION OF ml_predictions
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

CREATE INDEX idx_predictions_transaction_id ON ml_predictions(transaction_id);
CREATE INDEX idx_predictions_user_id ON ml_predictions(user_id);
CREATE INDEX idx_predictions_decision ON ml_predictions(decision);


-- === 4. Feature Snapshots (for audit) ===

CREATE TABLE IF NOT EXISTS ml_prediction_feature_snapshot (
    prediction_id UUID NOT NULL,
    feature_data JSONB NOT NULL,
    feature_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (prediction_id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE ml_prediction_feature_snapshot_2026_01 PARTITION OF ml_prediction_feature_snapshot
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE ml_prediction_feature_snapshot_2026_02 PARTITION OF ml_prediction_feature_snapshot
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE ml_prediction_feature_snapshot_2026_03 PARTITION OF ml_prediction_feature_snapshot
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE ml_prediction_feature_snapshot_2026_04 PARTITION OF ml_prediction_feature_snapshot
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE ml_prediction_feature_snapshot_2026_05 PARTITION OF ml_prediction_feature_snapshot
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE ml_prediction_feature_snapshot_2026_06 PARTITION OF ml_prediction_feature_snapshot
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');


-- === 5. Feature Store — User ===

CREATE TABLE IF NOT EXISTS feature_store_user (
    user_id UUID PRIMARY KEY,
    avg_transaction_amount_7d NUMERIC(18,2),
    transaction_velocity_1h INTEGER,
    transaction_velocity_24h INTEGER,
    failed_login_attempts_24h INTEGER,
    account_age_days INTEGER,
    historical_fraud_flag BOOLEAN,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_feature_user_last_updated ON feature_store_user(last_updated);


-- === 6. Feature Store — Device ===

CREATE TABLE IF NOT EXISTS feature_store_device (
    device_fingerprint VARCHAR(255) PRIMARY KEY,
    device_risk_score NUMERIC(5,4),
    device_fraud_count INTEGER,
    distinct_user_count INTEGER,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- === 7. Fraud Review Cases ===

CREATE TABLE IF NOT EXISTS fraud_review_cases (
    id UUID PRIMARY KEY,
    prediction_id UUID NOT NULL,
    reviewer_id UUID,
    review_decision VARCHAR(20),
    review_notes TEXT,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_review_cases_prediction_id ON fraud_review_cases(prediction_id);
CREATE INDEX idx_review_cases_reviewed_at ON fraud_review_cases(reviewed_at);


-- === 8. Drift Metrics ===

CREATE TABLE IF NOT EXISTS ml_drift_metrics (
    id UUID PRIMARY KEY,
    model_version_id VARCHAR(50) NOT NULL,
    psi_score NUMERIC(6,4),
    accuracy NUMERIC(5,4),
    fraud_rate NUMERIC(5,4),
    evaluation_window_start TIMESTAMPTZ,
    evaluation_window_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- === 9. Audit Logs (append-only) ===

CREATE TABLE IF NOT EXISTS ml_audit_logs (
    id UUID PRIMARY KEY,
    entity_type VARCHAR(50),
    entity_id UUID,
    action VARCHAR(50),
    performed_by UUID,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- === 10. AML Models ===

CREATE TABLE IF NOT EXISTS aml_models (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    artifact_path TEXT NOT NULL,
    training_dataset_hash TEXT NOT NULL,
    feature_schema_version VARCHAR(50) NOT NULL,
    hyperparameters JSONB,
    performance_metrics JSONB,
    status VARCHAR(20) CHECK (status IN ('DRAFT', 'STAGING', 'PRODUCTION', 'RETIRED')),
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);
CREATE INDEX idx_aml_models_status ON aml_models(status);


-- === 11. AML Predictions ===

CREATE TABLE IF NOT EXISTS aml_predictions (
    id UUID PRIMARY KEY,
    transaction_id UUID NOT NULL,
    user_id UUID NOT NULL,
    model_id VARCHAR(50),
    risk_score NUMERIC(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level VARCHAR(20),
    decision VARCHAR(20),
    rule_override BOOLEAN DEFAULT FALSE,
    top_risk_factors JSONB,
    feature_snapshot JSONB,
    threshold_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_aml_predictions_user ON aml_predictions(user_id);
CREATE INDEX idx_aml_predictions_transaction ON aml_predictions(transaction_id);
CREATE INDEX idx_aml_predictions_created_at ON aml_predictions(created_at);


-- === 12. AML User Risk Profile ===

CREATE TABLE IF NOT EXISTS aml_user_risk_profile (
    user_id UUID PRIMARY KEY,
    cumulative_risk_score NUMERIC(5,4),
    risk_category VARCHAR(20),
    risk_trend VARCHAR(20),
    last_transaction_at TIMESTAMPTZ,
    velocity_1h INTEGER,
    velocity_24h INTEGER,
    total_volume_24h NUMERIC(18,2),
    network_risk_score NUMERIC(5,4),
    sanctions_flag BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- === 13. AML Cases ===

CREATE TABLE IF NOT EXISTS aml_cases (
    id UUID PRIMARY KEY,
    entity_type VARCHAR(20) CHECK (entity_type IN ('USER', 'TRANSACTION', 'MERCHANT')),
    entity_id UUID NOT NULL,
    trigger_reason VARCHAR(500),
    risk_score NUMERIC(5,4),
    priority VARCHAR(20),
    status VARCHAR(30) CHECK (status IN ('OPEN', 'INVESTIGATING', 'ESCALATED', 'CLOSED_FALSE_POSITIVE', 'CLOSED_CONFIRMED', 'CLOSED_INCONCLUSIVE')),
    assigned_to UUID,
    notes TEXT,
    resolution VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);
CREATE INDEX idx_aml_cases_status ON aml_cases(status);
CREATE INDEX idx_aml_cases_entity ON aml_cases(entity_id);


-- === 14. AML Audit Logs ===

CREATE TABLE IF NOT EXISTS aml_audit_logs (
    id UUID PRIMARY KEY,
    entity_type VARCHAR(50),
    entity_id UUID,
    action VARCHAR(100),
    performed_by UUID,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- === 15. AML Threshold Config ===

CREATE TABLE IF NOT EXISTS aml_threshold_config (
    id UUID PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    critical_threshold NUMERIC(5,4),
    high_threshold NUMERIC(5,4),
    medium_threshold NUMERIC(5,4),
    structuring_window_hours INTEGER,
    structuring_txn_count INTEGER,
    velocity_multiplier NUMERIC(5,2),
    country_risk_weights JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    approved_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_aml_threshold_active ON aml_threshold_config(is_active)
    WHERE is_active = TRUE;


-- === 16. AML User Features ===

CREATE TABLE IF NOT EXISTS aml_user_features (
    user_id UUID PRIMARY KEY,
    avg_transaction_amount NUMERIC(18,2),
    std_transaction_amount NUMERIC(18,2),
    transaction_count_30d INTEGER,
    total_volume_30d NUMERIC(18,2),
    device_count_30d INTEGER,
    ip_count_30d INTEGER,
    high_risk_country_exposure INTEGER,
    historical_structuring_flag BOOLEAN,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);


-- === 17. AML Network Edges ===

CREATE TABLE IF NOT EXISTS aml_network_edges (
    id UUID PRIMARY KEY,
    source_user_id UUID,
    target_user_id UUID,
    relationship_type VARCHAR(50),
    weight NUMERIC(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_network_edges_source ON aml_network_edges(source_user_id);
CREATE INDEX idx_network_edges_target ON aml_network_edges(target_user_id);


-- === 18. AML Sanctions Entities ===

CREATE TABLE IF NOT EXISTS aml_sanctions_entities (
    id UUID PRIMARY KEY,
    name TEXT,
    aliases TEXT[],
    country VARCHAR(5),
    identifier TEXT,
    source VARCHAR(50),
    last_updated TIMESTAMPTZ
);
CREATE INDEX idx_sanctions_name ON aml_sanctions_entities USING gin(to_tsvector('english', name));
