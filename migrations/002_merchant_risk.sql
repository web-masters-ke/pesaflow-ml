-- Migration 002: Merchant Risk Tables
-- Adds merchant risk scoring, profiles, graph edges, and related tables.
-- Runs inside the shared 'pesaflow' database using ai_schema.

CREATE SCHEMA IF NOT EXISTS ai_schema;
SET search_path TO ai_schema;

-- === Merchant Risk Profiles ===
CREATE TABLE IF NOT EXISTS merchant_risk_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL UNIQUE,
    merchant_name VARCHAR(255),
    mcc_code VARCHAR(10),
    risk_score DECIMAL(5, 4) DEFAULT 0.0,
    risk_level VARCHAR(20) DEFAULT 'LOW',
    merchant_tier VARCHAR(20) DEFAULT 'STANDARD',
    avg_transaction_amount_30d DECIMAL(15, 2) DEFAULT 0,
    std_transaction_amount_30d DECIMAL(15, 2) DEFAULT 0,
    chargeback_rate_90d DECIMAL(5, 4) DEFAULT 0,
    refund_rate_90d DECIMAL(5, 4) DEFAULT 0,
    fraud_transaction_rate DECIMAL(5, 4) DEFAULT 0,
    total_transactions_30d INTEGER DEFAULT 0,
    total_volume_30d DECIMAL(20, 2) DEFAULT 0,
    unique_customers_30d INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_merchant_profiles_merchant_id ON merchant_risk_profiles(merchant_id);
CREATE INDEX idx_merchant_profiles_risk_level ON merchant_risk_profiles(risk_level);
CREATE INDEX idx_merchant_profiles_tier ON merchant_risk_profiles(merchant_tier);

-- === Merchant Risk Predictions (partitioned by month) ===
CREATE TABLE IF NOT EXISTS merchant_risk_predictions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    merchant_tier VARCHAR(20) NOT NULL,
    model_version VARCHAR(50),
    feature_snapshot JSONB,
    top_risk_factors TEXT[],
    rule_overrides TEXT[],
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE merchant_risk_predictions_2026_01 PARTITION OF merchant_risk_predictions
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE merchant_risk_predictions_2026_02 PARTITION OF merchant_risk_predictions
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE merchant_risk_predictions_2026_03 PARTITION OF merchant_risk_predictions
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE merchant_risk_predictions_2026_04 PARTITION OF merchant_risk_predictions
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE merchant_risk_predictions_2026_05 PARTITION OF merchant_risk_predictions
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE merchant_risk_predictions_2026_06 PARTITION OF merchant_risk_predictions
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

CREATE INDEX idx_merchant_pred_merchant_id ON merchant_risk_predictions(merchant_id, created_at DESC);

-- === Merchant Tier History ===
CREATE TABLE IF NOT EXISTS merchant_tier_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    previous_tier VARCHAR(20),
    new_tier VARCHAR(20) NOT NULL,
    reason VARCHAR(500),
    changed_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_merchant_tier_history ON merchant_tier_history(merchant_id, created_at DESC);

-- === Merchant Graph Edges (for Graph Risk Engine) ===
CREATE TABLE IF NOT EXISTS merchant_graph_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_merchant_id UUID NOT NULL,
    target_merchant_id UUID NOT NULL,
    edge_type VARCHAR(50) NOT NULL,  -- SHARED_CUSTOMER, SHARED_DEVICE, SHARED_IP, TRANSACTION_FLOW
    weight DECIMAL(5, 4) DEFAULT 1.0,
    shared_entity_count INTEGER DEFAULT 0,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    UNIQUE(source_merchant_id, target_merchant_id, edge_type)
);

CREATE INDEX idx_graph_edges_source ON merchant_graph_edges(source_merchant_id);
CREATE INDEX idx_graph_edges_target ON merchant_graph_edges(target_merchant_id);
CREATE INDEX idx_graph_edges_type ON merchant_graph_edges(edge_type);

-- === Merchant Graph Node Metrics ===
CREATE TABLE IF NOT EXISTS merchant_graph_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    pagerank_score DECIMAL(10, 8) DEFAULT 0,
    community_id INTEGER,
    degree_centrality DECIMAL(8, 6) DEFAULT 0,
    betweenness_centrality DECIMAL(8, 6) DEFAULT 0,
    cluster_risk_score DECIMAL(5, 4) DEFAULT 0,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_graph_metrics_merchant ON merchant_graph_metrics(merchant_id);
CREATE INDEX idx_graph_metrics_community ON merchant_graph_metrics(community_id);

-- === Threshold Configuration Audit ===
CREATE TABLE IF NOT EXISTS threshold_config_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL,  -- fraud, aml, merchant
    config_key VARCHAR(100) NOT NULL,
    old_value DECIMAL(5, 4),
    new_value DECIMAL(5, 4) NOT NULL,
    changed_by UUID,
    reason VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_threshold_audit_model ON threshold_config_audit(model_type, created_at DESC);

-- === Alert Log ===
CREATE TABLE IF NOT EXISTS alert_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL,  -- FRAUD_CRITICAL, AML_BLOCK, MERCHANT_BLOCKED, DRIFT_DETECTED
    severity VARCHAR(20) NOT NULL,    -- INFO, WARNING, CRITICAL
    channel VARCHAR(20) NOT NULL,     -- SLACK, EMAIL, PAGERDUTY, KAFKA
    recipient VARCHAR(255),
    subject VARCHAR(500),
    message TEXT,
    entity_type VARCHAR(50),
    entity_id VARCHAR(255),
    delivered BOOLEAN DEFAULT FALSE,
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alert_log_type ON alert_log(alert_type, created_at DESC);
CREATE INDEX idx_alert_log_severity ON alert_log(severity, delivered);

-- === Batch Scoring Jobs ===
CREATE TABLE IF NOT EXISTS batch_scoring_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL,    -- FRAUD_RESCORE, AML_RESCORE, MERCHANT_RESCORE
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING, RUNNING, COMPLETED, FAILED
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    initiated_by UUID,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_batch_jobs_status ON batch_scoring_jobs(status, created_at DESC);

-- === Regulatory Export Log ===
CREATE TABLE IF NOT EXISTS regulatory_export_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_type VARCHAR(50) NOT NULL,  -- SAR, STR, CTR
    format VARCHAR(20) NOT NULL,       -- CSV, JSON, XML
    record_count INTEGER DEFAULT 0,
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    file_path VARCHAR(500),
    file_hash VARCHAR(64),
    generated_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_regulatory_export_type ON regulatory_export_log(export_type, created_at DESC);
