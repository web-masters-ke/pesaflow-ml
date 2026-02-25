-- Migration 003: Label Feedback Loop, Active Learning, and Semi-Supervised Learning
-- Adds label columns to prediction tables (required by DataMaturityDetector),
-- active learning queue, pseudo-label storage, and audit tables.

BEGIN;

-- =============================================================================
-- 1. Add label columns to ALL prediction tables
--    These match exactly what DataMaturityDetector queries (WHERE label IS NOT NULL)
-- =============================================================================

ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS label SMALLINT CHECK (label IN (0, 1));
ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS labeled_at TIMESTAMPTZ;
ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS labeled_by UUID;
ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS label_source VARCHAR(30);

ALTER TABLE aml_predictions ADD COLUMN IF NOT EXISTS label SMALLINT CHECK (label IN (0, 1));
ALTER TABLE aml_predictions ADD COLUMN IF NOT EXISTS labeled_at TIMESTAMPTZ;
ALTER TABLE aml_predictions ADD COLUMN IF NOT EXISTS labeled_by UUID;
ALTER TABLE aml_predictions ADD COLUMN IF NOT EXISTS label_source VARCHAR(30);

ALTER TABLE merchant_risk_predictions ADD COLUMN IF NOT EXISTS label SMALLINT CHECK (label IN (0, 1));
ALTER TABLE merchant_risk_predictions ADD COLUMN IF NOT EXISTS labeled_at TIMESTAMPTZ;
ALTER TABLE merchant_risk_predictions ADD COLUMN IF NOT EXISTS labeled_by UUID;
ALTER TABLE merchant_risk_predictions ADD COLUMN IF NOT EXISTS label_source VARCHAR(30);

-- Add label_propagated flag to case tables for idempotent propagation
ALTER TABLE aml_cases ADD COLUMN IF NOT EXISTS label_propagated BOOLEAN DEFAULT FALSE;
ALTER TABLE aml_cases ADD COLUMN IF NOT EXISTS prediction_id UUID;

-- Indexes for label queries
CREATE INDEX IF NOT EXISTS idx_ml_predictions_label ON ml_predictions (label) WHERE label IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ml_predictions_unlabeled ON ml_predictions (created_at DESC) WHERE label IS NULL;
CREATE INDEX IF NOT EXISTS idx_aml_predictions_label ON aml_predictions (label) WHERE label IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_aml_predictions_unlabeled ON aml_predictions (created_at DESC) WHERE label IS NULL;
CREATE INDEX IF NOT EXISTS idx_merchant_predictions_label ON merchant_risk_predictions (label) WHERE label IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_merchant_predictions_unlabeled ON merchant_risk_predictions (created_at DESC) WHERE label IS NULL;

-- =============================================================================
-- 2. Label audit history (append-only, immutable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS label_audit_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(20) NOT NULL CHECK (domain IN ('fraud', 'aml', 'merchant')),
    prediction_id UUID NOT NULL,
    previous_label SMALLINT,
    new_label SMALLINT NOT NULL CHECK (new_label IN (0, 1)),
    label_source VARCHAR(30) NOT NULL,
    labeled_by UUID,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_label_audit_prediction ON label_audit_history (prediction_id);
CREATE INDEX IF NOT EXISTS idx_label_audit_domain ON label_audit_history (domain, created_at DESC);

-- =============================================================================
-- 3. Active learning queue
-- =============================================================================

CREATE TABLE IF NOT EXISTS active_learning_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(20) NOT NULL CHECK (domain IN ('fraud', 'aml', 'merchant')),
    prediction_id UUID NOT NULL,
    entity_id UUID,
    risk_score NUMERIC(5,4),
    informativeness_score NUMERIC(6,4) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ASSIGNED', 'LABELED', 'EXPIRED', 'SKIPPED')),
    assigned_to UUID,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_al_queue_domain_status ON active_learning_queue (domain, status, informativeness_score DESC);
CREATE INDEX IF NOT EXISTS idx_al_queue_expires ON active_learning_queue (expires_at) WHERE status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_al_queue_prediction ON active_learning_queue (prediction_id);

-- =============================================================================
-- 4. Active learning configuration (seeded per domain)
-- =============================================================================

CREATE TABLE IF NOT EXISTS active_learning_config (
    domain VARCHAR(20) PRIMARY KEY CHECK (domain IN ('fraud', 'aml', 'merchant')),
    strategy VARCHAR(30) NOT NULL DEFAULT 'UNCERTAINTY',
    daily_budget INTEGER NOT NULL DEFAULT 100,
    weekly_budget INTEGER NOT NULL DEFAULT 500,
    uncertainty_threshold NUMERIC(4,3) DEFAULT 0.400,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed default configs
INSERT INTO active_learning_config (domain, strategy, daily_budget, weekly_budget, uncertainty_threshold)
VALUES
    ('fraud', 'DISAGREEMENT', 100, 500, 0.400),
    ('aml', 'UNCERTAINTY', 80, 400, 0.450),
    ('merchant', 'ENTROPY', 50, 250, 0.350)
ON CONFLICT (domain) DO NOTHING;

-- =============================================================================
-- 5. SSL pseudo-label storage
-- =============================================================================

CREATE TABLE IF NOT EXISTS ssl_pseudo_labels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(20) NOT NULL CHECK (domain IN ('fraud', 'aml', 'merchant')),
    prediction_id UUID NOT NULL,
    feature_snapshot JSONB NOT NULL,
    pseudo_label INTEGER NOT NULL CHECK (pseudo_label IN (0, 1)),
    pseudo_label_confidence NUMERIC(5,4) NOT NULL,
    label_source VARCHAR(50) NOT NULL,
    source_model_version VARCHAR(50),
    source_iteration INTEGER DEFAULT 0,
    is_validated BOOLEAN DEFAULT FALSE,
    human_label INTEGER CHECK (human_label IN (0, 1)),
    label_correct BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ssl_pseudo_domain ON ssl_pseudo_labels (domain, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ssl_pseudo_validated ON ssl_pseudo_labels (domain, is_validated) WHERE is_validated = TRUE;
CREATE INDEX IF NOT EXISTS idx_ssl_pseudo_prediction ON ssl_pseudo_labels (prediction_id);

-- =============================================================================
-- 6. SSL training run metrics
-- =============================================================================

CREATE TABLE IF NOT EXISTS ssl_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(20) NOT NULL CHECK (domain IN ('fraud', 'aml', 'merchant')),
    technique VARCHAR(50) NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    labeled_count INTEGER NOT NULL DEFAULT 0,
    pseudo_labeled_count INTEGER NOT NULL DEFAULT 0,
    pseudo_label_agreement_rate NUMERIC(5,4),
    pseudo_label_flip_rate NUMERIC(5,4),
    model_auc_labeled NUMERIC(5,4),
    model_auc_full NUMERIC(5,4),
    converged BOOLEAN DEFAULT FALSE,
    config_snapshot JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ssl_runs_domain ON ssl_training_runs (domain, created_at DESC);

-- =============================================================================
-- 7. Cross-domain label transfers
-- =============================================================================

CREATE TABLE IF NOT EXISTS ssl_cross_domain_transfers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_domain VARCHAR(20) NOT NULL,
    target_domain VARCHAR(20) NOT NULL,
    source_prediction_id UUID NOT NULL,
    target_prediction_id UUID,
    source_label INTEGER NOT NULL CHECK (source_label IN (0, 1)),
    transferred_label INTEGER NOT NULL CHECK (transferred_label IN (0, 1)),
    confidence_decay NUMERIC(5,4) NOT NULL,
    transfer_rule VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ssl_transfers_source ON ssl_cross_domain_transfers (source_domain, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ssl_transfers_target ON ssl_cross_domain_transfers (target_domain, created_at DESC);

-- =============================================================================
-- 8. Label statistics materialized view
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS label_statistics AS
SELECT
    'fraud' AS domain,
    COUNT(*) AS total_predictions,
    COUNT(*) FILTER (WHERE label IS NOT NULL) AS labeled_count,
    COUNT(*) FILTER (WHERE label = 1) AS positive_count,
    COUNT(*) FILTER (WHERE label = 0) AS negative_count,
    COUNT(*) FILTER (WHERE label IS NULL) AS unlabeled_count,
    ROUND(COUNT(*) FILTER (WHERE label IS NOT NULL)::NUMERIC / NULLIF(COUNT(*), 0), 4) AS label_rate
FROM ml_predictions
UNION ALL
SELECT
    'aml' AS domain,
    COUNT(*) AS total_predictions,
    COUNT(*) FILTER (WHERE label IS NOT NULL) AS labeled_count,
    COUNT(*) FILTER (WHERE label = 1) AS positive_count,
    COUNT(*) FILTER (WHERE label = 0) AS negative_count,
    COUNT(*) FILTER (WHERE label IS NULL) AS unlabeled_count,
    ROUND(COUNT(*) FILTER (WHERE label IS NOT NULL)::NUMERIC / NULLIF(COUNT(*), 0), 4) AS label_rate
FROM aml_predictions
UNION ALL
SELECT
    'merchant' AS domain,
    COUNT(*) AS total_predictions,
    COUNT(*) FILTER (WHERE label IS NOT NULL) AS labeled_count,
    COUNT(*) FILTER (WHERE label = 1) AS positive_count,
    COUNT(*) FILTER (WHERE label = 0) AS negative_count,
    COUNT(*) FILTER (WHERE label IS NULL) AS unlabeled_count,
    ROUND(COUNT(*) FILTER (WHERE label IS NOT NULL)::NUMERIC / NULLIF(COUNT(*), 0), 4) AS label_rate
FROM merchant_risk_predictions;

CREATE UNIQUE INDEX IF NOT EXISTS idx_label_stats_domain ON label_statistics (domain);

COMMIT;
