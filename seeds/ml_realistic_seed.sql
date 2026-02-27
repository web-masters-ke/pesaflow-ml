-- =================================================================
-- PesaFlow ML — Realistic Data Seed Script
-- =================================================================
--
-- Populates backend tables (merchant_schema, payment_schema,
-- fraud_schema, auth_schema) and ML feature stores (ai_schema)
-- with realistic Kenyan fintech transaction patterns.
--
-- Data volume:
--   5 orgs · 20 merchants · 50 users · 500 payment requests
--   500 transactions · 500 fraud scores · 50 user features
--   30 device profiles · 50 AML profiles · 20 merchant profiles
--
-- Patterns: 80% normal · 15% suspicious · 5% fraudulent
-- Currency: KES (amounts stored as BIGINT cents in backend)
-- Period:   Past 90 days
--
-- Idempotent: ON CONFLICT DO NOTHING on all inserts
--
-- Usage:
--   docker exec -i pesaflow-postgres psql -U pesaflow -d pesaflow \
--     < seeds/ml_realistic_seed.sql
-- =================================================================

BEGIN;

SELECT setseed(0.42);

-- =================================================================
-- 0. SCHEMA & TABLE CREATION  (idempotent — IF NOT EXISTS)
--    Creates backend schemas/tables when backend migrations
--    haven't run yet. Safe no-op when they already exist.
-- =================================================================

CREATE SCHEMA IF NOT EXISTS merchant_schema;
CREATE SCHEMA IF NOT EXISTS auth_schema;
CREATE SCHEMA IF NOT EXISTS payment_schema;
CREATE SCHEMA IF NOT EXISTS fraud_schema;

-- merchant_schema.organizations
CREATE TABLE IF NOT EXISTS merchant_schema.organizations (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        VARCHAR(255) NOT NULL,
  status      VARCHAR(20)  DEFAULT 'active',
  created_at  TIMESTAMPTZ  DEFAULT NOW(),
  updated_at  TIMESTAMPTZ  DEFAULT NOW()
);

-- merchant_schema.merchants
CREATE TABLE IF NOT EXISTS merchant_schema.merchants (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id  UUID NOT NULL REFERENCES merchant_schema.organizations(id),
  business_name    VARCHAR(255) NOT NULL,
  trading_name     VARCHAR(255),
  business_type    VARCHAR(50) NOT NULL,
  registration_no  VARCHAR(100),
  tax_id           VARCHAR(100),
  email            VARCHAR(255) NOT NULL,
  business_email   VARCHAR(255),
  phone            VARCHAR(20) NOT NULL,
  website          VARCHAR(255),
  status           VARCHAR(30)  DEFAULT 'pending_kyc',
  kyc_status       VARCHAR(30)  DEFAULT 'pending',
  risk_level       VARCHAR(20),
  onboarded_at     TIMESTAMPTZ,
  created_at       TIMESTAMPTZ  DEFAULT NOW(),
  updated_at       TIMESTAMPTZ  DEFAULT NOW()
);

-- auth_schema.users
CREATE TABLE IF NOT EXISTS auth_schema.users (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email       VARCHAR(255) UNIQUE NOT NULL,
  password    VARCHAR(255) NOT NULL,
  first_name  VARCHAR(100),
  last_name   VARCHAR(100),
  phone       VARCHAR(20),
  status      VARCHAR(20)  DEFAULT 'active',
  created_at  TIMESTAMPTZ  DEFAULT NOW(),
  updated_at  TIMESTAMPTZ  DEFAULT NOW()
);

-- payment_schema.payment_requests
CREATE TABLE IF NOT EXISTS payment_schema.payment_requests (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  merchant_id       UUID NOT NULL,
  merchant_reference VARCHAR(150) NOT NULL,
  idempotency_key   VARCHAR(100) UNIQUE NOT NULL,
  amount            BIGINT NOT NULL,
  currency          CHAR(3) DEFAULT 'KES',
  customer_phone    VARCHAR(20),
  customer_email    VARCHAR(255),
  customer_name     VARCHAR(255),
  description       TEXT,
  metadata          JSONB,
  preferred_channel VARCHAR(50),
  status            VARCHAR(30) DEFAULT 'pending',
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- payment_schema.payment_transactions
CREATE TABLE IF NOT EXISTS payment_schema.payment_transactions (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  payment_request_id    UUID NOT NULL REFERENCES payment_schema.payment_requests(id),
  merchant_id           UUID NOT NULL,
  transaction_reference VARCHAR(150) UNIQUE NOT NULL,
  provider_reference    VARCHAR(150),
  channel               VARCHAR(50) NOT NULL,
  provider              VARCHAR(100) NOT NULL,
  amount                BIGINT NOT NULL,
  fee                   BIGINT DEFAULT 0,
  net_amount            BIGINT NOT NULL,
  currency              CHAR(3) DEFAULT 'KES',
  status                VARCHAR(30) DEFAULT 'initiated',
  customer_phone        VARCHAR(20),
  customer_email        VARCHAR(255),
  customer_name         VARCHAR(255),
  description           TEXT,
  metadata              JSONB,
  initiated_at          TIMESTAMPTZ DEFAULT NOW(),
  completed_at          TIMESTAMPTZ,
  failed_at             TIMESTAMPTZ
);

-- fraud_schema.fraud_scores
CREATE TABLE IF NOT EXISTS fraud_schema.fraud_scores (
  id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  payment_transaction_id UUID NOT NULL REFERENCES payment_schema.payment_transactions(id),
  risk_score             INT NOT NULL,
  risk_level             VARCHAR(20) NOT NULL,
  decision               VARCHAR(20) NOT NULL,
  model_version          VARCHAR(50),
  feature_vector         JSONB,
  created_at             TIMESTAMPTZ DEFAULT NOW()
);

-- =================================================================
-- 1. ORGANIZATIONS  (merchant_schema) — 5 Kenyan businesses
-- =================================================================

INSERT INTO merchant_schema.organizations (id, name, status, created_at, updated_at)
VALUES
  ('10000001-5eed-4000-a000-000000000001'::uuid, 'Safari Digital Ltd',       'active', NOW() - INTERVAL '400 days', NOW()),
  ('10000001-5eed-4000-a000-000000000002'::uuid, 'Tusker Payments Corp',     'active', NOW() - INTERVAL '350 days', NOW()),
  ('10000001-5eed-4000-a000-000000000003'::uuid, 'Uhuru Commerce Ltd',       'active', NOW() - INTERVAL '280 days', NOW()),
  ('10000001-5eed-4000-a000-000000000004'::uuid, 'Jambo Financial Services', 'active', NOW() - INTERVAL '200 days', NOW()),
  ('10000001-5eed-4000-a000-000000000005'::uuid, 'Mara Tech Solutions',      'active', NOW() - INTERVAL '120 days', NOW())
ON CONFLICT DO NOTHING;


-- =================================================================
-- 2. MERCHANTS  (merchant_schema) — 20 across 5 orgs
--    12 LOW · 5 MEDIUM · 3 HIGH risk
-- =================================================================

INSERT INTO merchant_schema.merchants
  (id, organization_id, business_name, trading_name, business_type,
   email, phone, status, kyc_status, risk_level, onboarded_at, created_at, updated_at)
VALUES
  -- Org 1: Safari Digital — fintech
  ('20000001-5eed-4000-a000-000000000001'::uuid, '10000001-5eed-4000-a000-000000000001'::uuid,
   'Nairobi Fresh Mart', 'Fresh Mart', 'retail',
   'freshmart@pesaflow.test', '+254700100001', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '380 days', NOW() - INTERVAL '390 days', NOW()),

  ('20000001-5eed-4000-a000-000000000002'::uuid, '10000001-5eed-4000-a000-000000000001'::uuid,
   'Karen Restaurant & Grill', 'Karen Grill', 'hospitality',
   'karengrill@pesaflow.test', '+254700100002', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '360 days', NOW() - INTERVAL '370 days', NOW()),

  ('20000001-5eed-4000-a000-000000000003'::uuid, '10000001-5eed-4000-a000-000000000001'::uuid,
   'Westlands Electronics', 'WestElec', 'retail',
   'westelec@pesaflow.test', '+254700100003', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '350 days', NOW() - INTERVAL '360 days', NOW()),

  ('20000001-5eed-4000-a000-000000000004'::uuid, '10000001-5eed-4000-a000-000000000001'::uuid,
   'Mombasa Beach Hotel', 'Beach Hotel', 'hospitality',
   'beachhotel@pesaflow.test', '+254700100004', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '340 days', NOW() - INTERVAL '350 days', NOW()),

  -- Org 2: Tusker Payments — payment processor
  ('20000001-5eed-4000-a000-000000000005'::uuid, '10000001-5eed-4000-a000-000000000002'::uuid,
   'Safaricom Agent Kibera', 'Saf Agent', 'telecom',
   'safagent@pesaflow.test', '+254700100005', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '330 days', NOW() - INTERVAL '340 days', NOW()),

  ('20000001-5eed-4000-a000-000000000006'::uuid, '10000001-5eed-4000-a000-000000000002'::uuid,
   'KCB Bank Agency Westlands', 'KCB Agency', 'financial',
   'kcbagency@pesaflow.test', '+254700100006', 'active', 'approved', 'MEDIUM',
   NOW() - INTERVAL '300 days', NOW() - INTERVAL '310 days', NOW()),

  ('20000001-5eed-4000-a000-000000000007'::uuid, '10000001-5eed-4000-a000-000000000002'::uuid,
   'Kilimall Online Store', 'Kilimall', 'ecommerce',
   'kilimall@pesaflow.test', '+254700100007', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '280 days', NOW() - INTERVAL '290 days', NOW()),

  ('20000001-5eed-4000-a000-000000000008'::uuid, '10000001-5eed-4000-a000-000000000002'::uuid,
   'CBD Pharmacy Plus', 'CBD Pharma', 'healthcare',
   'cbdpharma@pesaflow.test', '+254700100008', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '270 days', NOW() - INTERVAL '280 days', NOW()),

  -- Org 3: Uhuru Commerce — e-commerce
  ('20000001-5eed-4000-a000-000000000009'::uuid, '10000001-5eed-4000-a000-000000000003'::uuid,
   'Nairobi Jewelry Palace', 'Jewelry Palace', 'retail',
   'jewelry@pesaflow.test', '+254700100009', 'active', 'approved', 'MEDIUM',
   NOW() - INTERVAL '260 days', NOW() - INTERVAL '270 days', NOW()),

  ('20000001-5eed-4000-a000-000000000010'::uuid, '10000001-5eed-4000-a000-000000000003'::uuid,
   'Tusker Auto Dealers', 'Tusker Auto', 'automotive',
   'tuskerauto@pesaflow.test', '+254700100010', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '240 days', NOW() - INTERVAL '250 days', NOW()),

  ('20000001-5eed-4000-a000-000000000011'::uuid, '10000001-5eed-4000-a000-000000000003'::uuid,
   'Karen Supermarket', 'Karen Super', 'retail',
   'karensuper@pesaflow.test', '+254700100011', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '220 days', NOW() - INTERVAL '230 days', NOW()),

  ('20000001-5eed-4000-a000-000000000012'::uuid, '10000001-5eed-4000-a000-000000000003'::uuid,
   'Lavington Boutique', 'Lavi Boutique', 'retail',
   'boutique@pesaflow.test', '+254700100012', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '200 days', NOW() - INTERVAL '210 days', NOW()),

  -- Org 4: Jambo Financial — microfinance (includes HIGH risk)
  ('20000001-5eed-4000-a000-000000000013'::uuid, '10000001-5eed-4000-a000-000000000004'::uuid,
   'Quick Bet Gaming', 'QuickBet', 'gambling',
   'quickbet@pesaflow.test', '+254700100013', 'active', 'approved', 'HIGH',
   NOW() - INTERVAL '180 days', NOW() - INTERVAL '190 days', NOW()),

  ('20000001-5eed-4000-a000-000000000014'::uuid, '10000001-5eed-4000-a000-000000000004'::uuid,
   'Lucky Star Casino', 'Lucky Star', 'gambling',
   'luckystar@pesaflow.test', '+254700100014', 'active', 'approved', 'HIGH',
   NOW() - INTERVAL '160 days', NOW() - INTERVAL '170 days', NOW()),

  ('20000001-5eed-4000-a000-000000000015'::uuid, '10000001-5eed-4000-a000-000000000004'::uuid,
   'Rapid Money Transfer', 'Rapid Money', 'money_transfer',
   'rapidmoney@pesaflow.test', '+254700100015', 'active', 'approved', 'HIGH',
   NOW() - INTERVAL '90 days',  NOW() - INTERVAL '100 days', NOW()),

  ('20000001-5eed-4000-a000-000000000016'::uuid, '10000001-5eed-4000-a000-000000000004'::uuid,
   'Gikomba Wholesale Traders', 'Gikomba Wholesale', 'wholesale',
   'gikomba@pesaflow.test', '+254700100016', 'active', 'approved', 'MEDIUM',
   NOW() - INTERVAL '140 days', NOW() - INTERVAL '150 days', NOW()),

  -- Org 5: Mara Tech — technology
  ('20000001-5eed-4000-a000-000000000017'::uuid, '10000001-5eed-4000-a000-000000000005'::uuid,
   'Ngong Road Petrol Station', 'Ngong Petrol', 'fuel',
   'ngongpetrol@pesaflow.test', '+254700100017', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '100 days', NOW() - INTERVAL '110 days', NOW()),

  ('20000001-5eed-4000-a000-000000000018'::uuid, '10000001-5eed-4000-a000-000000000005'::uuid,
   'Eastleigh Electronics Hub', 'Eastleigh Elec', 'retail',
   'eastleigh@pesaflow.test', '+254700100018', 'active', 'approved', 'MEDIUM',
   NOW() - INTERVAL '80 days',  NOW() - INTERVAL '90 days',  NOW()),

  ('20000001-5eed-4000-a000-000000000019'::uuid, '10000001-5eed-4000-a000-000000000005'::uuid,
   'Thika Hardware Supplies', 'Thika Hardware', 'retail',
   'thikahw@pesaflow.test', '+254700100019', 'active', 'approved', 'LOW',
   NOW() - INTERVAL '60 days',  NOW() - INTERVAL '70 days',  NOW()),

  ('20000001-5eed-4000-a000-000000000020'::uuid, '10000001-5eed-4000-a000-000000000005'::uuid,
   'Virtual Forex Trading', 'VFX Trading', 'financial',
   'vfxtrading@pesaflow.test', '+254700100020', 'active', 'approved', 'MEDIUM',
   NOW() - INTERVAL '45 days',  NOW() - INTERVAL '50 days',  NOW())
ON CONFLICT DO NOTHING;


-- =================================================================
-- 3. USERS  (auth_schema) — 50 Kenyan user profiles
--    01-40: Normal · 41-47: Suspicious · 48-50: Fraudulent
-- =================================================================
-- Password: bcrypt placeholder (not a real credential)

INSERT INTO auth_schema.users
  (id, email, password, first_name, last_name, phone, status, created_at, updated_at)
SELECT
  ('30000001-5eed-4000-a000-' || lpad(n::text, 12, '0'))::uuid,
  'seed.user.' || n || '@pesaflow.test',
  '$2b$12$K4jK8fL3nR2mP5qS7tV9wOxY1zA3bC5dE7fG9hI0jK2lM4nO6pQr',
  -- Kenyan first names (50 entries)
  (ARRAY[
    'Kamau','James','Odhiambo','Grace','Kipchoge','Mercy','Otieno','Faith',
    'Mwangi','Sarah','Kiptoo','Esther','Juma','Rose','Ochieng','Lucy',
    'Njoroge','Diana','Kariuki','Amina','Mutua','Joy','Kibet','Zipporah',
    'Kosgei','Peter','Barasa','Wanjiku','Kevin','Akinyi','Chebet','Nyambura',
    'Daniel','Atieno','Muthoni','Njeri','Martin','Wairimu','Dennis','Kemunto',
    'Hassan','Ibrahim','Ali','Omar','Fatma','Ahmed','Yusuf','Mohamed','Aisha','Halima'
  ])[n],
  -- Kenyan last names (50 entries)
  (ARRAY[
    'Kimani','Ouma','Rotich','Maina','Okello','Ngugi','Kiplagat','Abdullahi',
    'Wanyama','Mutiso','Onyango','Korir','Hassan','Wekesa','Langat','Cheruiyot',
    'Kinyanjui','Omondi','Musyoka','Njenga','Karanja','Oduor','Sang','Muturi',
    'Ndirangu','Wambua','Bore','Kipruto','Mutai','Mwenda','Gitonga','Njuguna',
    'Kamau','Achieng','Chepkoech','Jebet','Kipkemboi','Chepkirui','Kipyegon','Chepngetich',
    'Abdi','Osman','Hussein','Adan','Mohamed','Yusuf','Warsame','Jama','Farah','Ismail'
  ])[n],
  '+2547' || lpad((10000000 + (n * 1234567) % 89999999)::text, 8, '0'),
  'active',
  -- Account age: normal=90-730d, suspicious=14-63d, fraudulent=1-10d
  CASE
    WHEN n <= 40 THEN NOW() - (INTERVAL '1 day' * (90 + n * 16))
    WHEN n <= 47 THEN NOW() - (INTERVAL '1 day' * (14 + (n - 40) * 7))
    ELSE              NOW() - (INTERVAL '1 day' * (1  + (n - 47) * 3))
  END,
  NOW()
FROM generate_series(1, 50) AS n
ON CONFLICT DO NOTHING;


-- =================================================================
-- 4–6. TRANSACTIONS  — 500 payment requests + transactions + fraud scores
--      Uses a temp table so amount/channel stay consistent across all 3
-- =================================================================

CREATE TEMPORARY TABLE IF NOT EXISTS _seed_tx (
  n              INTEGER PRIMARY KEY,
  user_num       INTEGER  NOT NULL,
  merchant_num   INTEGER  NOT NULL,
  channel        VARCHAR(50)  NOT NULL,
  provider       VARCHAR(100) NOT NULL,
  amount_cents   BIGINT   NOT NULL,
  fee_cents      BIGINT   NOT NULL,
  net_cents      BIGINT   NOT NULL,
  tx_status      VARCHAR(30) NOT NULL,
  risk_score_int INTEGER  NOT NULL,
  risk_level     VARCHAR(20) NOT NULL,
  fraud_decision VARCHAR(20) NOT NULL,
  risk_category  VARCHAR(20) NOT NULL,
  tx_ts          TIMESTAMPTZ NOT NULL
);

-- Generate all 500 rows in a single pass
INSERT INTO _seed_tx
SELECT
  n,
  user_num,
  merchant_num,
  channel,
  provider,
  amount_cents,
  -- Fee: M-Pesa 1%, Airtel 1.5%, Card 2.5%, Bank 0.5%
  CASE channel
    WHEN 'MPESA'         THEN amount_cents / 100
    WHEN 'AIRTEL_MONEY'  THEN amount_cents * 15 / 1000
    WHEN 'CARD'          THEN amount_cents * 25 / 1000
    ELSE                      amount_cents * 5  / 1000
  END AS fee_cents,
  amount_cents - CASE channel
    WHEN 'MPESA'         THEN amount_cents / 100
    WHEN 'AIRTEL_MONEY'  THEN amount_cents * 15 / 1000
    WHEN 'CARD'          THEN amount_cents * 25 / 1000
    ELSE                      amount_cents * 5  / 1000
  END AS net_cents,
  tx_status,
  risk_score_int,
  CASE
    WHEN risk_score_int <= 25 THEN 'LOW'
    WHEN risk_score_int <= 50 THEN 'MEDIUM'
    WHEN risk_score_int <= 75 THEN 'HIGH'
    ELSE 'CRITICAL'
  END AS risk_level,
  CASE
    WHEN risk_score_int <= 25 THEN 'APPROVE'
    WHEN risk_score_int <= 75 THEN 'REVIEW'
    ELSE 'BLOCK'
  END AS fraud_decision,
  risk_category,
  tx_ts
FROM (
  SELECT
    n,
    -- User assignment: 80% normal (1-40), 15% suspicious (41-47), 5% fraudulent (48-50)
    CASE
      WHEN n <= 400 THEN 1  + ((n - 1)   % 40)
      WHEN n <= 475 THEN 41 + ((n - 401)  % 7)
      ELSE               48 + ((n - 476)  % 3)
    END AS user_num,
    -- Cycle through 20 merchants
    1 + ((n - 1) % 20) AS merchant_num,
    -- Channel: 50% M-Pesa, 20% Airtel, 20% Card, 10% Bank
    CASE
      WHEN n % 10 < 5 THEN 'MPESA'
      WHEN n % 10 < 7 THEN 'AIRTEL_MONEY'
      WHEN n % 10 < 9 THEN 'CARD'
      ELSE                  'BANK_TRANSFER'
    END AS channel,
    CASE
      WHEN n % 10 < 5 THEN 'Safaricom'
      WHEN n % 10 < 7 THEN 'Airtel Kenya'
      WHEN n % 10 < 9 THEN 'Stripe'
      ELSE                  'Pesalink'
    END AS provider,
    -- Amount in cents (KES)
    CASE
      -- Normal users: realistic Kenyan amounts
      WHEN n <= 400 THEN
        CASE
          WHEN n % 10 < 5 THEN  5000   + ((n * 997)  % 495001)    -- M-Pesa  KES 50–5K
          WHEN n % 10 < 7 THEN  10000  + ((n * 1009) % 290001)    -- Airtel  KES 100–3K
          WHEN n % 10 < 9 THEN  50000  + ((n * 1013) % 4950001)   -- Card    KES 500–50K
          ELSE                  1000000 + ((n * 1019) % 49000001)  -- Bank    KES 10K–500K
        END
      -- Suspicious: structuring near KES 1M threshold
      WHEN n <= 475 THEN
        CASE
          WHEN n % 4 = 0 THEN  80000000 + ((n * 137) % 19000001)  -- KES 800K–990K
          WHEN n % 4 = 1 THEN  500000   + ((n * 251) % 9500001)   -- KES 5K–100K
          WHEN n % 4 = 2 THEN  2000000  + ((n * 373) % 48000001)  -- KES 20K–500K
          ELSE                 90000000  + ((n * 509) % 9000001)   -- KES 900K–990K
        END
      -- Fraudulent: high-value rapid drains
      ELSE 5000000 + ((n * 743) % 95000001)                       -- KES 50K–1M
    END::bigint AS amount_cents,
    -- Transaction status
    CASE
      WHEN n % 20 = 0           THEN 'failed'
      WHEN n % 15 = 0           THEN 'expired'
      WHEN n > 475 AND n % 3 = 0 THEN 'failed'     -- some fraud attempts fail
      WHEN n % 50 = 0           THEN 'pending'
      ELSE                           'completed'
    END AS tx_status,
    -- Risk score (0–100 int for fraud_schema)
    CASE
      WHEN n <= 400 THEN  5  + ((n * 13) % 26)   -- 5–30  (LOW)
      WHEN n <= 475 THEN  30 + ((n * 17) % 41)   -- 30–70 (MEDIUM–HIGH)
      ELSE                60 + ((n * 19) % 41)    -- 60–100 (HIGH–CRITICAL)
    END AS risk_score_int,
    -- Risk category label
    CASE
      WHEN n <= 400 THEN 'normal'
      WHEN n <= 475 THEN 'suspicious'
      ELSE               'fraudulent'
    END AS risk_category,
    -- Timestamps: spread across 90 days
    CASE
      WHEN n <= 400 THEN
        -- Normal: business hours (8am–8pm EAT = 5am–5pm UTC)
        (NOW() - INTERVAL '1 day' * (90.0 * n / 400))
        + INTERVAL '1 hour' * (5 + (n % 12))
      WHEN n <= 475 THEN
        -- Suspicious: some late-night (1–5am EAT = 22–02 UTC)
        (NOW() - INTERVAL '1 day' * (60.0 * (n - 400) / 75))
        + INTERVAL '1 hour' * CASE
            WHEN n % 3 = 0 THEN (22 + (n % 4)) % 24
            ELSE 5 + (n % 12)
          END
      ELSE
        -- Fraudulent: last 14 days, clustered bursts
        NOW() - INTERVAL '1 hour' * ((500 - n) * 12 + (n % 5))
    END AS tx_ts
  FROM generate_series(1, 500) AS n
) base
ON CONFLICT DO NOTHING;


-- -----------------------------------------------------------------
-- 4. PAYMENT REQUESTS  (payment_schema)
-- -----------------------------------------------------------------

INSERT INTO payment_schema.payment_requests
  (id, merchant_id, merchant_reference, idempotency_key, amount, currency,
   customer_phone, customer_name, description, preferred_channel, status, created_at)
SELECT
  ('40000001-5eed-4000-a000-' || lpad(t.n::text, 12, '0'))::uuid,
  ('20000001-5eed-4000-a000-' || lpad(t.merchant_num::text, 12, '0'))::uuid,
  'SEED-MREF-' || lpad(t.n::text, 6, '0'),
  'seed-idem-' || lpad(t.n::text, 6, '0'),
  t.amount_cents,
  'KES',
  '+2547' || lpad((10000000 + (t.user_num * 1234567) % 89999999)::text, 8, '0'),
  'Customer ' || t.n,
  'Seed payment #' || t.n,
  t.channel,
  CASE t.tx_status
    WHEN 'expired' THEN 'expired'
    ELSE t.tx_status
  END,
  t.tx_ts
FROM _seed_tx t
ON CONFLICT DO NOTHING;


-- -----------------------------------------------------------------
-- 5. PAYMENT TRANSACTIONS  (payment_schema)
-- -----------------------------------------------------------------

INSERT INTO payment_schema.payment_transactions
  (id, payment_request_id, merchant_id, transaction_reference, provider_reference,
   channel, provider, amount, fee, net_amount, currency, status,
   customer_phone, customer_name, description, metadata,
   initiated_at, completed_at, failed_at)
SELECT
  ('50000001-5eed-4000-a000-' || lpad(t.n::text, 12, '0'))::uuid,
  ('40000001-5eed-4000-a000-' || lpad(t.n::text, 12, '0'))::uuid,
  ('20000001-5eed-4000-a000-' || lpad(t.merchant_num::text, 12, '0'))::uuid,
  'SEED-TXN-' || lpad(t.n::text, 6, '0'),
  CASE WHEN t.tx_status IN ('completed','failed') THEN 'PROV-' || lpad(t.n::text, 8, '0') END,
  t.channel,
  t.provider,
  t.amount_cents,
  t.fee_cents,
  t.net_cents,
  'KES',
  -- payment_transactions doesn't have 'expired' — map to 'failed'
  CASE t.tx_status WHEN 'expired' THEN 'failed' ELSE t.tx_status END,
  '+2547' || lpad((10000000 + (t.user_num * 1234567) % 89999999)::text, 8, '0'),
  'Customer ' || t.n,
  'Seed payment #' || t.n,
  -- Metadata: device + IP info for ML scoring
  jsonb_build_object(
    'device_fingerprint', 'dev-seed-' || lpad(
      CASE
        WHEN t.user_num <= 40 THEN 1 + ((t.user_num - 1) % 20)  -- 20 normal devices
        WHEN t.user_num <= 47 THEN 21 + ((t.user_num - 41) % 7) -- 7 suspicious devices
        ELSE 28 + ((t.user_num - 48) % 3)                        -- 3 shared fraud devices
      END::text, 2, '0'),
    'ip_address', '41.89.' || (100 + t.user_num % 50) || '.' || (1 + t.n % 254),
    'user_agent', 'PesaFlow/1.2 ' || CASE WHEN t.n % 3 = 0 THEN 'iOS' WHEN t.n % 3 = 1 THEN 'Android' ELSE 'Web' END,
    'geo_location', jsonb_build_object(
      'country', 'KE',
      'lat', -1.2864 + (t.n % 100)::float / 1000,
      'lng', 36.8172 + (t.n % 100)::float / 1000
    )
  ),
  t.tx_ts,
  CASE WHEN t.tx_status = 'completed' THEN t.tx_ts + INTERVAL '1 second' * (2 + t.n % 30) END,
  CASE WHEN t.tx_status IN ('failed','expired') THEN t.tx_ts + INTERVAL '1 second' * (1 + t.n % 10) END
FROM _seed_tx t
ON CONFLICT DO NOTHING;


-- -----------------------------------------------------------------
-- 6. FRAUD SCORES  (fraud_schema)
-- -----------------------------------------------------------------

INSERT INTO fraud_schema.fraud_scores
  (id, payment_transaction_id, risk_score, risk_level, decision,
   model_version, feature_vector, created_at)
SELECT
  ('60000001-5eed-4000-a000-' || lpad(t.n::text, 12, '0'))::uuid,
  ('50000001-5eed-4000-a000-' || lpad(t.n::text, 12, '0'))::uuid,
  t.risk_score_int,
  t.risk_level,
  t.fraud_decision,
  'heuristic-v1.0',
  jsonb_build_object(
    'amount',          t.amount_cents::float / 100,
    'channel',         t.channel,
    'risk_category',   t.risk_category,
    'user_num',        t.user_num,
    'merchant_num',    t.merchant_num,
    'seed_id',         t.n
  ),
  t.tx_ts + INTERVAL '100 milliseconds'
FROM _seed_tx t
ON CONFLICT DO NOTHING;

DROP TABLE IF EXISTS _seed_tx;


-- =================================================================
-- 7. FEATURE STORE — USERS  (ai_schema) — 50 user profiles
-- =================================================================

SET search_path TO ai_schema;

INSERT INTO feature_store_user
  (user_id, avg_transaction_amount_7d, transaction_velocity_1h,
   transaction_velocity_24h, failed_login_attempts_24h,
   account_age_days, historical_fraud_flag, last_updated)
SELECT
  ('30000001-5eed-4000-a000-' || lpad(n::text, 12, '0'))::uuid,
  CASE
    WHEN n <= 40 THEN (500  + n * 100)::numeric(18,2)               -- KES 600–4,500
    WHEN n <= 47 THEN (5000 + (n - 40) * 5000)::numeric(18,2)       -- KES 10K–40K
    ELSE              (50000 + (n - 47) * 20000)::numeric(18,2)      -- KES 70K–110K
  END,
  CASE
    WHEN n <= 40 THEN n % 3                -- 0–2 per hour
    WHEN n <= 47 THEN 5 + (n - 40) % 10   -- 5–11
    ELSE              12 + (n - 47) * 5    -- 17–27
  END,
  CASE
    WHEN n <= 40 THEN 1 + n % 8            -- 1–8 per day
    WHEN n <= 47 THEN 15 + (n - 40) * 4   -- 19–43
    ELSE              30 + (n - 47) * 10   -- 40–60
  END,
  CASE
    WHEN n <= 40 THEN 0
    WHEN n <= 47 THEN 1 + (n - 41) % 3    -- 1–3
    ELSE              5 + (n - 47) * 2     -- 7–11
  END,
  CASE
    WHEN n <= 40 THEN 90  + n * 16         -- 106–730 days
    WHEN n <= 47 THEN 14  + (n - 40) * 7   -- 21–63 days
    ELSE              1   + (n - 47) * 3    -- 4–10 days
  END,
  CASE WHEN n > 47 THEN true ELSE false END,
  NOW()
FROM generate_series(1, 50) AS n
ON CONFLICT (user_id) DO NOTHING;


-- =================================================================
-- 8. FEATURE STORE — DEVICES  (ai_schema) — 30 device profiles
-- =================================================================

INSERT INTO feature_store_device
  (device_fingerprint, device_risk_score, device_fraud_count,
   distinct_user_count, last_seen)
SELECT
  'dev-seed-' || lpad(n::text, 2, '0'),
  CASE
    WHEN n <= 20 THEN (0.01 + n * 0.007)::numeric(5,4)    -- 0.017–0.15  (clean)
    WHEN n <= 27 THEN (0.20 + (n - 20) * 0.04)::numeric(5,4) -- 0.24–0.48  (suspicious)
    ELSE              (0.60 + (n - 27) * 0.12)::numeric(5,4)  -- 0.72–0.96  (fraud)
  END,
  CASE
    WHEN n <= 20 THEN 0
    WHEN n <= 27 THEN (n - 20) % 2     -- 0 or 1
    ELSE              2 + (n - 27)      -- 3–5
  END,
  CASE
    WHEN n <= 20 THEN 1                 -- one user per device
    WHEN n <= 27 THEN 2 + (n - 21) % 2 -- 2–3 users
    ELSE              3 + (n - 27) * 2  -- 5–9 (device sharing!)
  END,
  NOW() - INTERVAL '1 hour' * (n * 3)
FROM generate_series(1, 30) AS n
ON CONFLICT (device_fingerprint) DO NOTHING;


-- =================================================================
-- 9. AML USER FEATURES  (ai_schema) — 50 profiles
-- =================================================================

INSERT INTO aml_user_features
  (user_id, avg_transaction_amount, std_transaction_amount,
   transaction_count_30d, total_volume_30d, device_count_30d,
   ip_count_30d, high_risk_country_exposure,
   historical_structuring_flag, last_updated)
SELECT
  ('30000001-5eed-4000-a000-' || lpad(n::text, 12, '0'))::uuid,
  -- avg amount
  CASE
    WHEN n <= 40 THEN (1000  + n * 500)::numeric(18,2)         -- KES 1,500–21,000
    WHEN n <= 47 THEN (50000 + (n-40) * 20000)::numeric(18,2)  -- KES 70K–190K
    ELSE              (200000 + (n-47) * 50000)::numeric(18,2)  -- KES 250K–350K
  END,
  -- std amount (30–50% of avg)
  CASE
    WHEN n <= 40 THEN (400  + n * 200)::numeric(18,2)
    WHEN n <= 47 THEN (20000 + (n-40) * 8000)::numeric(18,2)
    ELSE              (80000 + (n-47) * 20000)::numeric(18,2)
  END,
  -- tx count 30d
  CASE
    WHEN n <= 40 THEN 5  + n % 25          -- 5–29
    WHEN n <= 47 THEN 30 + (n - 40) * 10   -- 40–100
    ELSE              80 + (n - 47) * 40    -- 120–200
  END,
  -- total volume 30d (count * avg)
  CASE
    WHEN n <= 40 THEN ((5 + n % 25) * (1000 + n * 500))::numeric(18,2)
    WHEN n <= 47 THEN ((30 + (n-40)*10) * (50000 + (n-40)*20000))::numeric(18,2)
    ELSE              ((80 + (n-47)*40) * (200000 + (n-47)*50000))::numeric(18,2)
  END,
  -- device count
  CASE
    WHEN n <= 40 THEN 1
    WHEN n <= 47 THEN 2 + (n - 41) % 3   -- 2–4
    ELSE              5 + (n - 48) * 2     -- 5–9
  END,
  -- ip count
  CASE
    WHEN n <= 40 THEN 1 + n % 3           -- 1–3
    WHEN n <= 47 THEN 5 + (n - 40) * 2    -- 7–19
    ELSE              15 + (n - 47) * 5    -- 20–30
  END,
  -- high-risk country exposure
  CASE
    WHEN n <= 40 THEN 0
    WHEN n <= 47 THEN (n - 41) % 3        -- 0–2
    ELSE              2 + (n - 48)         -- 2–4
  END,
  -- structuring flag
  CASE WHEN n > 47 THEN true ELSE false END,
  NOW()
FROM generate_series(1, 50) AS n
ON CONFLICT (user_id) DO NOTHING;


-- =================================================================
-- 10. AML USER RISK PROFILES  (ai_schema) — 50 profiles
-- =================================================================

INSERT INTO aml_user_risk_profile
  (user_id, cumulative_risk_score, risk_category, risk_trend,
   last_transaction_at, velocity_1h, velocity_24h,
   total_volume_24h, network_risk_score, sanctions_flag, updated_at)
SELECT
  ('30000001-5eed-4000-a000-' || lpad(n::text, 12, '0'))::uuid,
  CASE
    WHEN n <= 40 THEN (0.05 + n * 0.005)::numeric(5,4)     -- 0.055–0.25
    WHEN n <= 47 THEN (0.40 + (n-40) * 0.06)::numeric(5,4) -- 0.46–0.82
    ELSE              (0.75 + (n-47) * 0.07)::numeric(5,4)  -- 0.82–0.96
  END,
  CASE
    WHEN n <= 40 THEN 'LOW'
    WHEN n <= 47 THEN CASE WHEN n <= 44 THEN 'MEDIUM' ELSE 'HIGH' END
    ELSE              'CRITICAL'
  END,
  CASE
    WHEN n <= 40 THEN 'STABLE'
    WHEN n <= 47 THEN 'INCREASING'
    ELSE              'INCREASING'
  END,
  NOW() - INTERVAL '1 hour' * (n % 48),
  -- velocity 1h
  CASE
    WHEN n <= 40 THEN n % 3
    WHEN n <= 47 THEN 5 + (n - 40)
    ELSE              15 + (n - 47) * 3
  END,
  -- velocity 24h
  CASE
    WHEN n <= 40 THEN 1 + n % 8
    WHEN n <= 47 THEN 15 + (n - 40) * 5
    ELSE              40 + (n - 47) * 8
  END,
  -- total volume 24h
  CASE
    WHEN n <= 40 THEN (5000  + n * 1000)::numeric(18,2)
    WHEN n <= 47 THEN (100000 + (n-40) * 50000)::numeric(18,2)
    ELSE              (500000 + (n-47) * 200000)::numeric(18,2)
  END,
  -- network risk score
  CASE
    WHEN n <= 40 THEN (0.02 + n * 0.003)::numeric(5,4)
    WHEN n <= 47 THEN (0.30 + (n-40) * 0.07)::numeric(5,4)
    ELSE              (0.70 + (n-47) * 0.08)::numeric(5,4)
  END,
  -- sanctions flag: only high-risk fraudulent
  CASE WHEN n = 50 THEN true ELSE false END,
  NOW()
FROM generate_series(1, 50) AS n
ON CONFLICT (user_id) DO NOTHING;


-- =================================================================
-- 11. MERCHANT RISK PROFILES  (ai_schema) — 20 merchant profiles
--     MCC codes match merchant business types
-- =================================================================

INSERT INTO merchant_risk_profiles
  (merchant_id, merchant_name, mcc_code,
   risk_score, risk_level, merchant_tier,
   avg_transaction_amount_30d, std_transaction_amount_30d,
   chargeback_rate_90d, refund_rate_90d, fraud_transaction_rate,
   total_transactions_30d, total_volume_30d, unique_customers_30d,
   created_at, updated_at)
VALUES
  -- LOW risk merchants (12)
  ('20000001-5eed-4000-a000-000000000001'::uuid, 'Nairobi Fresh Mart',      '5411',
   0.0800, 'LOW', 'STANDARD', 1200.00, 800.00,
   0.0015, 0.0080, 0.0005, 4500, 5400000.00, 1800, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000002'::uuid, 'Karen Restaurant & Grill', '5812',
   0.1200, 'LOW', 'STANDARD', 2500.00, 1500.00,
   0.0020, 0.0100, 0.0008, 3200, 8000000.00, 1200, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000003'::uuid, 'Westlands Electronics',    '5732',
   0.1500, 'LOW', 'STANDARD', 15000.00, 12000.00,
   0.0030, 0.0150, 0.0010, 800, 12000000.00, 500, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000004'::uuid, 'Mombasa Beach Hotel',      '7011',
   0.1100, 'LOW', 'STANDARD', 8000.00, 5000.00,
   0.0010, 0.0050, 0.0003, 600, 4800000.00, 400, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000005'::uuid, 'Safaricom Agent Kibera',   '4814',
   0.0500, 'LOW', 'STANDARD', 500.00, 300.00,
   0.0005, 0.0020, 0.0002, 12000, 6000000.00, 5000, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000007'::uuid, 'Kilimall Online Store',    '5999',
   0.1800, 'LOW', 'STANDARD', 3500.00, 4000.00,
   0.0035, 0.0200, 0.0012, 2500, 8750000.00, 1500, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000008'::uuid, 'CBD Pharmacy Plus',        '5912',
   0.0600, 'LOW', 'STANDARD', 800.00, 400.00,
   0.0008, 0.0030, 0.0001, 3000, 2400000.00, 2000, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000010'::uuid, 'Tusker Auto Dealers',      '5511',
   0.1400, 'LOW', 'STANDARD', 450000.00, 200000.00,
   0.0020, 0.0100, 0.0005, 50, 22500000.00, 45, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000011'::uuid, 'Karen Supermarket',        '5411',
   0.0900, 'LOW', 'STANDARD', 1800.00, 1000.00,
   0.0012, 0.0060, 0.0004, 5500, 9900000.00, 2200, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000012'::uuid, 'Lavington Boutique',       '5699',
   0.1300, 'LOW', 'STANDARD', 6000.00, 3500.00,
   0.0025, 0.0120, 0.0006, 400, 2400000.00, 300, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000017'::uuid, 'Ngong Road Petrol Station','5541',
   0.0700, 'LOW', 'STANDARD', 3000.00, 1500.00,
   0.0008, 0.0010, 0.0002, 6000, 18000000.00, 3000, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000019'::uuid, 'Thika Hardware Supplies',  '5251',
   0.1000, 'LOW', 'STANDARD', 5000.00, 3000.00,
   0.0018, 0.0080, 0.0005, 700, 3500000.00, 400, NOW(), NOW()),

  -- MEDIUM risk merchants (5)
  ('20000001-5eed-4000-a000-000000000006'::uuid, 'KCB Bank Agency',          '6012',
   0.3200, 'MEDIUM', 'ENHANCED', 50000.00, 40000.00,
   0.0060, 0.0100, 0.0020, 1500, 75000000.00, 800, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000009'::uuid, 'Nairobi Jewelry Palace',   '5944',
   0.3800, 'MEDIUM', 'ENHANCED', 85000.00, 60000.00,
   0.0080, 0.0300, 0.0025, 200, 17000000.00, 150, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000016'::uuid, 'Gikomba Wholesale',        '5131',
   0.2800, 'MEDIUM', 'ENHANCED', 25000.00, 20000.00,
   0.0050, 0.0180, 0.0015, 1800, 45000000.00, 600, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000018'::uuid, 'Eastleigh Electronics Hub','5732',
   0.4200, 'MEDIUM', 'ENHANCED', 20000.00, 18000.00,
   0.0070, 0.0250, 0.0030, 900, 18000000.00, 400, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000020'::uuid, 'Virtual Forex Trading',    '6211',
   0.4800, 'MEDIUM', 'ENHANCED', 150000.00, 120000.00,
   0.0090, 0.0200, 0.0035, 350, 52500000.00, 200, NOW(), NOW()),

  -- HIGH risk merchants (3)
  ('20000001-5eed-4000-a000-000000000013'::uuid, 'Quick Bet Gaming',         '7995',
   0.7200, 'HIGH', 'RESTRICTED', 5000.00, 8000.00,
   0.0350, 0.0800, 0.0200, 8000, 40000000.00, 2000, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000014'::uuid, 'Lucky Star Casino',        '7995',
   0.8100, 'HIGH', 'RESTRICTED', 12000.00, 15000.00,
   0.0500, 0.1200, 0.0400, 5000, 60000000.00, 1500, NOW(), NOW()),

  ('20000001-5eed-4000-a000-000000000015'::uuid, 'Rapid Money Transfer',     '6051',
   0.6800, 'HIGH', 'RESTRICTED', 80000.00, 70000.00,
   0.0250, 0.0100, 0.0150, 2000, 160000000.00, 800, NOW(), NOW())
ON CONFLICT (merchant_id) DO NOTHING;


-- =================================================================
-- 12. AML NETWORK EDGES  (ai_schema) — suspicious/fraudulent links
-- =================================================================

INSERT INTO aml_network_edges (id, source_user_id, target_user_id, relationship_type, weight)
VALUES
  -- Fraudulent ring: users 48-49-50 connected
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000048'::uuid,
   '30000001-5eed-4000-a000-000000000049'::uuid,
   'SHARED_DEVICE', 0.8500),
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000049'::uuid,
   '30000001-5eed-4000-a000-000000000050'::uuid,
   'RAPID_TRANSFER', 0.9200),
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000048'::uuid,
   '30000001-5eed-4000-a000-000000000050'::uuid,
   'SHARED_IP', 0.7800),
  -- Suspicious cluster: users 41-44 with transfers
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000041'::uuid,
   '30000001-5eed-4000-a000-000000000042'::uuid,
   'FREQUENT_TRANSFER', 0.5500),
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000042'::uuid,
   '30000001-5eed-4000-a000-000000000043'::uuid,
   'FREQUENT_TRANSFER', 0.4200),
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000043'::uuid,
   '30000001-5eed-4000-a000-000000000044'::uuid,
   'SHARED_MERCHANT', 0.3500),
  -- Cross-link: suspicious to fraudulent
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000045'::uuid,
   '30000001-5eed-4000-a000-000000000048'::uuid,
   'DIRECT_TRANSFER', 0.6800),
  -- Normal user connected to suspicious (false positive test)
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000010'::uuid,
   '30000001-5eed-4000-a000-000000000041'::uuid,
   'SHARED_MERCHANT', 0.1500),
  -- Circular transfer pattern (AML red flag)
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000046'::uuid,
   '30000001-5eed-4000-a000-000000000047'::uuid,
   'CIRCULAR_TRANSFER', 0.8800),
  (gen_random_uuid(),
   '30000001-5eed-4000-a000-000000000047'::uuid,
   '30000001-5eed-4000-a000-000000000046'::uuid,
   'CIRCULAR_TRANSFER', 0.8800)
ON CONFLICT DO NOTHING;


-- =================================================================
-- 13. MERCHANT GRAPH EDGES  (ai_schema) — merchant relationships
-- =================================================================

INSERT INTO merchant_graph_edges
  (source_merchant_id, target_merchant_id, edge_type, weight,
   shared_entity_count, first_seen_at, last_seen_at)
VALUES
  -- Gambling merchants share customers
  ('20000001-5eed-4000-a000-000000000013'::uuid,
   '20000001-5eed-4000-a000-000000000014'::uuid,
   'SHARED_CUSTOMER', 0.7200, 85,
   NOW() - INTERVAL '90 days', NOW()),
  -- Money transfer linked to gambling
  ('20000001-5eed-4000-a000-000000000015'::uuid,
   '20000001-5eed-4000-a000-000000000013'::uuid,
   'SHARED_CUSTOMER', 0.5500, 35,
   NOW() - INTERVAL '60 days', NOW()),
  -- Normal retail cluster
  ('20000001-5eed-4000-a000-000000000001'::uuid,
   '20000001-5eed-4000-a000-000000000011'::uuid,
   'SHARED_CUSTOMER', 0.3000, 120,
   NOW() - INTERVAL '200 days', NOW()),
  -- Suspicious electronics cluster
  ('20000001-5eed-4000-a000-000000000003'::uuid,
   '20000001-5eed-4000-a000-000000000018'::uuid,
   'SHARED_CUSTOMER', 0.4500, 40,
   NOW() - INTERVAL '45 days', NOW())
ON CONFLICT DO NOTHING;


-- =================================================================
-- 14. AML SANCTIONS ENTITIES  (ai_schema) — test screening data
-- =================================================================

INSERT INTO aml_sanctions_entities
  (id, name, aliases, country, identifier, source, last_updated)
VALUES
  (gen_random_uuid(), 'Test Sanctioned Entity',
   ARRAY['TSE', 'Test Entity LLC', 'T.S.E. Holdings'],
   'IR', 'OFAC-SEED-001', 'OFAC', NOW() - INTERVAL '30 days'),
  (gen_random_uuid(), 'Suspicious Corp International',
   ARRAY['SCI', 'SusCorp', 'Suspicious Corp Intl'],
   'KP', 'UN-SEED-002', 'UN', NOW() - INTERVAL '30 days'),
  (gen_random_uuid(), 'Al-Noor Trading Company',
   ARRAY['Al-Noor', 'Alnoor Trading', 'ANTC'],
   'SY', 'OFAC-SEED-003', 'OFAC', NOW() - INTERVAL '60 days'),
  (gen_random_uuid(), 'Bright Star Holdings',
   ARRAY['BSH', 'Bright Star', 'BS Holdings Ltd'],
   'AF', 'UN-SEED-004', 'UN', NOW() - INTERVAL '45 days'),
  (gen_random_uuid(), 'Golden Path Remittance',
   ARRAY['GPR', 'Golden Path', 'GP Remittance Co'],
   'YE', 'OFAC-SEED-005', 'OFAC', NOW() - INTERVAL '15 days')
ON CONFLICT DO NOTHING;


-- =================================================================
-- 15. MERCHANT GRAPH METRICS  (ai_schema) — precomputed graph scores
-- =================================================================

INSERT INTO merchant_graph_metrics
  (id, merchant_id, pagerank_score, community_id, degree_centrality,
   betweenness_centrality, cluster_risk_score, computed_at)
VALUES
  (gen_random_uuid(), '20000001-5eed-4000-a000-000000000013'::uuid,
   0.08500000, 1, 0.150000, 0.120000, 0.7500, NOW()),
  (gen_random_uuid(), '20000001-5eed-4000-a000-000000000014'::uuid,
   0.07200000, 1, 0.120000, 0.080000, 0.7500, NOW()),
  (gen_random_uuid(), '20000001-5eed-4000-a000-000000000015'::uuid,
   0.06800000, 1, 0.100000, 0.090000, 0.6800, NOW()),
  (gen_random_uuid(), '20000001-5eed-4000-a000-000000000001'::uuid,
   0.04500000, 2, 0.080000, 0.020000, 0.1200, NOW()),
  (gen_random_uuid(), '20000001-5eed-4000-a000-000000000011'::uuid,
   0.03800000, 2, 0.060000, 0.015000, 0.1200, NOW())
ON CONFLICT DO NOTHING;


COMMIT;

-- =================================================================
-- VERIFICATION QUERIES  (run after seed)
-- =================================================================

SELECT '--- SEED VERIFICATION ---' AS section;

SELECT 'merchant_schema.organizations' AS entity,
       count(*) AS total
FROM merchant_schema.organizations
WHERE id::text LIKE '%5eed%'
UNION ALL
SELECT 'merchant_schema.merchants',
       count(*)
FROM merchant_schema.merchants
WHERE id::text LIKE '%5eed%'
UNION ALL
SELECT 'auth_schema.users',
       count(*)
FROM auth_schema.users
WHERE id::text LIKE '%5eed%'
UNION ALL
SELECT 'payment_schema.payment_requests',
       count(*)
FROM payment_schema.payment_requests
WHERE id::text LIKE '%5eed%'
UNION ALL
SELECT 'payment_schema.payment_transactions',
       count(*)
FROM payment_schema.payment_transactions
WHERE id::text LIKE '%5eed%'
UNION ALL
SELECT 'fraud_schema.fraud_scores',
       count(*)
FROM fraud_schema.fraud_scores
WHERE id::text LIKE '%5eed%'
UNION ALL
SELECT 'ai_schema.feature_store_user',
       count(*)
FROM ai_schema.feature_store_user
WHERE user_id::text LIKE '%5eed%'
UNION ALL
SELECT 'ai_schema.feature_store_device',
       count(*)
FROM ai_schema.feature_store_device
WHERE device_fingerprint LIKE 'dev-seed-%'
UNION ALL
SELECT 'ai_schema.aml_user_features',
       count(*)
FROM ai_schema.aml_user_features
WHERE user_id::text LIKE '%5eed%'
UNION ALL
SELECT 'ai_schema.aml_user_risk_profile',
       count(*)
FROM ai_schema.aml_user_risk_profile
WHERE user_id::text LIKE '%5eed%'
UNION ALL
SELECT 'ai_schema.merchant_risk_profiles',
       count(*)
FROM ai_schema.merchant_risk_profiles
WHERE merchant_id::text LIKE '%5eed%';

SELECT '--- AMOUNT DISTRIBUTION (KES) ---' AS section;

SELECT
  CASE
    WHEN amount < 500000     THEN 'Under 5K'
    WHEN amount < 5000000    THEN '5K–50K'
    WHEN amount < 50000000   THEN '50K–500K'
    ELSE                          'Over 500K'
  END AS range_kes,
  count(*) AS txn_count,
  round(avg(amount) / 100.0, 2) AS avg_kes,
  round(min(amount) / 100.0, 2) AS min_kes,
  round(max(amount) / 100.0, 2) AS max_kes
FROM payment_schema.payment_transactions
WHERE id::text LIKE '%5eed%'
GROUP BY 1
ORDER BY min_kes;

SELECT '--- RISK DISTRIBUTION ---' AS section;

SELECT risk_level, count(*) AS total
FROM fraud_schema.fraud_scores
WHERE id::text LIKE '%5eed%'
GROUP BY risk_level
ORDER BY
  CASE risk_level
    WHEN 'LOW'      THEN 1
    WHEN 'MEDIUM'   THEN 2
    WHEN 'HIGH'     THEN 3
    WHEN 'CRITICAL' THEN 4
  END;

SELECT '--- CHANNEL MIX ---' AS section;

SELECT channel, count(*) AS total,
       round(count(*)::numeric / 500 * 100, 1) AS pct
FROM payment_schema.payment_transactions
WHERE id::text LIKE '%5eed%'
GROUP BY channel
ORDER BY total DESC;

SELECT '--- SEED COMPLETE ---' AS result;
