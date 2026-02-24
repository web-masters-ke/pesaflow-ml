#!/usr/bin/env bash
# =============================================================================
# PesaFlow ML — Full-Stack Integration Test
# =============================================================================
# Starts backend infra (Postgres, Redis, Kafka), runs migrations,
# brings up the ML stack, seeds test data, and runs the full test suite.
#
# Usage: ./tests/run_full_test.sh
# Teardown: ./tests/run_full_test.sh --down
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="/Users/app/Documents/WEBMASTERS/pesa-flow-backend"
ML_PORT=18743
BASE_URL="http://localhost:${ML_PORT}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { printf "${CYAN}[%s]${NC} %s\n" "$(date +%H:%M:%S)" "$1"; }
ok()   { printf "${GREEN}[OK]${NC} %s\n" "$1"; }
fail() { printf "${RED}[FAIL]${NC} %s\n" "$1"; }
warn() { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }

# ------------------------------------------------------------------
# Teardown
# ------------------------------------------------------------------
if [ "${1:-}" = "--down" ]; then
  log "Tearing down ML stack..."
  cd "$ML_DIR" && docker compose down -v --remove-orphans 2>/dev/null || true
  log "Tearing down backend infra..."
  cd "$BACKEND_DIR" && docker compose down -v --remove-orphans 2>/dev/null || true
  # Clean up manually-created postgres container (port conflict workaround)
  docker rm -f pesaflow-postgres 2>/dev/null || true
  ok "All services stopped and volumes removed"
  exit 0
fi

# ------------------------------------------------------------------
# Step 1: Start backend infrastructure (Postgres, Redis, Kafka only)
# ------------------------------------------------------------------
log "Step 1/6 — Starting backend infrastructure..."
cd "$BACKEND_DIR"

# Start infra services. If Postgres port 5432 conflicts with local install,
# we still start it without host port binding (Docker networking is sufficient).
docker compose up -d postgres redis zookeeper kafka 2>&1 | tail -5 || true

if ! docker ps --format '{{.Names}}' | grep -q pesaflow-postgres; then
  warn "Postgres failed to start — likely host port 5432 conflict. Restarting without host port..."
  docker rm -f pesaflow-postgres 2>/dev/null || true
  # Re-create without host port mapping (ML connects via Docker network, not localhost)
  docker run -d --name pesaflow-postgres \
    --network pesa-flow-backend_pesaflow-network \
    -e POSTGRES_USER=pesaflow \
    -e POSTGRES_PASSWORD=pesaflow_secure_2024 \
    -e POSTGRES_DB=pesaflow \
    --health-cmd "pg_isready -U pesaflow" \
    --health-interval 5s \
    --health-timeout 3s \
    --health-retries 10 \
    postgres:16-alpine 2>&1 | tail -3
fi

log "Waiting for backend Postgres..."
PGRETRIES=0
until docker exec pesaflow-postgres pg_isready -U pesaflow -q 2>/dev/null; do
  PGRETRIES=$((PGRETRIES + 1))
  if [ "$PGRETRIES" -ge 30 ]; then
    fail "Postgres not ready after 60s"
    docker logs pesaflow-postgres 2>&1 | tail -10
    exit 1
  fi
  sleep 2
done
ok "Backend Postgres is ready"

log "Waiting for backend Redis..."
until docker exec pesaflow-redis redis-cli ping 2>/dev/null | grep -q PONG; do
  sleep 2
done
ok "Backend Redis is ready"

log "Waiting for backend Kafka..."
KAFKA_RETRIES=0
until docker exec pesaflow-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; do
  KAFKA_RETRIES=$((KAFKA_RETRIES + 1))
  if [ "$KAFKA_RETRIES" -ge 30 ]; then
    warn "Kafka not ready after 60s — continuing (ML will run without Kafka)"
    break
  fi
  sleep 2
done
if [ "$KAFKA_RETRIES" -lt 30 ]; then
  ok "Backend Kafka is ready"
fi

# ------------------------------------------------------------------
# Step 2: Run ML migrations on backend Postgres
# ------------------------------------------------------------------
log "Step 2/6 — Running ML schema migrations..."

for f in "$ML_DIR"/migrations/*.sql; do
  fname=$(basename "$f")
  log "  Running $fname..."
  docker exec -i pesaflow-postgres psql -U pesaflow -d pesaflow < "$f" 2>&1 | tail -3
done
ok "All migrations applied"

# ------------------------------------------------------------------
# Step 3: Seed test data
# ------------------------------------------------------------------
log "Step 3/6 — Seeding test data..."

docker exec -i pesaflow-postgres psql -U pesaflow -d pesaflow <<'SEED_SQL'
SET search_path TO ai_schema;

-- Seed user features (for feature extraction)
INSERT INTO feature_store_user (user_id, avg_transaction_amount_7d, transaction_velocity_1h, transaction_velocity_24h, failed_login_attempts_24h, account_age_days, historical_fraud_flag)
VALUES
  ('550e8400-e29b-41d4-a716-446655440001', 2500.00, 3, 12, 0, 365, false),
  ('d60e8400-e29b-41d4-a716-446655440010', 150000.00, 8, 45, 2, 30, true),
  ('a10e8400-e29b-41d4-a716-446655440050', 500.00, 1, 5, 0, 730, false)
ON CONFLICT (user_id) DO NOTHING;

-- Seed device features
INSERT INTO feature_store_device (device_fingerprint, device_risk_score, device_fraud_count, distinct_user_count)
VALUES
  ('fp_device_123abc', 0.15, 0, 1),
  ('fp_unknown_device', 0.85, 3, 7),
  ('fp_suspicious', 0.72, 2, 4)
ON CONFLICT (device_fingerprint) DO NOTHING;

-- Seed AML user features
INSERT INTO aml_user_features (user_id, avg_transaction_amount, std_transaction_amount, transaction_count_30d, total_volume_30d, device_count_30d, ip_count_30d, high_risk_country_exposure, historical_structuring_flag)
VALUES
  ('550e8400-e29b-41d4-a716-446655440001', 25000.00, 15000.00, 45, 1125000.00, 2, 3, 0, false),
  ('d60e8400-e29b-41d4-a716-446655440010', 500000.00, 300000.00, 120, 60000000.00, 8, 15, 5, true)
ON CONFLICT (user_id) DO NOTHING;

-- Seed AML user risk profiles
INSERT INTO aml_user_risk_profile (user_id, cumulative_risk_score, risk_category, risk_trend, velocity_1h, velocity_24h, total_volume_24h, network_risk_score, sanctions_flag)
VALUES
  ('550e8400-e29b-41d4-a716-446655440001', 0.35, 'LOW', 'STABLE', 2, 8, 50000.00, 0.1, false),
  ('d60e8400-e29b-41d4-a716-446655440010', 0.82, 'HIGH', 'INCREASING', 12, 55, 8500000.00, 0.75, false)
ON CONFLICT (user_id) DO NOTHING;

-- Seed merchant risk profiles
INSERT INTO merchant_risk_profiles (merchant_id, merchant_name, mcc_code, risk_score, risk_level, merchant_tier, avg_transaction_amount_30d, chargeback_rate_90d, refund_rate_90d, fraud_transaction_rate, total_transactions_30d, total_volume_30d, unique_customers_30d)
VALUES
  ('550e8400-e29b-41d4-a716-446655440005', 'Nairobi Supermart', '5411', 0.22, 'LOW', 'STANDARD', 1500.00, 0.002, 0.01, 0.001, 5000, 7500000.00, 2000),
  ('f70e8400-e29b-41d4-a716-446655440099', 'QuickBet Gaming', '7995', 0.78, 'HIGH', 'RESTRICTED', 25000.00, 0.08, 0.15, 0.05, 800, 20000000.00, 300)
ON CONFLICT (merchant_id) DO NOTHING;

-- Seed AML network edges
INSERT INTO aml_network_edges (id, source_user_id, target_user_id, relationship_type, weight)
VALUES
  (gen_random_uuid(), '550e8400-e29b-41d4-a716-446655440001', 'd60e8400-e29b-41d4-a716-446655440010', 'DIRECT_TRANSFER', 0.6),
  (gen_random_uuid(), 'd60e8400-e29b-41d4-a716-446655440010', 'a10e8400-e29b-41d4-a716-446655440050', 'DIRECT_TRANSFER', 0.9)
ON CONFLICT DO NOTHING;

-- Seed merchant graph edges
INSERT INTO merchant_graph_edges (source_merchant_id, target_merchant_id, edge_type, weight, shared_entity_count)
VALUES
  ('550e8400-e29b-41d4-a716-446655440005', 'f70e8400-e29b-41d4-a716-446655440099', 'SHARED_CUSTOMER', 0.4, 12)
ON CONFLICT DO NOTHING;

-- Seed sanctions entities
INSERT INTO aml_sanctions_entities (id, name, aliases, country, identifier, source, last_updated)
VALUES
  (gen_random_uuid(), 'Test Sanctioned Entity', ARRAY['TSE', 'Test Entity LLC'], 'IR', 'OFAC-12345', 'OFAC', NOW()),
  (gen_random_uuid(), 'Suspicious Corp International', ARRAY['SCI', 'SusCorp'], 'KP', 'UN-67890', 'UN', NOW())
ON CONFLICT DO NOTHING;

SELECT 'Seed data inserted successfully' AS result;
SEED_SQL

ok "Test data seeded"

# ------------------------------------------------------------------
# Step 4: Start ML stack
# ------------------------------------------------------------------
log "Step 4/6 — Building and starting ML service..."
cd "$ML_DIR"

# Start only the ML service + its direct dependency (ml-postgres).
# Kafka, MLflow, and Airflow are not needed for API integration tests.
docker compose up -d --build pesaflow-ml ml-postgres redis 2>&1 | tail -10

log "Waiting for ML API to become healthy..."
ML_RETRIES=0
until curl -sf "$BASE_URL/health/live" >/dev/null 2>&1; do
  ML_RETRIES=$((ML_RETRIES + 1))
  if [ "$ML_RETRIES" -ge 90 ]; then
    fail "ML API did not become healthy after 180s"
    docker compose logs pesaflow-ml 2>&1 | tail -30
    exit 1
  fi
  sleep 2
done
ok "ML API is healthy at $BASE_URL"

# ------------------------------------------------------------------
# Step 5: Quick readiness verification
# ------------------------------------------------------------------
log "Step 5/6 — Verifying service readiness..."

READY_RESPONSE=$(curl -s "$BASE_URL/health/ready")
echo "$READY_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$READY_RESPONSE"

# Check DB connectivity by trying to create and list a case
DB_CHECK=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/v1/aml/cases" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "USER",
    "entity_id": "550e8400-e29b-41d4-a716-446655440001",
    "risk_score": 0.50,
    "trigger_reason": "Integration test — verifying DB connectivity",
    "priority": "LOW"
  }')

if [ "$DB_CHECK" = "201" ]; then
  ok "Database connectivity verified (case created)"
else
  warn "Database returned HTTP $DB_CHECK (case management may not work)"
fi

# ------------------------------------------------------------------
# Step 6: Run full test suite
# ------------------------------------------------------------------
log "Step 6/6 — Running full test suite..."
echo ""

"$SCRIPT_DIR/curl_test.sh" "$BASE_URL"
TEST_EXIT=$?

echo ""
echo "============================================================"
if [ "$TEST_EXIT" -eq 0 ]; then
  ok "ALL TESTS PASSED"
else
  warn "Some tests failed (see above)"
fi
echo "============================================================"
echo ""
log "Services still running. Useful commands:"
echo "  View ML logs:     docker compose -f $ML_DIR/docker-compose.yml logs -f pesaflow-ml"
echo "  View all logs:    docker compose -f $ML_DIR/docker-compose.yml logs -f"
echo "  Teardown:         $SCRIPT_DIR/run_full_test.sh --down"
echo "  ML API:           $BASE_URL"
echo "  ML API docs:      $BASE_URL/docs"
echo "  MLflow UI:        http://localhost:15937"
echo "  Airflow UI:       http://localhost:28080 (admin/admin)"
echo ""

exit $TEST_EXIT
