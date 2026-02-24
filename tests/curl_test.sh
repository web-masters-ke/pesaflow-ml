#!/usr/bin/env bash
# =============================================================================
# PesaFlow ML — Comprehensive API Test Suite (curl)
# Usage: ./tests/curl_test.sh [base_url]
# =============================================================================
set -euo pipefail

BASE="${1:-http://localhost:8000}"
PASS=0
FAIL=0
TOTAL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test runner
run_test() {
  local name="$1"
  local expected_status="$2"
  shift 2
  TOTAL=$((TOTAL + 1))

  printf "${CYAN}[%02d] %-55s${NC}" "$TOTAL" "$name"

  # Capture response + status code
  local response
  local http_code
  response=$(curl -s -w "\n%{http_code}" "$@" 2>&1) || true
  http_code=$(echo "$response" | tail -n1)
  local body
  body=$(echo "$response" | sed '$d')

  if [ "$http_code" = "$expected_status" ]; then
    PASS=$((PASS + 1))
    printf "${GREEN}PASS${NC} (HTTP %s)\n" "$http_code"
  else
    FAIL=$((FAIL + 1))
    printf "${RED}FAIL${NC} (expected %s, got %s)\n" "$expected_status" "$http_code"
    echo "  Response: $(echo "$body" | head -c 200)"
  fi
}

echo ""
echo "============================================================"
echo "  PesaFlow ML API Test Suite"
echo "  Target: $BASE"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# 1. HEALTH CHECKS
# ------------------------------------------------------------------
echo "${YELLOW}--- Health Checks ---${NC}"

run_test "GET /health/live" "200" \
  "$BASE/health/live"

run_test "GET /health/ready" "200" \
  "$BASE/health/ready"

run_test "GET /metrics (prometheus)" "200" \
  "$BASE/metrics"

# ------------------------------------------------------------------
# 2. FRAUD SCORING
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- Fraud Scoring ---${NC}"

run_test "POST fraud/score — standard transaction" "200" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": 1500.50,
    "currency": "KES",
    "transaction_type": "TRANSFER",
    "device_fingerprint": "fp_device_123abc",
    "ip_address": "192.168.1.100",
    "geo_location": {"country": "KE", "lat": -1.2921, "lng": 36.8219},
    "merchant_id": "550e8400-e29b-41d4-a716-446655440005",
    "metadata": {"channel": "mobile_app"}
  }'

run_test "POST fraud/score — high-value suspicious" "200" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "a50e8400-e29b-41d4-a716-446655440099",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": 999999.99,
    "currency": "KES",
    "transaction_type": "WITHDRAWAL",
    "device_fingerprint": "fp_unknown_device",
    "ip_address": "203.0.113.42",
    "geo_location": {"country": "NG", "lat": 6.5244, "lng": 3.3792},
    "metadata": {"channel": "web", "new_device": true}
  }'

run_test "POST fraud/score — minimal fields" "200" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "b50e8400-e29b-41d4-a716-446655440001",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": 50.00,
    "currency": "KES",
    "transaction_type": "PAYMENT"
  }'

run_test "POST fraud/score — micro transaction" "200" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "b50e8400-e29b-41d4-a716-446655440002",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": 1.00,
    "currency": "KES",
    "transaction_type": "PAYMENT"
  }'

run_test "POST fraud/score — missing required fields (422)" "422" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{"amount": 100.00}'

run_test "POST fraud/score — negative amount (422)" "422" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": -500.00,
    "currency": "KES",
    "transaction_type": "PAYMENT"
  }'

run_test "POST fraud/score — empty body (422)" "422" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{}'

run_test "POST fraud/score/batch — 3 transactions" "200" \
  -X POST "$BASE/api/v1/ml/fraud/score/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "c50e8400-0001-41d4-a716-446655440001",
        "user_id": "550e8400-e29b-41d4-a716-446655440001",
        "amount": 500.00,
        "currency": "KES",
        "transaction_type": "PAYMENT"
      },
      {
        "transaction_id": "c50e8400-0002-41d4-a716-446655440002",
        "user_id": "550e8400-e29b-41d4-a716-446655440001",
        "amount": 75000.00,
        "currency": "KES",
        "transaction_type": "TRANSFER",
        "device_fingerprint": "fp_suspicious",
        "ip_address": "10.0.0.1"
      },
      {
        "transaction_id": "c50e8400-0003-41d4-a716-446655440003",
        "user_id": "d60e8400-e29b-41d4-a716-446655440010",
        "amount": 250000.00,
        "currency": "USD",
        "transaction_type": "WITHDRAWAL",
        "geo_location": {"country": "SO", "lat": 2.0469, "lng": 45.3182}
      }
    ]
  }'

run_test "GET fraud/explanation/{id} (no prior prediction)" "404" \
  "$BASE/api/v1/ml/fraud/explanation/550e8400-e29b-41d4-a716-446655440000"

# ------------------------------------------------------------------
# 3. AML SCORING
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- AML Scoring ---${NC}"

run_test "POST aml/score — standard transaction" "200" \
  -X POST "$BASE/api/v1/aml/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "d50e8400-e29b-41d4-a716-446655440010",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "sender_wallet_id": "e50e8400-e29b-41d4-a716-446655440020",
    "receiver_wallet_id": "e50e8400-e29b-41d4-a716-446655440021",
    "amount": 500000.00,
    "currency": "KES",
    "device_id": "device_abc123",
    "ip_address": "192.168.1.50",
    "geo_location": {"country": "KE", "lat": -1.2921, "lng": 36.8219},
    "channel": "app",
    "metadata": {"source": "mobile_money"}
  }'

run_test "POST aml/score — cross-border high-risk" "200" \
  -X POST "$BASE/api/v1/aml/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "d50e8400-e29b-41d4-a716-446655440011",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "sender_wallet_id": "e50e8400-e29b-41d4-a716-446655440020",
    "receiver_wallet_id": "f60e8400-e29b-41d4-a716-446655440030",
    "amount": 2000000.00,
    "currency": "USD",
    "device_id": "device_new_xyz",
    "ip_address": "203.0.113.99",
    "geo_location": {"country": "SO", "lat": 2.0469, "lng": 45.3182},
    "channel": "web",
    "metadata": {"cross_border": true, "recipient_country": "SO"}
  }'

run_test "POST aml/score — minimal fields" "200" \
  -X POST "$BASE/api/v1/aml/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "d50e8400-e29b-41d4-a716-446655440012",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": 1000.00,
    "currency": "KES"
  }'

run_test "POST aml/score/batch — 2 transactions" "200" \
  -X POST "$BASE/api/v1/aml/score/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "d50e8400-0001-41d4-a716-446655440001",
        "user_id": "550e8400-e29b-41d4-a716-446655440001",
        "amount": 10000.00,
        "currency": "KES",
        "channel": "app"
      },
      {
        "transaction_id": "d50e8400-0002-41d4-a716-446655440002",
        "user_id": "d60e8400-e29b-41d4-a716-446655440010",
        "amount": 1500000.00,
        "currency": "KES",
        "device_id": "device_unknown",
        "ip_address": "10.0.0.99",
        "channel": "API"
      }
    ]
  }'

run_test "GET aml/explanation/{id} (no prior prediction)" "404" \
  "$BASE/api/v1/aml/explanation/d50e8400-e29b-41d4-a716-446655440010"

# ------------------------------------------------------------------
# 4. MERCHANT RISK SCORING
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- Merchant Risk ---${NC}"

run_test "POST merchant/score — standard merchant" "200" \
  -X POST "$BASE/api/v1/ml/merchant/score" \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "550e8400-e29b-41d4-a716-446655440005",
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "amount": 2500.00,
    "currency": "KES",
    "customer_id": "550e8400-e29b-41d4-a716-446655440001",
    "mcc_code": "5411",
    "geo_location": {"country": "KE", "lat": -1.2921, "lng": 36.8219},
    "metadata": {"business_type": "retail"}
  }'

run_test "POST merchant/score — high-risk gambling MCC" "200" \
  -X POST "$BASE/api/v1/ml/merchant/score" \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "f70e8400-e29b-41d4-a716-446655440099",
    "amount": 950000.00,
    "currency": "KES",
    "mcc_code": "7995",
    "geo_location": {"country": "NG", "lat": 6.5244, "lng": 3.3792},
    "metadata": {"business_type": "gambling", "chargebacks_90d": 15}
  }'

run_test "POST merchant/score — minimal (just merchant_id)" "200" \
  -X POST "$BASE/api/v1/ml/merchant/score" \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "550e8400-e29b-41d4-a716-446655440005"
  }'

run_test "POST merchant/score/batch — 2 merchants" "200" \
  -X POST "$BASE/api/v1/ml/merchant/score/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "merchants": [
      {
        "merchant_id": "550e8400-e29b-41d4-a716-446655440005",
        "amount": 1000.00,
        "currency": "KES",
        "mcc_code": "5411"
      },
      {
        "merchant_id": "f70e8400-e29b-41d4-a716-446655440099",
        "amount": 500000.00,
        "currency": "KES",
        "mcc_code": "7995"
      }
    ]
  }'

run_test "GET merchant/explanation/{id} (no prior prediction)" "404" \
  "$BASE/api/v1/ml/merchant/explanation/550e8400-e29b-41d4-a716-446655440005"

# ------------------------------------------------------------------
# 5. AML CASE MANAGEMENT
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- AML Case Management ---${NC}"

# Create a case and capture case_id
CASE_RESPONSE=$(curl -s -X POST "$BASE/api/v1/aml/cases" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "USER",
    "entity_id": "550e8400-e29b-41d4-a716-446655440001",
    "risk_score": 0.85,
    "trigger_reason": "Sanctions match found in screening — high-value cross-border transfer",
    "priority": "CRITICAL"
  }' 2>&1)
CASE_ID=$(echo "$CASE_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('case_id',''))" 2>/dev/null || echo "")

if [ -n "$CASE_ID" ]; then
  TOTAL=$((TOTAL + 1))
  PASS=$((PASS + 1))
  printf "${CYAN}[%02d] %-55s${GREEN}PASS${NC} (HTTP 201, case_id=%s)\n" "$TOTAL" "POST aml/cases — create case" "$CASE_ID"
else
  TOTAL=$((TOTAL + 1))
  FAIL=$((FAIL + 1))
  printf "${CYAN}[%02d] %-55s${RED}FAIL${NC}\n" "$TOTAL" "POST aml/cases — create case"
  echo "  Response: $(echo "$CASE_RESPONSE" | head -c 200)"
  CASE_ID="00000000-0000-0000-0000-000000000000"
fi

run_test "GET aml/cases — list cases" "200" \
  "$BASE/api/v1/aml/cases?status=OPEN&limit=10&offset=0"

run_test "GET aml/cases/{id} — get case detail" "200" \
  "$BASE/api/v1/aml/cases/$CASE_ID"

run_test "PUT aml/cases/{id} — update to INVESTIGATING" "200" \
  -X PUT "$BASE/api/v1/aml/cases/$CASE_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "INVESTIGATING",
    "assigned_to": "a10e8400-e29b-41d4-a716-446655440200",
    "notes": "Investigation in progress — reviewing transaction patterns for last 90 days"
  }'

# Create a second case for variety
run_test "POST aml/cases — create TRANSACTION case" "201" \
  -X POST "$BASE/api/v1/aml/cases" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "TRANSACTION",
    "entity_id": "d50e8400-e29b-41d4-a716-446655440010",
    "risk_score": 0.72,
    "trigger_reason": "Structuring pattern detected — multiple sub-threshold transfers",
    "priority": "HIGH"
  }'

# ------------------------------------------------------------------
# 6. ADMIN ENDPOINTS
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- Admin ---${NC}"

run_test "GET admin/models — list models" "200" \
  "$BASE/api/v1/ml/admin/models"

run_test "GET admin/thresholds — get thresholds" "200" \
  "$BASE/api/v1/ml/admin/thresholds"

run_test "PUT admin/thresholds — update fraud thresholds" "200" \
  -X PUT "$BASE/api/v1/ml/admin/thresholds" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "fraud",
    "thresholds": {
      "approve_below": 0.55,
      "review_above": 0.78,
      "block_above": 0.92
    },
    "reason": "Q1 2024 performance review — adjusting for lower false positive rate"
  }'

run_test "GET admin/thresholds/history — fraud" "200" \
  "$BASE/api/v1/ml/admin/thresholds/history?model_type=fraud&limit=20"

run_test "GET admin/feature-importance/fraud (heuristic)" "200" \
  "$BASE/api/v1/ml/admin/feature-importance/fraud"

run_test "GET admin/feature-importance/aml (heuristic)" "200" \
  "$BASE/api/v1/ml/admin/feature-importance/aml"

run_test "GET admin/feature-importance/merchant (heuristic)" "200" \
  "$BASE/api/v1/ml/admin/feature-importance/merchant"

run_test "GET admin/alerts" "200" \
  "$BASE/api/v1/ml/admin/alerts?limit=10"

# ------------------------------------------------------------------
# 7. BATCH RESCORE & EXPORT
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- Batch Rescore & Export ---${NC}"

run_test "POST batch/fraud/rescore" "200" \
  -X POST "$BASE/api/v1/ml/batch/fraud/rescore" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "e50e8400-0001-41d4-a716-446655440001",
        "user_id": "550e8400-e29b-41d4-a716-446655440001",
        "amount": 5000.00,
        "currency": "KES",
        "transaction_type": "PAYMENT"
      },
      {
        "transaction_id": "e50e8400-0002-41d4-a716-446655440002",
        "user_id": "550e8400-e29b-41d4-a716-446655440001",
        "amount": 120000.00,
        "currency": "KES",
        "transaction_type": "TRANSFER"
      }
    ]
  }'

run_test "POST batch/aml/rescore" "200" \
  -X POST "$BASE/api/v1/ml/batch/aml/rescore" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "e50e8400-0003-41d4-a716-446655440003",
        "user_id": "550e8400-e29b-41d4-a716-446655440001",
        "amount": 80000.00,
        "currency": "KES",
        "channel": "app"
      }
    ]
  }'

run_test "POST batch/merchant/rescore" "200" \
  -X POST "$BASE/api/v1/ml/batch/merchant/rescore" \
  -H "Content-Type: application/json" \
  -d '{
    "merchants": [
      {
        "merchant_id": "550e8400-e29b-41d4-a716-446655440005",
        "amount": 3000.00,
        "currency": "KES",
        "mcc_code": "5411"
      }
    ]
  }'

run_test "POST export/sar — SAR report" "200" \
  -X POST "$BASE/api/v1/ml/batch/export/sar?start_date=2024-01-01T00:00:00&end_date=2024-12-31T23:59:59&export_format=JSON&min_risk_score=0.75"

run_test "POST export/str — STR report" "200" \
  -X POST "$BASE/api/v1/ml/batch/export/str?start_date=2024-01-01T00:00:00&end_date=2024-12-31T23:59:59&export_format=CSV"

# ------------------------------------------------------------------
# 8. EDGE CASES & ERROR HANDLING
# ------------------------------------------------------------------
echo ""
echo "${YELLOW}--- Edge Cases & Errors ---${NC}"

run_test "GET /nonexistent — 404" "404" \
  "$BASE/nonexistent"

run_test "POST fraud/score — invalid JSON (422)" "422" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d 'not-json'

run_test "POST fraud/score — invalid UUID format (422)" "422" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "not-a-uuid",
    "user_id": "also-not-uuid",
    "amount": 100.00,
    "currency": "KES",
    "transaction_type": "PAYMENT"
  }'

run_test "POST fraud/score — currency too long (422)" "422" \
  -X POST "$BASE/api/v1/ml/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "amount": 100.00,
    "currency": "TOOLONG",
    "transaction_type": "PAYMENT"
  }'

run_test "POST aml/score — missing required fields (422)" "422" \
  -X POST "$BASE/api/v1/aml/score" \
  -H "Content-Type: application/json" \
  -d '{"amount": 100.00}'

run_test "POST merchant/score — missing merchant_id (422)" "422" \
  -X POST "$BASE/api/v1/ml/merchant/score" \
  -H "Content-Type: application/json" \
  -d '{"amount": 100.00, "currency": "KES"}'

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
printf "  Total:  %d\n" "$TOTAL"
printf "  ${GREEN}Passed: %d${NC}\n" "$PASS"
printf "  ${RED}Failed: %d${NC}\n" "$FAIL"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
