#!/usr/bin/env bash
# =================================================================
# PesaFlow ML — Seed All Data Stores
# =================================================================
#
# Seeds both offline (PostgreSQL) and online (Redis) feature stores
# with realistic Kenyan fintech transaction data.
#
# Usage:
#   bash seeds/seed_all.sh                     # default: Docker containers
#   bash seeds/seed_all.sh --local             # local dev (localhost)
#   bash seeds/seed_all.sh --pg-only           # PostgreSQL only
#   bash seeds/seed_all.sh --redis-only        # Redis only
#
# Environment overrides:
#   PG_CONTAINER   — Postgres container name  (default: pesaflow-postgres)
#   PG_USER        — Postgres user            (default: pesaflow)
#   PG_DB          — Postgres database        (default: pesaflow)
#   ML_CONTAINER   — ML service container     (default: pesaflow-ml)
#   REDIS_URL      — Redis connection URL     (default: redis://pesaflow-redis:6379/2)
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_CONTAINER="${PG_CONTAINER:-pesaflow-postgres}"
PG_USER="${PG_USER:-pesaflow}"
PG_DB="${PG_DB:-pesaflow}"
ML_CONTAINER="${ML_CONTAINER:-pesaflow-ml}"
REDIS_URL="${REDIS_URL:-redis://pesaflow-redis:6379/2}"

SEED_PG=true
SEED_REDIS=true
LOCAL_MODE=false

for arg in "$@"; do
  case "$arg" in
    --pg-only)    SEED_REDIS=false ;;
    --redis-only) SEED_PG=false ;;
    --local)      LOCAL_MODE=true ;;
    --help|-h)
      head -20 "$0" | grep '^#' | sed 's/^# *//'
      exit 0
      ;;
  esac
done

echo "==========================================="
echo " PesaFlow ML — Seed Realistic Data"
echo "==========================================="
echo ""

# -----------------------------------------------------------------
# PostgreSQL — Offline Feature Store
# -----------------------------------------------------------------
if $SEED_PG; then
  echo "--- [1/2] PostgreSQL (offline features) ---"
  echo ""

  if $LOCAL_MODE; then
    echo "  Mode: local psql → localhost"
    psql -U "$PG_USER" -d "$PG_DB" < "$SCRIPT_DIR/ml_realistic_seed.sql"
  else
    echo "  Mode: docker exec → $PG_CONTAINER"
    docker exec -i "$PG_CONTAINER" \
      psql -U "$PG_USER" -d "$PG_DB" \
      < "$SCRIPT_DIR/ml_realistic_seed.sql"
  fi

  echo ""
  echo "  PostgreSQL seed complete."
  echo ""
fi

# -----------------------------------------------------------------
# Redis — Online Feature Store
# -----------------------------------------------------------------
if $SEED_REDIS; then
  echo "--- [2/2] Redis (online features) ---"
  echo ""

  if $LOCAL_MODE; then
    echo "  Mode: local python3"
    REDIS_URL="${REDIS_URL:-redis://localhost:26379/0}" \
      python3 "$SCRIPT_DIR/redis_online_seed.py"
  else
    echo "  Mode: docker exec → $ML_CONTAINER (piping Python script)"
    cat "$SCRIPT_DIR/redis_online_seed.py" | \
      docker exec -i -e REDIS_URL="$REDIS_URL" \
        "$ML_CONTAINER" python3 -
  fi

  echo ""
  echo "  Redis seed complete."
  echo ""
fi

echo "==========================================="
echo " Seed finished successfully."
echo ""
echo " Next steps:"
echo "   1. bash tests/curl_test.sh https://ml.pesaflow.ai"
echo "   2. Score seeded transactions through fraud/AML/merchant endpoints"
echo "   3. Check /api/v1/ml/labels/statistics for COLD maturity"
echo "==========================================="
