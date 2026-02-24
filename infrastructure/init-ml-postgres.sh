#!/bin/bash
set -e

# Create the Airflow database (MLflow database is created by POSTGRES_DB)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE pesaflow_airflow;
    GRANT ALL PRIVILEGES ON DATABASE pesaflow_airflow TO $POSTGRES_USER;
EOSQL

echo "=== ML Postgres initialized: pesaflow_mlflow + pesaflow_airflow ==="
