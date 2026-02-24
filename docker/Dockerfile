# ============================================================
# Stage 1: Builder — compile wheels in a throwaway layer
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/runtime.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout 300 --prefix=/install -r requirements.txt


# ============================================================
# Stage 2: Runtime — minimal production image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app appuser

COPY --from=builder /install /usr/local

# Copy ONLY runtime source — no training, tests, migrations
COPY serving/    ./serving/
COPY models/     ./models/
COPY feature_engineering/ ./feature_engineering/
COPY monitoring/ ./monitoring/
COPY config/     ./config/

RUN mkdir -p /app/model_artifacts/fraud \
             /app/model_artifacts/aml \
             /app/model_artifacts/merchant \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


# ============================================================
# Stage 3: Training — full image for model training jobs
# ============================================================
FROM runtime AS training

USER root

COPY requirements/training.txt /tmp/training.txt
RUN pip install --no-cache-dir --timeout 300 -r /tmp/training.txt && rm /tmp/training.txt

COPY training/ ./training/
COPY airflow/   ./airflow/
COPY migrations/ ./migrations/

RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "training"]
