"""Prometheus Metrics for Pesaflow ML Fraud & AML Engine."""

from prometheus_client import Counter, Gauge, Histogram


class PesaflowMetrics:
    """Centralized Prometheus metrics for fraud & AML scoring."""

    def __init__(self, prefix: str = "pesaflow_ml"):
        # === Fraud Scoring Metrics ===
        self.fraud_scoring_latency = Histogram(
            f"{prefix}_fraud_scoring_latency_ms",
            "Fraud scoring latency in milliseconds",
            ["decision"],
            buckets=(10, 25, 50, 100, 150, 200, 300, 500, 1000),
        )

        self.fraud_score_distribution = Histogram(
            f"{prefix}_fraud_score_distribution",
            "Distribution of fraud risk scores",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        self.fraud_decisions = Counter(
            f"{prefix}_fraud_decisions_total",
            "Total fraud decisions by type",
            ["decision", "risk_level"],
        )

        self.fraud_errors = Counter(
            f"{prefix}_fraud_errors_total",
            "Total fraud scoring errors",
        )

        # === AML Scoring Metrics ===
        self.aml_scoring_latency = Histogram(
            f"{prefix}_aml_scoring_latency_ms",
            "AML scoring latency in milliseconds",
            ["decision"],
            buckets=(10, 25, 50, 100, 150, 200, 300, 500, 1000),
        )

        self.aml_score_distribution = Histogram(
            f"{prefix}_aml_score_distribution",
            "Distribution of AML risk scores",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        self.aml_decisions = Counter(
            f"{prefix}_aml_decisions_total",
            "Total AML decisions by type",
            ["decision", "risk_level"],
        )

        self.aml_cases_created = Counter(
            f"{prefix}_aml_cases_created_total",
            "Total AML cases created",
            ["priority"],
        )

        self.aml_errors = Counter(
            f"{prefix}_aml_errors_total",
            "Total AML scoring errors",
        )

        # === Merchant Risk Metrics ===
        self.merchant_scoring_latency = Histogram(
            f"{prefix}_merchant_scoring_latency_ms",
            "Merchant risk scoring latency in milliseconds",
            ["decision"],
            buckets=(10, 25, 50, 100, 150, 200, 300, 500, 1000),
        )

        self.merchant_score_distribution = Histogram(
            f"{prefix}_merchant_score_distribution",
            "Distribution of merchant risk scores",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        self.merchant_decisions = Counter(
            f"{prefix}_merchant_decisions_total",
            "Total merchant risk decisions by type",
            ["decision", "risk_level"],
        )

        self.merchant_tier_assignments = Counter(
            f"{prefix}_merchant_tier_assignments_total",
            "Merchant tier assignments",
            ["tier"],
        )

        self.merchant_errors = Counter(
            f"{prefix}_merchant_errors_total",
            "Total merchant scoring errors",
        )

        # === Sanctions Screening ===
        self.sanctions_matches = Counter(
            f"{prefix}_sanctions_matches_total",
            "Total sanctions matches",
            ["match_type"],
        )

        # === Model Metrics ===
        self.model_version_active = Gauge(
            f"{prefix}_model_version_active",
            "Currently active model version",
            ["model_name", "version"],
        )

        self.drift_index = Gauge(
            f"{prefix}_drift_index",
            "Model drift PSI score",
            ["model_name"],
        )

        self.false_positive_rate = Gauge(
            f"{prefix}_false_positive_rate",
            "Estimated false positive rate",
            ["model_name"],
        )

        # === Feature Store ===
        self.feature_retrieval_latency = Histogram(
            f"{prefix}_feature_retrieval_latency_ms",
            "Feature retrieval latency",
            ["source"],
            buckets=(5, 10, 20, 30, 40, 50, 100),
        )

        self.feature_cache_hits = Counter(
            f"{prefix}_feature_cache_hits_total",
            "Feature cache hit count",
            ["feature_type"],
        )

        self.feature_cache_misses = Counter(
            f"{prefix}_feature_cache_misses_total",
            "Feature cache miss count",
            ["feature_type"],
        )

    # === Recording Methods ===

    def record_fraud_scoring(self, latency_ms: int, decision: str, risk_level: str, score: float) -> None:
        self.fraud_scoring_latency.labels(decision=decision).observe(latency_ms)
        self.fraud_score_distribution.observe(score)
        self.fraud_decisions.labels(decision=decision, risk_level=risk_level).inc()

    def record_fraud_error(self) -> None:
        self.fraud_errors.inc()

    def record_aml_scoring(self, latency_ms: int, decision: str, risk_level: str, score: float) -> None:
        self.aml_scoring_latency.labels(decision=decision).observe(latency_ms)
        self.aml_score_distribution.observe(score)
        self.aml_decisions.labels(decision=decision, risk_level=risk_level).inc()

    def record_aml_error(self) -> None:
        self.aml_errors.inc()

    def record_case_created(self, priority: str) -> None:
        self.aml_cases_created.labels(priority=priority).inc()

    def record_merchant_scoring(self, latency_ms: int, decision: str, risk_level: str, score: float, tier: str) -> None:
        self.merchant_scoring_latency.labels(decision=decision).observe(latency_ms)
        self.merchant_score_distribution.observe(score)
        self.merchant_decisions.labels(decision=decision, risk_level=risk_level).inc()
        self.merchant_tier_assignments.labels(tier=tier).inc()

    def record_merchant_error(self) -> None:
        self.merchant_errors.inc()

    def record_sanctions_match(self, match_type: str) -> None:
        self.sanctions_matches.labels(match_type=match_type).inc()
