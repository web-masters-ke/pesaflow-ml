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

        # === Label Feedback Metrics ===
        self.labels_submitted = Counter(
            f"{prefix}_labels_submitted_total",
            "Total labels submitted",
            ["domain", "label_source"],
        )

        self.labels_propagated = Counter(
            f"{prefix}_labels_propagated_total",
            "Total labels auto-propagated from cases",
            ["domain"],
        )

        self.label_flips = Counter(
            f"{prefix}_label_flips_total",
            "Times a label was changed (overwritten)",
            ["domain"],
        )

        self.labeled_count = Gauge(
            f"{prefix}_labeled_count",
            "Current count of labeled predictions",
            ["domain"],
        )

        # === Active Learning Metrics ===
        self.al_queue_size = Gauge(
            f"{prefix}_al_queue_size",
            "Active learning queue size",
            ["domain"],
        )

        self.al_labels_from_queue = Counter(
            f"{prefix}_al_labels_from_queue_total",
            "Labels submitted via active learning queue",
            ["domain"],
        )

        self.al_budget_used_daily = Gauge(
            f"{prefix}_al_budget_used_daily",
            "Daily active learning budget used",
            ["domain"],
        )

        # === Semi-Supervised Learning Metrics ===
        self.ssl_pseudo_labels_generated = Counter(
            f"{prefix}_ssl_pseudo_labels_generated_total",
            "Pseudo-labels generated by SSL techniques",
            ["domain", "technique"],
        )

        self.ssl_pseudo_label_accuracy = Gauge(
            f"{prefix}_ssl_pseudo_label_accuracy",
            "Validated accuracy of pseudo-labels",
            ["domain", "technique"],
        )

        self.ssl_pseudo_label_flip_rate = Gauge(
            f"{prefix}_ssl_pseudo_label_flip_rate",
            "Rate of pseudo-label flips between iterations",
            ["domain", "technique"],
        )

        self.ssl_training_runs_total = Counter(
            f"{prefix}_ssl_training_runs_total",
            "Total SSL training runs",
            ["domain", "technique"],
        )

        self.ssl_model_auc = Gauge(
            f"{prefix}_ssl_model_auc",
            "AUC of SSL-trained model on labeled holdout",
            ["domain", "technique"],
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

    # === Label & AL Recording Methods ===

    def record_label_submitted(self, domain: str, label_source: str) -> None:
        self.labels_submitted.labels(domain=domain, label_source=label_source).inc()

    def record_label_propagated(self, domain: str) -> None:
        self.labels_propagated.labels(domain=domain).inc()

    def record_label_flip(self, domain: str) -> None:
        self.label_flips.labels(domain=domain).inc()

    def set_labeled_count(self, domain: str, count: int) -> None:
        self.labeled_count.labels(domain=domain).set(count)

    def set_al_queue_size(self, domain: str, size: int) -> None:
        self.al_queue_size.labels(domain=domain).set(size)

    def record_al_label(self, domain: str) -> None:
        self.al_labels_from_queue.labels(domain=domain).inc()

    def set_al_budget_used(self, domain: str, used: int) -> None:
        self.al_budget_used_daily.labels(domain=domain).set(used)

    def record_ssl_pseudo_labels(self, domain: str, technique: str, count: int = 1) -> None:
        self.ssl_pseudo_labels_generated.labels(domain=domain, technique=technique).inc(count)

    def set_ssl_pseudo_label_accuracy(self, domain: str, technique: str, accuracy: float) -> None:
        self.ssl_pseudo_label_accuracy.labels(domain=domain, technique=technique).set(accuracy)

    def set_ssl_pseudo_label_flip_rate(self, domain: str, technique: str, rate: float) -> None:
        self.ssl_pseudo_label_flip_rate.labels(domain=domain, technique=technique).set(rate)

    def record_ssl_training_run(self, domain: str, technique: str) -> None:
        self.ssl_training_runs_total.labels(domain=domain, technique=technique).inc()

    def set_ssl_model_auc(self, domain: str, technique: str, auc: float) -> None:
        self.ssl_model_auc.labels(domain=domain, technique=technique).set(auc)
