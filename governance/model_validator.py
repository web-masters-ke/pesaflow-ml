"""Model Validator — Pre-deployment validation checks for ML models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from governance.bias_detector import BiasDetector, BiasReport


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    actual_value: float | None = None
    threshold: float | None = None
    message: str = ""


@dataclass
class ModelValidationReport:
    """Full pre-deployment validation report."""

    model_name: str
    model_version: str
    validation_timestamp: str
    overall_pass: bool
    performance_checks: list[ValidationResult]
    stability_checks: list[ValidationResult]
    fairness_checks: list[ValidationResult]
    data_quality_checks: list[ValidationResult]
    warnings: list[str] = field(default_factory=list)


class ModelValidator:
    """Pre-deployment validation gate for ML models."""

    def __init__(
        self,
        min_roc_auc: float = 0.90,
        min_precision: float = 0.85,
        min_recall: float = 0.80,
        max_score_drift_psi: float = 0.20,
        min_sample_size: int = 1000,
        max_missing_feature_rate: float = 0.05,
    ):
        self._min_auc = min_roc_auc
        self._min_precision = min_precision
        self._min_recall = min_recall
        self._max_psi = max_score_drift_psi
        self._min_samples = min_sample_size
        self._max_missing = max_missing_feature_rate

    def validate(
        self,
        model_name: str,
        model_version: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        features: np.ndarray,
        baseline_scores: np.ndarray | None = None,
        protected_attributes: dict[str, np.ndarray] | None = None,
    ) -> ModelValidationReport:
        """Run all pre-deployment validation checks."""
        performance = self._check_performance(y_true, y_pred, y_scores)
        stability = self._check_stability(y_scores, baseline_scores)
        fairness = self._check_fairness(model_name, y_true, y_pred, y_scores, protected_attributes)
        data_quality = self._check_data_quality(features, y_true)

        all_checks = performance + stability + fairness + data_quality
        overall_pass = all(c.passed for c in all_checks)

        warnings = [c.message for c in all_checks if not c.passed]

        report = ModelValidationReport(
            model_name=model_name,
            model_version=model_version,
            validation_timestamp=datetime.utcnow().isoformat(),
            overall_pass=overall_pass,
            performance_checks=performance,
            stability_checks=stability,
            fairness_checks=fairness,
            data_quality_checks=data_quality,
            warnings=warnings,
        )

        logger.info(
            f"Model validation for {model_name} {model_version}: "
            f"{'PASSED' if overall_pass else 'FAILED'} "
            f"({sum(c.passed for c in all_checks)}/{len(all_checks)} checks passed)"
        )

        return report

    def _check_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
    ) -> list[ValidationResult]:
        """Validate model performance metrics."""
        checks = []

        # ROC-AUC
        try:
            auc = roc_auc_score(y_true, y_scores)
            checks.append(ValidationResult(
                check_name="roc_auc",
                passed=auc >= self._min_auc,
                actual_value=auc,
                threshold=self._min_auc,
                message=f"ROC-AUC {auc:.4f} {'>=':} {self._min_auc}" if auc >= self._min_auc
                else f"ROC-AUC {auc:.4f} below threshold {self._min_auc}",
            ))
        except Exception as e:
            checks.append(ValidationResult(
                check_name="roc_auc",
                passed=False,
                message=f"ROC-AUC computation failed: {e}",
            ))

        # Precision
        try:
            prec = precision_score(y_true, y_pred, zero_division=0)
            checks.append(ValidationResult(
                check_name="precision",
                passed=prec >= self._min_precision,
                actual_value=prec,
                threshold=self._min_precision,
                message=f"Precision {prec:.4f}" if prec >= self._min_precision
                else f"Precision {prec:.4f} below threshold {self._min_precision}",
            ))
        except Exception as e:
            checks.append(ValidationResult(
                check_name="precision",
                passed=False,
                message=f"Precision computation failed: {e}",
            ))

        # Recall
        try:
            rec = recall_score(y_true, y_pred, zero_division=0)
            checks.append(ValidationResult(
                check_name="recall",
                passed=rec >= self._min_recall,
                actual_value=rec,
                threshold=self._min_recall,
                message=f"Recall {rec:.4f}" if rec >= self._min_recall
                else f"Recall {rec:.4f} below threshold {self._min_recall}",
            ))
        except Exception as e:
            checks.append(ValidationResult(
                check_name="recall",
                passed=False,
                message=f"Recall computation failed: {e}",
            ))

        # F1
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            checks.append(ValidationResult(
                check_name="f1_score",
                passed=True,  # Informational
                actual_value=f1,
                message=f"F1 score: {f1:.4f}",
            ))
        except Exception:
            pass

        return checks

    def _check_stability(
        self,
        y_scores: np.ndarray,
        baseline_scores: np.ndarray | None,
    ) -> list[ValidationResult]:
        """Check model stability via score distribution drift (PSI)."""
        checks = []

        if baseline_scores is None:
            checks.append(ValidationResult(
                check_name="score_drift_psi",
                passed=True,
                message="No baseline scores provided — skipping drift check",
            ))
            return checks

        psi = self._compute_psi(baseline_scores, y_scores)
        checks.append(ValidationResult(
            check_name="score_drift_psi",
            passed=psi <= self._max_psi,
            actual_value=psi,
            threshold=self._max_psi,
            message=f"Score PSI {psi:.4f}" if psi <= self._max_psi
            else f"Score PSI {psi:.4f} exceeds threshold {self._max_psi} — significant drift detected",
        ))

        # Score range check
        min_score = float(y_scores.min())
        max_score = float(y_scores.max())
        valid_range = 0.0 <= min_score and max_score <= 1.0
        checks.append(ValidationResult(
            check_name="score_range",
            passed=valid_range,
            message=f"Scores range [{min_score:.4f}, {max_score:.4f}]"
            if valid_range
            else f"Scores out of [0, 1] range: [{min_score:.4f}, {max_score:.4f}]",
        ))

        return checks

    def _check_fairness(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        protected_attributes: dict[str, np.ndarray] | None,
    ) -> list[ValidationResult]:
        """Run bias/fairness checks across protected attributes."""
        checks = []

        if protected_attributes is None:
            checks.append(ValidationResult(
                check_name="fairness_evaluation",
                passed=True,
                message="No protected attributes provided — skipping fairness checks",
            ))
            return checks

        detector = BiasDetector()

        for attr_name, attr_values in protected_attributes.items():
            try:
                report = detector.evaluate(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_scores=y_scores,
                    protected_attribute=attr_values,
                    attribute_name=attr_name,
                    model_name=model_name,
                )

                checks.append(ValidationResult(
                    check_name=f"fairness_{attr_name}",
                    passed=report.overall_fairness_pass,
                    actual_value=report.disparate_impact_ratio,
                    message=f"Fairness for '{attr_name}': DI={report.disparate_impact_ratio:.3f} "
                    f"FPR={report.fpr_parity_ratio:.3f} EO={report.equal_opportunity_ratio:.3f} "
                    f"{'PASS' if report.overall_fairness_pass else 'FAIL'}",
                ))

                if report.warnings:
                    for w in report.warnings:
                        checks.append(ValidationResult(
                            check_name=f"fairness_{attr_name}_warning",
                            passed=True,  # Warnings don't fail validation
                            message=w,
                        ))

            except Exception as e:
                checks.append(ValidationResult(
                    check_name=f"fairness_{attr_name}",
                    passed=False,
                    message=f"Fairness check failed for '{attr_name}': {e}",
                ))

        return checks

    def _check_data_quality(
        self,
        features: np.ndarray,
        y_true: np.ndarray,
    ) -> list[ValidationResult]:
        """Validate input data quality."""
        checks = []

        # Sample size
        n_samples = len(y_true)
        checks.append(ValidationResult(
            check_name="sample_size",
            passed=n_samples >= self._min_samples,
            actual_value=n_samples,
            threshold=self._min_samples,
            message=f"Sample size: {n_samples}"
            if n_samples >= self._min_samples
            else f"Sample size {n_samples} below minimum {self._min_samples}",
        ))

        # Missing values
        if isinstance(features, np.ndarray):
            missing_rate = np.isnan(features).mean()
            checks.append(ValidationResult(
                check_name="missing_features",
                passed=missing_rate <= self._max_missing,
                actual_value=missing_rate,
                threshold=self._max_missing,
                message=f"Missing feature rate: {missing_rate:.4f}"
                if missing_rate <= self._max_missing
                else f"Missing feature rate {missing_rate:.4f} exceeds threshold {self._max_missing}",
            ))

        # Class balance check
        positive_rate = y_true.mean()
        checks.append(ValidationResult(
            check_name="class_balance",
            passed=0.001 < positive_rate < 0.5,
            actual_value=positive_rate,
            message=f"Positive class rate: {positive_rate:.4f}"
            if 0.001 < positive_rate < 0.5
            else f"Extreme class imbalance: positive rate = {positive_rate:.4f}",
        ))

        # Feature variance check (detect constant features)
        if isinstance(features, np.ndarray) and features.ndim == 2:
            variances = np.nanvar(features, axis=0)
            zero_var_count = (variances == 0).sum()
            checks.append(ValidationResult(
                check_name="feature_variance",
                passed=zero_var_count == 0,
                actual_value=zero_var_count,
                message=f"All features have variance"
                if zero_var_count == 0
                else f"{zero_var_count} features have zero variance (constant)",
            ))

        return checks

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index between two score distributions."""
        eps = 1e-6

        breakpoints = np.linspace(0, 1, bins + 1)
        expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
        actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

        expected_pct = (expected_counts + eps) / (expected_counts.sum() + eps * bins)
        actual_pct = (actual_counts + eps) / (actual_counts.sum() + eps * bins)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)
