"""Bias Detector — Fairness metrics and disparate impact analysis for ML models."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FairnessMetrics:
    """Container for fairness evaluation results."""

    group_name: str
    group_value: str
    sample_size: int
    positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    true_positive_rate: float
    precision: float
    recall: float


@dataclass
class BiasReport:
    """Full bias analysis report."""

    model_name: str
    protected_attribute: str
    reference_group: str
    metrics_by_group: list[FairnessMetrics]
    disparate_impact_ratio: float
    fpr_parity_ratio: float
    equal_opportunity_ratio: float
    overall_fairness_pass: bool
    thresholds: dict[str, float]
    warnings: list[str] = field(default_factory=list)


class BiasDetector:
    """Detects bias and fairness issues in ML model predictions."""

    def __init__(
        self,
        disparate_impact_threshold: float = 0.80,
        fpr_parity_threshold: float = 0.80,
        equal_opportunity_threshold: float = 0.80,
        min_group_size: int = 30,
    ):
        self._di_threshold = disparate_impact_threshold
        self._fpr_threshold = fpr_parity_threshold
        self._eo_threshold = equal_opportunity_threshold
        self._min_group_size = min_group_size

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        protected_attribute: np.ndarray,
        attribute_name: str,
        model_name: str,
        reference_group: str | None = None,
    ) -> BiasReport:
        """Run full fairness evaluation across a protected attribute."""
        groups = np.unique(protected_attribute)
        warnings: list[str] = []

        if reference_group is None:
            # Use the most common group as reference
            unique, counts = np.unique(protected_attribute, return_counts=True)
            reference_group = str(unique[np.argmax(counts)])

        # Compute per-group metrics
        metrics_by_group: list[FairnessMetrics] = []
        for group in groups:
            mask = protected_attribute == group
            group_size = mask.sum()

            if group_size < self._min_group_size:
                warnings.append(f"Group '{group}' has only {group_size} samples (min: {self._min_group_size})")
                continue

            yt = y_true[mask]
            yp = y_pred[mask]

            tp = ((yp == 1) & (yt == 1)).sum()
            fp = ((yp == 1) & (yt == 0)).sum()
            tn = ((yp == 0) & (yt == 0)).sum()
            fn = ((yp == 0) & (yt == 1)).sum()

            positive_rate = yp.mean()
            fpr = fp / max(fp + tn, 1)
            fnr = fn / max(fn + tp, 1)
            tpr = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            recall = tpr

            metrics_by_group.append(
                FairnessMetrics(
                    group_name=attribute_name,
                    group_value=str(group),
                    sample_size=int(group_size),
                    positive_rate=float(positive_rate),
                    false_positive_rate=float(fpr),
                    false_negative_rate=float(fnr),
                    true_positive_rate=float(tpr),
                    precision=float(precision),
                    recall=float(recall),
                )
            )

        # Compute disparate impact ratio
        di_ratio = self._compute_disparate_impact(metrics_by_group, reference_group)

        # Compute FPR parity ratio
        fpr_ratio = self._compute_fpr_parity(metrics_by_group, reference_group)

        # Compute equal opportunity ratio
        eo_ratio = self._compute_equal_opportunity(metrics_by_group, reference_group)

        # Overall fairness assessment
        overall_pass = (
            di_ratio >= self._di_threshold and fpr_ratio >= self._fpr_threshold and eo_ratio >= self._eo_threshold
        )

        if not overall_pass:
            if di_ratio < self._di_threshold:
                warnings.append(f"Disparate impact ratio {di_ratio:.3f} below threshold {self._di_threshold}")
            if fpr_ratio < self._fpr_threshold:
                warnings.append(f"FPR parity ratio {fpr_ratio:.3f} below threshold {self._fpr_threshold}")
            if eo_ratio < self._eo_threshold:
                warnings.append(f"Equal opportunity ratio {eo_ratio:.3f} below threshold {self._eo_threshold}")

        report = BiasReport(
            model_name=model_name,
            protected_attribute=attribute_name,
            reference_group=reference_group,
            metrics_by_group=metrics_by_group,
            disparate_impact_ratio=di_ratio,
            fpr_parity_ratio=fpr_ratio,
            equal_opportunity_ratio=eo_ratio,
            overall_fairness_pass=overall_pass,
            thresholds={
                "disparate_impact": self._di_threshold,
                "fpr_parity": self._fpr_threshold,
                "equal_opportunity": self._eo_threshold,
            },
            warnings=warnings,
        )

        logger.info(
            f"Bias report for {model_name}/{attribute_name}: "
            f"DI={di_ratio:.3f} FPR={fpr_ratio:.3f} EO={eo_ratio:.3f} "
            f"Pass={overall_pass}"
        )

        return report

    def _compute_disparate_impact(self, metrics: list[FairnessMetrics], reference_group: str) -> float:
        """Compute disparate impact ratio (4/5ths rule)."""
        ref_rate = None
        min_ratio = 1.0

        for m in metrics:
            if m.group_value == reference_group:
                ref_rate = m.positive_rate
                break

        if ref_rate is None or ref_rate == 0:
            return 1.0

        for m in metrics:
            if m.group_value == reference_group:
                continue
            ratio = m.positive_rate / ref_rate if ref_rate > 0 else 1.0
            # DI can be > 1 or < 1; use min of ratio and 1/ratio
            di = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0
            min_ratio = min(min_ratio, di)

        return min_ratio

    def _compute_fpr_parity(self, metrics: list[FairnessMetrics], reference_group: str) -> float:
        """Compute FPR parity ratio across groups."""
        ref_fpr = None
        min_ratio = 1.0

        for m in metrics:
            if m.group_value == reference_group:
                ref_fpr = m.false_positive_rate
                break

        if ref_fpr is None:
            return 1.0

        for m in metrics:
            if m.group_value == reference_group:
                continue
            if ref_fpr == 0 and m.false_positive_rate == 0:
                continue
            max_fpr = max(ref_fpr, m.false_positive_rate)
            min_fpr = min(ref_fpr, m.false_positive_rate)
            ratio = min_fpr / max_fpr if max_fpr > 0 else 1.0
            min_ratio = min(min_ratio, ratio)

        return min_ratio

    def _compute_equal_opportunity(self, metrics: list[FairnessMetrics], reference_group: str) -> float:
        """Compute equal opportunity (TPR parity) ratio."""
        ref_tpr = None
        min_ratio = 1.0

        for m in metrics:
            if m.group_value == reference_group:
                ref_tpr = m.true_positive_rate
                break

        if ref_tpr is None or ref_tpr == 0:
            return 1.0

        for m in metrics:
            if m.group_value == reference_group:
                continue
            ratio = m.true_positive_rate / ref_tpr if ref_tpr > 0 else 1.0
            eo = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0
            min_ratio = min(min_ratio, eo)

        return min_ratio

    def generate_summary(self, reports: list[BiasReport]) -> dict:
        """Generate summary across all bias reports."""
        all_pass = all(r.overall_fairness_pass for r in reports)
        all_warnings = []
        for r in reports:
            all_warnings.extend(r.warnings)

        return {
            "total_evaluations": len(reports),
            "overall_fairness_pass": all_pass,
            "failing_attributes": [r.protected_attribute for r in reports if not r.overall_fairness_pass],
            "disparate_impact_scores": {r.protected_attribute: r.disparate_impact_ratio for r in reports},
            "fpr_parity_scores": {r.protected_attribute: r.fpr_parity_ratio for r in reports},
            "equal_opportunity_scores": {r.protected_attribute: r.equal_opportunity_ratio for r in reports},
            "warnings": all_warnings,
        }
