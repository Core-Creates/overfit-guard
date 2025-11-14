"""Statistical tests for overfitting detection."""

from typing import Dict, Any, List
import numpy as np
from scipy import stats
from overfit_guard.core.detector import BaseDetector, DetectionResult, OverfitSeverity


class StatisticalDetector(BaseDetector):
    """
    Uses statistical tests to detect overfitting.

    Applies various statistical methods:
    - Paired t-test between train and validation performance
    - Trend analysis with significance testing
    - Distribution comparison tests
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize statistical detector.

        Config options:
            - significance_level (float): P-value threshold (default: 0.05)
            - metric_name (str): Metric to monitor (default: 'loss')
            - min_samples (int): Minimum samples for testing (default: 10)
            - test_type (str): 'ttest', 'ks_test', or 'both' (default: 'both')
        """
        super().__init__(config)

        self.significance_level = self.config.get('significance_level', 0.05)
        self.metric_name = self.config.get('metric_name', 'loss')
        self.min_samples = self.config.get('min_samples', 10)
        self.test_type = self.config.get('test_type', 'both')

        self._train_samples: List[float] = []
        self._val_samples: List[float] = []

    def detect(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        **kwargs
    ) -> DetectionResult:
        """Detect overfitting using statistical tests."""

        train_value = train_metrics.get(self.metric_name)
        val_value = val_metrics.get(self.metric_name)

        if train_value is None or val_value is None:
            return DetectionResult(
                is_overfitting=False,
                severity=OverfitSeverity.NONE,
                confidence=0.0,
                metrics={},
                message=f"Missing metric: {self.metric_name}"
            )

        # Collect samples
        self._train_samples.append(train_value)
        self._val_samples.append(val_value)

        if len(self._train_samples) < self.min_samples:
            return DetectionResult(
                is_overfitting=False,
                severity=OverfitSeverity.NONE,
                confidence=0.0,
                metrics={'epoch': epoch},
                message="Insufficient samples for statistical testing"
            )

        # Run statistical tests
        test_results = {}
        is_overfitting = False
        confidence_scores = []

        # Paired t-test
        if self.test_type in ['ttest', 'both']:
            ttest_result = self._paired_ttest()
            test_results['ttest'] = ttest_result
            if ttest_result['significant']:
                is_overfitting = True
                confidence_scores.append(ttest_result['confidence'])

        # Kolmogorov-Smirnov test (distribution comparison)
        if self.test_type in ['ks_test', 'both']:
            ks_result = self._ks_test()
            test_results['ks_test'] = ks_result
            if ks_result['significant']:
                is_overfitting = True
                confidence_scores.append(ks_result['confidence'])

        # Calculate overall confidence
        confidence = max(confidence_scores) if confidence_scores else 0.0

        # Determine severity
        if confidence >= 0.9:
            severity = OverfitSeverity.SEVERE
        elif confidence >= 0.7:
            severity = OverfitSeverity.MODERATE
        elif confidence > 0.5:
            severity = OverfitSeverity.MILD
        else:
            severity = OverfitSeverity.NONE
            is_overfitting = False

        # Build metrics
        metrics = {
            **test_results,
            'num_samples': len(self._train_samples),
            'epoch': epoch
        }

        # Generate message
        messages = []
        if 'ttest' in test_results:
            messages.append(f"t-test p={test_results['ttest']['p_value']:.4f}")
        if 'ks_test' in test_results:
            messages.append(f"KS-test p={test_results['ks_test']['p_value']:.4f}")
        message = ", ".join(messages)

        # Recommendations
        recommendations = []
        if is_overfitting:
            recommendations.append("Statistical tests indicate significant performance gap")
            if severity.value >= OverfitSeverity.MODERATE.value:
                recommendations.append("Apply regularization or reduce model complexity")

        result = DetectionResult(
            is_overfitting=is_overfitting,
            severity=severity,
            confidence=confidence,
            metrics=metrics,
            message=message,
            recommendations=recommendations
        )

        self._add_to_history(result)
        return result

    def _paired_ttest(self) -> Dict[str, Any]:
        """Perform paired t-test between train and val metrics."""
        train_arr = np.array(self._train_samples)
        val_arr = np.array(self._val_samples)

        # Paired t-test
        statistic, p_value = stats.ttest_rel(val_arr, train_arr)

        # Significant if p < significance_level
        significant = p_value < self.significance_level

        # Confidence based on p-value
        confidence = 1.0 - p_value if significant else 0.0

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'confidence': confidence
        }

    def _ks_test(self) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        train_arr = np.array(self._train_samples)
        val_arr = np.array(self._val_samples)

        # KS test
        statistic, p_value = stats.ks_2samp(train_arr, val_arr)

        # Significant if p < significance_level
        significant = p_value < self.significance_level

        # Confidence based on both p-value and KS statistic
        confidence = (1.0 - p_value) * statistic if significant else 0.0

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'confidence': min(1.0, confidence)
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.history = []
        self._train_samples = []
        self._val_samples = []
