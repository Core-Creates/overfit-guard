"""Cross-validation based overfitting detector."""

from typing import Dict, Any, List, Optional
import numpy as np
from overfit_guard.core.detector import BaseDetector, DetectionResult, OverfitSeverity


class CrossValidationDetector(BaseDetector):
    """
    Detects overfitting using cross-validation metrics variance.

    This detector analyzes variance across CV folds to identify overfitting.
    High variance suggests the model is overfitting to specific data splits.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize CV detector.

        Config options:
            - variance_threshold_mild (float): Threshold for mild overfitting (default: 0.05)
            - variance_threshold_moderate (float): Threshold for moderate (default: 0.10)
            - variance_threshold_severe (float): Threshold for severe (default: 0.20)
            - metric_name (str): Metric to monitor (default: 'loss')
            - min_folds (int): Minimum folds required (default: 3)
        """
        super().__init__(config)

        self.variance_threshold_mild = self.config.get('variance_threshold_mild', 0.05)
        self.variance_threshold_moderate = self.config.get('variance_threshold_moderate', 0.10)
        self.variance_threshold_severe = self.config.get('variance_threshold_severe', 0.20)
        self.metric_name = self.config.get('metric_name', 'loss')
        self.min_folds = self.config.get('min_folds', 3)

        self._fold_scores: List[float] = []

    def detect(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        **kwargs
    ) -> DetectionResult:
        """
        Detect overfitting using CV fold variance.

        Additional kwargs:
            - cv_scores (List[float]): Cross-validation scores for current epoch
        """

        cv_scores = kwargs.get('cv_scores')

        # If CV scores provided, use them
        if cv_scores is not None and len(cv_scores) >= self.min_folds:
            return self._analyze_cv_scores(cv_scores, epoch)

        # Otherwise, fall back to train/val comparison
        # (Accumulate validation scores as pseudo-folds)
        val_value = val_metrics.get(self.metric_name)
        if val_value is not None:
            self._fold_scores.append(val_value)

        if len(self._fold_scores) < self.min_folds:
            return DetectionResult(
                is_overfitting=False,
                severity=OverfitSeverity.NONE,
                confidence=0.0,
                metrics={'epoch': epoch},
                message="Insufficient CV data"
            )

        return self._analyze_cv_scores(self._fold_scores, epoch)

    def _analyze_cv_scores(
        self,
        scores: List[float],
        epoch: int
    ) -> DetectionResult:
        """Analyze variance in CV scores."""

        scores_arr = np.array(scores)
        mean_score = float(np.mean(scores_arr))
        std_score = float(np.std(scores_arr))
        variance = float(np.var(scores_arr))

        # Calculate coefficient of variation
        cv_coef = std_score / mean_score if mean_score != 0 else 0

        # Determine severity based on variance/CV coefficient
        is_overfitting = False
        severity = OverfitSeverity.NONE
        confidence = 0.0

        if cv_coef >= self.variance_threshold_severe:
            severity = OverfitSeverity.SEVERE
            is_overfitting = True
            confidence = min(1.0, cv_coef / self.variance_threshold_severe)
        elif cv_coef >= self.variance_threshold_moderate:
            severity = OverfitSeverity.MODERATE
            is_overfitting = True
            confidence = cv_coef / self.variance_threshold_moderate * 0.8
        elif cv_coef >= self.variance_threshold_mild:
            severity = OverfitSeverity.MILD
            is_overfitting = True
            confidence = cv_coef / self.variance_threshold_mild * 0.6

        metrics = {
            'cv_mean': mean_score,
            'cv_std': std_score,
            'cv_variance': variance,
            'cv_coefficient': cv_coef,
            'num_folds': len(scores),
            'epoch': epoch
        }

        message = (
            f"CV analysis: mean={mean_score:.4f}, std={std_score:.4f}, "
            f"coef_var={cv_coef:.4f}"
        )

        recommendations = []
        if is_overfitting:
            recommendations.append("High variance across folds indicates overfitting")
            recommendations.append("Consider stronger regularization or more training data")
            if severity == OverfitSeverity.SEVERE:
                recommendations.append("Model may be too complex - reduce capacity")

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

    def reset(self) -> None:
        """Reset detector state."""
        self.history = []
        self._fold_scores = []

    def add_cv_fold(self, score: float) -> None:
        """Manually add a CV fold score."""
        self._fold_scores.append(score)
