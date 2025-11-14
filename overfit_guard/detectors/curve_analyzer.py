"""Learning curve analyzer for detecting overfitting patterns."""

from typing import Dict, Any, List
import numpy as np
from overfit_guard.core.detector import BaseDetector, DetectionResult, OverfitSeverity


class LearningCurveAnalyzer(BaseDetector):
    """
    Analyzes learning curves to detect overfitting patterns.

    This detector looks for patterns like:
    - Decreasing training loss with increasing validation loss
    - Plateauing or diverging curves
    - Early signs of overfitting based on curve trends
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize learning curve analyzer.

        Config options:
            - lookback_window (int): Number of epochs to analyze (default: 10)
            - metric_name (str): Primary metric to monitor (default: 'loss')
            - divergence_threshold (float): Threshold for curve divergence (default: 0.05)
            - trend_threshold (float): Threshold for trend detection (default: 0.01)
            - min_epochs (int): Minimum epochs before detection (default: 5)
        """
        super().__init__(config)

        self.lookback_window = self.config.get('lookback_window', 10)
        self.metric_name = self.config.get('metric_name', 'loss')
        self.divergence_threshold = self.config.get('divergence_threshold', 0.05)
        self.trend_threshold = self.config.get('trend_threshold', 0.01)
        self.min_epochs = self.config.get('min_epochs', 5)

        self._train_curve: List[float] = []
        self._val_curve: List[float] = []

    def detect(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        **kwargs
    ) -> DetectionResult:
        """Detect overfitting by analyzing learning curve patterns."""

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

        # Update curves
        self._train_curve.append(train_value)
        self._val_curve.append(val_value)

        # Need minimum epochs for analysis
        if len(self._train_curve) < self.min_epochs:
            return DetectionResult(
                is_overfitting=False,
                severity=OverfitSeverity.NONE,
                confidence=0.0,
                metrics={'epoch': epoch},
                message="Insufficient data for curve analysis"
            )

        # Get recent window
        window = min(self.lookback_window, len(self._train_curve))
        recent_train = self._train_curve[-window:]
        recent_val = self._val_curve[-window:]

        # Calculate trends (linear regression slope)
        train_trend = self._calculate_trend(recent_train)
        val_trend = self._calculate_trend(recent_val)

        # Calculate divergence
        divergence = self._calculate_divergence(recent_train, recent_val)

        # Determine overfitting patterns
        is_overfitting = False
        severity = OverfitSeverity.NONE
        confidence = 0.0
        patterns = []

        # Pattern 1: Train improving, val degrading (for loss metrics)
        if train_trend < -self.trend_threshold and val_trend > self.trend_threshold:
            is_overfitting = True
            patterns.append("diverging_trends")
            confidence += 0.4

        # Pattern 2: Significant divergence
        if divergence > self.divergence_threshold:
            is_overfitting = True
            patterns.append("curve_divergence")
            confidence += 0.3

        # Pattern 3: Val plateauing while train improving
        if train_trend < -self.trend_threshold and abs(val_trend) < self.trend_threshold / 2:
            is_overfitting = True
            patterns.append("val_plateau")
            confidence += 0.3

        # Determine severity based on confidence
        if confidence >= 0.7:
            severity = OverfitSeverity.SEVERE
        elif confidence >= 0.4:
            severity = OverfitSeverity.MODERATE
        elif confidence > 0:
            severity = OverfitSeverity.MILD

        confidence = min(1.0, confidence)

        # Build metrics
        metrics = {
            'train_trend': train_trend,
            'val_trend': val_trend,
            'divergence': divergence,
            'patterns': patterns,
            'epoch': epoch
        }

        # Generate message
        message = (
            f"Curve analysis: train_trend={train_trend:.4f}, "
            f"val_trend={val_trend:.4f}, divergence={divergence:.4f}"
        )

        # Recommendations
        recommendations = []
        if is_overfitting:
            if "diverging_trends" in patterns:
                recommendations.append("Consider early stopping")
            if "curve_divergence" in patterns:
                recommendations.append("Apply stronger regularization")
            if "val_plateau" in patterns:
                recommendations.append("Model may have reached capacity - reduce complexity")

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

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values using linear regression."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)

        return slope

    def _calculate_divergence(self, train_values: List[float], val_values: List[float]) -> float:
        """Calculate divergence between train and val curves."""
        train_arr = np.array(train_values)
        val_arr = np.array(val_values)

        # Calculate mean absolute difference
        diff = np.abs(val_arr - train_arr)
        return float(np.mean(diff))

    def reset(self) -> None:
        """Reset detector state."""
        self.history = []
        self._train_curve = []
        self._val_curve = []
