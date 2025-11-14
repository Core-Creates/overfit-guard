"""Train-validation gap detector for overfitting."""

from typing import Dict, Any
from overfit_guard.core.detector import BaseDetector, DetectionResult, OverfitSeverity


class TrainValGapDetector(BaseDetector):
    """
    Detects overfitting by monitoring the gap between training and validation metrics.

    This detector calculates the relative and absolute differences between
    training and validation performance.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize gap detector.

        Config options:
            - gap_threshold_mild (float): Threshold for mild overfitting (default: 0.05)
            - gap_threshold_moderate (float): Threshold for moderate (default: 0.10)
            - gap_threshold_severe (float): Threshold for severe (default: 0.20)
            - metric_name (str): Primary metric to monitor (default: 'loss')
            - use_relative_gap (bool): Use relative gap vs absolute (default: True)
            - window_size (int): Number of epochs to average over (default: 1)
        """
        super().__init__(config)

        # Thresholds
        self.gap_threshold_mild = self.config.get('gap_threshold_mild', 0.05)
        self.gap_threshold_moderate = self.config.get('gap_threshold_moderate', 0.10)
        self.gap_threshold_severe = self.config.get('gap_threshold_severe', 0.20)

        # Settings
        self.metric_name = self.config.get('metric_name', 'loss')
        self.use_relative_gap = self.config.get('use_relative_gap', True)
        self.window_size = self.config.get('window_size', 1)

        # State
        self._train_history = []
        self._val_history = []

    def detect(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        **kwargs
    ) -> DetectionResult:
        """Detect overfitting based on train-validation gap."""

        # Get primary metric values
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

        # Store in history
        self._train_history.append(train_value)
        self._val_history.append(val_value)

        # Calculate gap over window
        window_train = self._train_history[-self.window_size:]
        window_val = self._val_history[-self.window_size:]

        avg_train = sum(window_train) / len(window_train)
        avg_val = sum(window_val) / len(window_val)

        # Calculate gap
        if self.use_relative_gap and avg_train != 0:
            gap = abs(avg_val - avg_train) / abs(avg_train)
        else:
            gap = abs(avg_val - avg_train)

        # Determine severity
        if gap >= self.gap_threshold_severe:
            severity = OverfitSeverity.SEVERE
            is_overfitting = True
            confidence = min(1.0, gap / self.gap_threshold_severe)
        elif gap >= self.gap_threshold_moderate:
            severity = OverfitSeverity.MODERATE
            is_overfitting = True
            confidence = gap / self.gap_threshold_moderate * 0.8
        elif gap >= self.gap_threshold_mild:
            severity = OverfitSeverity.MILD
            is_overfitting = True
            confidence = gap / self.gap_threshold_mild * 0.6
        else:
            severity = OverfitSeverity.NONE
            is_overfitting = False
            confidence = 0.0

        # Build metrics dictionary
        metrics = {
            'gap': gap,
            'train_value': avg_train,
            'val_value': avg_val,
            'epoch': epoch
        }

        # Generate message and recommendations
        message = (
            f"Train-val gap: {gap:.4f} "
            f"(train: {avg_train:.4f}, val: {avg_val:.4f})"
        )

        recommendations = []
        if is_overfitting:
            recommendations.append("Apply regularization (L1/L2, dropout)")
            if severity.value >= OverfitSeverity.MODERATE.value:
                recommendations.append("Reduce model complexity")
                recommendations.append("Increase training data or augmentation")

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
        self._train_history = []
        self._val_history = []
