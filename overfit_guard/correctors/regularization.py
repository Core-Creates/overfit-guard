"""Regularization-based overfitting corrector."""

from typing import Dict, Any, List
from overfit_guard.core.corrector import BaseCorrector, CorrectionResult, CorrectionType
from overfit_guard.core.detector import DetectionResult, OverfitSeverity


class RegularizationCorrector(BaseCorrector):
    """
    Applies regularization techniques to reduce overfitting.

    Supports:
    - L1/L2 weight regularization
    - Dropout adjustment
    - Early stopping
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize regularization corrector.

        Config options:
            - enable_weight_decay (bool): Enable L2 regularization (default: True)
            - enable_dropout (bool): Enable dropout adjustment (default: True)
            - enable_early_stopping (bool): Enable early stopping (default: True)
            - weight_decay_step (float): Increment for weight decay (default: 0.001)
            - dropout_step (float): Increment for dropout (default: 0.1)
            - early_stop_patience (int): Epochs to wait before stopping (default: 10)
        """
        super().__init__(config)

        self.enable_weight_decay = self.config.get('enable_weight_decay', True)
        self.enable_dropout = self.config.get('enable_dropout', True)
        self.enable_early_stopping = self.config.get('enable_early_stopping', True)

        self.weight_decay_step = self.config.get('weight_decay_step', 0.001)
        self.dropout_step = self.config.get('dropout_step', 0.1)
        self.early_stop_patience = self.config.get('early_stop_patience', 10)

        self._current_weight_decay = 0.0
        self._current_dropout = 0.0
        self._no_improvement_count = 0
        self._best_val_metric = float('inf')

    def correct(
        self,
        model: Any,
        detection_result: DetectionResult,
        **kwargs
    ) -> CorrectionResult:
        """Apply regularization corrections to the model."""

        actions_taken: List[str] = []
        parameters_changed: Dict[str, Any] = {}
        success = True

        # Determine correction strength based on severity
        severity = detection_result.severity
        multiplier = self._get_severity_multiplier(severity)

        # Apply weight decay
        if self.enable_weight_decay:
            new_weight_decay = self._current_weight_decay + (
                self.weight_decay_step * multiplier
            )
            self._current_weight_decay = min(0.1, new_weight_decay)  # Cap at 0.1

            parameters_changed['weight_decay'] = self._current_weight_decay
            actions_taken.append(
                f"Increased weight decay to {self._current_weight_decay:.4f}"
            )

        # Apply dropout (framework-specific implementation needed)
        if self.enable_dropout:
            new_dropout = self._current_dropout + (self.dropout_step * multiplier)
            self._current_dropout = min(0.5, new_dropout)  # Cap at 0.5

            parameters_changed['dropout'] = self._current_dropout
            actions_taken.append(
                f"Increased dropout to {self._current_dropout:.2f}"
            )

        # Check for early stopping
        if self.enable_early_stopping:
            val_metric = detection_result.metrics.get('val_value', float('inf'))

            if val_metric < self._best_val_metric:
                self._best_val_metric = val_metric
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            if self._no_improvement_count >= self.early_stop_patience:
                parameters_changed['should_stop'] = True
                actions_taken.append(
                    f"Early stopping triggered (patience: {self.early_stop_patience})"
                )

        message = f"Applied {len(actions_taken)} regularization corrections"

        result = CorrectionResult(
            success=success,
            correction_type=CorrectionType.REGULARIZATION,
            actions_taken=actions_taken,
            parameters_changed=parameters_changed,
            message=message
        )

        self._add_to_history(result)
        return result

    def can_correct(self, model: Any) -> bool:
        """Check if regularization can be applied (always possible)."""
        return True

    def _get_severity_multiplier(self, severity: OverfitSeverity) -> float:
        """Get correction strength multiplier based on severity."""
        if severity == OverfitSeverity.SEVERE:
            return 2.0
        elif severity == OverfitSeverity.MODERATE:
            return 1.5
        elif severity == OverfitSeverity.MILD:
            return 1.0
        else:
            return 0.5

    def reset(self) -> None:
        """Reset corrector state."""
        super().reset()
        self._current_weight_decay = 0.0
        self._current_dropout = 0.0
        self._no_improvement_count = 0
        self._best_val_metric = float('inf')

    def get_parameters(self) -> Dict[str, Any]:
        """Get current regularization parameters."""
        return {
            'weight_decay': self._current_weight_decay,
            'dropout': self._current_dropout,
            'no_improvement_count': self._no_improvement_count,
            'best_val_metric': self._best_val_metric
        }
