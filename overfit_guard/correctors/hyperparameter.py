"""Hyperparameter tuning corrector for overfitting."""

from typing import Dict, Any, List
from overfit_guard.core.corrector import BaseCorrector, CorrectionResult, CorrectionType
from overfit_guard.core.detector import DetectionResult, OverfitSeverity


class HyperparameterCorrector(BaseCorrector):
    """
    Adjusts hyperparameters to reduce overfitting.

    Modifies:
    - Learning rate
    - Batch size
    - Optimizer parameters
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize hyperparameter corrector.

        Config options:
            - enable_lr_adjustment (bool): Adjust learning rate (default: True)
            - enable_batch_size_adjustment (bool): Adjust batch size (default: True)
            - lr_reduction_factor (float): Factor to reduce LR (default: 0.5)
            - batch_size_increase_factor (int): Factor to increase batch (default: 2)
            - min_lr (float): Minimum learning rate (default: 1e-6)
            - max_batch_size (int): Maximum batch size (default: 512)
        """
        super().__init__(config)

        self.enable_lr_adjustment = self.config.get('enable_lr_adjustment', True)
        self.enable_batch_size_adjustment = self.config.get(
            'enable_batch_size_adjustment', True
        )
        self.lr_reduction_factor = self.config.get('lr_reduction_factor', 0.5)
        self.batch_size_increase_factor = self.config.get('batch_size_increase_factor', 2)
        self.min_lr = self.config.get('min_lr', 1e-6)
        self.max_batch_size = self.config.get('max_batch_size', 512)

        self._current_lr: float = 0.001  # Default, should be overridden
        self._current_batch_size: int = 32  # Default, should be overridden
        self._lr_reductions = 0
        self._batch_increases = 0

    def correct(
        self,
        model: Any,
        detection_result: DetectionResult,
        **kwargs
    ) -> CorrectionResult:
        """Apply hyperparameter corrections."""

        actions_taken: List[str] = []
        parameters_changed: Dict[str, Any] = {}
        success = True

        severity = detection_result.severity

        # Get current hyperparameters from kwargs if provided
        current_lr = kwargs.get('current_lr', self._current_lr)
        current_batch_size = kwargs.get('current_batch_size', self._current_batch_size)

        # Adjust learning rate
        if self.enable_lr_adjustment:
            # Reduce learning rate based on severity
            reduction_factor = self.lr_reduction_factor
            if severity == OverfitSeverity.SEVERE:
                reduction_factor = self.lr_reduction_factor ** 2  # More aggressive

            new_lr = max(self.min_lr, current_lr * reduction_factor)

            if new_lr != current_lr:
                self._current_lr = new_lr
                self._lr_reductions += 1
                parameters_changed['learning_rate'] = new_lr
                actions_taken.append(
                    f"Reduced learning rate from {current_lr:.6f} to {new_lr:.6f}"
                )

        # Adjust batch size
        if self.enable_batch_size_adjustment:
            # Increasing batch size can reduce overfitting (acts as regularization)
            if severity.value >= OverfitSeverity.MODERATE.value:
                new_batch_size = min(
                    self.max_batch_size,
                    current_batch_size * self.batch_size_increase_factor
                )

                if new_batch_size != current_batch_size:
                    self._current_batch_size = new_batch_size
                    self._batch_increases += 1
                    parameters_changed['batch_size'] = new_batch_size
                    actions_taken.append(
                        f"Increased batch size from {current_batch_size} to {new_batch_size}"
                    )

        # Additional optimizer adjustments
        optimizer_params = self._generate_optimizer_recommendations(severity)
        if optimizer_params:
            parameters_changed['optimizer_params'] = optimizer_params
            actions_taken.append(f"Recommended optimizer adjustments: {optimizer_params}")

        message = f"Applied {len(actions_taken)} hyperparameter corrections"

        result = CorrectionResult(
            success=success,
            correction_type=CorrectionType.HYPERPARAMETER,
            actions_taken=actions_taken,
            parameters_changed=parameters_changed,
            message=message
        )

        self._add_to_history(result)
        return result

    def can_correct(self, model: Any) -> bool:
        """Check if hyperparameter corrections can be applied."""
        # Hyperparameter changes are always possible
        return True

    def _generate_optimizer_recommendations(
        self,
        severity: OverfitSeverity
    ) -> Dict[str, Any]:
        """Generate optimizer parameter recommendations."""
        recommendations = {}

        if severity == OverfitSeverity.SEVERE:
            recommendations['momentum'] = 0.9
            recommendations['beta1'] = 0.9  # For Adam
            recommendations['beta2'] = 0.999
            recommendations['weight_decay'] = 0.01
        elif severity == OverfitSeverity.MODERATE:
            recommendations['weight_decay'] = 0.001
            recommendations['momentum'] = 0.9

        return recommendations

    def reset(self) -> None:
        """Reset corrector state."""
        super().reset()
        self._current_lr = 0.001
        self._current_batch_size = 32
        self._lr_reductions = 0
        self._batch_increases = 0

    def get_parameters(self) -> Dict[str, Any]:
        """Get current hyperparameter state."""
        return {
            'current_lr': self._current_lr,
            'current_batch_size': self._current_batch_size,
            'lr_reductions': self._lr_reductions,
            'batch_increases': self._batch_increases
        }

    def set_current_hyperparameters(self, lr: float = None, batch_size: int = None) -> None:
        """Set current hyperparameters for tracking."""
        if lr is not None:
            self._current_lr = lr
        if batch_size is not None:
            self._current_batch_size = batch_size
