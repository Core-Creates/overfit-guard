"""Data augmentation corrector for overfitting."""

from typing import Dict, Any, List, Callable, Optional
from overfit_guard.core.corrector import BaseCorrector, CorrectionResult, CorrectionType
from overfit_guard.core.detector import DetectionResult, OverfitSeverity


class AugmentationCorrector(BaseCorrector):
    """
    Applies data augmentation to reduce overfitting.

    This corrector can:
    - Enable/increase augmentation strength
    - Apply synthetic data generation
    - Adjust augmentation parameters
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize augmentation corrector.

        Config options:
            - augmentation_strategies (List[str]): Available strategies
            - initial_strength (float): Starting augmentation strength (default: 0.1)
            - strength_increment (float): Strength increase per correction (default: 0.1)
            - max_strength (float): Maximum augmentation strength (default: 0.8)
            - custom_augmentation_fn (Callable): Custom augmentation function
        """
        super().__init__(config)

        self.augmentation_strategies = self.config.get('augmentation_strategies', [
            'rotation', 'flip', 'crop', 'noise', 'mixup'
        ])
        self.initial_strength = self.config.get('initial_strength', 0.1)
        self.strength_increment = self.config.get('strength_increment', 0.1)
        self.max_strength = self.config.get('max_strength', 0.8)
        self.custom_augmentation_fn: Optional[Callable] = self.config.get(
            'custom_augmentation_fn'
        )

        self._current_strength = self.initial_strength
        self._active_strategies: List[str] = []

    def correct(
        self,
        model: Any,
        detection_result: DetectionResult,
        **kwargs
    ) -> CorrectionResult:
        """Apply data augmentation corrections."""

        actions_taken: List[str] = []
        parameters_changed: Dict[str, Any] = {}
        success = True

        severity = detection_result.severity

        # Increase augmentation strength
        if severity.value >= OverfitSeverity.MODERATE.value:
            increment = self.strength_increment * (
                2.0 if severity == OverfitSeverity.SEVERE else 1.0
            )
            self._current_strength = min(
                self.max_strength,
                self._current_strength + increment
            )

            parameters_changed['augmentation_strength'] = self._current_strength
            actions_taken.append(
                f"Increased augmentation strength to {self._current_strength:.2f}"
            )

        # Enable additional augmentation strategies
        if len(self._active_strategies) < len(self.augmentation_strategies):
            # Add strategies based on severity
            num_to_add = 1
            if severity == OverfitSeverity.SEVERE:
                num_to_add = 2

            for strategy in self.augmentation_strategies:
                if strategy not in self._active_strategies and num_to_add > 0:
                    self._active_strategies.append(strategy)
                    actions_taken.append(f"Enabled augmentation: {strategy}")
                    num_to_add -= 1

            parameters_changed['active_strategies'] = self._active_strategies

        # Apply custom augmentation if provided
        if self.custom_augmentation_fn is not None:
            try:
                self.custom_augmentation_fn(model, self._current_strength)
                actions_taken.append("Applied custom augmentation function")
            except Exception as e:
                success = False
                actions_taken.append(f"Custom augmentation failed: {str(e)}")

        # Provide augmentation configuration
        parameters_changed['augmentation_config'] = {
            'strength': self._current_strength,
            'strategies': self._active_strategies,
        }

        message = (
            f"Applied data augmentation: {len(self._active_strategies)} strategies "
            f"at strength {self._current_strength:.2f}"
        )

        result = CorrectionResult(
            success=success,
            correction_type=CorrectionType.AUGMENTATION,
            actions_taken=actions_taken,
            parameters_changed=parameters_changed,
            message=message
        )

        self._add_to_history(result)
        return result

    def can_correct(self, model: Any) -> bool:
        """Check if augmentation can be applied."""
        # Can always apply augmentation, though effect depends on user integration
        return True

    def reset(self) -> None:
        """Reset corrector state."""
        super().reset()
        self._current_strength = self.initial_strength
        self._active_strategies = []

    def get_parameters(self) -> Dict[str, Any]:
        """Get current augmentation parameters."""
        return {
            'strength': self._current_strength,
            'active_strategies': self._active_strategies,
            'available_strategies': self.augmentation_strategies
        }
