"""Architecture modification corrector for overfitting."""

from typing import Dict, Any, List
from overfit_guard.core.corrector import BaseCorrector, CorrectionResult, CorrectionType
from overfit_guard.core.detector import DetectionResult, OverfitSeverity


class ArchitectureCorrector(BaseCorrector):
    """
    Modifies model architecture to reduce overfitting.

    Strategies:
    - Reduce layer dimensions
    - Prune layers
    - Simplify architecture
    - Add normalization layers
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize architecture corrector.

        Config options:
            - enable_pruning (bool): Enable layer pruning (default: True)
            - enable_dimension_reduction (bool): Reduce layer dimensions (default: True)
            - enable_normalization (bool): Add batch/layer norm (default: True)
            - dimension_reduction_factor (float): Factor to reduce dims (default: 0.8)
            - min_layer_size (int): Minimum layer dimension (default: 32)
        """
        super().__init__(config)

        self.enable_pruning = self.config.get('enable_pruning', True)
        self.enable_dimension_reduction = self.config.get('enable_dimension_reduction', True)
        self.enable_normalization = self.config.get('enable_normalization', True)
        self.dimension_reduction_factor = self.config.get('dimension_reduction_factor', 0.8)
        self.min_layer_size = self.config.get('min_layer_size', 32)

        self._pruning_history: List[str] = []
        self._dimension_changes: Dict[str, int] = {}

    def correct(
        self,
        model: Any,
        detection_result: DetectionResult,
        **kwargs
    ) -> CorrectionResult:
        """Apply architecture modifications to reduce overfitting."""

        actions_taken: List[str] = []
        parameters_changed: Dict[str, Any] = {}
        success = True

        severity = detection_result.severity

        # Architecture modifications are significant, so only apply for moderate+ severity
        if severity.value < OverfitSeverity.MODERATE.value:
            return CorrectionResult(
                success=False,
                correction_type=CorrectionType.ARCHITECTURE,
                actions_taken=["Severity too low for architecture changes"],
                parameters_changed={},
                message="Architecture changes require moderate or higher severity"
            )

        # Recommendations for architecture changes
        # (Actual implementation would be framework-specific)

        if self.enable_dimension_reduction:
            reduction_factor = self.dimension_reduction_factor
            if severity == OverfitSeverity.SEVERE:
                reduction_factor = 0.7  # More aggressive

            parameters_changed['dimension_reduction_factor'] = reduction_factor
            actions_taken.append(
                f"Recommend reducing layer dimensions by factor {reduction_factor:.2f}"
            )

        if self.enable_normalization:
            actions_taken.append(
                "Recommend adding batch normalization or layer normalization"
            )
            parameters_changed['add_normalization'] = True

        if self.enable_pruning and severity == OverfitSeverity.SEVERE:
            actions_taken.append(
                "Recommend pruning unnecessary layers or connections"
            )
            parameters_changed['enable_pruning'] = True

        # Provide specific architectural recommendations
        recommendations = self._generate_architecture_recommendations(severity)
        parameters_changed['recommendations'] = recommendations
        actions_taken.extend(recommendations)

        message = (
            f"Generated {len(recommendations)} architecture modification "
            f"recommendations (severity: {severity.name})"
        )

        result = CorrectionResult(
            success=success,
            correction_type=CorrectionType.ARCHITECTURE,
            actions_taken=actions_taken,
            parameters_changed=parameters_changed,
            message=message
        )

        self._add_to_history(result)
        return result

    def can_correct(self, model: Any) -> bool:
        """
        Check if architecture corrections can be applied.

        Note: Architecture changes typically require model rebuilding,
        so this is more of a recommendation system.
        """
        return True

    def _generate_architecture_recommendations(
        self,
        severity: OverfitSeverity
    ) -> List[str]:
        """Generate specific architecture recommendations based on severity."""
        recommendations = []

        if severity == OverfitSeverity.SEVERE:
            recommendations.extend([
                "Consider reducing model depth (remove 1-2 layers)",
                "Reduce hidden layer dimensions by 30-50%",
                "Replace large fully-connected layers with smaller ones",
                "Add strong regularization layers (BatchNorm + Dropout)"
            ])
        elif severity == OverfitSeverity.MODERATE:
            recommendations.extend([
                "Reduce layer dimensions by 20-30%",
                "Add batch normalization after each layer",
                "Consider using residual connections for better generalization"
            ])
        else:  # MILD
            recommendations.extend([
                "Add batch normalization to stabilize training",
                "Consider slight dimension reduction (10-20%)"
            ])

        return recommendations

    def reset(self) -> None:
        """Reset corrector state."""
        super().reset()
        self._pruning_history = []
        self._dimension_changes = {}

    def get_parameters(self) -> Dict[str, Any]:
        """Get current architecture modification state."""
        return {
            'pruning_history': self._pruning_history,
            'dimension_changes': self._dimension_changes,
            'dimension_reduction_factor': self.dimension_reduction_factor
        }
