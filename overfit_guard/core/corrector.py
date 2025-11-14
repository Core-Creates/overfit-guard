"""Base corrector class for applying overfitting corrections."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum


class CorrectionType(Enum):
    """Types of corrections that can be applied."""
    REGULARIZATION = "regularization"
    AUGMENTATION = "augmentation"
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"
    EARLY_STOPPING = "early_stopping"
    OTHER = "other"


class CorrectionResult:
    """Result of applying a correction."""

    def __init__(
        self,
        success: bool,
        correction_type: CorrectionType,
        actions_taken: List[str],
        parameters_changed: Dict[str, Any],
        message: str = ""
    ):
        """
        Initialize correction result.

        Args:
            success: Whether correction was successfully applied
            correction_type: Type of correction applied
            actions_taken: List of actions performed
            parameters_changed: Dictionary of changed parameters
            message: Human-readable message
        """
        self.success = success
        self.correction_type = correction_type
        self.actions_taken = actions_taken
        self.parameters_changed = parameters_changed
        self.message = message

    def __repr__(self) -> str:
        return (
            f"CorrectionResult(success={self.success}, "
            f"type={self.correction_type.value}, "
            f"actions={len(self.actions_taken)})"
        )


class BaseCorrector(ABC):
    """Base class for all overfitting correctors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize corrector with configuration.

        Args:
            config: Configuration dictionary for the corrector
        """
        self.config = config or {}
        self.history: List[CorrectionResult] = []
        self._is_enabled = True
        self._correction_count = 0

    @abstractmethod
    def correct(
        self,
        model: Any,
        detection_result: Any,
        **kwargs
    ) -> CorrectionResult:
        """
        Apply correction to the model.

        Args:
            model: The model to correct (type depends on framework)
            detection_result: DetectionResult that triggered this correction
            **kwargs: Additional corrector-specific arguments

        Returns:
            CorrectionResult object
        """
        pass

    @abstractmethod
    def can_correct(self, model: Any) -> bool:
        """
        Check if this corrector can be applied to the given model.

        Args:
            model: The model to check

        Returns:
            True if correction can be applied, False otherwise
        """
        pass

    def reset(self) -> None:
        """Reset corrector state."""
        self.history = []
        self._correction_count = 0

    def enable(self) -> None:
        """Enable the corrector."""
        self._is_enabled = True

    def disable(self) -> None:
        """Disable the corrector."""
        self._is_enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if corrector is enabled."""
        return self._is_enabled

    @property
    def correction_count(self) -> int:
        """Get number of corrections applied."""
        return self._correction_count

    def get_history(self) -> List[CorrectionResult]:
        """Get correction history."""
        return self.history

    def _add_to_history(self, result: CorrectionResult) -> None:
        """Add correction result to history."""
        if result.success:
            self._correction_count += 1
        self.history.append(result)

        # Keep only last N results
        max_history = self.config.get('max_history', 100)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
