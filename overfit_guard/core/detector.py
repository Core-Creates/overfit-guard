"""Base detector class for overfitting detection."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum


class OverfitSeverity(Enum):
    """Severity levels for overfitting detection."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class DetectionResult:
    """Result of an overfitting detection check."""

    def __init__(
        self,
        is_overfitting: bool,
        severity: OverfitSeverity,
        confidence: float,
        metrics: Dict[str, float],
        message: str = "",
        recommendations: Optional[List[str]] = None
    ):
        """
        Initialize detection result.

        Args:
            is_overfitting: Whether overfitting is detected
            severity: Severity level of overfitting
            confidence: Confidence score (0-1)
            metrics: Dictionary of relevant metrics
            message: Human-readable message
            recommendations: List of recommended actions
        """
        self.is_overfitting = is_overfitting
        self.severity = severity
        self.confidence = confidence
        self.metrics = metrics
        self.message = message
        self.recommendations = recommendations or []

    def __repr__(self) -> str:
        return (
            f"DetectionResult(is_overfitting={self.is_overfitting}, "
            f"severity={self.severity.name}, confidence={self.confidence:.3f})"
        )


class BaseDetector(ABC):
    """Base class for all overfitting detectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize detector with configuration.

        Args:
            config: Configuration dictionary for the detector
        """
        self.config = config or {}
        self.history: List[DetectionResult] = []
        self._is_enabled = True

    @abstractmethod
    def detect(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        **kwargs
    ) -> DetectionResult:
        """
        Detect overfitting based on provided metrics.

        Args:
            train_metrics: Training metrics (e.g., {'loss': 0.1, 'accuracy': 0.95})
            val_metrics: Validation metrics
            epoch: Current epoch/iteration number
            **kwargs: Additional detector-specific arguments

        Returns:
            DetectionResult object
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state (e.g., clear history)."""
        pass

    def enable(self) -> None:
        """Enable the detector."""
        self._is_enabled = True

    def disable(self) -> None:
        """Disable the detector."""
        self._is_enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self._is_enabled

    def get_history(self) -> List[DetectionResult]:
        """Get detection history."""
        return self.history

    def _add_to_history(self, result: DetectionResult) -> None:
        """Add detection result to history."""
        self.history.append(result)

        # Keep only last N results to prevent memory issues
        max_history = self.config.get('max_history', 1000)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
