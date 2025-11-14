"""Main monitoring class that orchestrates detectors and correctors."""

from typing import Dict, Any, List, Optional, Callable
import logging
from overfit_guard.core.detector import BaseDetector, DetectionResult, OverfitSeverity
from overfit_guard.core.corrector import BaseCorrector, CorrectionResult


logger = logging.getLogger(__name__)


class OverfitMonitor:
    """
    Main class for monitoring and correcting overfitting.

    This class orchestrates multiple detectors and correctors, managing
    the detection-correction pipeline.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseDetector]] = None,
        correctors: Optional[List[BaseCorrector]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the overfit monitor.

        Args:
            detectors: List of detector instances
            correctors: List of corrector instances
            config: Configuration dictionary
        """
        self.detectors = detectors or []
        self.correctors = correctors or []
        self.config = config or {}

        # Configuration
        self.auto_correct = self.config.get('auto_correct', False)
        self.min_severity = OverfitSeverity[
            self.config.get('min_severity_for_correction', 'MODERATE')
        ]
        self.correction_cooldown = self.config.get('correction_cooldown', 5)

        # State
        self._last_correction_epoch = -self.correction_cooldown
        self._detection_history: List[Dict[str, Any]] = []
        self._callbacks: Dict[str, List[Callable]] = {
            'on_detection': [],
            'on_correction': [],
            'on_overfitting': []
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )

    def add_detector(self, detector: BaseDetector) -> None:
        """Add a detector to the monitor."""
        self.detectors.append(detector)
        logger.info(f"Added detector: {detector.__class__.__name__}")

    def add_corrector(self, corrector: BaseCorrector) -> None:
        """Add a corrector to the monitor."""
        self.correctors.append(corrector)
        logger.info(f"Added corrector: {corrector.__class__.__name__}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for specific events.

        Args:
            event: Event name ('on_detection', 'on_correction', 'on_overfitting')
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def check(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        model: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check for overfitting using all enabled detectors.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch: Current epoch number
            model: Model object (required if auto_correct is True)
            **kwargs: Additional arguments passed to detectors

        Returns:
            Dictionary containing detection results and any corrections applied
        """
        results = {
            'epoch': epoch,
            'detections': [],
            'is_overfitting': False,
            'max_severity': OverfitSeverity.NONE,
            'corrections': []
        }

        # Run all enabled detectors
        for detector in self.detectors:
            if not detector.is_enabled:
                continue

            try:
                detection = detector.detect(train_metrics, val_metrics, epoch, **kwargs)
                results['detections'].append({
                    'detector': detector.__class__.__name__,
                    'result': detection
                })

                # Track if any detector found overfitting
                if detection.is_overfitting:
                    results['is_overfitting'] = True
                    if detection.severity.value > results['max_severity'].value:
                        results['max_severity'] = detection.severity

                # Trigger callbacks
                for callback in self._callbacks['on_detection']:
                    callback(detection)

            except Exception as e:
                logger.error(f"Error in detector {detector.__class__.__name__}: {e}")

        # Apply corrections if overfitting detected
        if results['is_overfitting']:
            logger.warning(
                f"Overfitting detected at epoch {epoch} "
                f"(severity: {results['max_severity'].name})"
            )

            # Trigger overfitting callbacks
            for callback in self._callbacks['on_overfitting']:
                callback(results)

            # Apply corrections if enabled and cooldown period has passed
            if self.auto_correct and model is not None:
                if epoch - self._last_correction_epoch >= self.correction_cooldown:
                    if results['max_severity'].value >= self.min_severity.value:
                        corrections = self._apply_corrections(model, results['detections'])
                        results['corrections'] = corrections
                        if corrections:
                            self._last_correction_epoch = epoch
                else:
                    logger.info(
                        f"Skipping correction (cooldown: {self.correction_cooldown} epochs)"
                    )

        # Store in history
        self._detection_history.append(results)

        return results

    def _apply_corrections(
        self,
        model: Any,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply corrections based on detection results.

        Args:
            model: Model to correct
            detections: List of detection results

        Returns:
            List of correction results
        """
        corrections = []

        for corrector in self.correctors:
            if not corrector.is_enabled:
                continue

            # Check if corrector can be applied
            if not corrector.can_correct(model):
                continue

            try:
                # Use the most severe detection result
                most_severe = max(
                    detections,
                    key=lambda d: d['result'].severity.value
                )

                correction = corrector.correct(model, most_severe['result'])
                corrections.append({
                    'corrector': corrector.__class__.__name__,
                    'result': correction
                })

                if correction.success:
                    logger.info(
                        f"Applied correction: {corrector.__class__.__name__} - "
                        f"{correction.message}"
                    )

                # Trigger callbacks
                for callback in self._callbacks['on_correction']:
                    callback(correction)

            except Exception as e:
                logger.error(f"Error in corrector {corrector.__class__.__name__}: {e}")

        return corrections

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring activity."""
        total_checks = len(self._detection_history)
        overfitting_count = sum(
            1 for h in self._detection_history if h['is_overfitting']
        )
        correction_count = sum(
            len(h['corrections']) for h in self._detection_history
        )

        return {
            'total_checks': total_checks,
            'overfitting_detected': overfitting_count,
            'corrections_applied': correction_count,
            'overfitting_rate': overfitting_count / total_checks if total_checks > 0 else 0,
            'active_detectors': len([d for d in self.detectors if d.is_enabled]),
            'active_correctors': len([c for c in self.correctors if c.is_enabled])
        }

    def reset(self) -> None:
        """Reset all detectors, correctors, and monitoring state."""
        for detector in self.detectors:
            detector.reset()
        for corrector in self.correctors:
            corrector.reset()
        self._detection_history = []
        self._last_correction_epoch = -self.correction_cooldown
        logger.info("Monitor reset complete")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full detection history."""
        return self._detection_history
