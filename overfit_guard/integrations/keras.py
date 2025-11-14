"""TensorFlow/Keras integration for overfit-guard."""

from typing import Dict, Any, Optional
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from overfit_guard.core.monitor import OverfitMonitor


logger = logging.getLogger(__name__)


if TF_AVAILABLE:
    class OverfitGuardCallback(keras.callbacks.Callback):
        """
        Keras callback for overfit monitoring.

        Usage:
            from overfit_guard.integrations.keras import create_keras_monitor

            monitor_callback = create_keras_monitor(auto_correct=True)

            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                callbacks=[monitor_callback],
                epochs=100
            )
        """

        def __init__(
            self,
            monitor: OverfitMonitor,
            train_metric_names: Optional[list] = None,
            val_metric_names: Optional[list] = None,
            verbose: bool = True
        ):
            """
            Initialize Keras callback.

            Args:
                monitor: OverfitMonitor instance
                train_metric_names: Training metric names to monitor
                val_metric_names: Validation metric names to monitor
                verbose: Print messages
            """
            super().__init__()
            self.monitor_obj = monitor
            self.train_metric_names = train_metric_names or ['loss']
            self.val_metric_names = val_metric_names or ['val_loss']
            self.verbose = verbose
            self.should_stop = False

        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
            """Called at the end of each epoch."""
            if logs is None:
                return

            # Extract train and validation metrics
            train_metrics = {}
            val_metrics = {}

            for key, value in logs.items():
                if key.startswith('val_'):
                    # Validation metric
                    metric_name = key[4:]  # Remove 'val_' prefix
                    val_metrics[metric_name] = float(value)
                else:
                    # Training metric
                    train_metrics[key] = float(value)

            # Ensure we have matching metrics
            if not train_metrics or not val_metrics:
                return

            # Check for overfitting
            results = self.monitor_obj.check(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
                model=self.model,
            )

            # Apply corrections
            if results['corrections']:
                self._apply_corrections(results['corrections'])

            # Check for early stopping
            for correction in results['corrections']:
                if correction['result'].parameters_changed.get('should_stop', False):
                    self.model.stop_training = True
                    self.should_stop = True
                    if self.verbose:
                        logger.info("Early stopping triggered by overfit-guard")

            if self.verbose and results['is_overfitting']:
                logger.warning(
                    f"Epoch {epoch}: Overfitting detected "
                    f"(severity: {results['max_severity'].name})"
                )

        def _apply_corrections(self, corrections: list) -> None:
            """Apply corrections to Keras model."""
            for correction in corrections:
                result = correction['result']
                params = result.parameters_changed

                # Apply learning rate changes
                if 'learning_rate' in params:
                    new_lr = params['learning_rate']
                    if hasattr(self.model.optimizer, 'learning_rate'):
                        keras.backend.set_value(
                            self.model.optimizer.learning_rate,
                            new_lr
                        )
                        if self.verbose:
                            logger.info(f"Updated learning rate to {new_lr:.6f}")

                # Note: Other corrections like dropout require model architecture changes


def create_keras_monitor(
    config: Optional[Dict[str, Any]] = None,
    auto_correct: bool = False,
    verbose: bool = True
) -> 'OverfitGuardCallback':
    """
    Create a Keras-integrated overfit monitor.

    Args:
        config: Configuration dictionary
        auto_correct: Enable automatic corrections
        verbose: Print messages

    Returns:
        OverfitGuardCallback instance

    Example:
        callback = create_keras_monitor(auto_correct=True)
        model.fit(X, y, validation_split=0.2, callbacks=[callback])
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. Install with: pip install tensorflow"
        )

    from overfit_guard.detectors.gap_detector import TrainValGapDetector
    from overfit_guard.detectors.curve_analyzer import LearningCurveAnalyzer
    from overfit_guard.correctors.regularization import RegularizationCorrector
    from overfit_guard.correctors.hyperparameter import HyperparameterCorrector
    from overfit_guard.utils.config import Config

    # Load configuration
    cfg = Config(config)

    # Create detectors
    detectors = []
    if cfg.is_detector_enabled('gap_detector'):
        detectors.append(TrainValGapDetector(cfg.get_detector_config('gap_detector')))
    if cfg.is_detector_enabled('curve_analyzer'):
        detectors.append(LearningCurveAnalyzer(cfg.get_detector_config('curve_analyzer')))

    # Create correctors
    correctors = []
    if cfg.is_corrector_enabled('regularization'):
        correctors.append(RegularizationCorrector(cfg.get_corrector_config('regularization')))
    if cfg.is_corrector_enabled('hyperparameter'):
        correctors.append(HyperparameterCorrector(cfg.get_corrector_config('hyperparameter')))

    # Create monitor
    monitor_config = {
        'auto_correct': auto_correct,
        'min_severity_for_correction': cfg.get('min_severity_for_correction'),
        'correction_cooldown': cfg.get('correction_cooldown'),
        'log_level': cfg.get('log_level')
    }

    monitor = OverfitMonitor(
        detectors=detectors,
        correctors=correctors,
        config=monitor_config
    )

    return OverfitGuardCallback(monitor, verbose=verbose)


else:
    # Placeholder when TensorFlow is not available
    class OverfitGuardCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TensorFlow is not installed. Install with: pip install tensorflow"
            )

    def create_keras_monitor(*args, **kwargs):
        raise ImportError(
            "TensorFlow is not installed. Install with: pip install tensorflow"
        )
