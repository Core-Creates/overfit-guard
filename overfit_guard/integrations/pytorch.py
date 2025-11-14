"""PyTorch integration for overfit-guard."""

from typing import Dict, Any, Optional, Callable
import logging

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from overfit_guard.core.monitor import OverfitMonitor


logger = logging.getLogger(__name__)


class PyTorchCallback:
    """
    PyTorch callback for overfit monitoring.

    Usage:
        monitor = OverfitMonitor(...)
        callback = PyTorchCallback(monitor)

        for epoch in range(num_epochs):
            # Training loop
            train_loss = train_one_epoch(...)
            val_loss = validate(...)

            # Check for overfitting
            callback.on_epoch_end(
                epoch,
                model,
                {'loss': train_loss},
                {'loss': val_loss}
            )
    """

    def __init__(
        self,
        monitor: OverfitMonitor,
        optimizer: Optional[Any] = None,
        verbose: bool = True
    ):
        """
        Initialize PyTorch callback.

        Args:
            monitor: OverfitMonitor instance
            optimizer: PyTorch optimizer (for applying hyperparameter corrections)
            verbose: Print messages
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

        self.monitor = monitor
        self.optimizer = optimizer
        self.verbose = verbose
        self.should_stop = False

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            model: PyTorch model
            train_metrics: Training metrics
            val_metrics: Validation metrics
            **kwargs: Additional arguments

        Returns:
            Detection results dictionary
        """
        # Check for overfitting
        results = self.monitor.check(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=epoch,
            model=model,
            **kwargs
        )

        # Apply corrections if any
        if results['corrections']:
            self._apply_corrections(model, results['corrections'])

        # Check for early stopping
        for correction in results['corrections']:
            if correction['result'].parameters_changed.get('should_stop', False):
                self.should_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")

        if self.verbose and results['is_overfitting']:
            logger.warning(
                f"Epoch {epoch}: Overfitting detected "
                f"(severity: {results['max_severity'].name})"
            )

        return results

    def _apply_corrections(
        self,
        model: nn.Module,
        corrections: list
    ) -> None:
        """Apply corrections to model and optimizer."""
        for correction in corrections:
            result = correction['result']
            params = result.parameters_changed

            # Apply learning rate changes
            if 'learning_rate' in params and self.optimizer is not None:
                new_lr = params['learning_rate']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                if self.verbose:
                    logger.info(f"Updated learning rate to {new_lr:.6f}")

            # Apply weight decay changes
            if 'weight_decay' in params and self.optimizer is not None:
                new_wd = params['weight_decay']
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = new_wd
                if self.verbose:
                    logger.info(f"Updated weight decay to {new_wd:.6f}")

            # Note: Dropout and architecture changes require model modification
            # These are handled as recommendations


class OverfitGuardHook:
    """
    PyTorch hook for automatic overfitting detection during training.

    This uses PyTorch hooks to automatically monitor training.
    """

    def __init__(
        self,
        model: nn.Module,
        monitor: OverfitMonitor,
        optimizer: Optional[Any] = None
    ):
        """
        Initialize hook.

        Args:
            model: PyTorch model
            monitor: OverfitMonitor instance
            optimizer: PyTorch optimizer
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        self.model = model
        self.monitor = monitor
        self.optimizer = optimizer
        self.callback = PyTorchCallback(monitor, optimizer)

    def __call__(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        **kwargs
    ) -> Dict[str, Any]:
        """Call the callback."""
        return self.callback.on_epoch_end(
            epoch,
            self.model,
            train_metrics,
            val_metrics,
            **kwargs
        )


def create_pytorch_monitor(
    model: nn.Module,
    optimizer: Any = None,
    config: Optional[Dict[str, Any]] = None,
    auto_correct: bool = False
) -> PyTorchCallback:
    """
    Create a PyTorch-integrated overfit monitor.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        auto_correct: Enable automatic corrections

    Returns:
        PyTorchCallback instance
    """
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

    return PyTorchCallback(monitor, optimizer)
