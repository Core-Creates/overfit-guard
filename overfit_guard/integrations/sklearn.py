"""Scikit-learn integration for overfit-guard."""

from typing import Dict, Any, Optional, Callable
import logging

try:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from overfit_guard.core.monitor import OverfitMonitor


logger = logging.getLogger(__name__)


class SklearnMonitorWrapper:
    """
    Wrapper for scikit-learn models with overfit monitoring.

    Usage:
        from overfit_guard.integrations.sklearn import create_sklearn_monitor

        monitor = create_sklearn_monitor()

        # During iterative training (e.g., SGDClassifier, MLPClassifier)
        for epoch in range(n_epochs):
            model.partial_fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            monitor.check_iteration(
                epoch,
                {'accuracy': train_score},
                {'accuracy': val_score},
                model
            )
    """

    def __init__(
        self,
        monitor: OverfitMonitor,
        metric_name: str = 'score',
        higher_is_better: bool = True,
        verbose: bool = True
    ):
        """
        Initialize sklearn monitor wrapper.

        Args:
            monitor: OverfitMonitor instance
            metric_name: Name of the metric being tracked
            higher_is_better: Whether higher scores are better
            verbose: Print messages
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install with: pip install scikit-learn"
            )

        self.monitor = monitor
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.verbose = verbose
        self.should_stop = False

    def check_iteration(
        self,
        iteration: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: Any = None
    ) -> Dict[str, Any]:
        """
        Check for overfitting at a given iteration.

        Args:
            iteration: Current iteration/epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            model: Scikit-learn model

        Returns:
            Detection results
        """
        # For sklearn, we often want to invert scores to losses for consistency
        # (overfitting detectors expect lower=better by default)
        if self.higher_is_better:
            train_metrics_converted = {
                k: 1.0 - v for k, v in train_metrics.items()
            }
            val_metrics_converted = {
                k: 1.0 - v for k, v in val_metrics.items()
            }
        else:
            train_metrics_converted = train_metrics
            val_metrics_converted = val_metrics

        # Check for overfitting
        results = self.monitor.check(
            train_metrics=train_metrics_converted,
            val_metrics=val_metrics_converted,
            epoch=iteration,
            model=model
        )

        # Check for early stopping
        for correction in results.get('corrections', []):
            if correction['result'].parameters_changed.get('should_stop', False):
                self.should_stop = True
                if self.verbose:
                    logger.info("Early stopping recommended by overfit-guard")

        if self.verbose and results['is_overfitting']:
            logger.warning(
                f"Iteration {iteration}: Overfitting detected "
                f"(severity: {results['max_severity'].name})"
            )

        return results

    def check_cross_validation(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check for overfitting using cross-validation.

        Args:
            model: Scikit-learn model
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Detection results with CV analysis
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        # Get training score
        model.fit(X, y)
        train_score = model.score(X, y)

        # Convert to loss format if needed
        if self.higher_is_better:
            train_loss = 1.0 - train_score
            val_loss = 1.0 - cv_scores.mean()
        else:
            train_loss = train_score
            val_loss = cv_scores.mean()

        # Check using CV detector
        results = self.monitor.check(
            train_metrics={'loss': train_loss},
            val_metrics={'loss': val_loss},
            epoch=0,
            model=model,
            cv_scores=cv_scores.tolist()
        )

        if self.verbose and results['is_overfitting']:
            logger.warning(
                f"Cross-validation: Overfitting detected "
                f"(severity: {results['max_severity'].name})"
            )

        return results


def create_sklearn_monitor(
    config: Optional[Dict[str, Any]] = None,
    metric_name: str = 'score',
    higher_is_better: bool = True,
    verbose: bool = True
) -> SklearnMonitorWrapper:
    """
    Create a scikit-learn-integrated overfit monitor.

    Args:
        config: Configuration dictionary
        metric_name: Name of the metric being tracked
        higher_is_better: Whether higher scores are better
        verbose: Print messages

    Returns:
        SklearnMonitorWrapper instance

    Example:
        monitor = create_sklearn_monitor()

        for epoch in range(100):
            model.partial_fit(X_train, y_train)
            monitor.check_iteration(
                epoch,
                {'accuracy': model.score(X_train, y_train)},
                {'accuracy': model.score(X_val, y_val)},
                model
            )
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Install with: pip install scikit-learn"
        )

    from overfit_guard.detectors.gap_detector import TrainValGapDetector
    from overfit_guard.detectors.curve_analyzer import LearningCurveAnalyzer
    from overfit_guard.detectors.cv_detector import CrossValidationDetector
    from overfit_guard.correctors.regularization import RegularizationCorrector
    from overfit_guard.utils.config import Config

    # Load configuration
    cfg = Config(config)

    # Create detectors
    detectors = []
    if cfg.is_detector_enabled('gap_detector'):
        detectors.append(TrainValGapDetector(cfg.get_detector_config('gap_detector')))
    if cfg.is_detector_enabled('curve_analyzer'):
        detectors.append(LearningCurveAnalyzer(cfg.get_detector_config('curve_analyzer')))
    if cfg.is_detector_enabled('cv_detector'):
        detectors.append(CrossValidationDetector(cfg.get_detector_config('cv_detector')))

    # Create correctors (limited for sklearn)
    correctors = []
    if cfg.is_corrector_enabled('regularization'):
        correctors.append(RegularizationCorrector(cfg.get_corrector_config('regularization')))

    # Create monitor
    monitor_config = {
        'auto_correct': False,  # Manual for sklearn
        'log_level': cfg.get('log_level')
    }

    monitor = OverfitMonitor(
        detectors=detectors,
        correctors=correctors,
        config=monitor_config
    )

    return SklearnMonitorWrapper(
        monitor,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        verbose=verbose
    )
