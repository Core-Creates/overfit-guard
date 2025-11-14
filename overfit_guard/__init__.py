"""
Overfit Guard - A micro-library to detect and correct overfitting in ML models.
"""

__version__ = "0.1.0"

from overfit_guard.core.monitor import OverfitMonitor
from overfit_guard.core.detector import BaseDetector
from overfit_guard.core.corrector import BaseCorrector

# Import detectors
from overfit_guard.detectors.gap_detector import TrainValGapDetector
from overfit_guard.detectors.curve_analyzer import LearningCurveAnalyzer
from overfit_guard.detectors.cv_detector import CrossValidationDetector
from overfit_guard.detectors.statistical import StatisticalDetector

# Import correctors
from overfit_guard.correctors.regularization import RegularizationCorrector
from overfit_guard.correctors.augmentation import AugmentationCorrector
from overfit_guard.correctors.architecture import ArchitectureCorrector
from overfit_guard.correctors.hyperparameter import HyperparameterCorrector

__all__ = [
    "OverfitMonitor",
    "BaseDetector",
    "BaseCorrector",
    # Detectors
    "TrainValGapDetector",
    "LearningCurveAnalyzer",
    "CrossValidationDetector",
    "StatisticalDetector",
    # Correctors
    "RegularizationCorrector",
    "AugmentationCorrector",
    "ArchitectureCorrector",
    "HyperparameterCorrector",
]
