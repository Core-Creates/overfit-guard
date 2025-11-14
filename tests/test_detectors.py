"""Tests for detector implementations."""

import pytest
import numpy as np
from overfit_guard.detectors.gap_detector import TrainValGapDetector
from overfit_guard.detectors.curve_analyzer import LearningCurveAnalyzer
from overfit_guard.detectors.cv_detector import CrossValidationDetector
from overfit_guard.detectors.statistical import StatisticalDetector
from overfit_guard.core.detector import OverfitSeverity


class TestTrainValGapDetector:
    """Tests for TrainValGapDetector."""

    def test_no_overfitting(self):
        """Test when there's no overfitting."""
        detector = TrainValGapDetector()

        result = detector.detect(
            {'loss': 0.5},
            {'loss': 0.52},
            epoch=0
        )

        assert result.is_overfitting is False
        assert result.severity == OverfitSeverity.NONE

    def test_mild_overfitting(self):
        """Test mild overfitting detection."""
        detector = TrainValGapDetector({
            'gap_threshold_mild': 0.05
        })

        result = detector.detect(
            {'loss': 0.1},
            {'loss': 0.17},  # 70% gap
            epoch=0
        )

        assert result.is_overfitting is True
        assert result.severity == OverfitSeverity.MILD

    def test_severe_overfitting(self):
        """Test severe overfitting detection."""
        detector = TrainValGapDetector()

        result = detector.detect(
            {'loss': 0.1},
            {'loss': 0.35},  # 250% gap
            epoch=0
        )

        assert result.is_overfitting is True
        assert result.severity == OverfitSeverity.SEVERE

    def test_missing_metric(self):
        """Test handling of missing metrics."""
        detector = TrainValGapDetector()

        result = detector.detect(
            {'accuracy': 0.9},
            {'accuracy': 0.8},
            epoch=0
        )

        assert result.is_overfitting is False

    def test_reset(self):
        """Test detector reset."""
        detector = TrainValGapDetector()

        detector.detect({'loss': 0.1}, {'loss': 0.2}, epoch=0)
        assert len(detector.history) == 1

        detector.reset()
        assert len(detector.history) == 0


class TestLearningCurveAnalyzer:
    """Tests for LearningCurveAnalyzer."""

    def test_insufficient_data(self):
        """Test with insufficient epochs."""
        detector = LearningCurveAnalyzer({'min_epochs': 5})

        result = detector.detect(
            {'loss': 0.5},
            {'loss': 0.52},
            epoch=0
        )

        assert result.is_overfitting is False

    def test_diverging_curves(self):
        """Test detection of diverging curves."""
        detector = LearningCurveAnalyzer({'min_epochs': 3})

        # Simulate diverging curves
        for i in range(10):
            train_loss = 0.5 - i * 0.05  # Decreasing
            val_loss = 0.5 + i * 0.03    # Increasing

            result = detector.detect(
                {'loss': train_loss},
                {'loss': val_loss},
                epoch=i
            )

        # Should detect overfitting in later epochs
        assert result.is_overfitting is True

    def test_stable_curves(self):
        """Test with stable, non-overfitting curves."""
        detector = LearningCurveAnalyzer({'min_epochs': 3})

        # Both curves improving together
        for i in range(10):
            loss = 0.5 - i * 0.03

            result = detector.detect(
                {'loss': loss},
                {'loss': loss + 0.01},
                epoch=i
            )

        # Should not detect severe overfitting
        assert result.severity.value <= OverfitSeverity.MILD.value


class TestCrossValidationDetector:
    """Tests for CrossValidationDetector."""

    def test_low_variance(self):
        """Test with low CV variance (good)."""
        detector = CrossValidationDetector()

        cv_scores = [0.85, 0.86, 0.84, 0.85, 0.86]

        result = detector.detect(
            {'loss': 0.15},
            {'loss': 0.15},
            epoch=0,
            cv_scores=cv_scores
        )

        assert result.is_overfitting is False

    def test_high_variance(self):
        """Test with high CV variance (overfitting)."""
        detector = CrossValidationDetector({
            'variance_threshold_moderate': 0.10
        })

        cv_scores = [0.5, 0.7, 0.3, 0.8, 0.4]  # High variance

        result = detector.detect(
            {'loss': 0.15},
            {'loss': 0.15},
            epoch=0,
            cv_scores=cv_scores
        )

        assert result.is_overfitting is True


class TestStatisticalDetector:
    """Tests for StatisticalDetector."""

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        detector = StatisticalDetector({'min_samples': 10})

        for i in range(5):
            result = detector.detect(
                {'loss': 0.1},
                {'loss': 0.2},
                epoch=i
            )

        assert result.is_overfitting is False

    def test_significant_difference(self):
        """Test with statistically significant difference."""
        detector = StatisticalDetector({'min_samples': 10})

        # Simulate consistent gap
        for i in range(15):
            result = detector.detect(
                {'loss': 0.1 + np.random.normal(0, 0.01)},
                {'loss': 0.3 + np.random.normal(0, 0.01)},
                epoch=i
            )

        # Should detect overfitting with statistical significance
        assert result.is_overfitting is True
