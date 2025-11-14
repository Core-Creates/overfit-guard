"""Tests for OverfitMonitor."""

import pytest
from overfit_guard.core.monitor import OverfitMonitor
from overfit_guard.detectors.gap_detector import TrainValGapDetector
from overfit_guard.correctors.regularization import RegularizationCorrector
from overfit_guard.core.detector import OverfitSeverity


class MockModel:
    """Mock model for testing."""
    pass


class TestOverfitMonitor:
    """Tests for OverfitMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        detector = TrainValGapDetector()
        corrector = RegularizationCorrector()

        monitor = OverfitMonitor(
            detectors=[detector],
            correctors=[corrector]
        )

        assert len(monitor.detectors) == 1
        assert len(monitor.correctors) == 1

    def test_check_no_overfitting(self):
        """Test check when no overfitting."""
        detector = TrainValGapDetector()
        monitor = OverfitMonitor(detectors=[detector])

        results = monitor.check(
            {'loss': 0.5},
            {'loss': 0.52},
            epoch=0
        )

        assert results['is_overfitting'] is False
        assert results['max_severity'] == OverfitSeverity.NONE
        assert len(results['corrections']) == 0

    def test_check_with_overfitting(self):
        """Test check when overfitting is detected."""
        detector = TrainValGapDetector()
        monitor = OverfitMonitor(detectors=[detector])

        results = monitor.check(
            {'loss': 0.1},
            {'loss': 0.35},  # Large gap
            epoch=0
        )

        assert results['is_overfitting'] is True
        assert results['max_severity'].value > OverfitSeverity.NONE.value

    def test_auto_correction(self):
        """Test automatic correction application."""
        detector = TrainValGapDetector()
        corrector = RegularizationCorrector()

        monitor = OverfitMonitor(
            detectors=[detector],
            correctors=[corrector],
            config={
                'auto_correct': True,
                'min_severity_for_correction': 'MILD'
            }
        )

        results = monitor.check(
            {'loss': 0.1},
            {'loss': 0.25},
            epoch=0,
            model=MockModel()
        )

        assert results['is_overfitting'] is True
        assert len(results['corrections']) > 0

    def test_correction_cooldown(self):
        """Test correction cooldown period."""
        detector = TrainValGapDetector()
        corrector = RegularizationCorrector()

        monitor = OverfitMonitor(
            detectors=[detector],
            correctors=[corrector],
            config={
                'auto_correct': True,
                'correction_cooldown': 5
            }
        )

        model = MockModel()

        # First correction
        results1 = monitor.check(
            {'loss': 0.1},
            {'loss': 0.35},
            epoch=0,
            model=model
        )

        # Try correction immediately (should be blocked)
        results2 = monitor.check(
            {'loss': 0.1},
            {'loss': 0.35},
            epoch=1,
            model=model
        )

        # After cooldown period
        results3 = monitor.check(
            {'loss': 0.1},
            {'loss': 0.35},
            epoch=6,
            model=model
        )

        assert len(results1.get('corrections', [])) > 0
        assert len(results2.get('corrections', [])) == 0
        assert len(results3.get('corrections', [])) > 0

    def test_callbacks(self):
        """Test callback registration and triggering."""
        detector = TrainValGapDetector()
        monitor = OverfitMonitor(detectors=[detector])

        callback_triggered = {'count': 0}

        def on_overfitting(result):
            callback_triggered['count'] += 1

        monitor.register_callback('on_overfitting', on_overfitting)

        # Trigger overfitting
        monitor.check(
            {'loss': 0.1},
            {'loss': 0.35},
            epoch=0
        )

        assert callback_triggered['count'] == 1

    def test_get_summary(self):
        """Test getting monitoring summary."""
        detector = TrainValGapDetector()
        monitor = OverfitMonitor(detectors=[detector])

        # Run several checks
        for i in range(10):
            gap = 0.1 + i * 0.03
            monitor.check(
                {'loss': 0.1},
                {'loss': 0.1 + gap},
                epoch=i
            )

        summary = monitor.get_summary()

        assert summary['total_checks'] == 10
        assert 'overfitting_detected' in summary
        assert 'overfitting_rate' in summary

    def test_reset(self):
        """Test monitor reset."""
        detector = TrainValGapDetector()
        monitor = OverfitMonitor(detectors=[detector])

        monitor.check({'loss': 0.1}, {'loss': 0.35}, epoch=0)

        assert len(monitor.get_history()) == 1

        monitor.reset()

        assert len(monitor.get_history()) == 0
        assert len(detector.history) == 0
