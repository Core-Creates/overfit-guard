"""Tests for corrector implementations."""

import pytest
from overfit_guard.correctors.regularization import RegularizationCorrector
from overfit_guard.correctors.augmentation import AugmentationCorrector
from overfit_guard.correctors.architecture import ArchitectureCorrector
from overfit_guard.correctors.hyperparameter import HyperparameterCorrector
from overfit_guard.core.detector import DetectionResult, OverfitSeverity
from overfit_guard.core.corrector import CorrectionType


class MockModel:
    """Mock model for testing."""
    pass


class TestRegularizationCorrector:
    """Tests for RegularizationCorrector."""

    def test_correction_application(self):
        """Test applying regularization correction."""
        corrector = RegularizationCorrector({
            'enable_weight_decay': True,
            'enable_dropout': True
        })

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.MODERATE,
            confidence=0.8,
            metrics={'gap': 0.15}
        )

        result = corrector.correct(MockModel(), detection)

        assert result.success is True
        assert result.correction_type == CorrectionType.REGULARIZATION
        assert len(result.actions_taken) > 0
        assert 'weight_decay' in result.parameters_changed

    def test_early_stopping(self):
        """Test early stopping trigger."""
        corrector = RegularizationCorrector({
            'enable_early_stopping': True,
            'early_stop_patience': 3
        })

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.MILD,
            confidence=0.6,
            metrics={'val_value': 0.5}
        )

        # Trigger multiple corrections without improvement
        for _ in range(5):
            result = corrector.correct(MockModel(), detection)

        assert result.parameters_changed.get('should_stop', False) is True

    def test_can_correct(self):
        """Test can_correct method."""
        corrector = RegularizationCorrector()
        assert corrector.can_correct(MockModel()) is True


class TestAugmentationCorrector:
    """Tests for AugmentationCorrector."""

    def test_strength_increase(self):
        """Test augmentation strength increases."""
        corrector = AugmentationCorrector({
            'initial_strength': 0.1,
            'strength_increment': 0.1
        })

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.MODERATE,
            confidence=0.7,
            metrics={}
        )

        result = corrector.correct(MockModel(), detection)

        assert 'augmentation_strength' in result.parameters_changed
        assert result.parameters_changed['augmentation_strength'] > 0.1

    def test_strategy_activation(self):
        """Test augmentation strategies are activated."""
        corrector = AugmentationCorrector({
            'augmentation_strategies': ['rotation', 'flip', 'crop']
        })

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.SEVERE,
            confidence=0.9,
            metrics={}
        )

        result = corrector.correct(MockModel(), detection)

        assert 'active_strategies' in result.parameters_changed
        assert len(result.parameters_changed['active_strategies']) > 0


class TestArchitectureCorrector:
    """Tests for ArchitectureCorrector."""

    def test_low_severity_skipped(self):
        """Test that mild severity doesn't trigger architecture changes."""
        corrector = ArchitectureCorrector()

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.MILD,
            confidence=0.5,
            metrics={}
        )

        result = corrector.correct(MockModel(), detection)

        assert result.success is False

    def test_recommendations_generated(self):
        """Test architecture recommendations are generated."""
        corrector = ArchitectureCorrector()

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.SEVERE,
            confidence=0.9,
            metrics={}
        )

        result = corrector.correct(MockModel(), detection)

        assert 'recommendations' in result.parameters_changed
        assert len(result.parameters_changed['recommendations']) > 0


class TestHyperparameterCorrector:
    """Tests for HyperparameterCorrector."""

    def test_learning_rate_reduction(self):
        """Test learning rate is reduced."""
        corrector = HyperparameterCorrector({
            'enable_lr_adjustment': True,
            'lr_reduction_factor': 0.5
        })

        corrector.set_current_hyperparameters(lr=0.001)

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.MODERATE,
            confidence=0.7,
            metrics={}
        )

        result = corrector.correct(
            MockModel(),
            detection,
            current_lr=0.001
        )

        assert 'learning_rate' in result.parameters_changed
        assert result.parameters_changed['learning_rate'] < 0.001

    def test_batch_size_increase(self):
        """Test batch size is increased."""
        corrector = HyperparameterCorrector({
            'enable_batch_size_adjustment': True,
            'batch_size_increase_factor': 2
        })

        detection = DetectionResult(
            is_overfitting=True,
            severity=OverfitSeverity.MODERATE,
            confidence=0.7,
            metrics={}
        )

        result = corrector.correct(
            MockModel(),
            detection,
            current_batch_size=32
        )

        assert 'batch_size' in result.parameters_changed
        assert result.parameters_changed['batch_size'] == 64
