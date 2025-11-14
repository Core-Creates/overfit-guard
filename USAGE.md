# Overfit Guard - Usage Guide

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Framework Integration](#framework-integration)
5. [Detection Strategies](#detection-strategies)
6. [Correction Strategies](#correction-strategies)
7. [Configuration](#configuration)
8. [Advanced Features](#advanced-features)

## Basic Concepts

Overfit Guard operates on a simple principle:

1. **Detection**: Monitor training metrics to detect overfitting
2. **Correction**: Apply appropriate corrections when overfitting is detected
3. **Automation**: Optionally automate the entire process

### Key Components

- **Detectors**: Identify overfitting using various strategies
- **Correctors**: Apply fixes to reduce overfitting
- **Monitor**: Orchestrates detectors and correctors
- **Integrations**: Framework-specific implementations

## Installation

```bash
# Basic installation
pip install overfit-guard

# With PyTorch support
pip install overfit-guard[pytorch]

# With all frameworks
pip install overfit-guard[all]
```

## Quick Start

### 1. Framework-Agnostic Usage

```python
from overfit_guard.core.monitor import OverfitMonitor
from overfit_guard.detectors.gap_detector import TrainValGapDetector
from overfit_guard.correctors.regularization import RegularizationCorrector

# Create monitor
monitor = OverfitMonitor(
    detectors=[TrainValGapDetector()],
    correctors=[RegularizationCorrector()],
    config={'auto_correct': False}
)

# In your training loop
for epoch in range(num_epochs):
    train_metrics = {'loss': train_loss}
    val_metrics = {'loss': val_loss}

    results = monitor.check(train_metrics, val_metrics, epoch)

    if results['is_overfitting']:
        print(f"Overfitting detected! Severity: {results['max_severity'].name}")
```

### 2. PyTorch Integration

```python
from overfit_guard.integrations.pytorch import create_pytorch_monitor

# Create monitor
callback = create_pytorch_monitor(
    model=model,
    optimizer=optimizer,
    auto_correct=True
)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    callback.on_epoch_end(
        epoch, model,
        {'loss': train_loss},
        {'loss': val_loss}
    )

    if callback.should_stop:
        break
```

### 3. Keras Integration

```python
from overfit_guard.integrations.keras import create_keras_monitor

# Create callback
callback = create_keras_monitor(auto_correct=True)

# Train with callback
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[callback],
    epochs=100
)
```

### 4. Scikit-learn Integration

```python
from overfit_guard.integrations.sklearn import create_sklearn_monitor

# Create monitor
monitor = create_sklearn_monitor()

# Iterative training
model = MLPClassifier(max_iter=1, warm_start=True)

for epoch in range(n_epochs):
    model.fit(X_train, y_train)

    monitor.check_iteration(
        epoch,
        {'accuracy': model.score(X_train, y_train)},
        {'accuracy': model.score(X_val, y_val)}
    )
```

## Detection Strategies

### Train-Validation Gap Detector

Monitors the gap between training and validation metrics.

```python
from overfit_guard.detectors.gap_detector import TrainValGapDetector

detector = TrainValGapDetector({
    'gap_threshold_mild': 0.05,        # 5% gap = mild
    'gap_threshold_moderate': 0.10,    # 10% gap = moderate
    'gap_threshold_severe': 0.20,      # 20% gap = severe
    'metric_name': 'loss',
    'use_relative_gap': True,
    'window_size': 1
})
```

**When to use**: Best for general-purpose overfitting detection during training.

### Learning Curve Analyzer

Analyzes trends in learning curves.

```python
from overfit_guard.detectors.curve_analyzer import LearningCurveAnalyzer

detector = LearningCurveAnalyzer({
    'lookback_window': 10,             # Analyze last 10 epochs
    'divergence_threshold': 0.05,
    'trend_threshold': 0.01,
    'min_epochs': 5
})
```

**When to use**: When you want to detect diverging trends early.

### Cross-Validation Detector

Uses variance across CV folds.

```python
from overfit_guard.detectors.cv_detector import CrossValidationDetector

detector = CrossValidationDetector({
    'variance_threshold_moderate': 0.10,
    'min_folds': 3
})
```

**When to use**: For model selection or when using k-fold cross-validation.

### Statistical Detector

Applies statistical tests.

```python
from overfit_guard.detectors.statistical import StatisticalDetector

detector = StatisticalDetector({
    'significance_level': 0.05,
    'test_type': 'both',  # 'ttest', 'ks_test', or 'both'
    'min_samples': 10
})
```

**When to use**: When you need rigorous statistical validation.

## Correction Strategies

### Regularization Corrector

Applies L1/L2, dropout, and early stopping.

```python
from overfit_guard.correctors.regularization import RegularizationCorrector

corrector = RegularizationCorrector({
    'enable_weight_decay': True,
    'enable_dropout': True,
    'enable_early_stopping': True,
    'weight_decay_step': 0.001,
    'dropout_step': 0.1,
    'early_stop_patience': 10
})
```

**Effect**: Gradually increases regularization strength.

### Data Augmentation Corrector

Adjusts augmentation parameters.

```python
from overfit_guard.correctors.augmentation import AugmentationCorrector

corrector = AugmentationCorrector({
    'augmentation_strategies': ['rotation', 'flip', 'crop', 'noise'],
    'initial_strength': 0.1,
    'strength_increment': 0.1,
    'max_strength': 0.8
})
```

**Effect**: Increases data augmentation to improve generalization.

### Architecture Corrector

Recommends architecture changes.

```python
from overfit_guard.correctors.architecture import ArchitectureCorrector

corrector = ArchitectureCorrector({
    'dimension_reduction_factor': 0.8,
    'min_layer_size': 32
})
```

**Effect**: Provides recommendations (requires manual implementation).

### Hyperparameter Corrector

Adjusts learning rate and batch size.

```python
from overfit_guard.correctors.hyperparameter import HyperparameterCorrector

corrector = HyperparameterCorrector({
    'enable_lr_adjustment': True,
    'enable_batch_size_adjustment': True,
    'lr_reduction_factor': 0.5,
    'batch_size_increase_factor': 2
})
```

**Effect**: Reduces learning rate and increases batch size.

## Configuration

### Using Configuration Files

Create a `config.json`:

```json
{
  "auto_correct": true,
  "min_severity_for_correction": "MODERATE",
  "correction_cooldown": 5,
  "log_level": "INFO",
  "detectors": {
    "gap_detector": {
      "enabled": true,
      "gap_threshold_moderate": 0.10
    },
    "curve_analyzer": {
      "enabled": true,
      "lookback_window": 10
    }
  },
  "correctors": {
    "regularization": {
      "enabled": true,
      "early_stop_patience": 10
    },
    "hyperparameter": {
      "enabled": true,
      "lr_reduction_factor": 0.5
    }
  }
}
```

Load and use:

```python
from overfit_guard.utils.config import Config

config = Config.from_file('config.json')
monitor = create_pytorch_monitor(model, optimizer, config.to_dict())
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_correct` | bool | False | Enable automatic corrections |
| `min_severity_for_correction` | str | 'MODERATE' | Minimum severity to trigger correction |
| `correction_cooldown` | int | 5 | Epochs to wait between corrections |
| `log_level` | str | 'INFO' | Logging level |

## Advanced Features

### Custom Detectors

```python
from overfit_guard.core.detector import BaseDetector, DetectionResult, OverfitSeverity

class CustomDetector(BaseDetector):
    def detect(self, train_metrics, val_metrics, epoch, **kwargs):
        # Your custom detection logic
        is_overfitting = your_logic()

        return DetectionResult(
            is_overfitting=is_overfitting,
            severity=OverfitSeverity.MODERATE,
            confidence=0.8,
            metrics={'custom_metric': 0.5}
        )

    def reset(self):
        pass
```

### Callbacks

```python
def on_overfitting_detected(result):
    print(f"Overfitting! Severity: {result['max_severity'].name}")
    # Send alert, save model, etc.

monitor.register_callback('on_overfitting', on_overfitting_detected)
monitor.register_callback('on_correction', lambda r: print(f"Correction applied"))
```

### Monitoring Without Auto-Correction

```python
# Detection only mode
monitor = OverfitMonitor(
    detectors=[TrainValGapDetector()],
    correctors=[],  # No correctors
    config={'auto_correct': False}
)

results = monitor.check(train_metrics, val_metrics, epoch)

if results['is_overfitting']:
    # Handle manually
    print("Recommendations:", results['detections'][0]['result'].recommendations)
```

### Getting Summary Statistics

```python
summary = monitor.get_summary()

print(f"Total checks: {summary['total_checks']}")
print(f"Overfitting detected: {summary['overfitting_detected']} times")
print(f"Corrections applied: {summary['corrections_applied']}")
print(f"Overfitting rate: {summary['overfitting_rate']:.2%}")
```

## Best Practices

1. **Start with detection only**: Disable auto-correction initially to understand patterns
2. **Tune thresholds**: Adjust detection thresholds based on your domain
3. **Use multiple detectors**: Combine different detection strategies for robustness
4. **Monitor correction effects**: Track whether corrections actually help
5. **Set appropriate cooldowns**: Avoid over-correcting by setting reasonable cooldown periods

## Troubleshooting

### Issue: Too many false positives

**Solution**: Increase detection thresholds or use more conservative detectors.

```python
detector = TrainValGapDetector({
    'gap_threshold_moderate': 0.15,  # Increased from 0.10
    'gap_threshold_severe': 0.30     # Increased from 0.20
})
```

### Issue: Corrections not helping

**Solution**: Check if corrections are actually being applied to your training loop.

```python
# Verify corrections
results = monitor.check(...)
for correction in results['corrections']:
    print(f"Applied: {correction['corrector']}")
    print(f"Changes: {correction['result'].parameters_changed}")
```

### Issue: Early stopping too aggressive

**Solution**: Increase patience parameter.

```python
corrector = RegularizationCorrector({
    'early_stop_patience': 20  # Increased from 10
})
```

## Examples

See the `examples/` directory for complete working examples:
- `pytorch_example.py`
- `keras_example.py`
- `sklearn_example.py`
