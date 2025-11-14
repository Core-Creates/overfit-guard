# Overfit Guard

A comprehensive micro-library for detecting and correcting overfitting in machine learning models across multiple frameworks.

## Features

- **Multi-Framework Support**: Works with PyTorch, TensorFlow/Keras, and Scikit-learn
- **Automatic Detection**: Multiple detection strategies including train/val gap, learning curve analysis, cross-validation, and statistical tests
- **Auto-Correction**: Applies regularization, data augmentation, architecture adjustments, and hyperparameter tuning
- **Flexible Integration**: Callbacks, hooks, decorators, and standalone monitoring
- **Configuration-Driven**: Easy to customize via configuration files or dictionaries

## Installation

```bash
pip install overfit-guard
```

### Framework-Specific Installation

```bash
# For PyTorch support
pip install overfit-guard[pytorch]

# For TensorFlow/Keras support
pip install overfit-guard[tensorflow]

# For Scikit-learn support
pip install overfit-guard[sklearn]

# For all frameworks
pip install overfit-guard[all]
```

### Development Installation

```bash
git clone https://github.com/yourusername/overfit-guard.git
cd overfit-guard
pip install -e .[dev]
```

## Quick Start

### PyTorch Example

```python
import torch
from overfit_guard.integrations.pytorch import create_pytorch_monitor

# Create model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# Create monitor with auto-correction
monitor = create_pytorch_monitor(
    model=model,
    optimizer=optimizer,
    auto_correct=True
)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # Check for overfitting
    results = monitor.on_epoch_end(
        epoch, model,
        {'loss': train_loss},
        {'loss': val_loss}
    )

    if monitor.should_stop:
        break  # Early stopping triggered
```

### Keras Example

```python
from overfit_guard.integrations.keras import create_keras_monitor

# Create model
model = keras.Sequential([...])
model.compile(optimizer='adam', loss='mse')

# Create monitor callback
monitor = create_keras_monitor(auto_correct=True)

# Train with callback
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[monitor],
    epochs=100
)
```

### Scikit-learn Example

```python
from overfit_guard.integrations.sklearn import create_sklearn_monitor

# Create monitor
monitor = create_sklearn_monitor()

# Iterative training with monitoring
model = MLPClassifier(max_iter=1, warm_start=True)

for epoch in range(n_epochs):
    model.fit(X_train, y_train)

    results = monitor.check_iteration(
        epoch,
        {'accuracy': model.score(X_train, y_train)},
        {'accuracy': model.score(X_val, y_val)},
        model
    )

    if monitor.should_stop:
        break
```

## Detection Strategies

### 1. Train-Validation Gap Detector
Monitors the gap between training and validation metrics.

```python
from overfit_guard.detectors.gap_detector import TrainValGapDetector

detector = TrainValGapDetector({
    'gap_threshold_mild': 0.05,
    'gap_threshold_moderate': 0.10,
    'gap_threshold_severe': 0.20
})
```

### 2. Learning Curve Analyzer
Analyzes trends in learning curves to detect divergence patterns.

```python
from overfit_guard.detectors.curve_analyzer import LearningCurveAnalyzer

detector = LearningCurveAnalyzer({
    'lookback_window': 10,
    'divergence_threshold': 0.05
})
```

### 3. Cross-Validation Detector
Uses variance across CV folds to identify overfitting.

```python
from overfit_guard.detectors.cv_detector import CrossValidationDetector

detector = CrossValidationDetector({
    'variance_threshold_moderate': 0.10
})
```

### 4. Statistical Detector
Applies statistical tests (t-test, KS-test) for rigorous detection.

```python
from overfit_guard.detectors.statistical import StatisticalDetector

detector = StatisticalDetector({
    'significance_level': 0.05,
    'test_type': 'both'
})
```

## Correction Strategies

### 1. Regularization Corrector
Applies L1/L2 regularization, dropout, and early stopping.

```python
from overfit_guard.correctors.regularization import RegularizationCorrector

corrector = RegularizationCorrector({
    'enable_weight_decay': True,
    'enable_dropout': True,
    'enable_early_stopping': True
})
```

### 2. Data Augmentation Corrector
Increases augmentation strength dynamically.

```python
from overfit_guard.correctors.augmentation import AugmentationCorrector

corrector = AugmentationCorrector({
    'augmentation_strategies': ['rotation', 'flip', 'crop'],
    'max_strength': 0.8
})
```

### 3. Architecture Corrector
Provides recommendations for architecture modifications.

```python
from overfit_guard.correctors.architecture import ArchitectureCorrector

corrector = ArchitectureCorrector({
    'dimension_reduction_factor': 0.8
})
```

### 4. Hyperparameter Corrector
Adjusts learning rate and batch size.

```python
from overfit_guard.correctors.hyperparameter import HyperparameterCorrector

corrector = HyperparameterCorrector({
    'lr_reduction_factor': 0.5,
    'batch_size_increase_factor': 2
})
```

## Configuration

### Using Configuration Files

```python
from overfit_guard.utils.config import Config

# Load from JSON file
config = Config.from_file('config.json')

# Access values
auto_correct = config.get('auto_correct')
detector_config = config.get_detector_config('gap_detector')

# Save configuration
config.save('my_config.json')
```

### Configuration Structure

```json
{
  "auto_correct": true,
  "min_severity_for_correction": "MODERATE",
  "correction_cooldown": 5,
  "detectors": {
    "gap_detector": {
      "enabled": true,
      "gap_threshold_moderate": 0.10
    }
  },
  "correctors": {
    "regularization": {
      "enabled": true,
      "early_stop_patience": 10
    }
  }
}
```

## Advanced Usage

### Manual Monitor Setup

```python
from overfit_guard.core.monitor import OverfitMonitor
from overfit_guard.detectors.gap_detector import TrainValGapDetector
from overfit_guard.correctors.regularization import RegularizationCorrector

# Create components
detector = TrainValGapDetector()
corrector = RegularizationCorrector()

# Create monitor
monitor = OverfitMonitor(
    detectors=[detector],
    correctors=[corrector],
    config={'auto_correct': True}
)

# Use monitor
results = monitor.check(train_metrics, val_metrics, epoch, model)
```

### Callbacks

```python
# Register callbacks for events
monitor.register_callback('on_detection', lambda result: print(result))
monitor.register_callback('on_overfitting', lambda result: log_to_file(result))
monitor.register_callback('on_correction', lambda result: notify_user(result))
```

## API Reference

### OverfitMonitor

Main class for orchestrating detection and correction.

- `check(train_metrics, val_metrics, epoch, model)` - Check for overfitting
- `get_summary()` - Get monitoring summary
- `reset()` - Reset all state

### BaseDetector

Abstract base class for detectors.

- `detect(train_metrics, val_metrics, epoch)` - Detect overfitting
- `reset()` - Reset detector state

### BaseCorrector

Abstract base class for correctors.

- `correct(model, detection_result)` - Apply correction
- `can_correct(model)` - Check if correction is applicable

## Examples

See the `examples/` directory for complete working examples:

- `pytorch_example.py` - PyTorch integration
- `keras_example.py` - TensorFlow/Keras integration
- `sklearn_example.py` - Scikit-learn integration

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=overfit_guard --cov-report=html
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{overfit_guard,
  title = {Overfit Guard: A Micro-Library for ML Overfitting Detection and Correction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/overfit-guard}
}
```

## Support

- Documentation: [https://overfit-guard.readthedocs.io](https://overfit-guard.readthedocs.io)
- Issues: [https://github.com/yourusername/overfit-guard/issues](https://github.com/yourusername/overfit-guard/issues)
- Discussions: [https://github.com/yourusername/overfit-guard/discussions](https://github.com/yourusername/overfit-guard/discussions)
