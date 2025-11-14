# Overfit Guard - Quick Reference

## Installation

```bash
pip install overfit-guard[all]
```

## 30-Second Start

### PyTorch
```python
from overfit_guard.integrations.pytorch import create_pytorch_monitor

monitor = create_pytorch_monitor(model, optimizer, auto_correct=True)

# In training loop:
monitor.on_epoch_end(epoch, model, {'loss': train_loss}, {'loss': val_loss})
```

### Keras
```python
from overfit_guard.integrations.keras import create_keras_monitor

model.fit(X, y, validation_split=0.2,
          callbacks=[create_keras_monitor(auto_correct=True)])
```

### Scikit-learn
```python
from overfit_guard.integrations.sklearn import create_sklearn_monitor

monitor = create_sklearn_monitor()
monitor.check_iteration(epoch, train_metrics, val_metrics, model)
```

## Core Classes

### OverfitMonitor
```python
from overfit_guard.core.monitor import OverfitMonitor

monitor = OverfitMonitor(detectors=[], correctors=[], config={})
results = monitor.check(train_metrics, val_metrics, epoch, model)
```

### Detectors

| Class | Purpose |
|-------|---------|
| `TrainValGapDetector` | Monitor train/val metric gap |
| `LearningCurveAnalyzer` | Analyze curve trends |
| `CrossValidationDetector` | CV variance analysis |
| `StatisticalDetector` | Statistical hypothesis tests |

### Correctors

| Class | Action |
|-------|--------|
| `RegularizationCorrector` | Apply L1/L2, dropout, early stop |
| `AugmentationCorrector` | Adjust augmentation |
| `ArchitectureCorrector` | Recommend architecture changes |
| `HyperparameterCorrector` | Tune LR and batch size |

## Configuration Quick Examples

### Minimal Detection Only
```python
config = {'auto_correct': False}
```

### Aggressive Auto-Correction
```python
config = {
    'auto_correct': True,
    'min_severity_for_correction': 'MILD',
    'correction_cooldown': 3
}
```

### Custom Thresholds
```python
config = {
    'detectors': {
        'gap_detector': {
            'gap_threshold_mild': 0.03,
            'gap_threshold_severe': 0.15
        }
    }
}
```

## Common Patterns

### Pattern 1: Monitor with Early Stopping
```python
monitor = create_pytorch_monitor(model, optimizer, auto_correct=True)

for epoch in range(max_epochs):
    # ... training code ...
    monitor.on_epoch_end(epoch, model, train_metrics, val_metrics)

    if monitor.should_stop:
        print("Early stopping triggered")
        break
```

### Pattern 2: Manual Correction
```python
monitor = OverfitMonitor(detectors=[...], config={'auto_correct': False})

results = monitor.check(train_metrics, val_metrics, epoch)

if results['is_overfitting']:
    if results['max_severity'] == OverfitSeverity.SEVERE:
        # Take manual action
        reduce_model_capacity()
```

### Pattern 3: With Callbacks
```python
def alert_on_overfit(result):
    send_email(f"Overfitting detected: {result['max_severity']}")

monitor.register_callback('on_overfitting', alert_on_overfit)
```

### Pattern 4: Tracking and Logging
```python
results = monitor.check(train_metrics, val_metrics, epoch, model)

# Log to experiment tracker
wandb.log({
    'overfitting_detected': results['is_overfitting'],
    'severity': results['max_severity'].name,
    'corrections_applied': len(results['corrections'])
})
```

## Severity Levels

| Level | Value | Typical Gap |
|-------|-------|-------------|
| NONE | 0 | < 5% |
| MILD | 1 | 5-10% |
| MODERATE | 2 | 10-20% |
| SEVERE | 3 | > 20% |

## Detection Results

```python
{
    'epoch': 10,
    'is_overfitting': True,
    'max_severity': OverfitSeverity.MODERATE,
    'detections': [
        {
            'detector': 'TrainValGapDetector',
            'result': DetectionResult(...)
        }
    ],
    'corrections': [
        {
            'corrector': 'RegularizationCorrector',
            'result': CorrectionResult(...)
        }
    ]
}
```

## Useful Methods

### Monitor Methods
- `monitor.check()` - Check for overfitting
- `monitor.get_summary()` - Get statistics
- `monitor.reset()` - Reset state
- `monitor.register_callback()` - Add callbacks
- `monitor.add_detector()` - Add detector
- `monitor.add_corrector()` - Add corrector

### Detector Methods
- `detector.detect()` - Run detection
- `detector.reset()` - Clear history
- `detector.enable()` / `disable()` - Toggle detector

### Corrector Methods
- `corrector.correct()` - Apply correction
- `corrector.can_correct()` - Check applicability
- `corrector.get_parameters()` - Get current state

## Tips & Tricks

1. **Start Conservative**: Begin with `auto_correct=False` to observe patterns
2. **Tune Thresholds**: Adjust based on your specific use case
3. **Use Multiple Detectors**: Combine strategies for robustness
4. **Monitor Corrections**: Check if they actually help
5. **Set Appropriate Cooldowns**: Avoid over-correcting
6. **Log Everything**: Track all detections and corrections
7. **Review Recommendations**: Architecture corrector provides useful insights

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many false positives | Increase thresholds |
| Missing detections | Decrease thresholds or add more detectors |
| Corrections not working | Check framework integration |
| Early stopping too early | Increase patience |
| No corrections applied | Check severity threshold and cooldown |

## Examples Location

```
examples/
├── pytorch_example.py
├── keras_example.py
└── sklearn_example.py
```

## Documentation

- README.md - Overview and installation
- USAGE.md - Comprehensive usage guide
- QUICKREF.md - This file
- API docs - See source code docstrings

## Links

- GitHub: https://github.com/yourusername/overfit-guard
- Issues: https://github.com/yourusername/overfit-guard/issues
- PyPI: https://pypi.org/project/overfit-guard/
