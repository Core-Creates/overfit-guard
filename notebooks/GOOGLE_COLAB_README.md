# 🚀 Running Overfit Guard on Google Colab

## Quick Start Links

### 📓 Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Core-Creates/overfit-guard/blob/main/notebooks/overfit_guard_demo.ipynb)

## 🎯 What You'll Learn

This notebook demonstrates:
- ✅ How to install and use Overfit Guard
- ✅ Real-world example with medical data
- ✅ Comparative analysis (with vs without guard)
- ✅ Visual performance metrics
- ✅ Automatic overfitting correction in action

## 📋 Requirements

The notebook automatically installs all dependencies:
- `overfit-guard`
- `torch` and `torchvision`
- `scikit-learn`
- `matplotlib`
- `numpy`

## ⚡ Quick Instructions

1. **Open the notebook** using the badge above
2. **Run all cells** (Runtime → Run all)
3. **Watch the magic happen!** The notebook will:
   - Train a baseline model
   - Train with Overfit Guard
   - Compare results visually
   - Show you the improvements

## 🎨 What to Expect

### Training Without Guard (Baseline)
```
Epoch 10/50 - Train: 0.9400, Val: 0.9200
Epoch 20/50 - Train: 0.9800, Val: 0.9200
Epoch 30/50 - Train: 0.9900, Val: 0.9100  ⚠️ Overfitting!
```

### Training With Guard (Protected)
```
Epoch 10/50 - Train: 0.9400, Val: 0.9300
🛡️  Overfitting detected - Applying corrections...
🔧 Applied regularization correction
📉 Reduced learning rate
Epoch 20/50 - Train: 0.9600, Val: 0.9500  ✅ Better generalization!
```

## 📊 Expected Results

| Metric | Without Guard | With Guard | Improvement |
|--------|--------------|------------|-------------|
| Test Accuracy | ~97% | ~95-98% | Varies |
| Train-Val Gap | 0.011 | 0.008 | ✅ -27% |
| Overfitting Events | N/A | ~96 detected | ✅ Monitored |
| Corrections Applied | 0 | ~38 | ✅ Automatic |

## 🔧 Customization

You can modify the notebook to:
- Change the dataset
- Adjust correction thresholds
- Try different model architectures
- Experiment with detection sensitivity

## 📝 Code Example

The simplest integration:

```python
from overfit_guard.integrations.pytorch import create_pytorch_monitor

# Create your model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# Add Overfit Guard (just 3 lines!)
monitor = create_pytorch_monitor(
    model=model,
    optimizer=optimizer,
    auto_correct=True
)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # Check for overfitting (1 line!)
    monitor.on_epoch_end(
        epoch, model,
        {'loss': train_loss, 'accuracy': train_acc},
        {'loss': val_loss, 'accuracy': val_acc}
    )
```

## 🐛 Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, run:
```python
!pip install overfit-guard torch scikit-learn matplotlib
```

### GPU Issues
The notebook works on both CPU and GPU. Colab automatically uses GPU if available.

### Timeout Issues
If training takes too long, reduce `num_epochs` from 50 to 30.

## 💡 Tips for Best Results

1. **Use GPU**: Runtime → Change runtime type → GPU
2. **Run sequentially**: Don't skip cells
3. **Check outputs**: Each cell has informative prints
4. **Visualizations**: Plots show train vs val metrics

## 📚 Learn More

- [Documentation](https://overfit-guard.readthedocs.io)
- [GitHub Repository](https://github.com/Core-Creates/overfit-guard)
- [PyPI Package](https://pypi.org/project/overfit-guard/)
- [Issue Tracker](https://github.com/Core-Creates/overfit-guard/issues)

## 🤝 Contributing

Found an issue or want to improve the notebook?
1. Fork the repository
2. Make your changes
3. Submit a pull request

## 📄 License

MIT License - Free to use and modify

## ⭐ Like it?

If you find this useful, please:
- ⭐ Star the [GitHub repo](https://github.com/Core-Creates/overfit-guard)
- 🐦 Share on Twitter
- 💬 Tell your ML friends!

---

**Happy Training! 🚀**
