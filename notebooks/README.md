# Overfit Guard - Jupyter Notebooks

This folder contains interactive Jupyter notebooks demonstrating Overfit Guard capabilities.

## Available Notebooks

### 1. **Quick Start Demo** (`overfit_guard_colab_demo.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Core-Creates/overfit-guard/blob/main/notebooks/overfit_guard_colab_demo.ipynb)

**Best for:** First-time users, understanding the basics

**Features:**
- Ready-to-run breast cancer classification example
- Side-by-side comparison (with/without guard)
- Clear visualizations and metrics
- Complete tutorial format

**Dataset:** Wisconsin Breast Cancer (569 samples, 30 features, binary classification)

**Framework:** PyTorch

**Time to run:** ~5 minutes

---

### 2. **Universal Dataset Demo** (`universal_dataset_demo.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Core-Creates/overfit-guard/blob/main/notebooks/universal_dataset_demo.ipynb)

**Best for:** Testing with your own data, exploring different frameworks

**Features:**
- **Use ANY dataset** - built-in or upload custom CSV
- **All frameworks** - PyTorch, Keras, or scikit-learn
- **Multiple datasets** included:
  - Classification: Breast Cancer, Digits, Wine, Iris
  - Regression: Diabetes, California Housing
- Easy configuration via simple variables
- Comprehensive comparison and analysis

**Custom Dataset Support:**
```python
# Just set these variables:
DATASET = 'custom'
CUSTOM_CSV_PATH = 'your_data.csv'
TARGET_COLUMN = 'target'
FRAMEWORK = 'pytorch'  # or 'keras' or 'sklearn'
```

**Time to run:** ~10 minutes

---

## Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badges above - no setup required!

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install -e ".[dev]"

# Start Jupyter
jupyter notebook notebooks/

# Open any .ipynb file
```

### Option 3: VSCode
1. Install Jupyter extension
2. Open `.ipynb` file
3. Select Python kernel
4. Run cells

---

## What Each Notebook Demonstrates

### Core Features
Both notebooks demonstrate:
- ✅ Automatic overfitting detection
- ✅ Real-time monitoring during training
- ✅ Automatic corrections (regularization, learning rate adjustment)
- ✅ Early stopping when needed
- ✅ Comprehensive metrics and visualizations
- ✅ Side-by-side comparison

### Detection Methods
- Train-validation gap monitoring
- Learning curve analysis
- Statistical significance tests
- Cross-validation variance

### Correction Strategies
- Regularization (L1/L2, dropout, weight decay)
- Learning rate adjustment
- Early stopping
- Architecture recommendations

---

## Built-in Datasets

The Universal Demo includes these datasets:

| Dataset | Type | Samples | Features | Classes/Target |
|---------|------|---------|----------|----------------|
| `breast_cancer` | Classification | 569 | 30 | 2 (malignant/benign) |
| `digits` | Classification | 1,797 | 64 | 10 (0-9) |
| `wine` | Classification | 178 | 13 | 3 (wine types) |
| `iris` | Classification | 150 | 4 | 3 (flower types) |
| `diabetes` | Regression | 442 | 10 | Disease progression |
| `california_housing` | Regression | 20,640 | 8 | Median house value |

---

## Using Your Own Data

### Requirements for Custom CSV:
1. **Format**: CSV file with headers
2. **Target column**: One column for prediction target
3. **Feature columns**: All other columns (numeric)
4. **No missing values** (or handle before upload)

### Example CSV structure:
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,0
...
```

### Upload in Colab:
```python
DATASET = 'custom'
TARGET_COLUMN = 'target'  # Your target column name
# Notebook will prompt for file upload
```

### Use local file:
```python
DATASET = 'custom'
CUSTOM_CSV_PATH = '/path/to/your/data.csv'
TARGET_COLUMN = 'target'
```

---

## Configuration Options

### Dataset Selection
```python
DATASET = 'breast_cancer'  # or any built-in dataset name, or 'custom'
```

### Framework Selection
```python
FRAMEWORK = 'pytorch'  # Options: 'pytorch', 'keras', 'sklearn'
```

### Training Settings
```python
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

### Custom Dataset
```python
CUSTOM_CSV_PATH = 'my_data.csv'
TARGET_COLUMN = 'target'  # Name of your target column
```

---

## Expected Results

### Typical Outputs:

**Without Overfit Guard:**
- May achieve slightly higher test accuracy
- Larger train-validation gap
- No automatic intervention
- Unmonitored overfitting

**With Overfit Guard:**
- More robust generalization
- Smaller train-validation gap (~20-30% reduction)
- Automatic corrections applied
- Early stopping when beneficial
- Detailed overfitting statistics

### Example Metrics:
```
WITHOUT Guard:
  Test Accuracy: 97.67%
  Train-Val Gap: 1.10%

WITH Guard:
  Test Accuracy: 95.35%
  Train-Val Gap: 0.85% (↓23%)
  Detections: 96 events
  Corrections: 38 applied
```

---

## Troubleshooting

### Colab Issues

**Installation fails:**
```python
# Try installing dependencies separately
!pip install torch scikit-learn matplotlib
!pip install git+https://github.com/Core-Creates/overfit-guard.git
```

**Out of memory:**
```python
# Reduce batch size
BATCH_SIZE = 16
```

**Slow training:**
```python
# Reduce epochs
NUM_EPOCHS = 50
```

### Local Jupyter Issues

**Kernel dies:**
- Restart kernel and run all cells
- Check available memory
- Reduce dataset size

**Module not found:**
```bash
# Reinstall in development mode
pip install -e ".[dev]"
```

**GPU not detected (PyTorch):**
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
```

---

## Performance Tips

### For faster training:
1. Use smaller datasets for testing
2. Reduce `NUM_EPOCHS`
3. Increase `BATCH_SIZE` (if memory allows)
4. Use `sklearn` framework (fastest for small datasets)

### For better results:
1. Use more epochs for complex datasets
2. Try different frameworks
3. Adjust learning rate
4. Experiment with different datasets

---

## Next Steps

After running the notebooks:

1. **Try with your own data** - Use the Universal Demo
2. **Explore different frameworks** - Compare PyTorch vs Keras vs sklearn
3. **Integrate into your code** - See examples folder
4. **Read the docs** - Check the main README.md
5. **Run real-world tests** - See `examples/real_world_test.py`

---

## Additional Resources

### Documentation
- Main README: `../README.md`
- Analysis Report: `../ANALYSIS_REPORT.md`
- Executive Summary: `../EXECUTIVE_SUMMARY.md`

### Code Examples
- PyTorch: `../examples/pytorch_example.py`
- Keras: `../examples/keras_example.py`
- Scikit-learn: `../examples/sklearn_example.py`
- Real-world test: `../examples/real_world_test.py`

### Support
- GitHub Issues: https://github.com/Core-Creates/overfit-guard/issues
- PyPI: https://pypi.org/project/overfit-guard/

---

## Contributing

Found an issue or want to add a new notebook?

1. Fork the repository
2. Create your notebook
3. Test in both Colab and local Jupyter
4. Submit a pull request

---

**Happy Learning! 📚🚀**
