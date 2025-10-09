# Quick Start Guide

Get up and running with Parent Ion Classifier in 5 minutes.

## Installation (2 minutes)

### For Users: Install from Conda

```bash
# Create environment and install
conda create -n parent_classifier python=3.10
conda activate parent_classifier
conda install -c ketzahle parent_ion_classifier

# Download models (one-time)
parent-ion-classifier download
```

### For Developers: Install from Source

```bash
# Clone repository
git clone https://github.com/Kezahle/parent_ion_classifier.git
cd parent_ion_classifier

# Create development environment
conda env create -f environment.yml
conda activate parent_classifier_dev

# Install in development mode
pip install -e .

# Download models
parent-ion-classifier download
```

## Your First Classification (3 minutes)

### 1. Prepare Your Data

Your data should be a Python dictionary with this structure:

```python
import pandas as pd

spectra_dict = {
    'spectrum_001': {
        'positive': pd.DataFrame({
            'mz': [100.5, 200.3, 150.7, 300.1, 250.9],
            'MS1': [1000, 800, 1200, 600, 900],
            'MS2': [2000, 1500, 2500, 1000, 1800]
        }),
        'negative': None  # or another DataFrame for dual-mode
    }
}
```

### 2. Save Your Data

```python
import pickle

with open('my_spectra.pkl', 'wb') as f:
    pickle.dump(spectra_dict, f)
```

### 3. Run Classification

```bash
parent-ion-classifier classify -i my_spectra.pkl -o results.pkl
```

### 4. Load Results

```python
import pickle

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access predictions
predictions = results['spectrum_001']['single_model_prediction']
print(predictions[['mz', 'MS1', 'MS2', 'parent']])
```

The `parent` column contains probabilities (0-1) that each peak is the parent ion.

## Common Use Cases

### Single Ionization Mode

```python
from parent_ion_classifier.classifier import classify_parent_ions

# Classify positive mode spectra
classify_parent_ions(
    input_file='positive_spectra.pkl',
    output_file='positive_results.pkl',
    normalization_method='softmax_per_ionization'
)
```

### Dual Ionization Mode

```python
spectra_dict = {
    'spectrum_001': {
        'positive': positive_df,  # DataFrame with positive mode data
        'negative': negative_df   # DataFrame with negative mode data
    }
}

# Classification will use the dual-mode model automatically
```

### Using Python API

```python
from parent_ion_classifier.models import load_model
from parent_ion_classifier.classifier import MSNetModels, process_spectra

# Load models once
models = MSNetModels(
    positive=load_model('TopIntensityMSNet_positive'),
    negative=load_model('TopIntensityMSNet_negative'),
    merged=load_model('TopIntensityMSNet_merged'),
    dual=load_model('DualModeMSNet')
)

# Process multiple spectra
process_spectra('softmax_per_ionization', models, spectra_dict)

# Access results
for spectrum_id, data in spectra_dict.items():
    predictions = data['single_model_prediction']
    parent_mz = predictions.loc[predictions['parent'].idxmax(), 'mz']
    print(f"{spectrum_id}: Parent ion at m/z {parent_mz:.2f}")
```

## Normalization Methods

Choose the normalization that fits your use case:

| Method | Description | Best For |
|--------|-------------|----------|
| `softmax_per_ionization` | Separate softmax per mode | **Recommended for most cases** |
| `softmax` | Softmax over all peaks | Single certain parent ion |
| `sigmoid` | Independent probabilities | Multiple possible parents |
| `none` | Raw model outputs | Custom post-processing |

```bash
# Use different normalization
parent-ion-classifier classify -i input.pkl -o output.pkl -n sigmoid
```

## Useful Commands

```bash
# List all available models
parent-ion-classifier list

# Check which models are downloaded
parent-ion-classifier status

# Download specific model
parent-ion-classifier download --model DualModeMSNet

# Show cache location
parent-ion-classifier cache directory

# Run tests
parent-ion-classifier test

# Get help
parent-ion-classifier --help
parent-ion-classifier classify --help
```

## Next Steps

- **Read the full [README](../README.md)** for detailed features
- **Check [INSTALLATION](INSTALLATION.md)** for detailed setup and troubleshooting
- **See [API Documentation](API.md)** for advanced usage
- **Read [CONTRIBUTING](CONTRIBUTING.md)** if you want to develop

## Common Issues

### Models won't download
```bash
# Check status
parent-ion-classifier status

# Try downloading test models first
parent-ion-classifier download --group test_models
```

### Command not found
```bash
# Make sure environment is activated
conda activate parent_classifier

# Verify installation
conda list | grep parent_ion_classifier
```

### Import errors
```python
# Check if package is installed
import parent_ion_classifier
print(parent_ion_classifier.__version__)
```

## Getting Help

- **Documentation:** Check the [docs/](.) folder
- **Issues:** [GitHub Issues](https://github.com/Kezahle/parent_ion_classifier/issues)
- **Package:** [Anaconda.org](https://anaconda.org/ketzahle/parent_ion_classifier)

Happy classifying! 🎉