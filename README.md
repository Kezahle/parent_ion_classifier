# Parent Ion Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conda Version](https://img.shields.io/conda/vn/ketzahle/parent_ion_classifier.svg)](https://anaconda.org/ketzahle/parent_ion_classifier)
[![Conda Downloads](https://img.shields.io/conda/dn/ketzahle/parent_ion_classifier.svg)](https://anaconda.org/ketzahle/parent_ion_classifier)
[![Conda Platform](https://img.shields.io/conda/pn/ketzahle/parent_ion_classifier.svg)](https://anaconda.org/ketzahle/parent_ion_classifier)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

A Python package for classifying parent ions in mass spectrometry (MS/MS) experiments using deep learning models.

Developed as part of a research thesis at the Open University of Israel with collaboration with the Weizmann Institute of Science, this package leverages PyTorch-based neural networks to detect and classify parent ions from MS data, providing insights for chemical analysis and compound identification.

## Features

- **Deep learning models** for MS/MS parent ion classification
- **Multiple ionization modes**: Support for positive, negative, merged, and dual-mode spectra
- **Automatic model management**: Downloads and caches models from Hugging Face Hub
- **Multiple normalization methods**: Flexible output normalization (softmax, sigmoid, per-ionization)
- **Command-line interface**: Easy integration into analysis workflows
- **Cross-platform support**: Works on Linux, macOS (Intel & ARM), and Windows
- **Cross-platform reproducibility**: Deterministic results across different systems
- **Built-in testing**: Verify installation with included test suite

## Installation

### Option 1: Conda (Recommended)

The easiest way to install Parent Ion Classifier is via conda. This installs the package with all dependencies:

```bash
# Create a new environment (recommended)
conda create -n parent_classifier python=3.10
conda activate parent_classifier

# Install from conda
conda install -c ketzahle parent_ion_classifier
```

**That's it!** The package and all dependencies are now installed.

### Option 2: pip (Coming Soon)

```bash
pip install parent-ion-classifier
```

*Note: PyPI distribution coming soon. For now, please use conda or install from source.*

### Option 3: From Source (For Development)

If you want to contribute to development or modify the package:

```bash
# Clone the repository
git clone https://github.com/Kezahle/parent_ion_classifier.git
cd parent_ion_classifier

# Create development environment from YAML
conda env create -f environment.yml
conda activate parent_classifier_dev

# Install in development mode
pip install -e .
```

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed development setup instructions.

## Quick Start

### First-Time Setup

After installation, download the models (one-time setup):

```bash
# Download all production models (recommended)
parent-ion-classifier download

# Or download a specific model group
parent-ion-classifier download --group production_models
```

Models are cached locally and only need to be downloaded once.

### Command Line Usage

```bash
# Classify spectra from a pickle file
parent-ion-classifier classify -i input.pkl -o output.pkl

# Use different normalization methods
parent-ion-classifier classify -i input.pkl -o output.pkl -n softmax_per_ionization

# Check model status
parent-ion-classifier status

# Run tests to verify installation
parent-ion-classifier test
```

### Python API Usage

```python
import pandas as pd
from parent_ion_classifier import DataCanonizer, process_spectra
from parent_ion_classifier.models import load_model
from parent_ion_classifier.classifier import MSNetModels

# Load models (downloads automatically if needed)
models = MSNetModels(
    positive=load_model('TopIntensityMSNet_positive'),
    negative=load_model('TopIntensityMSNet_negative'),
    merged=load_model('TopIntensityMSNet_merged'),
    dual=load_model('DualModeMSNet')
)

# Prepare your data
spectra_dict = {
    '001': {
        'positive': pd.DataFrame({
            'mz': [100.5, 200.3, 150.7],
            'MS1': [1000, 500, 750],
            'MS2': [2000, 1500, 1800]
        }),
        'negative': None  # or DataFrame for dual-mode
    }
}

# Process spectra
process_spectra('softmax_per_ionization', models, spectra_dict)

# Access predictions
predictions = spectra_dict['001']['single_model_prediction']
```

## Input Format

The classifier expects a dictionary of dictionaries with the following structure:

```python
{
    'spectrum_id': {
        'positive': pd.DataFrame or None,  # Columns: ['mz', 'MS1', 'MS2']
        'negative': pd.DataFrame or None   # Columns: ['mz', 'MS1', 'MS2']
    }
}
```

- At least one of 'positive' or 'negative' must be provided
- DataFrames must contain columns: `['mz', 'MS1', 'MS2']`
- Values represent mass-to-charge ratio and intensities

## Available Commands

### Classification
```bash
parent-ion-classifier classify -i INPUT -o OUTPUT [OPTIONS]

Options:
  -n, --norm {none,sigmoid,softmax,softmax_per_ionization}
                        Output normalization method (default: softmax_per_ionization)
  --overwrite          Allow overwriting output file
  --reset_output_index Reset DataFrame index in output
```

### Model Management
```bash
# List available models
parent-ion-classifier list

# List model groups
parent-ion-classifier list --groups

# Check model status
parent-ion-classifier status

# Download models
parent-ion-classifier download                    # Download all
parent-ion-classifier download --group test_models  # Download group
parent-ion-classifier download --model DualModeMSNet  # Download specific

# Cache management
parent-ion-classifier cache info                  # Show cache info
parent-ion-classifier cache directory             # Show cache location
parent-ion-classifier cache clear                 # Clear all cached models
parent-ion-classifier cache clear --model MODEL   # Clear specific model
```

### Testing
```bash
# Run built-in tests
parent-ion-classifier test

# Recreate test outputs (for development)
parent-ion-classifier test --recreate-output
```

## Available Models

### Production Models
- **DualModeMSNet**: Default dual-mode model (uses MultiHeaded_X_attention for best performance)
- **TopIntensityMSNet_merged**: Single ionization, trained on both modes
- **TopIntensityMSNet_positive**: Specialized for positive ionization
- **TopIntensityMSNet_negative**: Specialized for negative ionization

### Experimental Models
- **MultiHeaded_X_attention**: Best prediction performance but slower (same architecture as DualModeMSNet)
- **InterleavedEmbeddings**: Faster alternative for dual-mode with minimal performance trade-off

### Test Models
- **test_single**: For unit testing single ionization
- **test_dual**: For unit testing dual ionization

**Note:** For dual-mode spectra, `DualModeMSNet` uses the `MultiHeaded_X_attention` architecture by default for optimal prediction performance. Users processing large batches who prioritize speed can substitute `InterleavedEmbeddings` when calling `load_model()` without significantly affecting classification accuracy.

### Choosing Between Dual-Mode Models

When working with dual-mode spectra, you can choose between two models:

| Model | Performance | Speed | Best For |
|-------|------------|-------|----------|
| **DualModeMSNet** (default) | Best | Slower | High-accuracy requirements |
| **InterleavedEmbeddings** | Very good | Faster | Large batch processing, speed-critical workflows |

**Example - Using the faster model:**
```python
from parent_ion_classifier.models import load_model
from parent_ion_classifier.classifier import MSNetModels

# Use InterleavedEmbeddings for faster processing
models = MSNetModels(
    positive=load_model('TopIntensityMSNet_positive'),
    negative=load_model('TopIntensityMSNet_negative'),
    merged=load_model('TopIntensityMSNet_merged'),
    dual=load_model('InterleavedEmbeddings')  # Faster alternative
)
```

## Normalization Methods

- **`none`**: Raw model outputs
- **`sigmoid`**: Sigmoid activation applied to outputs
- **`softmax`**: Softmax over all predictions
- **`softmax_per_ionization`**: Separate softmax for positive and negative modes (recommended)

## Requirements

- Python ≥3.10
- PyTorch ≥2.2.0 (CPU version included)
- pandas
- numpy
- huggingface_hub ≥0.23
- tqdm
- scikit-learn
- matplotlib
- seaborn

All dependencies are automatically installed with conda.

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed installation instructions and troubleshooting
- **[Quick Start Guide](docs/QUICKSTART.md)**: Get started in 5 minutes
- **[API Reference](docs/API.md)**: Complete API documentation
- **[Architecture Guide](docs/ARCHITECTURE.md)**: Design decisions and internals
- **[Contributing Guide](docs/CONTRIBUTING.md)**: Development setup and guidelines

## Citation

If you use this package in your research, please cite:

```
Katz, A. (2025). Parent Ion Classifier: Deep Learning for MS/MS Parent Ion Classification.
Open University of Israel & Weizmann Institute of Science.
https://github.com/Kezahle/parent_ion_classifier
```

*Formal citation to be updated upon publication.*

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to get started.

## Support

- **Package Page**: [Anaconda.org](https://anaconda.org/ketzahle/parent_ion_classifier)
- **Issues**: [GitHub Issues](https://github.com/Kezahle/parent_ion_classifier/issues)
- **Source Code**: [GitHub Repository](https://github.com/Kezahle/parent_ion_classifier)

## Authors

- **Ariel Katz** - *Main Developer*

## Acknowledgments

Developed as part of a research thesis at the **Open University of Israel** with collaboration with the **Weizmann Institute of Science**.