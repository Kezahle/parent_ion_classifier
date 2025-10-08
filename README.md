# Parent Ion Classifier

A Python package for classifying parent ions in mass spectrometry (MS/MS) experiments using deep learning models.

Developed as part of research at the Weizmann Institute of Science, this package leverages PyTorch-based neural networks to detect and classify parent ions from MS data, providing insights for chemical analysis and compound identification.

## Features

- **Deep learning models** for MS/MS parent ion classification
- **Multiple ionization modes**: Support for positive, negative, merged, and dual-mode spectra
- **Automatic model management**: Downloads and caches models from Hugging Face Hub
- **Multiple normalization methods**: Flexible output normalization (softmax, sigmoid, per-ionization)
- **Command-line interface**: Easy integration into analysis workflows
- **Cross-platform reproducibility**: Deterministic results across different systems
- **Built-in testing**: Verify installation with included test suite

## Installation

### From Conda (Recommended)

```bash
# Create a new environment
conda create -n parent_classifier python=3.12
conda activate parent_classifier

# Install from local build
conda install parent_ion_classifier -c file:///path/to/bin
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Kezahle/parent_ion_classifier.git
cd parent_ion_classifier

# Build the package
./scripts/build_and_test.sh build

# Install in a new environment
conda create -n parent_classifier python=3.12
conda activate parent_classifier
conda install parent_ion_classifier -c "file://$PWD/bin"
```

## Quick Start

### Command Line Usage

```bash
# Classify spectra from a pickle file
parent-ion-classifier classify -i input.pkl -o output.pkl

# Use different normalization methods
parent-ion-classifier classify -i input.pkl -o output.pkl -n softmax_per_ionization

# Download models before classification
parent-ion-classifier download --group production_models

# Run tests to verify installation
parent-ion-classifier test
```

### Python API Usage

```python
import pandas as pd
from parent_ion_classifier import DataCanonizer, process_spectra
from parent_ion_classifier.models import load_model
from parent_ion_classifier.classifier import MSNetModels

# Load models
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
            'mz': [...],
            'MS1': [...],
            'MS2': [...]
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
- **DualModeMSNet**: Dual-mode spectra classification
- **TopIntensityMSNet_merged**: Single ionization, trained on both modes
- **TopIntensityMSNet_positive**: Specialized for positive ionization
- **TopIntensityMSNet_negative**: Specialized for negative ionization

### Experimental Models
- **MultiHeaded_X_attention**: Best performance with cross-attention
- **InterleavedEmbeddings**: Balanced complexity and performance

### Test Models
- **test_single**: For unit testing single ionization
- **test_dual**: For unit testing dual ionization

## Normalization Methods

- **`none`**: Raw model outputs
- **`sigmoid`**: Sigmoid activation applied to outputs
- **`softmax`**: Softmax over all predictions
- **`softmax_per_ionization`**: Separate softmax for positive and negative modes (recommended)

## Development

### Building from Source

```bash
# Build package
./scripts/build_script.sh

# Set up test environment
./scripts/test_script.sh

# Build and test in one command
./scripts/build_and_test.sh all
```

### Project Structure

```
parent_ion_classifier/
├── src/parent_ion_classifier/    # Main package
│   ├── config/                   # Configuration management
│   ├── models/                   # Model loading and management
│   ├── test/                     # Test data and fixtures
│   ├── classifier.py             # Core classification logic
│   ├── data_canonizer.py         # Data preprocessing
│   ├── main.py                   # CLI entry point
│   └── utils.py                  # Utility functions
├── scripts/                      # Build and test scripts
├── conda-recipe/                 # Conda packaging
├── docs/                         # Documentation
└── tests/                        # Additional tests (if any)
```

### Running Tests

Tests ensure cross-platform reproducibility and validate model outputs:

```bash
conda activate test_env
parent-ion-classifier test
```

Tests compare outputs across different normalization methods and verify deterministic behavior.

## Requirements

- Python ≥3.10
- PyTorch ≥2.2.0
- pandas <2.0.0
- numpy <2
- huggingface_hub ≥0.23
- tqdm

## Citation

If you use this package in your research, please cite:

```
[Citation information to be added]
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- **Issues**: [GitHub Issues](https://github.com/Kezahle/parent_ion_classifier/issues)
- **Documentation**: [GitHub Repository](https://github.com/Kezahle/parent_ion_classifier)

## Authors

- Ariel Katz

## Acknowledgments

Developed as part of a research Thesis at the Open University of Israel with collaboration with Weizmann Institute of Science.
