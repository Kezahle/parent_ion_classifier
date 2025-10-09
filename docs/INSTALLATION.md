# Installation Guide

Complete installation instructions for Parent Ion Classifier.

## Table of Contents

- [Quick Install (Conda)](#quick-install-conda)
- [pip Install](#pip-install-coming-soon)
- [Install from Source](#install-from-source)
- [Verifying Installation](#verifying-installation)
- [First-Time Setup](#first-time-setup)
- [Troubleshooting](#troubleshooting)
- [Uninstalling](#uninstalling)

## Quick Install (Conda)

**Recommended for most users.** This is the easiest and most reliable installation method.

### Prerequisites

- Anaconda or Miniconda installed
- Internet connection (for downloading package and models)

### Installation Steps

```bash
# 1. Create a new conda environment (recommended)
conda create -n parent_classifier python=3.10
conda activate parent_classifier

# 2. Install the package
conda install -c ketzahle parent_ion_classifier

# 3. Download models (one-time setup)
parent-ion-classifier download
```

**That's it!** Skip to [Verifying Installation](#verifying-installation).

### Alternative: Install Without Creating New Environment

```bash
# Activate your existing environment
conda activate myenv

# Install the package
conda install -c ketzahle parent_ion_classifier

# Download models
parent-ion-classifier download
```

### Specifying Python Version

```bash
# For Python 3.12
conda create -n parent_classifier python=3.12
conda activate parent_classifier
conda install -c ketzahle parent_ion_classifier

# For Python 3.10 (minimum supported)
conda create -n parent_classifier python=3.10
conda activate parent_classifier
conda install -c ketzahle parent_ion_classifier
```

## pip Install (Coming Soon)

PyPI distribution is planned for a future release.

```bash
# This will work in a future version:
pip install parent-ion-classifier
```

For now, please use conda or install from source.

## Install from Source

Install from source if you want to:
- Contribute to development
- Use the latest unreleased features
- Modify the package for your needs

### Option 1: Development Installation (Recommended for Contributors)

```bash
# 1. Clone the repository
git clone https://github.com/Kezahle/parent_ion_classifier.git
cd parent_ion_classifier

# 2. Create development environment from YAML file
conda env create -f environment.yml
conda activate parent_classifier_dev

# 3. Install in development mode
pip install -e .

# 4. Download models
parent-ion-classifier download
```

**Development mode (`-e`)** means:
- Changes to source code take effect immediately
- No need to reinstall after modifying code
- Perfect for development and testing

### Option 2: Regular Installation from Source

```bash
# 1. Clone the repository
git clone https://github.com/Kezahle/parent_ion_classifier.git
cd parent_ion_classifier

# 2. Create a conda environment
conda create -n parent_classifier python=3.10
conda activate parent_classifier

# 3. Install dependencies
conda install pytorch-cpu numpy pandas scikit-learn matplotlib seaborn tqdm requests onnx
pip install huggingface_hub

# 4. Install the package
pip install .

# 5. Download models
parent-ion-classifier download
```

### Option 3: Build Conda Package from Source

```bash
# 1. Clone and setup
git clone https://github.com/Kezahle/parent_ion_classifier.git
cd parent_ion_classifier

# 2. Create build environment
conda create -n build_env python=3.12 conda-build
conda activate build_env

# 3. Build the package
conda build conda-recipe --output-folder bin

# 4. Create installation environment
conda create -n parent_classifier python=3.10
conda activate parent_classifier

# 5. Install from local build
conda install parent_ion_classifier -c file://$(pwd)/bin

# 6. Download models
parent-ion-classifier download
```

## Verifying Installation

After installation, verify everything works:

### 1. Check Command Availability

```bash
# Check if commands are available
parent-ion-classifier --help
classify_parent_ion --help
```

Expected output: Help message showing available commands.

### 2. Check Python Import

```bash
python -c "import parent_ion_classifier; print('Import successful')"
```

Expected output: `Import successful`

### 3. Check Version

```bash
python -c "import parent_ion_classifier; print(parent_ion_classifier.__version__)"
```

Expected output: Version number (e.g., `1.0.0`)

### 4. Run Tests

```bash
parent-ion-classifier test
```

Expected output: All tests should pass with messages like:
```
✓ Normalization method 'none' test passed
✓ Normalization method 'sigmoid' test passed
✓ Normalization method 'softmax' test passed
✓ Normalization method 'softmax_per_ionization' test passed
All tests passed successfully!
```

## First-Time Setup

After installation, you need to download the models (one-time only).

### Download All Models

```bash
parent-ion-classifier download
```

This downloads all available models (~500 MB total).

### Download Specific Model Group

```bash
# Just production models (recommended)
parent-ion-classifier download --group production_models

# Just test models (for development)
parent-ion-classifier download --group test_models
```

### Download Specific Model

```bash
parent-ion-classifier download --model DualModeMSNet
```

### Check Download Status

```bash
# See what's downloaded
parent-ion-classifier status

# List all available models
parent-ion-classifier list

# Show cache information
parent-ion-classifier cache info
```

## Platform-Specific Notes

### macOS

Works on both Intel and Apple Silicon (M1/M2/M3) Macs:

```bash
# Same installation for both architectures
conda install -c ketzahle parent_ion_classifier
```

**Note:** The package is built as `noarch`, so it works on all Mac architectures.

### Linux

Tested on Ubuntu, CentOS, and other major distributions:

```bash
conda install -c ketzahle parent_ion_classifier
```

### Windows

The package should work on Windows, but has limited testing:

```bash
conda install -c ketzahle parent_ion_classifier
```

If you encounter issues on Windows, please report them on [GitHub Issues](https://github.com/Kezahle/parent_ion_classifier/issues).

## Troubleshooting

### Command Not Found

**Problem:** `parent-ion-classifier: command not found`

**Solutions:**
1. Make sure you activated the conda environment:
   ```bash
   conda activate parent_classifier
   ```

2. Verify installation:
   ```bash
   conda list | grep parent_ion_classifier
   ```

3. Reinstall if needed:
   ```bash
   conda install --force-reinstall -c ketzahle parent_ion_classifier
   ```

### Import Error

**Problem:** `ModuleNotFoundError: No module named 'parent_ion_classifier'`

**Solutions:**
1. Check you're in the correct environment:
   ```bash
   conda env list
   # The active environment has a * next to it
   ```

2. Verify Python can find the package:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. Reinstall the package:
   ```bash
   pip uninstall parent-ion-classifier
   conda install -c ketzahle parent_ion_classifier
   ```

### Model Download Fails

**Problem:** Models won't download or connection errors

**Solutions:**
1. Check internet connection

2. Try downloading specific models:
   ```bash
   parent-ion-classifier download --model test_single
   ```

3. Check HuggingFace Hub status (models are hosted there)

4. Manually set cache directory:
   ```bash
   export HF_HOME=/path/to/cache
   parent-ion-classifier download
   ```

5. Use a VPN if HuggingFace is blocked in your region

### Conda Solver Issues

**Problem:** Conda takes forever to solve environment or fails

**Solutions:**
1. Use mamba (faster conda alternative):
   ```bash
   conda install -n base conda-forge::mamba
   mamba install -c ketzahle parent_ion_classifier
   ```

2. Create a clean environment:
   ```bash
   conda create -n parent_classifier_clean python=3.10
   conda activate parent_classifier_clean
   conda install -c ketzahle parent_ion_classifier
   ```

3. Update conda:
   ```bash
   conda update -n base conda
   ```

### PyTorch Issues

**Problem:** PyTorch not working or CUDA errors

**Solution:** The package uses `pytorch-cpu` by default. If you need GPU support:

```bash
# Uninstall CPU version
conda remove pytorch-cpu

# Install GPU version (if you have CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Permission Errors

**Problem:** Permission denied when installing or downloading models

**Solutions:**
1. Don't use sudo with conda:
   ```bash
   # Wrong
   sudo conda install -c ketzahle parent_ion_classifier
   
   # Correct
   conda install -c ketzahle parent_ion_classifier
   ```

2. Check conda directory permissions:
   ```bash
   ls -la ~/miniconda3/envs/
   ```

3. Create environment in a writable location:
   ```bash
   conda create -p ./local_env python=3.10
   conda activate ./local_env
   conda install -c ketzahle parent_ion_classifier
   ```

### Test Failures

**Problem:** `parent-ion-classifier test` fails

**Solutions:**
1. Make sure test models are downloaded:
   ```bash
   parent-ion-classifier download --group test_models
   ```

2. Check model status:
   ```bash
   parent-ion-classifier status
   ```

3. Clear cache and re-download:
   ```bash
   parent-ion-classifier cache clear
   parent-ion-classifier download --group test_models
   ```

4. Report the issue on GitHub with the full error message

### Slow Performance

**Problem:** Classification is very slow

**Solutions:**
1. Use faster dual-mode model for large batches:
   ```python
   from parent_ion_classifier.models import load_model
   dual_model = load_model('InterleavedEmbeddings')  # Faster
   ```

2. Consider GPU version of PyTorch (see PyTorch Issues above)

3. Process in batches rather than one-by-one

## Upgrading

### Upgrade to Latest Version

```bash
# Activate environment
conda activate parent_classifier

# Update package
conda update -c ketzahle parent_ion_classifier

# Download any new models
parent-ion-classifier download
```

### Check for Updates

```bash
# Check current version
python -c "import parent_ion_classifier; print(parent_ion_classifier.__version__)"

# Check available versions on conda
conda search -c ketzahle parent_ion_classifier
```

### Upgrade from Development Installation

```bash
# Navigate to repository
cd parent_ion_classifier

# Pull latest changes
git pull origin main

# Reinstall
pip install -e . --force-reinstall
```

## Uninstalling

### Remove Package Only

```bash
# Activate environment
conda activate parent_classifier

# Uninstall package
conda remove parent_ion_classifier
```

### Remove Package and Environment

```bash
# Deactivate environment first
conda deactivate

# Remove entire environment
conda env remove -n parent_classifier
```

### Clear Downloaded Models

```bash
# Before uninstalling, optionally clear model cache
parent-ion-classifier cache clear

# Or manually delete cache directory
rm -rf ~/.cache/huggingface/hub/models--katz-ariel--parent_ion_classifier
```

## System Requirements

### Minimum Requirements

- **OS:** Linux, macOS 10.13+, or Windows 10+
- **Python:** 3.10 or higher
- **RAM:** 4 GB
- **Disk Space:** 2 GB (package + models)
- **Internet:** Required for initial setup and model downloads

### Recommended Requirements

- **OS:** Linux (Ubuntu 20.04+) or macOS 11+
- **Python:** 3.12
- **RAM:** 8 GB
- **Disk Space:** 5 GB
- **CPU:** Modern multi-core processor
- **GPU:** Optional, but speeds up inference

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues:** [GitHub Issues](https://github.com/Kezahle/parent_ion_classifier/issues)
2. **Ask a question:** [GitHub Discussions](https://github.com/Kezahle/parent_ion_classifier/discussions)
3. **Report a bug:** Open a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce
   - Output of `conda list`

## Next Steps

After successful installation:

1. Read the [README](../README.md) for usage examples
2. Check the [QUICKSTART](QUICKSTART.md) for a 5-minute guide
3. Review the [API Documentation](API.md) for detailed reference
4. Run `parent-ion-classifier --help` to see all commands

## Additional Resources

- **Package Page:** https://anaconda.org/ketzahle/parent_ion_classifier
- **Source Code:** https://github.com/Kezahle/parent_ion_classifier
- **Model Repository:** https://huggingface.co/katz-ariel/parent_ion_classifier
- **Documentation:** https://github.com/Kezahle/parent_ion_classifier/tree/main/docs