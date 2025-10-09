# Contributing to Parent Ion Classifier

Thank you for your interest in contributing to the Parent Ion Classifier project!

## Development Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git
- A GitHub account

### Quick Start

1. **Fork and clone the repository**
   ```bash
   # Fork on GitHub first, then:
   git clone https://github.com/YOUR_USERNAME/parent_ion_classifier.git
   cd parent_ion_classifier
   ```

2. **Create development environment**
   ```bash
   # Create environment from YAML file
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate parent_classifier_dev
   ```

3. **Install package in development mode**
   ```bash
   # Install with pip in editable mode
   pip install -e .
   
   # Verify installation
   parent-ion-classifier --help
   ```

4. **Set up pre-commit hooks** (Recommended)
   ```bash
   pre-commit install
   ```

   This will automatically run code quality checks before each commit:
   - Format code with black
   - Sort imports with isort
   - Check for common issues with ruff
   - Validate YAML/TOML files
   - Remove trailing whitespace

5. **Download models for testing**
   ```bash
   parent-ion-classifier download --group test_models
   ```

## Development Environment Details

The `environment.yml` file includes:

**Build Tools:**
- conda-build (for building conda packages)
- setuptools, pip

**Runtime Dependencies:**
- pytorch-cpu ≥2.2.0
- numpy, pandas
- huggingface_hub ≥0.23
- scikit-learn, matplotlib, seaborn
- tqdm, requests, onnx

**Development Tools:**
- pytest (testing)
- black, isort, ruff (code formatting and linting)
- mypy (type checking)
- pre-commit (git hooks)
- ipython (interactive development)

**Documentation:**
- sphinx, sphinx_rtd_theme

## Building the Conda Package

If you need to build a conda package for distribution:

### Use conda-build directly

```bash
# Activate development environment
conda activate parent_classifier_dev

# Build the package
conda build conda-recipe --output-folder bin

# Check the output
ls -lh bin/noarch/
```

### Installing Your Built Package

```bash
# Create test environment
conda create -n test_install python=3.10
conda activate test_install

# Install from local build
conda install parent_ion_classifier -c file://$(pwd)/bin

# Test installation
parent-ion-classifier --help
parent-ion-classifier test
```

## Code Style

We follow these style guidelines:

- **PEP 8** for general Python style
- **snake_case** for functions and variables
- **Type hints** on all function signatures
- **Docstrings** for all public functions and classes (Google style)
- **Line length**: 100 characters maximum
- **Single quotes** for all error messages and strings in code

**Pre-commit hooks will automatically enforce most style rules!**

### Example of Good Code Style

```python
def process_spectrum(
    spectrum_id: str,
    positive_df: pd.DataFrame | None,
    negative_df: pd.DataFrame | None,
    normalization: str = 'softmax_per_ionization'
) -> dict:
    """
    Process a single spectrum with given normalization.
    
    Args:
        spectrum_id: Unique identifier for the spectrum
        positive_df: DataFrame with positive mode data, or None
        negative_df: DataFrame with negative mode data, or None
        normalization: Normalization method to apply
        
    Returns:
        Dictionary containing processed spectrum with predictions
        
    Raises:
        ValueError: If both positive_df and negative_df are None
        
    Example:
        >>> df = pd.DataFrame({'mz': [100], 'MS1': [1000], 'MS2': [2000]})
        >>> result = process_spectrum('001', df, None)
    """
    if positive_df is None and negative_df is None:
        raise ValueError(f'Spectrum {spectrum_id} must have at least one mode')
    
    # Implementation here
    return result
```

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-model-support`
- `fix/normalization-bug`
- `docs/improve-api-reference`

### 2. Make Your Changes

- Write clear, self-documenting code
- Add type hints to all functions
- Write docstrings for public APIs
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run pre-commit checks
pre-commit run --all-files

# Run unit tests
parent-ion-classifier test

# Test manually
parent-ion-classifier classify -i test_data.pkl -o output.pkl

# Test in fresh environment (optional but recommended)
conda create -n test_changes python=3.10
conda activate test_changes
pip install -e .
parent-ion-classifier test
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add feature: description of changes"
# Pre-commit hooks run automatically here!
```

Write clear commit messages:
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Be specific: "Fix sigmoid normalization for dual-mode spectra"
- Reference issues if applicable: "Fix #123: Handle empty dataframes"

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub:
- Provide a clear description of changes
- Reference related issues
- Include screenshots for UI changes
- List any breaking changes

## Code Quality Tools

### Running Quality Checks Manually

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type check (optional)
mypy src/
```

### Pre-commit Hook Configuration

The `.pre-commit-config.yaml` file includes:
- **black**: Code formatter (100 char lines)
- **isort**: Import sorter
- **ruff**: Fast Python linter
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure newline at EOF
- **check-yaml**: Validate YAML files
- **check-toml**: Validate TOML files

## Testing

### Running Tests

```bash
# Run built-in tests
parent-ion-classifier test

# Run with pytest (if you add more tests)
pytest tests/ -v
```

### Adding New Tests

When adding new functionality:

1. **Add test data** to `src/parent_ion_classifier/test/`
2. **Update test cases** in `src/parent_ion_classifier/test.py`
3. **Ensure tests pass** on your platform
4. **Test cross-platform** if possible (Linux, macOS, Windows)

### Writing Good Tests

```python
def test_normalization_methods():
    """Test that all normalization methods work correctly."""
    # Arrange
    spectra_dict = create_test_spectra()
    models = load_test_models()
    
    # Act
    for method in ['none', 'sigmoid', 'softmax', 'softmax_per_ionization']:
        process_spectra(method, models, spectra_dict)
        
        # Assert
        assert 'single_model_prediction' in spectra_dict['001']
        assert spectra_dict['001']['single_model_prediction'] is not None
```

### Cross-Platform Testing

The project aims for reproducibility across platforms:
- Test on Linux and macOS if possible
- Ensure deterministic behavior
- Use appropriate tolerances for float comparisons: `rtol=1e-4, atol=1e-6`
- Avoid platform-specific operations

## Type Hints

All new code should include type hints:

```python
from typing import Optional, Dict, List
import pandas as pd

# Good - with type hints
def load_spectra(path: str, validate: bool = True) -> Dict[str, pd.DataFrame]:
    """Load spectra from file."""
    pass

# Bad - no type hints
def load_spectra(path, validate=True):
    """Load spectra from file."""
    pass
```

Use modern Python type hints:
- `str | None` instead of `Optional[str]` (Python 3.10+)
- `list[str]` instead of `List[str]` (Python 3.10+)
- `dict[str, int]` instead of `Dict[str, int]` (Python 3.10+)

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def canonize_spectrum(
    df: pd.DataFrame,
    n_peaks: int = 150,
    sort_by_ms2: bool = True
) -> pd.DataFrame:
    """
    Canonize MS spectrum for model input.
    
    This function standardizes the input spectrum by sorting peaks,
    selecting top-N by intensity, and normalizing values.
    
    Args:
        df: Input DataFrame with columns ['mz', 'MS1', 'MS2']
        n_peaks: Number of peaks to keep (default: 150)
        sort_by_ms2: If True, sort by MS2 intensity first (default: True)
        
    Returns:
        Canonized DataFrame with exactly n_peaks rows
        
    Raises:
        ValueError: If df doesn't contain required columns
        
    Example:
        >>> df = pd.DataFrame({'mz': [100, 200], 'MS1': [1, 2], 'MS2': [3, 4]})
        >>> canonized = canonize_spectrum(df, n_peaks=150)
        >>> len(canonized)
        150
    """
    pass
```

### Updating Documentation

- **README.md**: Update for user-facing changes
- **API.md**: Update for API changes
- **ARCHITECTURE.md**: Update for design changes
- **Docstrings**: Always update with code changes

## Error Messages

All error messages should:
- Use **single quotes** for consistency
- Be clear and actionable
- Suggest how to fix the problem

```python
# Good
if model_name not in available_models:
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Available models: {', '.join(available_models)}. "
        f"Run 'parent-ion-classifier list' to see all models."
    )

# Bad
if model_name not in available_models:
    raise ValueError("Model not found")
```

## Project Structure

```
parent_ion_classifier/
├── src/parent_ion_classifier/   # Main package source
│   ├── config/                  # Configuration management
│   ├── models/                  # Model loading and management
│   ├── test/                    # Test data and fixtures
│   ├── classifier.py            # Core classification logic
│   ├── data_canonizer.py        # Data preprocessing
│   ├── main.py                  # CLI entry point
│   ├── test.py                  # Unit tests
│   └── utils.py                 # Utility functions
├── conda-recipe/                # Conda packaging files
│   └── meta.yaml
├── docs/                        # Documentation
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   ├── INSTALLATION.md
│   └── QUICKSTART.md
├── tests/                       # Additional pytest tests (future)
├── environment.yml              # Conda development environment
├── pyproject.toml              # Package configuration
├── LICENSE                      # MIT License
└── README.md                    # Main documentation
```

## Release Process (For Maintainers)

### 1. Update Version

Update version in:
- `pyproject.toml`
- `conda-recipe/meta.yaml`
- `src/parent_ion_classifier/__init__.py` (if version constant exists)

### 2. Build Package

```bash
conda activate parent_classifier_dev
conda build conda-recipe --output-folder bin
```

### 3. Upload to Anaconda.org

```bash
anaconda login
anaconda upload bin/noarch/parent_ion_classifier-*.tar.bz2
```

### 4. Create GitHub Release

- Tag the release: `git tag v1.0.1`
- Push tag: `git push origin v1.0.1`
- Create release on GitHub
- Attach conda package to release
- Write release notes

### 5. Update Documentation

- Update README badges if needed
- Announce on relevant channels

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email the maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical aspects
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Every contribution helps make Parent Ion Classifier better. We appreciate your time and effort!