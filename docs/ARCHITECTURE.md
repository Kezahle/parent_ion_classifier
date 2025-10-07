# Architecture Documentation

This document describes the architecture and design decisions of the Parent Ion Classifier package.

## Overview

The Parent Ion Classifier is organized into modular components that handle different aspects of the classification pipeline:

```
Data Input → Canonization → Model Inference → Normalization → Output
```

## Package Structure

```
parent_ion_classifier/
├── config/              # Configuration management
│   ├── __init__.py
│   ├── model_config.py  # Model configuration loading
│   └── model_config.json # Model definitions
├── models/              # Model management
│   ├── __init__.py
│   ├── loader.py        # HuggingFace model loading
│   └── manager.py       # Generic model management
├── test/                # Test data and fixtures
│   ├── spectra_dict.pkl
│   └── spectra_dict_*_norm.pkl
├── classifier.py        # Core classification logic
├── data_canonizer.py    # Data preprocessing
├── main.py             # CLI entry point
├── test.py             # Unit tests
└── utils.py            # Utility functions
```

## Core Components

### 1. Data Canonization (`data_canonizer.py`)

**Purpose**: Standardize MS data for model input

**Key Features**:
- Deterministic sorting with tie-breaking using original index
- Top-N peak selection with parent ion preservation
- Intensity scaling and normalization
- Missing value handling
- Cross-platform reproducibility

**Design Decision**: The `_original_index` tie-breaker ensures identical results across different platforms/pandas versions when intensity values are equal.

```python
# Deterministic sorting example
df['_original_index'] = range(len(df))
df.sort_values(by=['MS2', 'MS1', '_original_index'], ascending=False)
```

### 2. Model Management (`models/`)

**Purpose**: Handle model downloading, caching, and loading from HuggingFace Hub

**Architecture**:
- `manager.py`: Generic reusable model management (can be used in other projects)
- `loader.py`: Project-specific wrapper with convenience functions
- Lazy initialization of ModelManager singleton
- Automatic model downloading with progress bars

**Key Features**:
- Cache-first approach (checks local cache before downloading)
- Model groups for organizing related models
- Revision pinning for reproducibility
- Cache management (info, clearing)

### 3. Configuration (`config/`)

**Purpose**: Centralized model configuration

**Format** (`model_config.json`):
```json
{
  "repo_id": "katz-ariel/parent_ion_classifier",
  "revision": "main",
  "models": {
    "model_name": {
      "filename": "model.pt_jit",
      "description": "Model description"
    }
  },
  "model_groups": {
    "production_models": ["model1", "model2"],
    "test_models": ["test_model"]
  }
}
```

**Design Decision**: Separating model configuration from code allows easy model updates without code changes.

### 4. Classification Pipeline (`classifier.py`)

**Purpose**: Core inference logic

**Key Components**:
- `MSNetModels`: Dataclass holding the 4 model types
- `process_spectra()`: Main entry point for batch processing
- `process_spectrum()`: Single spectrum processing
- Normalization methods: none, sigmoid, softmax, softmax_per_ionization

**Data Flow**:
```
Input Dict → DataCanonizer → Tensors → Models → Normalization → Output Dict
```

**Design Decision**: Models return raw logits; normalization is applied separately for flexibility.

### 5. Command-Line Interface (`main.py`)

**Purpose**: User-facing CLI

**Architecture**:
- Subcommand-based (classify, download, status, list, cache, test)
- Argparse for robust argument handling
- Handler functions for each command group

**Design Decision**: Subcommands provide a clean, extensible interface similar to git/conda.

## Key Design Patterns

### 1. Dependency Injection

Models are passed as parameters rather than loaded internally:

```python
def process_spectra(normalization_method: str, models: MSNetModels, spectra_dict: dict):
    # Uses provided models
```

**Benefits**:
- Easier testing with mock models
- Flexibility in model selection
- Clear dependencies

### 2. Immutable Configuration

Configuration is loaded once and stored in frozen dataclasses:

```python
@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    revision: str
    models: Dict[str, ModelSpec]
```

**Benefits**:
- Thread-safe
- Prevents accidental modification
- Clear contract

### 3. Separation of Concerns

Each module has a single responsibility:
- `data_canonizer.py`: Data preprocessing only
- `classifier.py`: Inference logic only
- `models/`: Model management only

### 4. Progressive Enhancement

Features are added without breaking existing functionality:
- `_original_index` for deterministic sorting (backwards compatible)
- Model groups (optional, defaults to all models)
- Multiple normalization methods (optional, defaults to softmax_per_ionization)

## Data Structures

### Input Format

```python
{
    'spectrum_id': {
        'positive': pd.DataFrame({'mz': [...], 'MS1': [...], 'MS2': [...]}),
        'negative': pd.DataFrame or None
    }
}
```

### Output Format

```python
{
    'spectrum_id': {
        'positive': pd.DataFrame,  # Original data
        'negative': pd.DataFrame or None,
        'single_model_prediction': pd.DataFrame,  # Predictions for single mode
        'merged_model_prediction': pd.DataFrame,  # Predictions from merged model
        'dual_model_prediction': pd.DataFrame  # Predictions for dual mode (if applicable)
    }
}
```

### Prediction DataFrames

Columns: `['mz', 'MS1', 'MS2', 'parent']`
- `parent`: Probability that this peak is the parent ion (0-1 after normalization)
- Rows correspond to canonized peaks
- NaN indicates missing/invalid peaks

## Testing Strategy

### Unit Tests (`test.py`)

- Tests all normalization methods
- Cross-platform reproducibility validation
- Tolerance-based comparison (rtol=1e-4, atol=1e-6)
- Deterministic test data

### Test Models

- `test_single`: Lightweight model for single ionization testing
- `test_dual`: Lightweight model for dual ionization testing
- Hosted on HuggingFace Hub for consistent testing

### Continuous Testing

```bash
parent-ion-classifier test
```

Runs automatically in conda build process (optional).

## Performance Considerations

### Memory

- Processes spectra one at a time (not batch)
- Models kept in memory once loaded
- DataFrame operations use in-place modifications where safe

### Speed

- Model loading is cached (loaded once per session)
- Batch processing with tqdm progress bars
- Efficient pandas operations (vectorized where possible)

### Scalability

Current design handles:
- 1000s of spectra per run
- Models up to ~500MB each
- Datasets with 100K+ peaks per spectrum (truncated to top-150)

## Future Extensibility

### Adding New Models

1. Add model to HuggingFace Hub
2. Update `model_config.json`
3. No code changes required

### Adding New Normalization Methods

1. Add method to `apply_output_normalization()` in `classifier.py`
2. Add choice to CLI in `main.py`
3. Add test case in `test.py`

### Supporting New File Formats

1. Add loader/saver to `utils.py`
2. Update CLI to accept new format
3. Maintain pickle as default for backwards compatibility

## Error Handling

### Philosophy

- Fail fast with clear error messages
- Validate inputs early
- Provide actionable error messages
- Use appropriate exception types

### Examples

```python
# Good: Clear, actionable
if model_name not in config.models:
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Available: {list(config.models.keys())}"
    )

# Good: Suggests fix
if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    print("Please check the path and try again.")
    sys.exit(1)
```

## Logging

Currently uses `print()` for user feedback. Future versions may add proper logging:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing spectra...")
```

## Dependencies

### Core

- PyTorch: Model inference
- Pandas: Data manipulation
- NumPy: Numerical operations
- HuggingFace Hub: Model downloads

### Philosophy

- Minimize dependencies
- Pin major versions for stability
- Use widely-adopted packages

## Versioning

Follows Semantic Versioning (SemVer):
- MAJOR: Breaking API changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes

Current: 1.0.0
