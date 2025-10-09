# Architecture Documentation

This document describes the architecture and design decisions of the Parent Ion Classifier package.

## Overview

The Parent Ion Classifier is organized into modular components that handle different aspects of the classification pipeline:

```
Data Input → Canonization → Model Inference → Normalization → Output
```

## Distribution Strategy

### Package Distribution

The package is distributed primarily via Anaconda.org:
- **Primary Channel**: https://anaconda.org/ketzahle/parent_ion_classifier
- **Installation**: `conda install -c ketzahle parent_ion_classifier`
- **Build Type**: `noarch` (platform-independent pure Python)
- **Package Size**: ~120 KB (lightweight)
- **Model Storage**: HuggingFace Hub (downloaded on-demand)
- **Total Models**: ~500 MB (cached locally after download)

This architecture keeps the package small while models are cached locally on first use.

### Supported Platforms

Pure Python implementation with `noarch` build ensures cross-platform compatibility:
- **Linux**: x86_64, ARM64
- **macOS**: Intel (x86_64), Apple Silicon (ARM64)
- **Windows**: x86_64

### Build System

The package uses modern Python packaging standards:
- **`pyproject.toml`**: Package metadata (PEP 517/518)
- **`conda-recipe/meta.yaml`**: Conda package specification
- **`noarch: python`**: Platform-independent build
- **No compiled extensions**: Pure Python for maximum compatibility

### Future Distribution

- **PyPI** (pip install): Planned for future release
- **Conda-Forge**: Potential submission to main conda channel

## Package Structure

```
parent_ion_classifier/
|-- config/              # Configuration management
|   |-- __init__.py
|   |-- model_config.py  # Model configuration loading
|   `-- model_config.json # Model definitions
|-- models/              # Model management
|   |-- __init__.py
|   |-- loader.py        # HuggingFace model loading
|   `-- manager.py       # Generic model management
|-- test/                # Test data and fixtures
|   |-- spectra_dict.pkl
|   `-- spectra_dict_*_norm.pkl
|-- classifier.py        # Core classification logic
|-- data_canonizer.py    # Data preprocessing
|-- main.py             # CLI entry point
|-- test.py             # Unit tests
`-- utils.py            # Utility functions
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

**Model Aliasing**: The configuration supports multiple model names referencing the same file for flexibility and backward compatibility. Specifically, `DualModeMSNet` and `MultiHeaded_X_attention` both point to `mh_x_attention.pt_jit`.

**Design Rationale**: Two dual-mode models were developed:
1. **MultiHeaded_X_attention** (`mh_x_attention.pt_jit`): Uses multi-headed cross-attention, providing marginally better prediction performance but with slower inference and training times
2. **InterleavedEmbeddings** (`interleaved.pt_jit`): Faster alternative with interleaved embeddings, offering good performance with quicker processing

`DualModeMSNet` is configured to use the MultiHeaded_X_attention architecture by default, prioritizing prediction accuracy for the standard use case. However, users processing large batches who prefer faster inference can easily substitute `InterleavedEmbeddings` by loading it directly:

```python
# Default: Best performance
dual_model = load_model('DualModeMSNet')

# Alternative: Faster processing
dual_model = load_model('InterleavedEmbeddings')
```

When a model is loaded, only one copy is downloaded and cached regardless of which alias is used.

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

**Design Decisions**:
- Separating model configuration from code allows easy model updates without code changes
- Model aliases (multiple names for same file) enable backward compatibility
- Model groups facilitate bulk operations like downloading all production models

### Import Order in __init__.py

Constants are defined before imports to avoid circular dependencies:
- `classifier.py` imports constants from `__init__.py`
- `__init__.py` imports functions from `classifier.py`
- `DataCanonizer` is imported directly in `classifier.py` to break the cycle

### 4. Classification Pipeline (`classifier.py`)

**Purpose**: Core inference logic

**Key Components**:
- `MSNetModels`: Dataclass holding the 4 model types
- `process_spectra()`: Main entry point for batch processing
- `process_spectrum()`: Single spectrum processing
- Normalization methods: none, sigmoid, softmax, softmax_per_ionization

**Data Flow**:
```
Input Dict → DataCanonizer → Tensors → Models → Raw Outputs → Normalization → Output Dict
```

**Design Decision**: Models return raw logits as tensors that must be squeezed (`.squeeze()`) before normalization. Normalization is applied separately for flexibility, allowing users to choose the most appropriate method for their use case.

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
- Use single quotes for all error messages (consistency)

### Examples

```python
# Good: Clear, actionable, single quotes
if model_name not in config.models:
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Available: {list(config.models.keys())}"
    )

# Good: Suggests fix
if not os.path.exists(input_file):
    print(f'File not found: {input_file}')
    print('Please check the path and try again.')
    sys.exit(1)
```

## Logging

Currently uses `print()` for user feedback. Future versions may add proper logging:

```python
import logging

logger = logging.getLogger(__name__)
logger.info('Processing spectra...')
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