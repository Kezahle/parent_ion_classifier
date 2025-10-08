# API Reference

Complete API documentation for the Parent Ion Classifier package.

## Table of Contents

- [Core Functions](#core-functions)
- [Data Preprocessing](#data-preprocessing)
- [Model Management](#model-management)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Core Functions

### classifier.classify_parent_ions

Main entry point for classification from files.

```python
def classify_parent_ions(
    input_file: str,
    output_file: str,
    normalization_method: str,
    reset_output_index: bool = False
) -> None
```

**Parameters**:
- `input_file` (str): Path to input pickle file containing spectra dictionary
- `output_file` (str): Path where results will be saved
- `normalization_method` (str): One of 'none', 'sigmoid', 'softmax', 'softmax_per_ionization'
- `reset_output_index` (bool): Whether to reset DataFrame indices in output

**Raises**:
- `OSError`: If input file cannot be loaded
- `Exception`: If models fail to load

**Example**:
```python
from parent_ion_classifier.classifier import classify_parent_ions

classify_parent_ions(
    input_file='data/spectra.pkl',
    output_file='results/predictions.pkl',
    normalization_method='softmax_per_ionization'
)
```

### classifier.process_spectra

Process a dictionary of spectra with loaded models.

```python
def process_spectra(
    normalization_method: str,
    models: MSNetModels,
    spectra_dict: dict,
    reset_output_index: bool = False
) -> None
```

**Parameters**:
- `normalization_method` (str): Normalization to apply
- `models` (MSNetModels): Loaded model collection
- `spectra_dict` (dict): Dictionary of spectra to process (modified in-place)
- `reset_output_index` (bool): Whether to reset indices

**Example**:
```python
from parent_ion_classifier.classifier import process_spectra, MSNetModels
from parent_ion_classifier.models import load_model

models = MSNetModels(
    positive=load_model('TopIntensityMSNet_positive'),
    negative=load_model('TopIntensityMSNet_negative'),
    merged=load_model('TopIntensityMSNet_merged'),
    dual=load_model('DualModeMSNet')
)

spectra_dict = {
    '001': {
        'positive': df_positive,
        'negative': None
    }
}

process_spectra('softmax_per_ionization', models, spectra_dict)
# spectra_dict now contains predictions
```

## Data Preprocessing

### DataCanonizer

Class for preprocessing MS data.

```python
class DataCanonizer:
    def __init__(
        self,
        N: int,
        data_columns: List[str],
        label_columns: List[str],
        mz_decimal_digits: int = 2,
        missing_val_replacement: int = -1,
        intensity_max: int = 10,
        mz_msb_digits: int = -1
    )
```

**Parameters**:
- `N`: Maximum number of rows (top-N by intensity)
- `data_columns`: Feature column names (must include 'mz', 'MS1', 'MS2')
- `label_columns`: Label column names (single-element list, e.g., ['parent'])
- `mz_decimal_digits`: Decimal places for m/z rounding
- `missing_val_replacement`: Value for missing data
- `intensity_max`: Maximum value for scaled intensities
- `mz_msb_digits`: Digits to keep in m/z integer part (-1 to disable)

### DataCanonizer.canonise_and_truncate_df

Main preprocessing method.

```python
def canonise_and_truncate_df(
    self,
    in_df: pd.DataFrame,
    reset_index: bool = False,
    sort_by_MS2_first: bool = True
) -> pd.DataFrame
```

**Parameters**:
- `in_df`: Input DataFrame with columns matching data_columns
- `reset_index`: Whether to reset index after processing
- `sort_by_MS2_first`: If True, sort by MS2 then MS1; otherwise MS1 then MS2

**Returns**:
- Canonized DataFrame with exactly N rows

**Example**:
```python
from parent_ion_classifier import DataCanonizer
import pandas as pd

dc = DataCanonizer(N=150, data_columns=['mz', 'MS1', 'MS2'],
                   label_columns=['parent'])

df = pd.DataFrame({
    'mz': [100.5, 200.3, 150.7, 300.1],
    'MS1': [1000, 500, 750, 2000],
    'MS2': [2000, 1500, 1800, 3000],
    'parent': [0, 1, 0, 0]
})

canonized = dc.canonise_and_truncate_df(df)
# Returns DataFrame with exactly 150 rows, sorted and scaled
```

### DataCanonizer.df_to_tensors

Convert DataFrame to PyTorch tensors.

```python
def df_to_tensors(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]
```

**Parameters**:
- `df`: Canonized DataFrame

**Returns**:
- Tuple of (data_tensor, label_tensor)
  - data_tensor: Shape (1, N, len(data_columns)), dtype float32
  - label_tensor: Shape (N,), dtype float32, or None if no labels

**Example**:
```python
data_tensor, labels = dc.df_to_tensors(canonized_df)
# data_tensor.shape: torch.Size([1, 150, 3])
```

## Model Management

### models.load_model

Load a model with automatic downloading.

```python
def load_model(
    model_type: str,
    device: str = "cpu",
    raise_exception: bool = False,
    auto_download: bool = True
) -> torch.jit.ScriptModule | None
```

**Parameters**:
- `model_type`: Name of model to load (e.g., 'DualModeMSNet')
- `device`: Device to load model on ('cpu' or 'cuda')
- `raise_exception`: If True, raises exceptions; if False, returns None on error
- `auto_download`: If True, automatically downloads missing models

**Returns**:
- Loaded PyTorch model or None

**Example**:
```python
from parent_ion_classifier.models import load_model

# Load with automatic download
model = load_model('DualModeMSNet')

# Load without auto-download (cache only)
model = load_model('DualModeMSNet', auto_download=False, raise_exception=True)
```

### models.download_model

Download a specific model.

```python
def download_model(model_name: str, force: bool = False) -> str
```

**Parameters**:
- `model_name`: Name of model to download
- `force`: If True, re-download even if cached

**Returns**:
- Path to downloaded model

**Example**:
```python
from parent_ion_classifier.models import download_model

path = download_model('DualModeMSNet')
print(f"Model downloaded to: {path}")
```

### models.download_models

Download multiple models.

```python
def download_models(
    model_names: Optional[List[str]] = None,
    group_name: Optional[str] = None,
    force: bool = False
) -> Dict[str, str]
```

**Parameters**:
- `model_names`: Specific models to download (None = all models)
- `group_name`: Download models from specific group
- `force`: If True, re-download even if cached

**Returns**:
- Dictionary mapping model names to local paths

**Example**:
```python
from parent_ion_classifier.models import download_models

# Download all models
all_models = download_models()

# Download specific group
prod_models = download_models(group_name='production_models')

# Download specific models
models = download_models(model_names=['DualModeMSNet', 'TopIntensityMSNet_merged'])
```

### models.get_available_models

Get list of available models.

```python
def get_available_models() -> List[str]
```

**Returns**:
- Sorted list of model names

**Example**:
```python
from parent_ion_classifier.models import get_available_models

models = get_available_models()
print(f"Available models: {models}")
```

### models.get_model_groups

Get list of model groups.

```python
def get_model_groups() -> List[str]
```

**Returns**:
- Sorted list of group names

**Example**:
```python
from parent_ion_classifier.models import get_model_groups

groups = get_model_groups()
# ['production_models', 'test_models', 'experimental_models']
```

### models.get_models_in_group

Get models in a specific group.

```python
def get_models_in_group(group_name: str) -> List[str]
```

**Parameters**:
- `group_name`: Name of the model group

**Returns**:
- List of model names in the group

**Example**:
```python
from parent_ion_classifier.models import get_models_in_group

models = get_models_in_group('production_models')
# ['DualModeMSNet', 'TopIntensityMSNet_merged', ...]
```

### models.is_model_cached

Check if a model is cached locally.

```python
def is_model_cached(model_name: str) -> bool
```

**Parameters**:
- `model_name`: Name of the model

**Returns**:
- True if cached, False otherwise

### models.get_cache_directory

Get the cache directory path.

```python
def get_cache_directory() -> str
```

**Returns**:
- Path to cache directory

### models.clear_model_cache

Clear cached models.

```python
def clear_model_cache(
    model_names: Optional[List[str]] = None,
    confirm: bool = True
) -> bool
```

**Parameters**:
- `model_names`: Specific models to clear (None = all)
- `confirm`: If True, asks for confirmation

**Returns**:
- True if successful

## Configuration

### config.get_config_data

Load model configuration.

```python
def get_config_data() -> ModelConfig
```

**Returns**:
- ModelConfig object with repo_id, revision, models, and model_groups

**Example**:
```python
from parent_ion_classifier.config import get_config_data

config = get_config_data()
print(f"Repository: {config.repo_id}")
print(f"Available models: {list(config.models.keys())}")
```

### ModelConfig

Configuration dataclass.

```python
@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    revision: str
    models: Dict[str, ModelSpec]
    model_groups: Dict[str, List[str]]
```

### ModelSpec

Model specification dataclass.

```python
@dataclass(frozen=True)
class ModelSpec:
    filename: str
    description: str = ""
```

## Utilities

### utils.unpickle_file

Load a pickle file.

```python
def unpickle_file(path: str) -> Any
```

**Parameters**:
- `path`: Path to pickle file

**Returns**:
- Unpickled Python object

**Example**:
```python
from parent_ion_classifier.utils import unpickle_file

data = unpickle_file('data/spectra.pkl')
```

### utils.pickle_file

Save data to pickle file.

```python
def pickle_file(data: Any, path: str) -> None
```

**Parameters**:
- `data`: Python object to pickle
- `path`: Destination path

**Example**:
```python
from parent_ion_classifier.utils import pickle_file

results = {'spectrum_001': predictions}
pickle_file(results, 'output/results.pkl')
```

## Data Classes

### MSNetModels

Container for the four model types.

```python
@dataclass
class MSNetModels:
    positive: nn.Module
    negative: nn.Module
    merged: nn.Module
    dual: nn.Module
```

**Example**:
```python
from parent_ion_classifier.classifier import MSNetModels
from parent_ion_classifier.models import load_model

models = MSNetModels(
    positive=load_model('TopIntensityMSNet_positive'),
    negative=load_model('TopIntensityMSNet_negative'),
    merged=load_model('TopIntensityMSNet_merged'),
    dual=load_model('DualModeMSNet')
)
```

## Constants

### Package-level Constants

```python
from parent_ion_classifier import (
    N,                              # 150 - Maximum peaks per spectrum
    DATA_COLUMNS,                   # ['mz', 'MS1', 'MS2']
    LABEL_COLUMN,                   # ['parent']
    MODEL_MISSING_VALUE,            # -1
    DUAL_OUTPUT_KEY,                # 'dual_model_prediction'
    MERGED_OUTPUT_KEY,              # 'merged_model_prediction'
    SINGLE_IONIZATION_OUTPUT_KEY    # 'single_model_prediction'
)
```

## Normalization Methods

### Available Normalization Methods

1. **`'none'`**: Raw model outputs (logits)
   - Use for custom post-processing

2. **`'sigmoid'`**: Sigmoid activation
   - Maps outputs to [0, 1] independently
   - Probabilities don't sum to 1

3. **`'softmax'`**: Softmax over all predictions
   - Probabilities sum to 1 across all peaks
   - Use when exactly one parent ion is expected

4. **`'softmax_per_ionization'`** (Recommended):
   - Separate softmax for positive and negative modes
   - Probabilities sum to 1 within each mode
   - Use for dual-mode spectra or when mode is uncertain

## Error Handling

### Common Exceptions

**ValueError**: Invalid model name, missing required data
```python
try:
    model = load_model('NonexistentModel', raise_exception=True)
except ValueError as e:
    print(f"Model not found: {e}")
```

**FileNotFoundError**: Input file doesn't exist
```python
try:
    data = unpickle_file('missing.pkl')
except FileNotFoundError:
    print("File not found!")
```

**LocalEntryNotFoundError**: Model not in cache and auto_download=False
```python
from huggingface_hub.utils import LocalEntryNotFoundError

try:
    model = load_jit_model('DualModeMSNet', auto_download=False)
except LocalEntryNotFoundError:
    print("Model not cached. Run: parent-ion-classifier download")
```

## Type Hints

The package uses type hints throughout. Import commonly used types:

```python
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import torch
```

## Best Practices

### 1. Load Models Once

```python
# Good: Load once, reuse
models = MSNetModels(...)
for spectrum in spectra_dict.values():
    process_spectrum(dc, method, models, spectrum)

# Bad: Load repeatedly
for spectrum in spectra_dict.values():
    models = MSNetModels(...)  # Slow!
    process_spectrum(dc, method, models, spectrum)
```

### 2. Check Cache Before Batch Processing

```python
from parent_ion_classifier.models import check_and_download_models

# Ensure models are available
check_and_download_models(group_name='production_models')

# Now process
classify_parent_ions(...)
```

### 3. Handle Missing Data

```python
spectra_dict = {
    '001': {
        'positive': df_pos if has_positive else None,
        'negative': df_neg if has_negative else None
    }
}
# At least one must be non-None
```

### 4. Use Context Managers for Files

```python
# The package handles this internally, but for custom I/O:
import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Performance Tips

1. **Batch Processing**: Process multiple spectra in one call
2. **Model Caching**: Models are cached after first load
3. **Device Selection**: Use GPU if available: `load_model(..., device='cuda')`
4. **Memory**: Process large datasets in chunks if memory-constrained

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions and internals
- [README.md](../README.md) - User guide and examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
