"""Parent Ion Classifier Package

A package for classifying parent ions in mass spectrometry (MS/MS) experiments
using deep learning models.
"""

TENSOR_PRINT_PRECISION = 4
TENSOR_PRINT_SCI_MODE = False

# Maximum number of rows outputted by an MS1/MS2 experiment
N = 150
# The number of channels (depth) outputted from the first layer of convolution
C = 10

LABEL_COLUMN = ["parent"]
MZ_COLUMN = "mz"
DATA_COLUMNS = [MZ_COLUMN] + ["MS1", "MS2"]

MODEL_MISSING_VALUE = -1
OUTPUT_KEY = "_model_prediction"
DUAL_OUTPUT_KEY = "dual" + OUTPUT_KEY
MERGED_OUTPUT_KEY = "merged" + OUTPUT_KEY
SINGLE_IONIZATION_OUTPUT_KEY = "single" + OUTPUT_KEY

# Import and expose key classes and functions
from .config import ModelConfig, get_config_data
from .data_canonizer import DataCanonizer
from .models import (
    ModelManager,
    ModelSpec,
    get_available_models,
    get_cache_directory,
    get_cache_info,
    load_model,
    print_cache_directory,
    print_cache_info,
)
from .classifier import process_spectra

__all__ = [
    # Constants
    "N",
    "C",
    "DATA_COLUMNS",
    "LABEL_COLUMN",
    "MODEL_MISSING_VALUE",
    "OUTPUT_KEY",
    "DUAL_OUTPUT_KEY",
    "MERGED_OUTPUT_KEY",
    "SINGLE_IONIZATION_OUTPUT_KEY",
    # Classes
    "DataCanonizer",
    "ModelManager",
    "ModelSpec",
    "ModelConfig",
    # Functions
    "process_spectra",
    "load_model",
    "get_available_models",
    "get_cache_directory",
    "print_cache_directory",
    "get_cache_info",
    "print_cache_info",
    "get_config_data",
]