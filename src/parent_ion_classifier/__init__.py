TENSOR_PRINT_PRECISION = 4
TENSOR_PRINT_SCI_MODE = False

# Maximum number of rows outputted by an MS1/MS2 experiment
N = 150
# The number of channels (depth) outputted from the first layer of convolution
C = 10

LABEL_COLUMN = ['parent']
MZ_COLUMN = 'mz'
DATA_COLUMNS = [MZ_COLUMN] + ["MS1", "MS2"]

MODEL_MISSING_VALUE = -1
OUTPUT_KEY = '_model_prediction'
DUAL_OUTPUT_KEY = 'dual' + OUTPUT_KEY
MERGED_OUTPUT_KEY = 'merged' + OUTPUT_KEY
SINGLE_IONIZATION_OUTPUT_KEY = 'single' + OUTPUT_KEY

# Expose key functions and classes from submodules to the top-level namespace.
from .models import ModelManager, ModelSpec, ModelConfig
from .config import ModelConfig, get_config_data
from .models import (
    load_model, 
    get_available_models, 
    get_cache_directory, 
    print_cache_directory,
    get_cache_info,
    print_cache_info
)
from .data_canonizer import DataCanonizer
from .classifier import process_spectra