"""Model loading and management for parent_ion_classifier."""

from .manager import ModelManager, ModelSpec, ModelConfig
from .loader import (
    get_model_manager,
    get_cache_directory,
    print_cache_directory,
    get_available_models,
    get_model_groups,
    get_models_in_group,
    is_model_cached,
    download_model,
    download_models,
    check_and_download_models,
    load_jit_model,
    load_model,
    clear_model_cache,
    get_cache_info,
    print_cache_info,
    print_model_status,
    get_model_info,
    get_missing_models,
    get_cached_models
)

__all__ = [
    # From manager
    'ModelManager',
    'ModelSpec',
    'ModelConfig',
    # From loader
    'get_model_manager',
    'get_cache_directory',
    'print_cache_directory',
    'get_available_models',
    'get_model_groups',
    'get_models_in_group',
    'is_model_cached',
    'download_model',
    'download_models',
    'check_and_download_models',
    'load_jit_model',
    'load_model',
    'clear_model_cache',
    'get_cache_info',
    'print_cache_info',
    'print_model_status',
    'get_model_info',
    'get_missing_models',
    'get_cached_models'
]