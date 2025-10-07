"""
Hugging Face Model Loader for Parent Ion Classifier

This module provides functions to load pre-trained models from Hugging Face Hub
for the parent ion classifier. It uses the generic ModelManager for downloading
and caching functionality.
"""

from __future__ import annotations
import sys
from typing import Optional, List, Dict, Any

import torch
from huggingface_hub.utils import LocalEntryNotFoundError

from ..config import get_config_data
from .manager import ModelManager


# Global model manager instance (initialized lazily)
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager instance, initializing it if needed.
    
    Returns:
        The global ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        config_data = get_config_data()
        _model_manager = ModelManager(config_data)
    
    return _model_manager


def get_cache_directory() -> str:
    """
    Get the current cache directory being used.
    
    Returns:
        Path to the cache directory
    """
    return get_model_manager().get_cache_directory()


def print_cache_directory() -> None:
    """Print the current cache directory."""
    get_model_manager().print_cache_directory()


def get_available_models() -> List[str]:
    """
    Returns a sorted list of all available model names from the configuration.

    Returns:
        List of available model names
    """
    return get_model_manager().get_available_models()


def get_model_groups() -> List[str]:
    """
    Returns a sorted list of all available model group names.

    Returns:
        List of model group names
    """
    return get_model_manager().get_model_groups()


def get_models_in_group(group_name: str) -> List[str]:
    """
    Get models belonging to a specific group.
    
    Args:
        group_name: Name of the model group
        
    Returns:
        List of model names in the group
    """
    return get_model_manager().get_models_in_group(group_name)


def is_model_cached(model_name: str) -> bool:
    """
    Check if a model is cached locally.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if cached, False otherwise
    """
    return get_model_manager().is_model_cached(model_name)


def download_model(model_name: str, force: bool = False) -> str:
    """
    Download a specific model.
    
    Args:
        model_name: Name of the model to download
        force: If True, re-download even if cached
        
    Returns:
        Path to the downloaded model
    """
    return get_model_manager().download_model(model_name, force=force)


def download_models(
    model_names: Optional[List[str]] = None,
    group_name: Optional[str] = None,
    force: bool = False
) -> Dict[str, str]:
    """
    Download multiple models.
    
    Args:
        model_names: Specific models to download. If None, downloads all models.
        group_name: Download models from a specific group. Ignored if model_names is provided.
        force: If True, re-download even if cached
        
    Returns:
        Dictionary mapping model names to local paths
    """
    return get_model_manager().download_models(
        model_names=model_names,
        group_name=group_name,
        force=force
    )


def check_and_download_models(
    model_names: Optional[List[str]] = None,
    group_name: Optional[str] = None
) -> bool:
    """
    Check for missing models and download them.
    
    Args:
        model_names: Specific models to check. If None, checks all models.
        group_name: Check models from a specific group. Ignored if model_names is provided.
        
    Returns:
        True if all specified models are available, False otherwise
    """
    manager = get_model_manager()
    
    # Determine which models to check
    if model_names is not None:
        models_to_check = model_names
    elif group_name is not None:
        models_to_check = manager.get_models_in_group(group_name)
    else:
        models_to_check = manager.get_available_models()
    
    # Find missing models
    missing_models = []
    for model_name in models_to_check:
        if not manager.is_model_cached(model_name):
            missing_models.append(model_name)
    
    if missing_models:
        print(f"Missing models: {missing_models}")
        print("Downloading missing models...")
        try:
            manager.download_models(model_names=missing_models)
            return True
        except Exception as e:
            print(f"Failed to download missing models: {e}", file=sys.stderr)
            return False
    
    return True


def load_jit_model(
    model_type: str, 
    device: str = "cpu", 
    local_files_only: bool | None = None,
    auto_download: bool = True
) -> torch.jit.ScriptModule:
    """
    Loads a TorchScript model from the Hugging Face Hub, with a cache-first approach.

    Args:
        model_type: The name of the model to load
        device: The device to map the model to ('cpu' or 'cuda')
        local_files_only: If True, only load from the local cache.
                         If False, always try to download. If None,
                         try cache first then download
        auto_download: If True, automatically download missing models

    Returns:
        The loaded TorchScript model

    Raises:
        ValueError: If model_type is unknown
        LocalEntryNotFoundError: If the model is not in the local cache and downloading is disabled
    """
    manager = get_model_manager()
    
    if model_type not in manager.config.models:
        raise ValueError(f"Unknown model type '{model_type}', available: {get_available_models()}")

    # Handle local_files_only logic
    if local_files_only is True:
        auto_download = False
    elif local_files_only is False:
        auto_download = True
    # If local_files_only is None, use the auto_download parameter

    # Try to load the model
    model = manager.load_model(
        model_type, 
        device=device, 
        auto_download=auto_download
    )
    
    if model is None:
        if local_files_only is True or not auto_download:
            raise LocalEntryNotFoundError(f"Model '{model_type}' not found in local cache")
        else:
            raise RuntimeError(f"Failed to load model '{model_type}'")
    
    return model


def load_model(
    model_type: str, 
    device: str = "cpu", 
    raise_exception: bool = False,
    auto_download: bool = True
) -> torch.jit.ScriptModule | None:
    """
    A user-facing function to load a pre-trained model with optional error handling.

    Args:
        model_type: The name of the model to load
        device: The device to load the model on
        raise_exception: If True, propagates any exception that
                        occurs during loading. If False, returns
                        None on failure and prints an error message
        auto_download: If True, automatically download missing models

    Returns:
        The loaded model, or None if loading fails and raise_exception is False
    """
    try:
        return load_jit_model(model_type, device=device, auto_download=auto_download)
    except Exception as e:
        if raise_exception:
            raise
        else:
            print(f"[ERROR] Could not load model '{model_type}': {e}", file=sys.stderr)
            return None


def clear_model_cache(
    model_names: Optional[List[str]] = None,
    confirm: bool = True
) -> bool:
    """
    Clear cached models from local storage.
    
    Args:
        model_names: Specific models to clear. If None, clears all models.
        confirm: If True, asks for confirmation before clearing
        
    Returns:
        True if successful, False otherwise
    """
    return get_model_manager().clear_cache(model_names=model_names, confirm=confirm)


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about cached models and disk usage.
    
    Returns:
        Dictionary with cache information including cache directory
    """
    return get_model_manager().get_cache_info()


def print_cache_info(verbose: bool = False) -> None:
    """
    Print cache information in a user-friendly format.
    
    Args:
        verbose: If True, show detailed file information
    """
    cache_info = get_cache_info()
    
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Repository: {cache_info['repo_id']}")
    print(f"Revision: {cache_info['revision']}")
    
    if cache_info.get('error'):
        print(f"Error: {cache_info['error']}")
        return
    
    if cache_info.get('cached'):
        size_mb = cache_info['total_size'] / (1024 * 1024)
        print(f"Total cache size: {size_mb:.1f} MB")
        print(f"Number of files: {cache_info['file_count']}")
        
        if verbose and cache_info.get('files'):
            print("\nCached files:")
            for file_info in cache_info['files']:
                file_size_mb = file_info['size'] / (1024 * 1024)
                print(f"  {file_info['filename']} ({file_size_mb:.1f} MB)")
    else:
        print("No cached files found")


def print_model_status(group_name: Optional[str] = None) -> None:
    """
    Print status of models.
    
    Args:
        group_name: Optional group name to show only models in that group
    """
    get_model_manager().print_status(group_name=group_name)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information including cache directory
    """
    return get_model_manager().get_model_info(model_name)


def get_missing_models(group_name: Optional[str] = None) -> List[str]:
    """
    Get list of models that are not cached.
    
    Args:
        group_name: Optional group name to check only models in that group
        
    Returns:
        List of missing model names
    """
    return get_model_manager().get_missing_models(group_name=group_name)


def get_cached_models() -> List[str]:
    """
    Get list of models that are currently cached.
    
    Returns:
        List of cached model names
    """
    return get_model_manager().get_cached_models()