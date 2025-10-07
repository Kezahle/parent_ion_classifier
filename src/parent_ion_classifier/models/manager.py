"""
Generic Model Manager for Hugging Face Models

This module provides a generic interface for downloading, caching, and managing
Hugging Face models. It's designed to be reusable across different projects
that need to work with HF models.
"""

from __future__ import annotations
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from tqdm import tqdm

import torch
from huggingface_hub import hf_hub_download, list_repo_files, scan_cache_dir
from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub import HfApi


@dataclass
class ModelSpec:
    """Specification for a single model."""
    filename: str
    description: str = ""


@dataclass
class ModelConfig:
    """Configuration for a collection of models."""
    repo_id: str
    revision: str
    models: Dict[str, ModelSpec]
    model_groups: Dict[str, List[str]]


class ModelManager:
    """
    Generic manager for Hugging Face models with caching and update capabilities.
    
    This class provides a unified interface for:
    - Downloading models from Hugging Face Hub
    - Caching models locally
    - Checking for updates
    - Loading models with appropriate error handling
    - Cache management and cleanup
    """
    
    def __init__(self, config: ModelConfig, cache_dir: Optional[str] = None):
        """
        Initialize the ModelManager.
        
        Args:
            config: Model configuration containing repo info and model specs
            cache_dir: Optional custom cache directory. If None, uses HF default.
        """
        self.config = config
        self.cache_dir = cache_dir
        self.hf_api = HfApi()
        
    def get_cache_directory(self) -> str:
        """
        Get the current cache directory being used.
        
        Returns:
            Path to the cache directory
        """
        if self.cache_dir is not None:
            return str(Path(self.cache_dir).resolve())
        else:
            # Get HF default cache directory
            from huggingface_hub import constants
            return str(Path(constants.HF_HUB_CACHE).resolve())
    
    def print_cache_directory(self) -> None:
        """Print the current cache directory."""
        print(f"Cache directory: {self.get_cache_directory()}")
        
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            Sorted list of model names
        """
        return sorted(self.config.models.keys())
    
    def get_model_groups(self) -> List[str]:
        """
        Get list of available model group names.
        
        Returns:
            Sorted list of model group names
        """
        return sorted(self.config.model_groups.keys())
    
    def get_models_in_group(self, group_name: str) -> List[str]:
        """
        Get models belonging to a specific group.
        
        Args:
            group_name: Name of the model group
            
        Returns:
            List of model names in the group
            
        Raises:
            ValueError: If group_name doesn't exist
        """
        if group_name not in self.config.model_groups:
            raise ValueError(f"Unknown model group '{group_name}'. Available: {self.get_model_groups()}")
        
        return self.config.model_groups[group_name]
    
    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if a model is already cached locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is cached, False otherwise
            
        Raises:
            ValueError: If model_name is not in configuration
        """
        if model_name not in self.config.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {self.get_available_models()}")
            
        filename = self.config.models[model_name].filename
        
        try:
            hf_hub_download(
                repo_id=self.config.repo_id,
                filename=filename,
                revision=self.config.revision,
                local_files_only=True,
                cache_dir=self.cache_dir
            )
            return True
        except LocalEntryNotFoundError:
            return False
    
    def get_cached_models(self) -> List[str]:
        """
        Get list of models that are currently cached.
        
        Returns:
            List of cached model names
        """
        cached = []
        for model_name in self.get_available_models():
            if self.is_model_cached(model_name):
                cached.append(model_name)
        return cached
    
    def get_missing_models(self, group_name: Optional[str] = None) -> List[str]:
        """
        Get list of models that are not cached.
        
        Args:
            group_name: Optional group name to check only models in that group
        
        Returns:
            List of missing model names
        """
        models_to_check = (
            self.get_models_in_group(group_name) if group_name 
            else self.get_available_models()
        )
        
        missing = []
        for model_name in models_to_check:
            if not self.is_model_cached(model_name):
                missing.append(model_name)
        return missing
    
    def download_model(
        self, 
        model_name: str, 
        force: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> str:
        """
        Download a specific model.
        
        Args:
            model_name: Name of the model to download
            force: If True, re-download even if cached
            progress_callback: Optional callback for progress updates (model_name, current, total)
            
        Returns:
            Path to the downloaded model file
            
        Raises:
            ValueError: If model_name is not in configuration
            RepositoryNotFoundError: If the repository doesn't exist
            Exception: For other download errors
        """
        if model_name not in self.config.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {self.get_available_models()}")
            
        filename = self.config.models[model_name].filename
        
        # Check if already cached and not forcing re-download
        if not force and self.is_model_cached(model_name):
            print(f"Model '{model_name}' already cached, skipping download.")
            return hf_hub_download(
                repo_id=self.config.repo_id,
                filename=filename,
                revision=self.config.revision,
                local_files_only=True,
                cache_dir=self.cache_dir
            )
        
        print(f"Downloading model '{model_name}' ({filename})...")
        
        try:
            model_path = hf_hub_download(
                repo_id=self.config.repo_id,
                filename=filename,
                revision=self.config.revision,
                cache_dir=self.cache_dir
            )
            print(f"Successfully downloaded '{model_name}'")
            return model_path
            
        except Exception as e:
            print(f"Failed to download model '{model_name}': {e}", file=sys.stderr)
            raise
    
    def download_models(
        self, 
        model_names: Optional[List[str]] = None,
        group_name: Optional[str] = None,
        force: bool = False, 
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, str]:
        """
        Download multiple models.
        
        Args:
            model_names: Specific models to download. If None, downloads all models.
            group_name: Download models from a specific group. Ignored if model_names is provided.
            force: If True, re-download even if cached
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping model names to their local paths
        """
        # Determine which models to download
        if model_names is not None:
            # Validate provided model names
            invalid_models = [name for name in model_names if name not in self.config.models]
            if invalid_models:
                raise ValueError(f"Unknown models: {invalid_models}. Available: {self.get_available_models()}")
            models_to_check = model_names
        elif group_name is not None:
            models_to_check = self.get_models_in_group(group_name)
        else:
            models_to_check = self.get_available_models()
        
        # Filter to only models that need downloading
        models_to_download = []
        for model_name in models_to_check:
            if force or not self.is_model_cached(model_name):
                models_to_download.append(model_name)
        
        if not models_to_download:
            print("All specified models are already cached.")
            return {}
        
        print(f"Downloading {len(models_to_download)} models...")
        
        results = {}
        with tqdm(models_to_download, desc="Downloading models") as pbar:
            for i, model_name in enumerate(pbar):
                pbar.set_description(f"Downloading {model_name}")
                try:
                    path = self.download_model(model_name, force=force)
                    results[model_name] = path
                    if progress_callback:
                        progress_callback(model_name, i + 1, len(models_to_download))
                except Exception as e:
                    print(f"Failed to download '{model_name}': {e}", file=sys.stderr)
                    # Continue with other downloads rather than stopping
        
        return results
    
    def check_for_updates(self) -> Dict[str, bool]:
        """
        Check if there are updates available for cached models.
        
        This is a simplified check - in practice, you might want to compare
        file hashes or modification times for more accurate update detection.
        
        Returns:
            Dictionary mapping model names to update availability (True if update available)
        """
        try:
            # Get remote file listing
            remote_files = list_repo_files(
                repo_id=self.config.repo_id,
                revision=self.config.revision
            )
            remote_filenames = set(remote_files)
            
            updates_available = {}
            for model_name, spec in self.config.models.items():
                # Check if file exists remotely and is not cached locally
                file_exists_remote = spec.filename in remote_filenames
                is_cached_local = self.is_model_cached(model_name)
                updates_available[model_name] = file_exists_remote and not is_cached_local
            
            return updates_available
            
        except Exception as e:
            print(f"Failed to check for updates: {e}", file=sys.stderr)
            return {}
    
    def load_model(
        self, 
        model_name: str, 
        device: str = "cpu",
        auto_download: bool = True,
        loader_func: Optional[Callable[[str, str], Any]] = None
    ) -> Optional[Any]:
        """
        Load a model, with optional automatic downloading.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            auto_download: If True, download model if not cached
            loader_func: Custom function to load the model file. 
                        Defaults to torch.jit.load for .pt/.pt_jit files
            
        Returns:
            Loaded model, or None if loading fails
            
        Raises:
            ValueError: If model_name is not in configuration
        """
        if model_name not in self.config.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {self.get_available_models()}")
        
        # Auto-download if not cached
        if auto_download and not self.is_model_cached(model_name):
            try:
                self.download_model(model_name)
            except Exception as e:
                print(f"Failed to auto-download '{model_name}': {e}", file=sys.stderr)
                return None
        
        # Get model path
        try:
            model_path = hf_hub_download(
                repo_id=self.config.repo_id,
                filename=self.config.models[model_name].filename,
                revision=self.config.revision,
                local_files_only=True,
                cache_dir=self.cache_dir
            )
        except LocalEntryNotFoundError:
            print(f"Model '{model_name}' not found in cache and auto-download failed", file=sys.stderr)
            return None
        
        # Load the model
        try:
            if loader_func:
                return loader_func(model_path, device)
            else:
                # Default loader for PyTorch models
                return torch.jit.load(model_path, map_location=device)
        except Exception as e:
            print(f"Failed to load model '{model_name}' from {model_path}: {e}", file=sys.stderr)
            return None
    
    def _find_repo_cache(self, cache_info):
        """Helper method to find the repository cache information."""
        for repo in cache_info.repos:
            if repo.repo_id == self.config.repo_id:
                return repo
        return None
    
    def _find_target_revision(self, repo_cache):
        """Helper method to find the target revision in the repository cache."""
        # Look through all revisions for the one matching our target revision
        for revision_info in repo_cache.revisions:
            if revision_info.commit_hash == self.config.revision or self.config.revision in revision_info.refs:
                return revision_info
        return None
    
    def clear_cache(
        self, 
        model_names: Optional[List[str]] = None, 
        confirm: bool = True
    ) -> bool:
        """
        Clear cached models from local storage.
        
        Args:
            model_names: Specific models to clear. If None, clears all models from this repo.
            confirm: If True, asks for confirmation before clearing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Scan cache to find relevant files
            cache_info = scan_cache_dir(cache_dir=self.cache_dir)
            repo_cache = self._find_repo_cache(cache_info)
            
            if repo_cache is None:
                print(f"No cached files found for repository '{self.config.repo_id}' in {self.get_cache_directory()}")
                return True
            
            # Find the target revision
            target_revision = self._find_target_revision(repo_cache)
            
            if target_revision is None:
                print(f"No cached files found for revision '{self.config.revision}' in repository '{self.config.repo_id}'")
                return True
            
            # Determine what to clear
            if model_names is None:
                # Clear entire repository cache for this revision
                files_to_clear = list(target_revision.files)
                clear_description = f"all cached models from {self.config.repo_id} (revision: {self.config.revision})"
            else:
                # Clear specific models
                target_filenames = set()
                for model_name in model_names:
                    if model_name not in self.config.models:
                        print(f"Warning: Unknown model '{model_name}', skipping")
                        continue
                    target_filenames.add(self.config.models[model_name].filename)
                
                files_to_clear = [
                    f for f in target_revision.files 
                    if f.file_name in target_filenames
                ]
                clear_description = f"models {model_names} from {self.config.repo_id} (revision: {self.config.revision})"
            
            if not files_to_clear:
                print("No matching files found in cache")
                return True
            
            # Calculate total size
            total_size = sum(f.size_on_disk for f in files_to_clear)
            size_mb = total_size / (1024 * 1024)
            
            print(f"Found {len(files_to_clear)} cached files ({size_mb:.1f} MB) in {self.get_cache_directory()}")
            
            # Confirm if requested
            if confirm:
                response = input(f"Clear {clear_description}? This will free {size_mb:.1f} MB. (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    print("Cache clearing cancelled")
                    return False
            
            # Delete the files using the commit hash
            delete_strategy = cache_info.delete_revisions(target_revision.commit_hash)
            delete_strategy.execute()
            
            print(f"Successfully cleared {len(files_to_clear)} files ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"Failed to clear cache: {e}", file=sys.stderr)
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models and disk usage.
        
        Returns:
            Dictionary with cache information
        """
        try:
            cache_info = scan_cache_dir(cache_dir=self.cache_dir)
            repo_cache = self._find_repo_cache(cache_info)
            
            if repo_cache is None:
                return {
                    'repo_id': self.config.repo_id,
                    'revision': self.config.revision,
                    'cache_dir': self.get_cache_directory(),
                    'cached': False,
                    'total_size': 0,
                    'file_count': 0,
                    'files': []
                }
            
            # Find the target revision
            target_revision = self._find_target_revision(repo_cache)
            
            if target_revision is None:
                return {
                    'repo_id': self.config.repo_id,
                    'revision': self.config.revision,
                    'cache_dir': self.get_cache_directory(),
                    'cached': False,
                    'total_size': 0,
                    'file_count': 0,
                    'files': []
                }
            
            # Get file information
            files_info = []
            total_size = 0
            
            for file_info in target_revision.files:
                files_info.append({
                    'filename': file_info.file_name,
                    'size': file_info.size_on_disk,
                    'path': str(file_info.file_path)
                })
                total_size += file_info.size_on_disk
            
            return {
                'repo_id': self.config.repo_id,
                'revision': self.config.revision,
                'cache_dir': self.get_cache_directory(),
                'cached': True,
                'total_size': total_size,
                'file_count': len(files_info),
                'files': files_info
            }
            
        except Exception as e:
            print(f"Failed to get cache info: {e}", file=sys.stderr)
            return {
                'repo_id': self.config.repo_id,
                'revision': self.config.revision,
                'cache_dir': self.get_cache_directory(),
                'error': str(e)
            }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.config.models:
            raise ValueError(f"Unknown model '{model_name}'")
        
        spec = self.config.models[model_name]
        is_cached = self.is_model_cached(model_name)
        
        # Find which groups contain this model
        model_groups = [
            group_name for group_name, models in self.config.model_groups.items()
            if model_name in models
        ]
        
        info = {
            'name': model_name,
            'filename': spec.filename,
            'description': spec.description,
            'cached': is_cached,
            'repo_id': self.config.repo_id,
            'revision': self.config.revision,
            'cache_dir': self.get_cache_directory(),
            'groups': model_groups
        }
        
        if is_cached:
            try:
                model_path = hf_hub_download(
                    repo_id=self.config.repo_id,
                    filename=spec.filename,
                    revision=self.config.revision,
                    local_files_only=True,
                    cache_dir=self.cache_dir
                )
                info['local_path'] = model_path
                info['file_size'] = Path(model_path).stat().st_size
            except Exception:
                pass
        
        return info
    
    def print_status(self, group_name: Optional[str] = None) -> None:
        """
        Print a status summary of models.
        
        Args:
            group_name: Optional group name to show only models in that group
        """
        print(f"\nModel Status for repository: {self.config.repo_id}")
        print(f"Revision: {self.config.revision}")
        print(f"Cache directory: {self.get_cache_directory()}")
        
        if group_name:
            print(f"Group: {group_name}")
            models_to_show = self.get_models_in_group(group_name)
        else:
            models_to_show = self.get_available_models()
            
        print("-" * 60)
        
        for model_name in models_to_show:
            info = self.get_model_info(model_name)
            status = "CACHED" if info['cached'] else "NOT CACHED"
            
            print(f"{model_name:<25} {status}")
            if info.get('description'):
                print(f"  Description: {info['description']}")
            if info.get('groups'):
                print(f"  Groups: {', '.join(info['groups'])}")
        
        cached_count = len([m for m in models_to_show if self.is_model_cached(m)])
        total_count = len(models_to_show)
        print(f"\nSummary: {cached_count}/{total_count} models cached")