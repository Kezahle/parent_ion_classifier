"""
Model Configuration Module

This module handles loading and parsing of model configuration from JSON files.
It supports both simple and enhanced configuration formats with backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

try:
    from importlib.resources import files  # Python 3.9+
except ImportError:
    from importlib_resources import files  # backport for <3.9

PACKAGE_NAME = "parent_ion_classifier.config"
MODELS_CONFIG_FILENAME = "model_config.json"


@dataclass(frozen=True)
class ModelSpec:
    """
    Data class for a single model's specification.

    This supports both simple filename-only specs and enhanced specs with metadata.
    """

    filename: str
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict | str) -> "ModelSpec":
        """
        Create ModelSpec from dictionary or string, handling multiple formats.

        Args:
            data: Dictionary containing model specification or filename string

        Returns:
            ModelSpec instance

        Raises:
            ValueError: If data format is invalid
        """
        if isinstance(data, str):
            # Simple format: just filename string
            return cls(filename=data)
        elif isinstance(data, dict):
            if "filename" in data:
                # Enhanced format: dictionary with filename and optional metadata
                return cls(filename=data["filename"], description=data.get("description", ""))
            elif len(data) == 1:
                # Legacy format: {"filename": "model.pt"}
                filename = list(data.values())[0]
                return cls(filename=filename)
            else:
                raise ValueError(f"Invalid model spec format: {data}")
        else:
            raise ValueError(f"Invalid model spec type: {type(data)}")


@dataclass(frozen=True)
class ModelConfig:
    """Data class for the overall model configuration."""

    repo_id: str
    revision: str
    models: Dict[str, ModelSpec]
    model_groups: Dict[str, List[str]]


def get_config_data() -> ModelConfig:
    """
    Loads and parses the models configuration from the model_config.json file.

    Returns:
        ModelConfig: The parsed model configuration object.

    Raises:
        RuntimeError: If the configuration file cannot be loaded or parsed.
    """
    try:
        cfg_text = (files(PACKAGE_NAME) / MODELS_CONFIG_FILENAME).read_text(encoding="utf-8")
        raw = json.loads(cfg_text)

        # Parse models with support for multiple formats
        models = {}
        for k, v in raw["models"].items():
            models[k] = ModelSpec.from_dict(v)

        # Parse model groups (optional)
        model_groups = raw.get("model_groups", {})

        return ModelConfig(
            repo_id=raw["repo_id"],
            revision=raw["revision"],
            models=models,
            model_groups=model_groups,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model configuration: {e}") from e


def get_models_by_group(config: ModelConfig, group_name: str) -> List[str]:
    """
    Get list of model names belonging to a specific group.

    Args:
        config: Model configuration
        group_name: Name of the model group

    Returns:
        List of model names in the group

    Raises:
        ValueError: If group_name doesn't exist
    """
    if group_name not in config.model_groups:
        raise ValueError(
            f"Unknown model group '{group_name}'. Available: {list(config.model_groups.keys())}"
        )

    return config.model_groups[group_name]


def get_model_groups(config: ModelConfig) -> List[str]:
    """
    Get list of available model group names.

    Args:
        config: Model configuration

    Returns:
        List of model group names
    """
    return list(config.model_groups.keys())


def validate_config(config: ModelConfig) -> List[str]:
    """
    Validate model configuration and return list of issues found.

    Args:
        config: Model configuration to validate

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    if not config.repo_id:
        issues.append("repo_id is required")

    if not config.revision:
        issues.append("revision is required")

    if not config.models:
        issues.append("at least one model must be configured")

    # Validate individual models
    for model_name, spec in config.models.items():
        if not spec.filename:
            issues.append(f"model '{model_name}' missing filename")

        if not spec.filename.endswith((".pt", ".pt_jit", ".pth", ".onnx")):
            issues.append(f"model '{model_name}' has unsupported file extension: {spec.filename}")

    # Validate model groups reference existing models
    for group_name, model_list in config.model_groups.items():
        for model_name in model_list:
            if model_name not in config.models:
                issues.append(f"model group '{group_name}' references unknown model '{model_name}'")

    return issues


def print_config_summary(config: ModelConfig) -> None:
    """
    Print a summary of the model configuration.

    Args:
        config: Model configuration to summarize
    """
    print("Model Configuration Summary")
    print(f"Repository: {config.repo_id}")
    print(f"Revision: {config.revision}")
    print(f"Models: {len(config.models)}")
    print("-" * 50)

    # Print models by group if groups are defined
    if config.model_groups:
        for group_name, model_list in config.model_groups.items():
            print(f"\n{group_name.replace('_', ' ').title()}:")
            for model_name in model_list:
                if model_name in config.models:
                    spec = config.models[model_name]
                    print(f"  {model_name:<25} {spec.filename}")
                    if spec.description:
                        print(f"    {spec.description}")
                else:
                    print(f"  {model_name:<25} [NOT FOUND]")

        # Print ungrouped models
        all_grouped = set()
        for model_list in config.model_groups.values():
            all_grouped.update(model_list)

        ungrouped = set(config.models.keys()) - all_grouped
        if ungrouped:
            print("\nOther Models:")
            for model_name in sorted(ungrouped):
                spec = config.models[model_name]
                print(f"  {model_name:<25} {spec.filename}")
                if spec.description:
                    print(f"    {spec.description}")
    else:
        # Print all models without grouping
        for name, spec in config.models.items():
            print(f"  {name:<25} {spec.filename}")
            if spec.description:
                print(f"    {spec.description}")


def export_config_template() -> str:
    """
    Export a template configuration as JSON string.

    Returns:
        JSON string template for model configuration
    """
    template = {
        "repo_id": "your-username/your-repo",
        "revision": "main",
        "models": {
            "example_model": {
                "filename": "model.pt_jit",
                "description": "Example model description",
            }
        },
        "model_groups": {"production_models": ["example_model"], "test_models": []},
    }

    return json.dumps(template, indent=2)
