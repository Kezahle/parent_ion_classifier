import argparse
import os
import sys

from .classifier import classify_parent_ions
from .models import (
    check_and_download_models,
    clear_model_cache,
    download_model,
    download_models,
    get_available_models,
    get_cached_models,
    get_model_groups,
    get_model_info,
    get_models_in_group,
    print_cache_directory,
    print_cache_info,
    print_model_status,
)
from .test import unittest


def handle_model_download(args: argparse.Namespace) -> None:
    """Handle model download commands."""
    try:
        if args.model_name:
            # Download specific model
            path = download_model(args.model_name, force=args.force)
            print(f"Model '{args.model_name}' downloaded to: {path}")
        elif args.group:
            # Download models from specific group
            downloaded = download_models(group_name=args.group, force=args.force)
            if downloaded:
                print(f"Downloaded {len(downloaded)} models from group '{args.group}':")
                for name, path in downloaded.items():
                    print(f"  {name}: {path}")
            else:
                print(f"All models in group '{args.group}' are already cached.")
        else:
            # Download all models
            downloaded = download_models(force=args.force)
            if downloaded:
                print(f"Downloaded {len(downloaded)} models:")
                for name, path in downloaded.items():
                    print(f"  {name}: {path}")
            else:
                print("All models are already cached.")
    except Exception as e:
        print(f"Failed to download models: {e}", file=sys.stderr)
        sys.exit(1)


def handle_model_status(args: argparse.Namespace) -> None:
    """Handle model status commands."""
    if args.model_name:
        # Show specific model info
        try:
            info = get_model_info(args.model_name)
            print(f"\nModel: {info['name']}")
            print(f"Filename: {info['filename']}")
            print(f"Description: {info['description'] or 'No description'}")
            print(f"Cached: {'Yes' if info['cached'] else 'No'}")
            print(f"Repository: {info['repo_id']}")
            print(f"Revision: {info['revision']}")
            print(f"Cache directory: {info['cache_dir']}")
            if info["groups"]:
                print(f"Groups: {', '.join(info['groups'])}")
            if info["cached"] and "local_path" in info:
                print(f"Local Path: {info['local_path']}")
                if "file_size" in info:
                    size_mb = info["file_size"] / (1024 * 1024)
                    print(f"File Size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"Error getting model info: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.group:
        # Show group status
        print_model_status(group_name=args.group)
    else:
        # Show all models status
        print_model_status()


def handle_model_list(args: argparse.Namespace) -> None:
    """Handle model list commands."""
    if args.groups:
        # List model groups
        groups = get_model_groups()
        if groups:
            print(f"Available model groups ({len(groups)}):")
            for group in groups:
                models_in_group = get_models_in_group(group)
                print(f"  {group} ({len(models_in_group)} models)")
                if args.verbose:
                    for model in models_in_group:
                        cached_status = "cached" if model in get_cached_models() else "not cached"
                        print(f"    - {model} ({cached_status})")
        else:
            print("No model groups defined")
    elif args.group:
        # List models in specific group
        try:
            models = get_models_in_group(args.group)
            cached_models = set(get_cached_models())
            print(f"Models in group '{args.group}' ({len(models)}):")
            for model in models:
                status = "cached" if model in cached_models else "not cached"
                print(f"  {model} ({status})")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # List all models
        models = get_available_models()
        cached_models = set(get_cached_models())
        print(f"Available models ({len(models)}):")
        for model in models:
            status = "cached" if model in cached_models else "not cached"
            print(f"  {model} ({status})")


def handle_model_cache(args: argparse.Namespace) -> None:
    """Handle model cache management commands."""
    if args.action == "clear":
        # Clear cache
        model_names = [args.model_name] if args.model_name else None
        success = clear_model_cache(model_names=model_names, confirm=not args.yes)
        if not success:
            sys.exit(1)
    elif args.action == "info":
        # Show cache info using the new print_cache_info function
        print_cache_info(verbose=args.verbose)
    elif args.action == "directory":
        # Show cache directory
        print_cache_directory()


def handle_classify_cmd(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Handle the main classification command."""
    # Validate required arguments
    if not args.infile:
        parser.print_help()
        print("Error: --infile is required")
        sys.exit(-1)

    input_file = args.infile
    output_file = args.outfile
    normalization = args.norm
    reset_output_index = args.reset_output_index

    # Validate input file path
    if not os.path.exists(input_file):
        print(f"no such file '{input_file}'")
        sys.exit(-1)

    if os.path.exists(output_file) and not args.overwrite:
        print(f"'{output_file}' already exists")
        sys.exit(-1)

    # Check and download required models before classification
    if not args.skip_model_check:
        print("Checking for required models...")
        # Check for production models (you can customize this based on your needs)
        required_group = getattr(args, "model_group", "production_models")
        try:
            groups = get_model_groups()
            if required_group in groups:
                success = check_and_download_models(group_name=required_group)
            else:
                # Fall back to checking specific models if group doesn't exist
                required_models = [
                    "DualModeMSNet",
                    "TopIntensityMSNet_merged",
                    "TopIntensityMSNet_positive",
                    "TopIntensityMSNet_negative",
                ]
                success = check_and_download_models(model_names=required_models)

            if not success:
                print(
                    "Failed to ensure required models are available. Use --skip-model-check to bypass."
                )
                sys.exit(1)
            print("Required models verified.")
        except Exception as e:
            print(f"Error checking models: {e}")
            if not args.skip_model_check:
                sys.exit(1)

    classify_parent_ions(input_file, output_file, normalization, reset_output_index)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Parent Ion Classifier - Classify parent ions in mass spectrometry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify spectra
  %(prog)s classify -i data.pkl -o results.pkl

  # Download all models
  %(prog)s download

  # Download models from a specific group
  %(prog)s download --group production_models

  # Download a specific model
  %(prog)s download --model DualModeMSNet

  # Check model status
  %(prog)s status

  # Check status of models in a group
  %(prog)s status --group production_models

  # List available models
  %(prog)s list

  # List model groups
  %(prog)s list --groups

  # Show cache directory
  %(prog)s cache directory

  # Show cache information
  %(prog)s cache info

  # Show detailed cache information
  %(prog)s cache info --verbose

  # Clear cache
  %(prog)s cache clear

  # Clear specific model from cache
  %(prog)s cache clear --model DualModeMSNet

  # Run tests
  %(prog)s test
        """,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Classification command
    classify_parser = subparsers.add_parser("classify", help="Classify parent ions from MS data")
    classify_parser.add_argument(
        "-i",
        "--infile",
        required=True,
        help="Path to the input file (a pickled dictionary of dictionaries)",
    )
    classify_parser.add_argument(
        "-o",
        "--outfile",
        default="results.pkl",
        help="Path to the output file (default: 'results.pkl')",
    )
    classify_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists",
    )
    classify_parser.add_argument(
        "-n",
        "--norm",
        type=str,
        default="softmax_per_ionization",
        choices=["none", "sigmoid", "softmax", "softmax_per_ionization"],
        help="Output normalization method (default: 'softmax_per_ionization')",
    )
    classify_parser.add_argument(
        "--reset_output_index", action="store_true", help="Reset the output DataFrame's index"
    )
    classify_parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip automatic model availability check and download",
    )
    classify_parser.add_argument(
        "--model-group",
        default="production_models",
        help="Model group to check for classification (default: 'production_models')",
    )

    # Model download command
    download_parser = subparsers.add_parser(
        "download", help="Download models from Hugging Face Hub"
    )
    download_group = download_parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--model", dest="model_name", help="Name of specific model to download"
    )
    download_group.add_argument("--group", help="Download all models from a specific group")
    download_parser.add_argument(
        "--force", action="store_true", help="Force re-download even if model is already cached"
    )

    # Model status command
    status_parser = subparsers.add_parser("status", help="Show model status information")
    status_group = status_parser.add_mutually_exclusive_group()
    status_group.add_argument(
        "--model", dest="model_name", help="Show detailed info for specific model"
    )
    status_group.add_argument("--group", help="Show status for models in specific group")

    # Model list command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--groups", action="store_true", help="List model groups instead of individual models"
    )
    list_parser.add_argument("--group", help="List models in a specific group")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show additional details")

    # Cache management command
    cache_parser = subparsers.add_parser("cache", help="Manage model cache")
    cache_subparsers = cache_parser.add_subparsers(dest="action", help="Cache actions")

    clear_parser = cache_subparsers.add_parser("clear", help="Clear cached models")
    clear_parser.add_argument("--model", dest="model_name", help="Clear specific model")
    clear_parser.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation")

    info_parser = cache_subparsers.add_parser("info", help="Show cache information")
    info_parser.add_argument("-v", "--verbose", action="store_true", help="Show file details")

    cache_subparsers.add_parser("directory", help="Show cache directory")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run unit tests")
    test_parser.add_argument(
        "--recreate-output", action="store_true", help="Recreate test output files"
    )

    return parser


def main() -> None:
    """Main function to parse arguments and execute commands."""
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Handle different commands
    try:
        if args.command == "classify":
            handle_classify_cmd(parser, args)
        elif args.command == "download":
            handle_model_download(args)
        elif args.command == "status":
            handle_model_status(args)
        elif args.command == "list":
            handle_model_list(args)
        elif args.command == "cache":
            if args.action:
                handle_model_cache(args)
            else:
                cache_parser = parser._subparsers._group_actions[0].choices["cache"]
                cache_parser.print_help()
        elif args.command == "test":
            unittest(args.recreate_output)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
