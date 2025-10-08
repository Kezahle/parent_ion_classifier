"""
Unit Testing Module for Parent Ion Classifier

This module provides comprehensive testing functionality for the spectra
processing pipeline, including cross-platform compatibility checks and
deterministic behavior validation.
"""

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from .classifier import MSNetModels, process_spectra
from .models import load_model
from .utils import pickle_file, unpickle_file


class DictComparisonError(Exception):
    """Custom exception for dictionary comparison mismatches."""

    pass


def compare_dicts(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    path: str = "",
    rtol: float = 1e-4,  # 0.01% relative tolerance for cross-platform compatibility
    atol: float = 1e-6,  # Absolute tolerance for values near zero
) -> None:
    """
    Compare two dictionaries recursively with platform-appropriate tolerances.

    This function performs deep comparison of nested dictionaries, handling
    DataFrames and floating-point values with configurable tolerances to
    ensure cross-platform reproducibility.

    Args:
        dict1: The first dictionary to compare
        dict2: The second dictionary to compare
        path: Current path in the nested structure (for error messages)
        rtol: Relative tolerance for floating-point comparisons (0.01% default)
        atol: Absolute tolerance for floating-point comparisons (for near-zero values)

    Raises:
        DictComparisonError: If there is a mismatch in keys or values

    Example:
        >>> d1 = {'a': 1.0, 'b': pd.DataFrame({'x': [1, 2]})}
        >>> d2 = {'a': 1.000001, 'b': pd.DataFrame({'x': [1, 2]})}
        >>> compare_dicts(d1, d2)  # Passes with default tolerance
    """
    # Check for key mismatches
    if set(dict1.keys()) != set(dict2.keys()):
        raise DictComparisonError(
            f"Keys mismatch at path '{path}':\n"
            f"  Missing in dict1: {set(dict2.keys()) - set(dict1.keys())}\n"
            f"  Missing in dict2: {set(dict1.keys()) - set(dict2.keys())}"
        )

    for key in dict1:
        current_path = f"{path}.{key}" if path else key

        val1 = dict1[key]
        val2 = dict2[key]

        # Handle nested dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            compare_dicts(val1, val2, current_path, rtol, atol)

        # Handle DataFrames
        elif isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(val1, val2, rtol=rtol, atol=atol)
            except AssertionError as e:
                diff_report = get_dataframe_differences(val1, val2, rtol=rtol, atol=atol)
                raise DictComparisonError(
                    f"DataFrames mismatch at path '{current_path}':\n{diff_report}"
                ) from e

        # Handle floating-point numbers (with tolerance)
        elif isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
            if not np.isclose(val1, val2, rtol=rtol, atol=atol):
                raise DictComparisonError(
                    f"Floating-point value mismatch at path '{current_path}': {val1} != {val2}"
                )

        # Handle other types
        elif val1 != val2:
            raise DictComparisonError(f"Value mismatch at path '{current_path}': {val1} != {val2}")


def get_dataframe_differences(
    df1: pd.DataFrame, df2: pd.DataFrame, rtol: float = 1e-4, atol: float = 1e-6
) -> str:
    """
    Generate a detailed report of differences between two DataFrames.

    This function provides human-readable diagnostics when DataFrames don't match,
    showing specific rows and columns where differences occur.

    Args:
        df1: The first DataFrame to compare
        df2: The second DataFrame to compare
        rtol: Relative tolerance for floating-point comparisons
        atol: Absolute tolerance for floating-point comparisons

    Returns:
        A detailed string report of the differences found

    Example:
        >>> df1 = pd.DataFrame({'x': [1.0, 2.0]})
        >>> df2 = pd.DataFrame({'x': [1.1, 2.0]})
        >>> report = get_dataframe_differences(df1, df2)
        >>> print(report)
        Column 'x' differs at 1 rows...
    """
    diff_report = []

    # Check shape
    if df1.shape != df2.shape:
        diff_report.append(f"Shapes differ: {df1.shape} vs {df2.shape}")
        return "\n".join(diff_report)  # Can't compare further if shapes differ

    # Compare column-by-column
    for column in df1.columns:
        if column not in df2.columns:
            diff_report.append(f"Column '{column}' is missing in df2")
            continue

        series1 = df1[column]
        series2 = df2[column]

        # Compare values
        if series1.dtype.kind in "fc" and series2.dtype.kind in "fc":  # Floating-point or complex
            # Use equal_nan=True to treat NaN values as equal
            mismatches = ~np.isclose(series1, series2, rtol=rtol, atol=atol, equal_nan=True)
        else:
            mismatches = series1 != series2

        if mismatches.any():
            diff_indices = np.where(mismatches)[0]

            # Show first few mismatches with actual values
            num_examples = min(5, len(diff_indices))
            sample_indices = diff_indices[:num_examples]
            examples = []
            for idx in sample_indices:
                val1 = series1.iloc[idx]
                val2 = series2.iloc[idx]
                if isinstance(val1, (float, np.floating)):
                    examples.append(
                        f"row {idx}: {val1:.6e} vs {val2:.6e} (diff: {abs(val1-val2):.6e})"
                    )
                else:
                    examples.append(f"row {idx}: {val1} vs {val2}")

            diff_report.append(
                f"Column '{column}' differs at {len(diff_indices)} rows "
                f"(tolerance: rtol={rtol}, atol={atol}).\n"
                f"  Examples:\n  " + "\n  ".join(examples)
            )

            if len(diff_indices) > num_examples:
                diff_report.append(f"  ... and {len(diff_indices) - num_examples} more")

    # Check for extra columns in df2
    extra_columns = set(df2.columns) - set(df1.columns)
    if extra_columns:
        diff_report.append(f"Extra columns in df2: {extra_columns}")

    # Combine and return report
    if not diff_report:
        return "No differences detected."
    return "\n".join(diff_report)


def unittest(recreate_test_output: bool = False) -> None:
    """
    Run comprehensive unit tests for the spectra processing pipeline.

    This function performs the following validation steps:
        1. Loads test models for single and dual ionization modes
        2. Iterates over different normalization methods to process spectra
        3. Compares processed spectra against expected outputs with platform-appropriate tolerances
        4. Optionally recreates test outputs if they're missing or outdated

    Args:
        recreate_test_output: If True, missing or failed test outputs are recreated.
                             Otherwise, the test fails with an error.

    Raises:
        SystemExit: If a model or input/output spectra file cannot be loaded
                   and recreate_test_output is False, or if tests fail.

    Example:
        >>> # Run tests with existing outputs
        >>> unittest()
        >>>
        >>> # Recreate test outputs (for development)
        >>> unittest(recreate_test_output=True)

    Note:
        This function uses relaxed tolerances (rtol=1e-4, atol=1e-6) to ensure
        cross-platform reproducibility between different systems and PyTorch versions.
    """

    print("=" * 80)
    print("Running unit tests")
    print("=" * 80)

    # Determine the input root directory for test data
    input_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "test"))

    # Load test models
    print("Loading test models...")
    models = MSNetModels(
        load_model("test_single", raise_exception=True),
        load_model("test_single", raise_exception=True),
        load_model("test_single", raise_exception=True),
        load_model("test_dual", raise_exception=True),
    )
    print("Test models loaded successfully.")

    # Define normalization methods to test
    normalization_methods = ["none", "sigmoid", "softmax", "softmax_per_ionization"]

    for normalization_method in normalization_methods:
        print(f"\nTesting normalization method: {normalization_method}")

        try:
            # Load input spectra
            in_path = os.path.join(input_root, "spectra_dict.pkl")
            spectra_dict = unpickle_file(in_path)
        except Exception as e:
            print(f"Failed to load input spectra from '{in_path}' - {e}")
            sys.exit(-1)

        try:
            # Load expected output spectra
            out_path = os.path.join(input_root, f"spectra_dict_{normalization_method}_norm.pkl")
            out_spectra_dict = unpickle_file(out_path)
        except Exception as e:
            if recreate_test_output:
                print(f"Failed to load output spectra from '{out_path}'. Recreating outputs.")
                process_spectra(normalization_method, models, spectra_dict)
                pickle_file(spectra_dict, out_path)
                print(f"Created new test output: {out_path}")
                continue
            else:
                print(f"Failed to load output spectra from '{out_path}' - {e}")
                print("Run with --recreate-output to generate test outputs")
                sys.exit(-1)

        # Process spectra and compare against expected output
        print("  Processing spectra...")
        process_spectra(normalization_method, models, spectra_dict)

        print("  Comparing results (rtol=1e-4, atol=1e-6)...")
        try:
            compare_dicts(spectra_dict, out_spectra_dict, path=normalization_method)
            print(f"  ✓ {normalization_method}: PASSED")
        except DictComparisonError as e:
            print(f"  ✗ {normalization_method}: FAILED")
            print(f"\n{e}\n")
            if recreate_test_output:
                print(f"  Recreating test output for {normalization_method}")
                pickle_file(spectra_dict, out_path)
                print(f"  Updated: {out_path}")
            else:
                sys.exit(-1)

    print("\n" + "=" * 80)
    print("Unit tests completed successfully!")
    print("=" * 80)
