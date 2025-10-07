import os
import sys
import pandas as pd
import numpy as np

from .utils import unpickle_file, pickle_file
from .classifier import process_spectra, MSNetModels
from .models import load_model
from .config import get_config_data
from huggingface_hub import delete_folder

class DictComparisonError(Exception):
    """Custom exception for dictionary comparison mismatches."""
    pass

def compare_dicts(
    dict1: dict, 
    dict2: dict, 
    path: str = "",
    rtol: float = 1e-4,  # 0.01% relative tolerance for cross-platform compatibility
    atol: float = 1e-6   # Absolute tolerance for values near zero
):
    """
    Compare two dictionaries recursively with platform-appropriate tolerances.
    
    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        path (str): Path to the current key for debugging purposes.
        rtol (float): Relative tolerance for floating-point comparisons.
        atol (float): Absolute tolerance for floating-point comparisons.

    Raises:
        DictComparisonError: If there is a mismatch in keys or values.
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
                    f"DataFrames mismatch at path '{current_path}':\n"
                    f"{diff_report}"
                ) from e

        # Handle floating-point numbers (with tolerance)
        elif isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
            if not np.isclose(val1, val2, rtol=rtol, atol=atol):
                raise DictComparisonError(
                    f"Floating-point value mismatch at path '{current_path}': {val1} != {val2}"
                )

        # Handle other types
        elif val1 != val2:
            raise DictComparisonError(
                f"Value mismatch at path '{current_path}': {val1} != {val2}"
            )

def get_dataframe_differences(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    rtol: float = 1e-4,
    atol: float = 1e-6
):
    """
    Generate a detailed report of differences between two DataFrames.
    
    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        rtol (float): The relative tolerance for floating-point comparisons.
        atol (float): The absolute tolerance for floating-point comparisons.

    Returns:
        str: A detailed report of the differences.
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
        if series1.dtype.kind in 'fc' and series2.dtype.kind in 'fc':  # Floating-point or complex
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
                    examples.append(f"row {idx}: {val1:.6e} vs {val2:.6e} (diff: {abs(val1-val2):.6e})")
                else:
                    examples.append(f"row {idx}: {val1} vs {val2}")
            
            diff_report.append(
                f"Column '{column}' differs at {len(diff_indices)} rows (tolerance: rtol={rtol}, atol={atol}).\n"
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

def unittest(recreate_test_output: bool = False):
    """
    Runs unit tests for the spectra processing pipeline.

    This function performs the following:
        1. Loads test models for single and dual ionization.
        2. Iterates over different normalization methods to process spectra.
        3. Compares processed spectra against expected outputs with platform-appropriate tolerances.

    Args:
        recreate_test_output (bool): If True, missing or failed test outputs are recreated. 
                                     Otherwise, the test fails with an error.

    Raises:
        SystemExit: If a model or input/output spectra file cannot be loaded 
                    and `recreate_test_output` is False.
    """

    print("=" * 80)
    print("Running unit tests")
    print("=" * 80)

    # Determine the input root directory for test data
    input_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test'))

    # Load test models
    print("Loading test models...")
    models = MSNetModels(
        load_model('test_single', raise_exception=True),
        load_model('test_single', raise_exception=True),
        load_model('test_single', raise_exception=True),
        load_model('test_dual', raise_exception=True)
    )
    print("Test models loaded successfully.")

    # Define normalization methods
    normalization_methods = ['none', 'sigmoid', 'softmax', 'softmax_per_ionization']

    for normalization_method in normalization_methods:
        print(f"\nTesting normalization method: {normalization_method}")
        
        try:
            # Load input spectra
            in_path = os.path.join(input_root, 'spectra_dict.pkl')
            spectra_dict = unpickle_file(in_path)
        except Exception as e:
            print(f"Failed to load input spectra from '{in_path}' - {e}")
            sys.exit(-1)

        try:
            # Load expected output spectra
            out_path = os.path.join(input_root, f'spectra_dict_{normalization_method}_norm.pkl')
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
        print(f"  Processing spectra...")
        process_spectra(normalization_method, models, spectra_dict)
        
        print(f"  Comparing results (rtol=1e-4, atol=1e-6)...")
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