"""
Utility Functions for Parent Ion Classifier

This module provides helper functions for file I/O operations,
particularly for loading and saving pickle files containing MS data.

The pickle format is used for efficient serialization of Python objects,
especially pandas DataFrames and dictionaries containing MS spectra data.
"""

import pickle
from pathlib import Path
from typing import Any


def unpickle_file(path: str) -> Any:
    """
    Load a pickled file and return its contents.

    Args:
        path: Filesystem path to the pickle file

    Returns:
        The unpickled Python object (typically a dict or DataFrame)

    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file is corrupted or not a valid pickle

    Example:
        >>> spectra_dict = unpickle_file('data/spectra.pkl')
        >>> print(type(spectra_dict))
        <class 'dict'>

    Note:
        This function loads the entire file into memory. For very large files,
        consider streaming or chunked processing approaches.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_file(data: Any, path: str) -> None:
    """
    Serialize a Python object to a pickle file.

    Uses the highest pickle protocol available for best performance and
    compatibility with modern Python versions.

    Args:
        data: Python object to serialize (dict, DataFrame, etc.)
        path: Destination filesystem path for the pickle file

    Raises:
        OSError: If unable to write to the specified path
        pickle.PicklingError: If the object cannot be pickled

    Example:
        >>> results = {'spectrum_001': processed_data}
        >>> pickle_file(results, 'output/results.pkl')

    Note:
        The file will be overwritten if it already exists. Use with caution.
        Creates parent directories if they don't exist.
    """
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
