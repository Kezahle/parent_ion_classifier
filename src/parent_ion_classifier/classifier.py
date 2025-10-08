"""
Core Classification Module for Parent Ion Classifier

This module contains the core logic for running inference on MSNet models,
processing mass spectrometry data, and applying various normalization methods
to the model's output. The functions are designed to work with the package's
internal data structures and model configurations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from . import (
    DATA_COLUMNS,
    DUAL_OUTPUT_KEY,
    LABEL_COLUMN,
    MERGED_OUTPUT_KEY,
    MODEL_MISSING_VALUE,
    SINGLE_IONIZATION_OUTPUT_KEY,
    N,
)
from .data_canonizer import DataCanonizer
from .models import load_model
from .utils import pickle_file, unpickle_file


@dataclass
class MSNetModels:
    """
    A data class to hold the four primary MSNet models used for classification.

    Attributes:
        positive: The model for positive ionization spectra
        negative: The model for negative ionization spectra
        merged: The model for combined positive/negative spectra
        dual: The model for dual-mode ionization spectra

    Example:
        >>> from parent_ion_classifier.models import load_model
        >>> models = MSNetModels(
        ...     positive=load_model('TopIntensityMSNet_positive'),
        ...     negative=load_model('TopIntensityMSNet_negative'),
        ...     merged=load_model('TopIntensityMSNet_merged'),
        ...     dual=load_model('DualModeMSNet')
        ... )
    """

    positive: nn.Module
    negative: nn.Module
    merged: nn.Module
    dual: nn.Module


def apply_output_normalization(
    input_t: torch.Tensor,
    output: torch.Tensor,
    canonized_pos_df: pd.DataFrame,
    canonized_neg_df: pd.DataFrame,
    single_ionization: bool,
    normalization_method: str = "none",
) -> pd.DataFrame:
    """
    Normalize the model's output tensor and combine it with a DataFrame.

    This function handles different normalization methods (sigmoid, softmax)
    and merges the processed predictions back into a pandas DataFrame,
    making sure to handle missing values appropriately.

    Args:
        input_t: The input tensor used for inference, containing original data
                 and placeholder values
        output: The raw output tensor from the model
        canonized_pos_df: The canonized DataFrame for positive spectra
        canonized_neg_df: The canonized DataFrame for negative spectra
        single_ionization: A flag indicating if the spectra are single-mode
        normalization_method: The normalization method to apply. One of:
            - 'none': Raw model outputs (no normalization)
            - 'sigmoid': Sigmoid activation applied to outputs
            - 'softmax': Softmax over all predictions
            - 'softmax_per_ionization': Separate softmax for positive and
              negative modes (recommended for dual-mode)

    Returns:
        A DataFrame with the normalized model predictions, including backup
        columns from the original data

    Raises:
        ValueError: If the output tensor's shape is invalid (not 1-dimensional)

    Example:
        >>> output_df = apply_output_normalization(
        ...     input_tensor, output_tensor,
        ...     pos_df, neg_df,
        ...     single_ionization=True,
        ...     normalization_method='softmax'
        ... )
    """
    if output.ndim != 1:
        raise ValueError(f"output tensor shape ({input_t.shape}) was invalid!")

    # Mask out the predictions for non-existent spectra
    mask = input_t.squeeze()[:, 0] == -1
    output[mask] = float("-inf")

    if normalization_method == "sigmoid":
        output = torch.sigmoid(output)
    elif normalization_method == "softmax" or single_ionization:
        output = F.softmax(output, dim=0)
    elif normalization_method == "softmax_per_ionization":
        # Apply softmax to positive and negative predictions separately
        pos_inference = output[:N]
        pos_normalized = F.softmax(pos_inference, dim=0)
        neg_inference = output[N:]
        neg_normalized = F.softmax(neg_inference, dim=0)
        output = torch.cat((pos_normalized, neg_normalized))
    # 'none' method is handled by passing through

    output[mask] = MODEL_MISSING_VALUE

    df_combined = pd.concat([canonized_pos_df, canonized_neg_df], axis=0)

    # Define columns to copy and their new names
    columns_to_copy = {f"{item}_backup": item for item in DATA_COLUMNS}
    # Copy and rename columns for the output DataFrame
    output_df = df_combined[list(columns_to_copy.keys())].rename(columns=columns_to_copy)
    output_df[LABEL_COLUMN[0]] = output.detach().numpy()
    output_df.replace(MODEL_MISSING_VALUE, np.nan, inplace=True)
    return output_df


def df_to_canonized_tensor(
    dc: DataCanonizer, df: Optional[pd.DataFrame], reset_output_index: bool = False
) -> Tuple[Optional[torch.Tensor], pd.DataFrame]:
    """
    Convert a pandas DataFrame to a canonized torch tensor.

    Args:
        dc: The DataCanonizer instance for processing
        df: The input DataFrame (can be None for missing spectra)
        reset_output_index: Whether to reset the index of the output DataFrame

    Returns:
        A tuple containing:
            - The canonized tensor (None if df is None)
            - The canonized DataFrame (empty if df is None)

    Raises:
        ValueError: If the final tensor shape does not match the desired shape

    Example:
        >>> dc = DataCanonizer(N=150, data_columns=DATA_COLUMNS, label_columns=LABEL_COLUMN)
        >>> tensor, canonized_df = df_to_canonized_tensor(dc, spectrum_df)
    """
    DESIRED_SHAPE = [1, 1, N, 3]
    data_tensor = None
    canonized_df = pd.DataFrame()
    if df is not None:
        canonized_df = dc.canonise_and_truncate_df(df.fillna(-1), reset_output_index)
        data_tensor, _ = dc.df_to_tensors(canonized_df)
        data_tensor = data_tensor.unsqueeze(0).cpu()
        if data_tensor.shape != torch.Size(DESIRED_SHAPE):
            raise ValueError(
                f"Tensor with shape {data_tensor.shape} doesn't match "
                f"required input shape ({DESIRED_SHAPE})"
            )
    return data_tensor, canonized_df


def get_canonized_tensor(
    dc: DataCanonizer, value: Dict, key: str, reset_output_index: bool = False
) -> Tuple[Optional[torch.Tensor], pd.DataFrame]:
    """
    A helper function to get a canonized tensor from a dictionary value.

    Args:
        dc: The DataCanonizer instance
        value: The dictionary containing the DataFrame to process
        key: The key to access the DataFrame within the dictionary
        reset_output_index: Whether to reset the index of the output DataFrame

    Returns:
        A tuple of (canonized_tensor, canonized_dataframe)
    """
    df = value.get(key)
    return df_to_canonized_tensor(dc, df, reset_output_index)


def predict_dual_ionization(
    normalization_method: str,
    models: MSNetModels,
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    value: Dict,
    canonized_pos_df: pd.DataFrame,
    canonized_neg_df: pd.DataFrame,
) -> None:
    """
    Perform dual-mode inference and update the input dictionary with results.

    Args:
        normalization_method: The normalization method to apply
        models: The collection of MSNet models
        pos_tensor: Tensor for positive spectra
        neg_tensor: Tensor for negative spectra
        value: The dictionary to update with prediction results
        canonized_pos_df: Canonized positive DataFrame
        canonized_neg_df: Canonized negative DataFrame

    Note:
        This function modifies the input dictionary in-place, adding the
        DUAL_OUTPUT_KEY with prediction results.
    """
    output_tensor = models.dual.forward_tensor(pos_tensor, neg_tensor).squeeze()
    concat_input_t = torch.cat((pos_tensor, neg_tensor), dim=2)
    value[DUAL_OUTPUT_KEY] = apply_output_normalization(
        concat_input_t,
        output_tensor,
        canonized_pos_df,
        canonized_neg_df,
        False,  # single_ionization == False
        normalization_method,
    )


def predict_normalized_single_ionization(
    normalization_method: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    canonized_pos_df: pd.DataFrame,
    canonized_neg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform single-mode inference and apply normalization.

    Args:
        normalization_method: The normalization method
        model: The single-mode model to use
        input_tensor: The input tensor for inference
        canonized_pos_df: Canonized positive DataFrame
        canonized_neg_df: Canonized negative DataFrame

    Returns:
        A DataFrame with the normalized predictions
    """
    output_tensor = model(input_tensor).squeeze()
    return apply_output_normalization(
        input_tensor,
        output_tensor,
        canonized_pos_df,
        canonized_neg_df,
        True,  # single_ionization == True
        normalization_method,
    )


def predict_single_ionization(
    normalization_method: str,
    models: MSNetModels,
    is_positive_spectra: bool,
    input_tensor: torch.Tensor,
    value: Dict,
    canonized_pos_df: pd.DataFrame,
    canonized_neg_df: pd.DataFrame,
) -> None:
    """
    Perform single-mode inference and update the input dictionary.

    Args:
        normalization_method: The normalization method
        models: The collection of MSNet models
        is_positive_spectra: True if spectra are positive, False if negative
        input_tensor: The input tensor for inference
        value: The dictionary to update with predictions
        canonized_pos_df: Canonized positive DataFrame
        canonized_neg_df: Canonized negative DataFrame

    Note:
        This function modifies the input dictionary in-place, adding both
        SINGLE_IONIZATION_OUTPUT_KEY and MERGED_OUTPUT_KEY with predictions.
    """
    value[SINGLE_IONIZATION_OUTPUT_KEY] = predict_normalized_single_ionization(
        normalization_method,
        models.positive if is_positive_spectra else models.negative,
        input_tensor,
        canonized_pos_df,
        canonized_neg_df,
    )

    value[MERGED_OUTPUT_KEY] = predict_normalized_single_ionization(
        normalization_method, models.merged, input_tensor, canonized_pos_df, canonized_neg_df
    )


def process_spectrum(
    dc: DataCanonizer,
    normalization_method: str,
    models: MSNetModels,
    value: Dict,
    reset_output_index: bool = False,
) -> None:
    """
    Process a single spectrum, apply classification, and store results.

    This function determines whether a spectrum is single-mode or dual-mode
    and dispatches to the appropriate prediction function.

    Args:
        dc: The DataCanonizer instance
        normalization_method: The normalization method to apply
        models: The collection of MSNet models
        value: A dictionary representing the spectrum data (modified in-place)
        reset_output_index: Whether to reset the output DataFrame's index

    Note:
        The input dictionary is modified in-place with prediction results.
    """

    pos_tensor, canonized_pos_df = get_canonized_tensor(
        dc, value, key="positive", reset_output_index=reset_output_index
    )
    neg_tensor, canonized_neg_df = get_canonized_tensor(
        dc, value, key="negative", reset_output_index=reset_output_index
    )

    if neg_tensor is None or pos_tensor is None:
        # It's a single ionization spectrum
        is_pos_spectra = neg_tensor is None
        input_tensor = pos_tensor if is_pos_spectra else neg_tensor
        predict_single_ionization(
            normalization_method,
            models,
            is_pos_spectra,
            input_tensor,
            value,
            canonized_pos_df,
            canonized_neg_df,
        )
    else:
        # It's a dual ionization spectrum
        predict_dual_ionization(
            normalization_method,
            models,
            pos_tensor,
            neg_tensor,
            value,
            canonized_pos_df,
            canonized_neg_df,
        )


def process_spectra(
    normalization_method: str,
    models: MSNetModels,
    spectra_dict: Dict,
    reset_output_index: bool = False,
) -> None:
    """
    Iterate through a dictionary of spectra and process each one.

    This is the main processing function that handles a collection of spectra,
    applying the specified models and normalization method to each spectrum.

    Args:
        normalization_method: The normalization method to apply. One of:
            - 'none': No normalization
            - 'sigmoid': Sigmoid activation
            - 'softmax': Softmax across all predictions
            - 'softmax_per_ionization': Separate softmax for pos/neg (recommended)
        models: The collection of MSNet models to use for predictions
        spectra_dict: A dictionary containing spectrum data to be processed.
                     Each key maps to a dict with 'positive' and/or 'negative'
                     DataFrames. Modified in-place with results.
        reset_output_index: Whether to reset the output DataFrame's index

    Raises:
        ValueError: If a spectrum dictionary is missing both 'positive' and 'negative' data

    Example:
        >>> from parent_ion_classifier import process_spectra, MSNetModels
        >>> from parent_ion_classifier.models import load_model
        >>>
        >>> # Load models
        >>> models = MSNetModels(
        ...     positive=load_model('TopIntensityMSNet_positive'),
        ...     negative=load_model('TopIntensityMSNet_negative'),
        ...     merged=load_model('TopIntensityMSNet_merged'),
        ...     dual=load_model('DualModeMSNet')
        ... )
        >>>
        >>> # Process spectra
        >>> process_spectra('softmax_per_ionization', models, spectra_dict)
        >>>
        >>> # Access results
        >>> prediction = spectra_dict['001']['single_model_prediction']

    Note:
        The spectra_dict is modified in-place with prediction results.
        Use a copy if you need to preserve the original data.
    """
    dc = DataCanonizer(N, DATA_COLUMNS, LABEL_COLUMN)

    for key, value in tqdm(spectra_dict.items(), desc="Processing spectra"):
        # Validate that at least one of 'positive' or 'negative' is not None
        if value.get("positive") is None and value.get("negative") is None:
            raise ValueError(f"Neither 'positive' nor 'negative' is provided for {key}.")
        process_spectrum(dc, normalization_method, models, value, reset_output_index)


def classify_parent_ions(
    input_file: str,
    output_file: str,
    normalization_method: str,
    reset_output_index: bool = False,
) -> None:
    """
    Classify parent ions from an input file and save the results.

    This is the main entry point for the classification process. It loads
    the necessary models, processes the input spectra, and saves the output.

    Args:
        input_file: The path to the input spectra file (e.g., a pickle file)
        output_file: The path to save the classified spectra
        normalization_method: The normalization method to apply
        reset_output_index: Whether to reset the output DataFrame's index

    Raises:
        OSError: If the input spectra file cannot be loaded
        Exception: If a model fails to load

    Example:
        >>> classify_parent_ions(
        ...     'data/input_spectra.pkl',
        ...     'results/output.pkl',
        ...     'softmax_per_ionization'
        ... )

    Note:
        This function automatically downloads models if they're not cached.
        The process may take several minutes depending on the number of spectra.
    """

    print("Loading models...")
    # Load models with automatic download if needed
    # If any model fails to load, it will raise an exception and stop the process.
    models = MSNetModels(
        load_model("TopIntensityMSNet_positive", raise_exception=True, auto_download=True),
        load_model("TopIntensityMSNet_negative", raise_exception=True, auto_download=True),
        load_model("TopIntensityMSNet_merged", raise_exception=True, auto_download=True),
        load_model("DualModeMSNet", raise_exception=True, auto_download=True),
    )
    print("Models loaded successfully.")

    print(f"Loading input spectra from {input_file}...")
    try:
        spectra_dict = unpickle_file(input_file)
    except Exception as e:
        raise OSError(f"Failed to load input spectra from {input_file} - {e}")

    print(f"Processing {len(spectra_dict)} spectra...")
    process_spectra(
        normalization_method, models, spectra_dict, reset_output_index=reset_output_index
    )

    print(f"Saving results to {output_file}...")
    pickle_file(spectra_dict, output_file)
    print("Classification completed successfully!")
