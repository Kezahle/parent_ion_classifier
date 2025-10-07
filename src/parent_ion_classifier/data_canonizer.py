"""
Data Canonization Module for Mass Spectrometry Data

This module provides the DataCanonizer class for preprocessing and standardizing
mass spectrometry data before model inference. It handles:
- Data sorting and truncation
- m/z value rounding and canonization
- Intensity scaling and normalization
- Missing value handling
- Deterministic ordering for reproducibility

The canonization process ensures that MS data is in a consistent format
suitable for neural network input, regardless of the original data structure.

Example:
    >>> from parent_ion_classifier import DataCanonizer
    >>> dc = DataCanonizer(N=150, data_columns=['mz', 'MS1', 'MS2'], 
    ...                    label_columns=['parent'])
    >>> canonized_df = dc.canonise_and_truncate_df(raw_df)
    >>> tensor, labels = dc.df_to_tensors(canonized_df)
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List


class DataCanonizer:
    """
    Preprocesses and standardizes mass spectrometry data for model input.
    
    This class handles the complete pipeline of transforming raw MS data into
    a format suitable for neural network inference. It ensures deterministic
    behavior across platforms through stable sorting and consistent processing.
    
    Attributes:
        N (int): Maximum number of rows to keep (top-N by intensity)
        data_columns (List[str]): Column names for feature data (e.g., ['mz', 'MS1', 'MS2'])
        label_column (str): Column name for labels (e.g., 'parent')
        missing_val_replacement (int): Value to use for missing data (default: -1)
        intensity_max (int): Maximum value for scaled intensities (default: 10)
        mz_decimal_digits (int): Decimal places to round m/z values (default: 2)
        mz_msb_digits (int): Most significant digits to keep in m/z (default: -1, disabled)
    
    Example:
        >>> dc = DataCanonizer(N=150, data_columns=['mz', 'MS1', 'MS2'],
        ...                    label_columns=['parent'])
        >>> df = pd.DataFrame({
        ...     'mz': [100.5, 200.3, 150.7],
        ...     'MS1': [1000, 500, 750],
        ...     'MS2': [2000, 1500, 1800],
        ...     'parent': [0, 1, 0]
        ... })
        >>> canonized = dc.canonise_and_truncate_df(df)
        >>> tensor, labels = dc.df_to_tensors(canonized)
    """
    
    def __init__(self, N: int, data_columns: List[str], label_columns: List[str],
                 mz_decimal_digits: int = 2, missing_val_replacement: int = -1,
                 intensity_max: int = 10, mz_msb_digits: int = -1):
        """
        Initialize the DataCanonizer.
        
        Args:
            N: Maximum number of rows to retain (top-N by intensity)
            data_columns: Names of feature columns (must include 'mz', 'MS1', 'MS2')
            label_columns: Names of label columns (must be single-element list)
            mz_decimal_digits: Number of decimal places for m/z rounding
            missing_val_replacement: Value to substitute for missing data
            intensity_max: Maximum value for intensity scaling
            mz_msb_digits: Number of most significant digits to keep in m/z integer part
                          (-1 disables this feature)
        
        Raises:
            AssertionError: If label_columns contains more than one column
        """
        self.N = N
        self.data_columns = data_columns
        assert (len(label_columns) == 1), f"Too many label columns provided: {label_columns}"
        self.label_column = label_columns[0]
        self.missing_val_replacement = missing_val_replacement
        self.intensity_max = intensity_max
        self.mz_decimal_digits = mz_decimal_digits
        self.mz_msb_digits = mz_msb_digits

    def _has_label_column(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains the label column."""
        return self.label_column in df.columns

    def _has_parent_ions(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains any parent ion labels (value = 1)."""
        return (df[self.label_column] == 1).any()

    def _validate_input(self, df: pd.DataFrame):
        """
        Validate input DataFrame has required structure.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            AssertionError: If validation fails (missing columns, no parent ions, etc.)
        """
        assert self.N > 0, f"{self.N=} is too small!"
        assert all(col in df.columns for col in self.data_columns), \
            f"Data frame missing required columns {self.data_columns}"
        if self._has_label_column(df):
            assert self._has_parent_ions(df), "Input DataFrame lacks parent ions"

    def _add_missing_parents(self, df: pd.DataFrame, truncated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all parent ions are included even if they fall outside top-N.
        
        When truncating to top-N rows, parent ions (label=1) that fall outside
        the top-N are added back by removing non-parent rows.
        
        Args:
            df: Original full DataFrame
            truncated_df: DataFrame already truncated to N rows
            
        Returns:
            DataFrame with all parent ions included
        """
        if not self._has_label_column(df):
            truncated_df[self.label_column] = 0
            return truncated_df

        missing_parents = df[(df[self.label_column] == 1) & (df.index >= self.N)]
        if not missing_parents.empty:
            truncated_df = self._remove_last_n_non_parents(truncated_df, len(missing_parents))
            truncated_df = pd.concat([truncated_df, missing_parents], ignore_index=True)
        return truncated_df

    def _remove_last_n_non_parents(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Remove the last n non-parent rows from DataFrame.
        
        Args:
            df: Input DataFrame
            n: Number of non-parent rows to remove
            
        Returns:
            DataFrame with n non-parent rows removed from the end
        """
        non_parent_indices = df[df[self.label_column] != 1].tail(n).index
        return df.drop(non_parent_indices)

    def _truncate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Truncate DataFrame to top-N rows while preserving parent ions.
        
        Args:
            df: Input DataFrame (must be pre-sorted by intensity)
            
        Returns:
            Truncated DataFrame with at most N rows, including all parent ions
        """
        truncated_df = df.head(self.N).copy()
        return self._add_missing_parents(df, truncated_df)

    def _scale_df(self, df: pd.DataFrame, cols: List[str], reset_index: bool = False):
        """
        Scale intensity columns to [0, intensity_max] range.
        
        First normalizes to [0, 1] if any values exceed 1, then scales to
        the configured intensity_max value.
        
        Args:
            df: DataFrame to scale (modified in-place)
            cols: Column names to scale (typically ['MS1', 'MS2'])
            reset_index: Whether to reset the DataFrame index
            
        Note:
            Modifies DataFrame in-place
        """
        # Scale cols down if any value is > 1
        if (df[cols] > 1).any().any():
            max_val = df[cols].max(skipna=True).max()
            df[cols] = (df[cols] / max_val)

        # Scale cols to self.intensity_max
        df[cols] = self.intensity_max * df[cols]
        if reset_index:
            df.reset_index(drop=True, inplace=True)

    def _truncate_mz(self, mz: float) -> float:
        """
        Truncate m/z value to specified number of most significant digits.
        
        Args:
            mz: Mass-to-charge ratio value
            
        Returns:
            Truncated m/z value, or original if mz_msb_digits is disabled
            
        Example:
            >>> dc = DataCanonizer(N=150, ..., mz_msb_digits=3)
            >>> dc._truncate_mz(12345.678)
            345.678  # Keeps only 3 most significant digits of integer part
        """
        if mz < 0 or self.mz_msb_digits == -1:
            return mz
        integer_part = int(np.floor(mz))
        truncated = integer_part % (10 ** self.mz_msb_digits)
        return truncated + (mz - integer_part)

    def _canonize_mz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply m/z truncation and sort by m/z in descending order.
        
        Uses original index as tie-breaker for deterministic sorting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame sorted by m/z (and original index for ties)
        """
        if self.mz_msb_digits != -1:
            df['mz'] = df['mz'].apply(self._truncate_mz)
        
        # Sort by mz with original index as tie-breaker for deterministic ordering
        if '_original_index' in df.columns:
            return df.sort_values(by=['mz', '_original_index'], ascending=False)
        else:
            return df.sort_values(by='mz', ascending=False)

    def _remove_insignificant_rows(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Remove rows where all intensity values are near-zero.
        
        Args:
            df: Input DataFrame
            cols: Columns to check for insignificance (typically ['MS1', 'MS2'])
            
        Returns:
            DataFrame with insignificant rows removed
        """
        threshold = 1e-10
        condition = (df[cols].abs() < threshold).all(axis=1)
        return df[~condition]

    def canonise_and_truncate_df(self, in_df: pd.DataFrame,
                                 reset_index: bool = False,
                                 sort_by_MS2_first: bool = True) -> pd.DataFrame:
        """
        Complete canonization pipeline for MS data.
        
        This is the main entry point for data preprocessing. It performs:
        1. Data validation
        2. m/z rounding
        3. Deterministic sorting by intensity (with original index tie-breaker)
        4. Truncation to top-N rows (preserving parent ions)
        5. Removal of insignificant peaks
        6. m/z canonization and re-sorting
        7. Intensity scaling
        8. Missing value handling
        9. Padding to exactly N rows
        
        Args:
            in_df: Input DataFrame with columns matching data_columns
            reset_index: Whether to reset index after processing
            sort_by_MS2_first: If True, sort by MS2 then MS1; otherwise MS1 then MS2
            
        Returns:
            Canonized DataFrame with exactly N rows, ready for model input
            
        Raises:
            AssertionError: If input validation fails
            
        Example:
            >>> df = pd.DataFrame({
            ...     'mz': [100.5, 200.3, 150.7],
            ...     'MS1': [1000, 500, 750],
            ...     'MS2': [2000, 1500, 1800]
            ... })
            >>> canonized = dc.canonise_and_truncate_df(df)
            >>> len(canonized)
            150
        """
        # Make sure to only manipulate a copy of the input
        df = in_df.copy()
        self._validate_input(df)

        df[self.data_columns] = df[self.data_columns].astype(float)
        df.replace(self.missing_val_replacement, np.nan, inplace=True)
        
        # Create backup columns for later reconstruction
        for column in self.data_columns:
            df[column + '_backup'] = df[column].astype(float)

        df['mz'] = df['mz'].round(self.mz_decimal_digits)

        # Preserve original index for deterministic sorting when values are tied
        df['_original_index'] = range(len(df))

        sort_cols = ['MS2', 'MS1'] if sort_by_MS2_first else ['MS1', 'MS2']
        # Add _original_index as final tie-breaker for fully deterministic sorting
        df.sort_values(by=sort_cols + ['_original_index'], ascending=False, inplace=True)

        truncated_df = self._truncate_df(df)
        truncated_df = self._remove_insignificant_rows(truncated_df, sort_cols)
        truncated_df = self._canonize_mz(truncated_df)

        self._scale_df(truncated_df, sort_cols, reset_index)

        truncated_df.fillna(self.missing_val_replacement, inplace=True)

        # Remove the helper column before final processing
        if '_original_index' in truncated_df.columns:
            truncated_df = truncated_df.drop('_original_index', axis=1)

        # Pad rows if necessary
        pad_rows = self.N - len(truncated_df)
        if pad_rows > 0:
            padding = pd.DataFrame({col: self.missing_val_replacement for col in truncated_df.columns},
                                   index=range(pad_rows))
            truncated_df = pd.concat([truncated_df, padding])

        return truncated_df

    def df_to_tensors(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert canonized DataFrame to PyTorch tensors.
        
        Args:
            df: Canonized DataFrame with data_columns and optional label_column
            
        Returns:
            Tuple of (data_tensor, label_tensor):
                - data_tensor: Shape (1, N, len(data_columns)), dtype float32
                - label_tensor: Shape (N,), dtype float32, or None if no labels
                
        Example:
            >>> data_tensor, labels = dc.df_to_tensors(canonized_df)
            >>> data_tensor.shape
            torch.Size([1, 150, 3])
        """
        data = torch.tensor(df[self.data_columns].values, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(df[self.label_column].values, dtype=torch.float32).squeeze() \
            if self.label_column in df.columns else None
        return data, label