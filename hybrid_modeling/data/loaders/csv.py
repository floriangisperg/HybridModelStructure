"""
Generic CSV data loader.

This module provides a general-purpose data loader for CSV files,
without assumptions about specific column names or data structures.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from hybrid_modeling.core.data import DataLoader, ExperimentDataset, ExperimentRun
from hybrid_modeling.data.loaders.excel import GenericExcelRun, GenericExcelDataset


class GenericCSVLoader(DataLoader):
    """
    Generic data loader for CSV files.

    This loader can handle any CSV file with time series data,
    without assumptions about specific column names or data structure.
    """

    def __init__(self,
                 time_col: str,
                 state_cols: List[str],
                 control_cols: List[str],
                 run_id_col: Optional[str] = None,
                 delimiter: str = ',',
                 preprocessing: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None):
        """
        Initialize generic CSV loader.

        Args:
            time_col: Name of time column
            state_cols: Names of state columns
            control_cols: Names of control columns
            run_id_col: Optional name of run ID column (if None, use index)
            delimiter: CSV delimiter character
            preprocessing: Optional function to preprocess data
        """
        self.time_col = time_col
        self.state_cols = state_cols
        self.control_cols = control_cols
        self.run_id_col = run_id_col
        self.delimiter = delimiter
        self.preprocessing = preprocessing

    def load(self, file_path: str,
             run_ids: Optional[List[str]] = None,
             max_runs: Optional[int] = None) -> GenericExcelDataset:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            run_ids: Optional list of specific run IDs to load
            max_runs: Optional maximum number of runs to load

        Returns:
            GenericExcelDataset (compatible with Excel dataset)
        """
        # Read CSV file
        data = pd.read_csv(file_path, delimiter=self.delimiter)

        # Apply preprocessing if provided
        if self.preprocessing:
            data = self.preprocessing(data)

        # Handle run identification
        if self.run_id_col:
            # Use specified run ID column
            all_run_ids = data[self.run_id_col].unique()

            # Filter run IDs if specified
            if run_ids:
                # Keep only requested run IDs that exist in the data
                all_run_ids = [rid for rid in all_run_ids if str(rid) in run_ids]

            # Limit number of runs if max_runs specified
            if max_runs:
                all_run_ids = all_run_ids[:max_runs]

            # Create runs
            runs = []
            for run_id in all_run_ids:
                # Filter data for this run
                run_data = data[data[self.run_id_col] == run_id].copy()

                # Skip empty runs
                if len(run_data) == 0:
                    continue

                # Sort by time
                if self.time_col in run_data.columns:
                    run_data = run_data.sort_values(self.time_col)

                # Create run
                run = GenericExcelRun(
                    run_id=str(run_id),
                    data=run_data,
                    time_col=self.time_col,
                    state_cols=self.state_cols,
                    control_cols=self.control_cols
                )

                runs.append(run)
        else:
            # If no run ID column, treat entire file as a single run
            # Sort by time
            if self.time_col in data.columns:
                data = data.sort_values(self.time_col)

            # Create a single run
            run = GenericExcelRun(
                run_id="run_1",
                data=data,
                time_col=self.time_col,
                state_cols=self.state_cols,
                control_cols=self.control_cols
            )

            runs = [run]

        return GenericExcelDataset(runs)


def create_multiple_runs_from_csv(file_path: str,
                                  time_col: str,
                                  state_cols: List[str],
                                  control_cols: List[str],
                                  run_separator_func: Callable[
                                      [pd.DataFrame], List[pd.DataFrame]]) -> GenericExcelDataset:
    """
    Create multiple runs from a single CSV file using a custom separation function.

    Args:
        file_path: Path to CSV file
        time_col: Name of time column
        state_cols: Names of state columns
        control_cols: Names of control columns
        run_separator_func: Function that splits DataFrame into multiple run DataFrames

    Returns:
        GenericExcelDataset
    """
    # Read CSV file
    data = pd.read_csv(file_path)

    # Split into run DataFrames
    run_dfs = run_separator_func(data)

    # Create runs
    runs = []
    for i, run_data in enumerate(run_dfs):
        # Skip empty runs
        if len(run_data) == 0:
            continue

        # Sort by time
        if time_col in run_data.columns:
            run_data = run_data.sort_values(time_col)

        # Create run
        run = GenericExcelRun(
            run_id=f"run_{i + 1}",
            data=run_data,
            time_col=time_col,
            state_cols=state_cols,
            control_cols=control_cols
        )

        runs.append(run)

    return GenericExcelDataset(runs)


def load_multiple_csv_files(file_paths: List[str],
                            loader: GenericCSVLoader,
                            **kwargs) -> GenericExcelDataset:
    """
    Load data from multiple CSV files.

    Args:
        file_paths: List of paths to CSV files
        loader: GenericCSVLoader instance
        **kwargs: Additional arguments to pass to loader.load

    Returns:
        GenericExcelDataset
    """
    all_runs = []

    for file_path in file_paths:
        # Load data from this file
        dataset = loader.load(file_path, **kwargs)

        # Add runs to list
        all_runs.extend(dataset.runs)

    return GenericExcelDataset(all_runs)


def split_data_by_time_gaps(data: pd.DataFrame,
                            time_col: str,
                            gap_threshold: float) -> List[pd.DataFrame]:
    """
    Split data into runs based on gaps in time.

    Args:
        data: DataFrame containing data
        time_col: Name of time column
        gap_threshold: Threshold for time gaps to split runs

    Returns:
        List of DataFrames, one for each run
    """
    # Sort by time
    sorted_data = data.sort_values(time_col).copy()

    # Calculate time differences
    time_diffs = sorted_data[time_col].diff()

    # Find gaps
    gap_indices = time_diffs[time_diffs > gap_threshold].index.tolist()

    # Add first and last indices
    split_indices = [sorted_data.index[0]] + gap_indices + [sorted_data.index[-1] + 1]

    # Split data
    run_dfs = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]

        run_df = sorted_data.loc[start_idx:end_idx - 1]
        run_dfs.append(run_df)

    return run_dfs