"""
Generic Excel data loader.

This module provides a general-purpose data loader for Excel files,
without assumptions about specific column names or data structures.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from hybrid_modeling.core.data import DataLoader, ExperimentDataset, ExperimentRun


class GenericExcelRun(ExperimentRun):
    """A single experiment run from an Excel file."""

    def __init__(self,
                run_id: str,
                data: pd.DataFrame,
                time_col: str,
                state_cols: List[str],
                control_cols: List[str],
                initial_conditions: Optional[Dict[str, float]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize generic experiment run.

        Args:
            run_id: Run identifier
            data: DataFrame containing run data
            time_col: Name of time column
            state_cols: Names of state columns
            control_cols: Names of control columns
            initial_conditions: Optional initial conditions (default: first row of states)
            metadata: Optional metadata
        """
        self._id = run_id
        self._data = data
        self._time_col = time_col
        self._state_cols = state_cols
        self._control_cols = control_cols
        self._metadata = metadata or {}

        # Set initial conditions if not provided
        if initial_conditions is None:
            self._initial_conditions = {}
            first_row = data.iloc[0]

            for col in state_cols:
                if col in first_row:
                    self._initial_conditions[col] = float(first_row[col])
        else:
            self._initial_conditions = initial_conditions

    @property
    def id(self) -> str:
        """Get the ID of this run."""
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        """Get the dataset for this run."""
        return self._data

    @property
    def initial_conditions(self) -> Dict[str, float]:
        """Get initial conditions for this run."""
        return self._initial_conditions

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for this run."""
        return self._metadata

    @property
    def time_points(self) -> np.ndarray:
        """Get time points for this run."""
        return np.array(self._data[self._time_col])

    @property
    def state_values(self) -> Dict[str, np.ndarray]:
        """Get state values for this run."""
        return {col: np.array(self._data[col]) for col in self._state_cols if col in self._data.columns}

    @property
    def control_values(self) -> Dict[str, np.ndarray]:
        """Get control values for this run."""
        return {col: np.array(self._data[col]) for col in self._control_cols if col in self._data.columns}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation of the run
        """
        run_dict = {
            'run_id': self._id,
            'times': self.time_points,
        }

        # Add state values
        for name, values in self.state_values.items():
            # Use original column name
            run_dict[name] = values

            # Also add as state variable
            clean_name = name.split('(')[0].strip()
            if clean_name != name:
                run_dict[clean_name] = values

        # Add control values
        for name, values in self.control_values.items():
            # Use original column name
            run_dict[name] = values

            # Also add as control variable
            clean_name = name.split('(')[0].strip()
            if clean_name != name:
                run_dict[clean_name] = values

        # Add time arrays for compatibility
        run_dict['states_times'] = self.time_points
        run_dict['controls_times'] = self.time_points

        # Add metadata and initial conditions
        run_dict.update(self.metadata)
        run_dict['initial_conditions'] = self.initial_conditions

        return run_dict


class GenericExcelDataset(ExperimentDataset):
    """A collection of experiment runs from Excel."""

    def __init__(self, runs: List[GenericExcelRun]):
        """
        Initialize generic experiment dataset.

        Args:
            runs: List of experiment runs
        """
        self._runs = runs
        self._run_map = {run.id: run for run in runs}
        self._norm_params = self._calculate_normalization()

    def _calculate_normalization(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate normalization parameters.

        Returns:
            Dictionary of normalization parameters
        """
        norm_params = {}

        # Collect all column names
        all_cols = set()
        for run in self._runs:
            all_cols.update(run.data.columns)

        # Calculate mean and std for each column
        for col in all_cols:
            # Collect values from all runs
            values = []
            for run in self._runs:
                if col in run.data.columns:
                    values.append(run.data[col].dropna().values)

            if values:
                # Concatenate values
                all_values = np.concatenate(values)

                # Calculate mean and std
                mean = float(np.mean(all_values))
                std = float(np.std(all_values))

                # Add to normalization parameters
                norm_params[col] = {'mean': mean, 'std': std}

                # Also add a version without units for easier reference
                clean_name = col.split('(')[0].strip()
                if clean_name != col:
                    norm_params[clean_name] = {'mean': mean, 'std': std}

        return norm_params

    def get_run(self, run_id: str) -> GenericExcelRun:
        """Get an experiment run by ID."""
        return self._run_map[run_id]

    @property
    def runs(self) -> List[GenericExcelRun]:
        """Get all experiment runs."""
        return self._runs

    @property
    def num_runs(self) -> int:
        """Get the number of runs."""
        return len(self._runs)

    @property
    def normalization_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get normalization parameters for all features."""
        return self._norm_params

    def get_run_dicts(self) -> List[Dict[str, Any]]:
        """
        Get all runs as dictionaries.

        Returns:
            List of run dictionaries
        """
        run_dicts = []
        for run in self._runs:
            run_dict = run.to_dict()
            run_dict['norm_params'] = self._norm_params
            run_dicts.append(run_dict)

        return run_dicts


class GenericExcelLoader(DataLoader):
    """
    Generic data loader for Excel files.

    This loader can handle any Excel file with time series data,
    without assumptions about specific column names or data structure.
    """

    def __init__(self,
                time_col: str,
                state_cols: List[str],
                control_cols: List[str],
                run_id_col: Optional[str] = None,
                sheet_name: Optional[Union[str, int]] = 0,
                preprocessing: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None):
        """
        Initialize generic Excel loader.

        Args:
            time_col: Name of time column
            state_cols: Names of state columns
            control_cols: Names of control columns
            run_id_col: Optional name of run ID column (if None, use index)
            sheet_name: Name or index of sheet to load (default: 0)
            preprocessing: Optional function to preprocess data
        """
        self.time_col = time_col
        self.state_cols = state_cols
        self.control_cols = control_cols
        self.run_id_col = run_id_col
        self.sheet_name = sheet_name
        self.preprocessing = preprocessing

    def load(self, file_path: str,
           run_ids: Optional[List[str]] = None,
           max_runs: Optional[int] = None) -> GenericExcelDataset:
        """
        Load data from Excel file.

        Args:
            file_path: Path to Excel file
            run_ids: Optional list of specific run IDs to load
            max_runs: Optional maximum number of runs to load

        Returns:
            GenericExcelDataset
        """
        # Read Excel file
        data = pd.read_excel(file_path, sheet_name=self.sheet_name)

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


def load_multiple_excel_files(file_paths: List[str],
                            loader: GenericExcelLoader,
                            **kwargs) -> GenericExcelDataset:
    """
    Load data from multiple Excel files.

    Args:
        file_paths: List of paths to Excel files
        loader: GenericExcelLoader instance
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