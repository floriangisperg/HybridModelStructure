# hybrid_models/data.py

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Set


class TimeSeriesDataLoader:
    """
    Generic loader for time series data suitable for hybrid modeling with standardized nomenclature.

    Nomenclature:
    - X variables: Observed uncontrolled (dependent) variables (process state/response)
    - W variables: Controlled independent variables
    - F variables: A subset of W that are flows/feeds affecting mass balance
    - Z variables: Independent variables expressing process conditions constant throughout a run
    - Y variables: Variables typically measured only once near the end of the experiment (e.g., CQAs)
    """

    def __init__(
            self,
            time_column: str,
            x_variables: List[str],
            w_variables: Optional[List[str]] = None,
            f_variables: Optional[List[str]] = None,
            z_variables: Optional[List[str]] = None,
            y_variables: Optional[List[str]] = None,
            run_id_column: Optional[str] = None
    ):
        """
        Initialize the data loader with column specifications.

        Args:
            time_column: Name of the column containing time values
            x_variables: Names of columns containing observed state variables
            w_variables: Names of columns containing controlled variables
            f_variables: Names of columns containing flow/feed variables (subset of w_variables)
            z_variables: Names of columns containing constant process conditions
            y_variables: Names of columns containing end-of-run measurements
            run_id_column: Name of the column containing run/experiment identifiers
        """
        self.time_column = time_column
        self.x_variables = x_variables
        self.w_variables = w_variables or []
        self.f_variables = f_variables or []
        self.z_variables = z_variables or []
        self.y_variables = y_variables or []
        self.run_id_column = run_id_column

        # Ensure f_variables are a subset of w_variables
        if not set(self.f_variables).issubset(set(self.w_variables)) and self.f_variables:
            # Add any f_variables not in w_variables to w_variables
            missing_w = set(self.f_variables) - set(self.w_variables)
            self.w_variables.extend(list(missing_w))

    def load_from_excel(
            self,
            file_path: str,
            run_ids: Optional[List[Any]] = None,
            max_runs: Optional[int] = None,
            sheet_name: Union[str, int] = 0
    ) -> List[Dict]:
        """
        Load time series data from an Excel file.

        Args:
            file_path: Path to the Excel file
            run_ids: Specific run IDs to load (if None, load based on max_runs)
            max_runs: Maximum number of runs to load (if run_ids is None)
            sheet_name: Name or index of the sheet to load

        Returns:
            List of dictionaries containing data for each run
        """
        # Load data from Excel
        data = pd.read_excel(file_path, sheet_name=sheet_name)

        # Process and return the data
        return self._process_dataframe(data, run_ids, max_runs)

    def load_from_csv(
            self,
            file_path: str,
            run_ids: Optional[List[Any]] = None,
            max_runs: Optional[int] = None,
            **kwargs
    ) -> List[Dict]:
        """
        Load time series data from a CSV file.

        Args:
            file_path: Path to the CSV file
            run_ids: Specific run IDs to load (if None, load based on max_runs)
            max_runs: Maximum number of runs to load (if run_ids is None)
            **kwargs: Additional arguments to pass to pd.read_csv

        Returns:
            List of dictionaries containing data for each run
        """
        # Load data from CSV
        data = pd.read_csv(file_path, **kwargs)

        # Process and return the data
        return self._process_dataframe(data, run_ids, max_runs)

    def _process_dataframe(
            self,
            data: pd.DataFrame,
            run_ids: Optional[List[Any]] = None,
            max_runs: Optional[int] = None
    ) -> List[Dict]:
        """
        Process a DataFrame into the required format for hybrid modeling.

        Args:
            data: DataFrame containing the time series data
            run_ids: Specific run IDs to load (if None, load based on max_runs)
            max_runs: Maximum number of runs to load (if run_ids is None)

        Returns:
            List of dictionaries containing data for each run
        """
        # Check if run ID column exists
        if self.run_id_column is not None:
            if self.run_id_column not in data.columns:
                raise ValueError(f"Run ID column '{self.run_id_column}' not found in DataFrame")

            # Get all unique run IDs
            all_run_ids = data[self.run_id_column].unique()

            # Select run IDs to process
            if run_ids is not None:
                # Use specified run IDs
                selected_run_ids = run_ids
            elif max_runs is not None:
                # Use first max_runs
                selected_run_ids = all_run_ids[:max_runs]
            else:
                # Use all run IDs
                selected_run_ids = all_run_ids
        else:
            # If no run ID column is specified, treat the entire dataset as a single run
            selected_run_ids = [None]

        # Process each run
        runs = []
        for run_id in selected_run_ids:
            if run_id is not None:
                # Filter for the specified run
                run_data = data[data[self.run_id_column] == run_id].copy()
            else:
                # Use the entire dataset
                run_data = data.copy()

            # Sort by time
            run_data = run_data.sort_values(self.time_column)

            # Process the run data
            processed_run = self._process_single_run(run_data, run_id)
            runs.append(processed_run)

        return runs

    def _process_single_run(self, run_data: pd.DataFrame, run_id: Any) -> Dict:
        """
        Process a single run's data.

        Args:
            run_data: DataFrame containing a single run's data
            run_id: Identifier for this run

        Returns:
            Dictionary containing processed data for this run
        """
        # Extract time values
        times = jnp.array(run_data[self.time_column].values)

        # Extract X variables (observed state variables)
        x_states = {}
        for col in self.x_variables:
            if col in run_data.columns:
                # Extract the state data, handling NaN values if present
                state_data = run_data[[self.time_column, col]].dropna(subset=[col])
                state_times = jnp.array(state_data[self.time_column].values)
                state_values = jnp.array(state_data[col].values)

                x_states[col] = {
                    'times': state_times,
                    'values': state_values
                }

        # Extract W variables (controlled variables)
        w_controls = {}
        for col in self.w_variables:
            if col in run_data.columns:
                # Handle NaN values in controls by forward-filling
                control_data = run_data[[self.time_column, col]].copy()
                control_data[col] = control_data[col].ffill()

                control_times = jnp.array(control_data[self.time_column].values)
                control_values = jnp.array(control_data[col].values)

                w_controls[col] = {
                    'times': control_times,
                    'values': control_values
                }

        # Extract F variables (feed/flow variables - subset of W)
        f_variables = {}
        for col in self.f_variables:
            if col in w_controls:
                f_variables[col] = w_controls[col]

        # Extract Z variables (constant process conditions)
        z_conditions = {}
        for col in self.z_variables:
            if col in run_data.columns:
                # Take the first non-NaN value as the constant condition
                value = run_data[col].dropna().iloc[0] if not run_data[col].dropna().empty else None
                if value is not None:
                    z_conditions[col] = float(value)

        # Extract Y variables (end-of-run measurements)
        y_measurements = {}
        for col in self.y_variables:
            if col in run_data.columns:
                # Take the last non-NaN value as the end measurement
                value = run_data[col].dropna().iloc[-1] if not run_data[col].dropna().empty else None
                if value is not None:
                    y_measurements[col] = float(value)

        # Create processed run data
        processed_run = {
            'run_id': run_id,
            'times': times,
            'X': x_states,  # Observed state variables
            'W': w_controls,  # Controlled variables
            'F': f_variables,  # Flow/feed variables (subset of W)
            'Z': z_conditions,  # Constant process conditions
            'Y': y_measurements  # End-of-run measurements
        }

        return processed_run

    def calculate_normalization_params(self, runs: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate normalization parameters from a list of runs.

        Args:
            runs: List of processed run dictionaries

        Returns:
            Dictionary of normalization parameters
        """
        norm_params = {
            'X': {},
            'W': {},
            'F': {},
            'Z': {},
            'Y': {}
        }

        # Collect all values for each variable type
        all_values = {
            'X': {col: [] for col in self.x_variables},
            'W': {col: [] for col in self.w_variables},
            'F': {col: [] for col in self.f_variables},
            'Z': {col: [] for col in self.z_variables},
            'Y': {col: [] for col in self.y_variables}
        }

        # Collect values from all runs
        for run in runs:
            # X variables
            for col in self.x_variables:
                if col in run['X']:
                    all_values['X'][col].extend(run['X'][col]['values'])

            # W variables
            for col in self.w_variables:
                if col in run['W']:
                    all_values['W'][col].extend(run['W'][col]['values'])

            # F variables (already a subset of W, but kept separate for clarity)
            for col in self.f_variables:
                if col in run['F']:
                    all_values['F'][col].extend(run['F'][col]['values'])

            # Z variables
            for col in self.z_variables:
                if col in run['Z']:
                    all_values['Z'][col].append(run['Z'][col])

            # Y variables
            for col in self.y_variables:
                if col in run['Y']:
                    all_values['Y'][col].append(run['Y'][col])

        # Calculate mean and std for each variable
        for var_type in ['X', 'W', 'F', 'Z', 'Y']:
            for col, values in all_values[var_type].items():
                if values:
                    values_array = jnp.array(values)
                    norm_params[var_type][col] = {
                        'mean': float(jnp.mean(values_array)),
                        'std': float(jnp.std(values_array)) or 1.0  # Use 1.0 if std is 0
                    }

        return norm_params

    def apply_normalization(self, runs: List[Dict], norm_params: Dict) -> List[Dict]:
        """
        Apply normalization to a list of runs.

        Args:
            runs: List of processed run dictionaries
            norm_params: Dictionary of normalization parameters

        Returns:
            List of normalized run dictionaries
        """
        normalized_runs = []

        for run in runs:
            normalized_run = {
                'run_id': run['run_id'],
                'times': run['times'],
                'X': {},
                'W': {},
                'F': {},
                'Z': {},
                'Y': {}
            }

            # Normalize X variables
            for col, data in run['X'].items():
                if col in norm_params['X']:
                    normalized_values = (data['values'] - norm_params['X'][col]['mean']) / norm_params['X'][col]['std']
                    normalized_run['X'][col] = {
                        'times': data['times'],
                        'values': normalized_values,
                        'original_values': data['values']
                    }
                else:
                    normalized_run['X'][col] = data

            # Normalize W variables
            for col, data in run['W'].items():
                if col in norm_params['W']:
                    normalized_values = (data['values'] - norm_params['W'][col]['mean']) / norm_params['W'][col]['std']
                    normalized_run['W'][col] = {
                        'times': data['times'],
                        'values': normalized_values,
                        'original_values': data['values']
                    }
                else:
                    normalized_run['W'][col] = data

            # Normalize F variables
            for col, data in run['F'].items():
                if col in norm_params['F']:
                    normalized_values = (data['values'] - norm_params['F'][col]['mean']) / norm_params['F'][col]['std']
                    normalized_run['F'][col] = {
                        'times': data['times'],
                        'values': normalized_values,
                        'original_values': data['values']
                    }
                else:
                    normalized_run['F'][col] = data

            # Normalize Z variables
            for col, value in run['Z'].items():
                if col in norm_params['Z']:
                    normalized_value = (value - norm_params['Z'][col]['mean']) / norm_params['Z'][col]['std']
                    normalized_run['Z'][col] = normalized_value
                    normalized_run['Z'][f'{col}_original'] = value
                else:
                    normalized_run['Z'][col] = value

            # Normalize Y variables
            for col, value in run['Y'].items():
                if col in norm_params['Y']:
                    normalized_value = (value - norm_params['Y'][col]['mean']) / norm_params['Y'][col]['std']
                    normalized_run['Y'][col] = normalized_value
                    normalized_run['Y'][f'{col}_original'] = value
                else:
                    normalized_run['Y'][col] = value

            normalized_runs.append(normalized_run)

        return normalized_runs