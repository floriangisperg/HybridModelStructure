# hybrid_models/data.py

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any


class TimeSeriesDataLoader:
    """
    Generic loader for time series data suitable.

    Uses standard process modeling nomenclature:
    - Z variables: Independent variables constant throughout the run (process conditions)
    - X variables: Observed uncontrolled/dependent variables (process states)
    - W variables: Controlled independent variables
    - F variables: Subset of W variables representing flows or feeds
    - Y variables: Variables typically measured once near the end (e.g., quality attributes)
    """

    def __init__(
            self,
            time_column: str,
            x_columns: List[str],  # Process states (previously state_columns)
            w_columns: Optional[List[str]] = None,  # Controlled variables
            f_columns: Optional[List[str]] = None,  # Flow/feed variables
            z_columns: Optional[List[str]] = None,  # Constant process conditions
            y_columns: Optional[List[str]] = None,  # End-of-run measurements
            run_id_column: Optional[str] = None
    ):
        """
        Initialize the data loader with column specifications.

        Args:
            time_column: Name of the column containing time values
            x_columns: Names of columns containing process state variables
            w_columns: Names of columns containing controlled variables
            f_columns: Names of columns containing flow/feed variables
            z_columns: Names of columns containing constant process conditions
            y_columns: Names of columns containing end-of-run measurements
            run_id_column: Name of the column containing run/experiment identifiers
        """
        self.time_column = time_column
        self.x_columns = x_columns
        self.w_columns = w_columns or []
        self.f_columns = f_columns or []
        self.z_columns = z_columns or []
        self.y_columns = y_columns or []
        self.run_id_column = run_id_column

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

        # Process X variables (process states)
        x_vars = {}
        for col in self.x_columns:
            if col in run_data.columns:
                # Extract data, handling NaN values
                state_data = run_data[[self.time_column, col]].dropna(subset=[col])
                state_times = jnp.array(state_data[self.time_column].values)
                state_values = jnp.array(state_data[col].values)

                x_vars[col] = {
                    'times': state_times,
                    'values': state_values
                }

        # Process W variables (controlled variables)
        w_vars = {}
        for col in self.w_columns:
            if col in run_data.columns:
                # Handle NaN values by forward-filling
                control_data = run_data[[self.time_column, col]].copy()
                control_data[col] = control_data[col].ffill()

                control_times = jnp.array(control_data[self.time_column].values)
                control_values = jnp.array(control_data[col].values)

                w_vars[col] = {
                    'times': control_times,
                    'values': control_values
                }

        # Process F variables (flow/feed variables, subset of W)
        f_vars = {}
        for col in self.f_columns:
            if col in run_data.columns:
                # Handle NaN values by forward-filling
                feed_data = run_data[[self.time_column, col]].copy()
                feed_data[col] = feed_data[col].ffill()

                feed_times = jnp.array(feed_data[self.time_column].values)
                feed_values = jnp.array(feed_data[col].values)

                f_vars[col] = {
                    'times': feed_times,
                    'values': feed_values
                }

        # Process Z variables (constant process conditions)
        z_vars = {}
        for col in self.z_columns:
            if col in run_data.columns:
                # For Z variables, we expect a constant value, but take the first non-NaN value to be safe
                z_value = run_data[col].dropna().iloc[0] if not run_data[col].dropna().empty else None
                if z_value is not None:
                    z_vars[col] = float(z_value)

        # Process Y variables (end-of-run measurements)
        y_vars = {}
        for col in self.y_columns:
            if col in run_data.columns:
                # For Y variables, we expect a measurement at the end of the run
                # Take the last non-NaN value
                y_value = run_data[col].dropna().iloc[-1] if not run_data[col].dropna().empty else None
                if y_value is not None:
                    y_vars[col] = float(y_value)

        # Create processed run data
        processed_run = {
            'run_id': run_id,
            'times': times,
            'X': x_vars,  # Process states
            'W': w_vars,  # Controlled variables
            'F': f_vars,  # Flow/feed variables
            'Z': z_vars,  # Constant process conditions
            'Y': y_vars  # End-of-run measurements
        }

        return processed_run


# Add to hybrid_models/data.py

def calculate_normalization_params(runs: List[Dict], variables: Dict[str, List[str]] = None) -> Dict[str, float]:
    """
    Calculate normalization parameters (mean, std) for specified variables across all runs.

    Args:
        runs: List of run dictionaries from TimeSeriesDataLoader
        variables: Dictionary mapping variable types ('X', 'W', 'F') to lists of variable names
                   If None, calculates for all variables in the runs

    Returns:
        Dictionary of normalization parameters (var_mean, var_std for each variable)
    """
    norm_params = {}

    # Default to process all variables if none specified
    if variables is None:
        variables = {
            'X': [],
            'W': [],
            'F': []
        }

        # Collect all variable names from the first run
        if runs:
            for var_type in ['X', 'W', 'F']:
                if var_type in runs[0]:
                    variables[var_type] = list(runs[0][var_type].keys())

    # Process each variable type
    for var_type, var_names in variables.items():
        for var_name in var_names:
            # Collect all values for this variable across runs
            all_values = []

            for run in runs:
                if var_type in run and var_name in run[var_type]:
                    all_values.append(run[var_type][var_name]['values'])

            if all_values:
                # Concatenate all values and calculate statistics
                all_values_array = jnp.concatenate(all_values)
                mean_val = float(jnp.mean(all_values_array))
                std_val = float(jnp.std(all_values_array))

                # Store normalization parameters
                norm_params[f"{var_name}_mean"] = mean_val
                norm_params[f"{var_name}_std"] = max(std_val, 1e-8)  # Avoid division by zero

    return norm_params


class DatasetPreparer:
    """
    Prepares datasets for ODE model training and evaluation from structured run data.
    """

    def __init__(self, norm_params: Dict[str, float] = None):
        """
        Initialize the dataset preparer.

        Args:
            norm_params: Optional normalization parameters to use
        """
        self.norm_params = norm_params

    def prepare_ode_datasets(
            self,
            runs: List[Dict],
            state_names: List[str],
            input_names: Dict[str, List[str]] = None,
            calculate_derivatives: bool = True
    ) -> List[Dict]:
        """
        Prepare datasets for ODE model training from structured run data.

        Args:
            runs: List of run dictionaries from TimeSeriesDataLoader
            state_names: Names of state variables to include in the dataset
            input_names: Dictionary mapping input types ('W', 'F', 'Z') to lists of input names
                         If None, includes all available inputs
            calculate_derivatives: Whether to calculate derivatives for inputs (e.g., feed rates)

        Returns:
            List of datasets ready for ODE training
        """
        datasets = []

        # Default input configuration if none provided
        if input_names is None:
            input_names = {
                'W': [],
                'F': [],
                'Z': []
            }

            # Collect all input names from the first run
            if runs:
                for input_type in ['W', 'F', 'Z']:
                    if input_type in runs[0]:
                        if input_type == 'Z':
                            input_names[input_type] = list(runs[0][input_type].keys())
                        else:
                            input_names[input_type] = list(runs[0][input_type].keys())

        for run in runs:
            # Create initial state dictionary
            initial_state = {}
            for state_name in state_names:
                if state_name in run['X']:
                    initial_state[state_name] = float(run['X'][state_name]['values'][0])
                else:
                    raise ValueError(f"State variable '{state_name}' not found in run data")

            # Find common evaluation times (from the first state variable)
            ref_state = run['X'][state_names[0]]
            eval_times = ref_state['times']

            # Create dictionary for time-dependent inputs
            time_dependent_inputs = {}

            # Process W variables (controlled variables)
            for var_name in input_names.get('W', []):
                if var_name in run['W']:
                    var_data = run['W'][var_name]
                    time_dependent_inputs[var_name] = (var_data['times'], var_data['values'])

            # Process F variables (flow/feed variables)
            for var_name in input_names.get('F', []):
                if var_name in run['F']:
                    var_data = run['F'][var_name]
                    time_dependent_inputs[var_name] = (var_data['times'], var_data['values'])

                    # Calculate derivatives (rates) if requested
                    if calculate_derivatives:
                        var_rate_name = f"{var_name}_rate"
                        var_rate = calculate_rate(var_data['times'], var_data['values'])
                        time_dependent_inputs[var_rate_name] = (var_data['times'], var_rate)

            # Create static inputs dictionary for Z variables
            static_inputs = {}
            for var_name in input_names.get('Z', []):
                if var_name in run['Z']:
                    static_inputs[var_name] = run['Z'][var_name]

            # Create dataset dictionary
            dataset = {
                'initial_state': initial_state,
                'times': eval_times,
                'time_dependent_inputs': time_dependent_inputs,
                'static_inputs': static_inputs
            }

            # Add true state values for loss calculation
            for state_name in state_names:
                if state_name in run['X']:
                    dataset[f'{state_name}_true'] = run['X'][state_name]['values']

            datasets.append(dataset)

        return datasets

    def normalize_dataset(self, dataset: Dict) -> Dict:
        """
        Normalize a dataset using stored normalization parameters.

        Args:
            dataset: Dataset dictionary to normalize

        Returns:
            Normalized dataset
        """
        if self.norm_params is None:
            return dataset  # Return unchanged if no normalization parameters

        normalized = dataset.copy()

        # Normalize initial state
        if 'initial_state' in normalized:
            normalized_initial_state = {}
            for state, value in normalized['initial_state'].items():
                mean_key = f"{state}_mean"
                std_key = f"{state}_std"
                if mean_key in self.norm_params and std_key in self.norm_params:
                    normalized_value = (value - self.norm_params[mean_key]) / self.norm_params[std_key]
                    normalized_initial_state[state] = normalized_value
                else:
                    normalized_initial_state[state] = value
            normalized['initial_state'] = normalized_initial_state

        # Normalize time-dependent inputs
        if 'time_dependent_inputs' in normalized:
            normalized_time_inputs = {}
            for var, (times, values) in normalized['time_dependent_inputs'].items():
                mean_key = f"{var}_mean"
                std_key = f"{var}_std"
                if mean_key in self.norm_params and std_key in self.norm_params:
                    normalized_values = (values - self.norm_params[mean_key]) / self.norm_params[std_key]
                    normalized_time_inputs[var] = (times, normalized_values)
                else:
                    normalized_time_inputs[var] = (times, values)
            normalized['time_dependent_inputs'] = normalized_time_inputs

        # Normalize static inputs
        if 'static_inputs' in normalized:
            normalized_static_inputs = {}
            for var, value in normalized['static_inputs'].items():
                mean_key = f"{var}_mean"
                std_key = f"{var}_std"
                if mean_key in self.norm_params and std_key in self.norm_params:
                    normalized_value = (value - self.norm_params[mean_key]) / self.norm_params[std_key]
                    normalized_static_inputs[var] = normalized_value
                else:
                    normalized_static_inputs[var] = value
            normalized['static_inputs'] = normalized_static_inputs

        # Normalize true state values (used for loss calculation)
        for key in normalized:
            if key.endswith('_true'):
                state = key[:-5]  # Remove '_true'
                mean_key = f"{state}_mean"
                std_key = f"{state}_std"
                if mean_key in self.norm_params and std_key in self.norm_params:
                    normalized[key] = (normalized[key] - self.norm_params[mean_key]) / self.norm_params[std_key]

        return normalized