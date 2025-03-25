"""
Data loading utilities for hybrid modeling.
"""

import pandas as pd
import jax.numpy as jnp
import abc
from typing import Dict, List, Optional, Any, Union, Tuple


class DataLoader(abc.ABC):
    """Base class for data loaders."""

    @abc.abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load data from source.

        Returns:
            List of run data dictionaries
        """
        pass


class ExcelExperimentLoader(DataLoader):
    """Data loader for experiments stored in Excel format."""

    def __init__(self, file_path: str, run_ids: Optional[List[str]] = None, max_runs: int = 5):
        """
        Initialize Excel experiment loader.

        Args:
            file_path: Path to the Excel file
            run_ids: Optional list of specific run IDs to load
            max_runs: Maximum number of runs to load if run_ids is None
        """
        self.file_path = file_path
        self.run_ids = run_ids
        self.max_runs = max_runs

    def load(self) -> List[Dict[str, Any]]:
        """
        Load experiment data from Excel file.

        Returns:
            List of run data dictionaries
        """
        # Load data
        data = pd.read_excel(self.file_path)

        # Get run IDs
        all_run_ids = data['RunID'].unique()

        # Use specified run IDs or first max_runs
        if self.run_ids is None:
            run_ids = all_run_ids[:self.max_runs]
        else:
            run_ids = self.run_ids

        # Process each run
        runs = []
        for run_id in run_ids:
            # Filter for the specified run
            run_data = data[data['RunID'] == run_id].sort_values('feedtimer(h)')

            # Extract state measurements (irregularly sampled)
            states_df = run_data[['feedtimer(h)', 'CDW(g/L)', 'Produktsol(g/L)']].dropna()
            states_times = jnp.array(states_df['feedtimer(h)'].values)
            X = jnp.array(states_df['CDW(g/L)'].values)
            P = jnp.array(states_df['Produktsol(g/L)'].values)

            # Extract extended control inputs
            controls_df = run_data[['feedtimer(h)', 'Temp(°C)', 'Feed(L)', 'InductorMASS(mg)',
                                    'Inductor(yesno)', 'Base(L)', 'Reaktorvolumen(L)']]

            # Handle potential NaN values in controls by forward-filling
            controls_df = controls_df.ffill()

            # Get inducer switch (Iy/n): 0 for no induction, 1 for induction
            controls_df['Iy/n'] = controls_df['Inductor(yesno)'].astype(int)

            # Convert to JAX arrays
            controls_times = jnp.array(controls_df['feedtimer(h)'].values)
            temp = jnp.array(controls_df['Temp(°C)'].values)
            feed = jnp.array(controls_df['Feed(L)'].values)
            inductor_mass = jnp.array(controls_df['InductorMASS(mg)'].values)
            inductor_switch = jnp.array(controls_df['Iy/n'].values)
            base = jnp.array(controls_df['Base(L)'].values)
            reactor_volume = jnp.array(controls_df['Reaktorvolumen(L)'].values)

            # Create processed run data
            runs.append({
                'run_id': run_id,
                'states_times': states_times,
                'X': X,
                'P': P,
                'controls_times': controls_times,
                'temp': temp,
                'feed': feed,
                'inductor_mass': inductor_mass,
                'inductor_switch': inductor_switch,
                'base': base,
                'reactor_volume': reactor_volume
            })

        # Calculate normalization parameters from all runs
        all_X = jnp.concatenate([run['X'] for run in runs])
        all_P = jnp.concatenate([run['P'] for run in runs])
        all_temp = jnp.concatenate([run['temp'] for run in runs])
        all_feed = jnp.concatenate([run['feed'] for run in runs])
        all_inductor_mass = jnp.concatenate([run['inductor_mass'] for run in runs])
        all_base = jnp.concatenate([run['base'] for run in runs])
        all_reactor_volume = jnp.concatenate([run['reactor_volume'] for run in runs])

        norm_params = {
            'temp_mean': float(all_temp.mean()),
            'temp_std': float(all_temp.std()),
            'feed_mean': float(all_feed.mean()),
            'feed_std': float(all_feed.std()),
            'inductor_mass_mean': float(all_inductor_mass.mean()),
            'inductor_mass_std': float(all_inductor_mass.std()),
            'base_mean': float(all_base.mean()),
            'base_std': float(all_base.std()),
            'reactor_volume_mean': float(all_reactor_volume.mean()),
            'reactor_volume_std': float(all_reactor_volume.std()),
            'X_mean': float(all_X.mean()),
            'X_std': float(all_X.std()),
            'P_mean': float(all_P.mean()),
            'P_std': float(all_P.std()),
        }

        # Add norm_params to each run
        for run in runs:
            run['norm_params'] = norm_params

        return runs


class CSVExperimentLoader(DataLoader):
    """Data loader for experiments stored in CSV format."""

    def __init__(self, file_path: str, time_col: str, run_id_col: str,
                 run_ids: Optional[List[str]] = None, max_runs: int = 5):
        """
        Initialize CSV experiment loader.

        Args:
            file_path: Path to the CSV file
            time_col: Name of the column containing time values
            run_id_col: Name of the column containing run IDs
            run_ids: Optional list of specific run IDs to load
            max_runs: Maximum number of runs to load if run_ids is None
        """
        self.file_path = file_path
        self.time_col = time_col
        self.run_id_col = run_id_col
        self.run_ids = run_ids
        self.max_runs = max_runs

    def load(self) -> List[Dict[str, Any]]:
        """
        Load experiment data from CSV file.

        Returns:
            List of run data dictionaries
        """
        # Load data
        data = pd.read_csv(self.file_path)

        # Get run IDs
        all_run_ids = data[self.run_id_col].unique()

        # Use specified run IDs or first max_runs
        if self.run_ids is None:
            run_ids = all_run_ids[:self.max_runs]
        else:
            run_ids = self.run_ids

        # Process each run
        runs = []
        for run_id in run_ids:
            # Filter for the specified run
            run_data = data[data[self.run_id_col] == run_id].sort_values(self.time_col)

            # Here you would need to customize the extraction of states and controls
            # to match your CSV structure. This is a placeholder:

            # Example: extract time, X (biomass), and P (product)
            # You would need to adjust column names to match your data
            times = jnp.array(run_data[self.time_col].values)
            # Assuming your CSV has columns named 'Biomass' and 'Product'
            X = jnp.array(run_data.get('Biomass', pd.Series([0] * len(run_data))).values)
            P = jnp.array(run_data.get('Product', pd.Series([0] * len(run_data))).values)

            # Create processed run data with minimal structure
            # You would need to expand this with your actual control variables
            runs.append({
                'run_id': run_id,
                'states_times': times,
                'X': X,
                'P': P,
                # Add other necessary fields
            })

        # Calculate normalization parameters
        # This would need to be customized for your data structure

        return runs


def create_split(runs: List[Dict[str, Any]],
                 test_fraction: float = 0.2,
                 random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split runs into training and testing sets.

    Args:
        runs: List of run data dictionaries
        test_fraction: Fraction of runs to use for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_runs, test_runs)
    """
    import numpy as np
    np.random.seed(random_seed)

    n_runs = len(runs)
    n_test = max(1, int(n_runs * test_fraction))

    # Randomly select indices for test set
    test_indices = np.random.choice(n_runs, n_test, replace=False)
    test_indices = set(test_indices)

    train_runs = [run for i, run in enumerate(runs) if i not in test_indices]
    test_runs = [run for i, run in enumerate(runs) if i in test_indices]

    return train_runs, test_runs