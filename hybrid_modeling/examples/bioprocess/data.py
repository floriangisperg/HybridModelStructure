"""
Bioprocess-specific data loading utilities.

This module provides specialized data loaders for bioprocess experimental data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from hybrid_modeling.data.loaders.excel import GenericExcelLoader, GenericExcelRun, GenericExcelDataset


class BioprocessDataPreprocessor:
    """
    Preprocessor for bioprocess data.

    This class handles preprocessing specific to bioprocess data,
    such as handling missing values, deriving new features, etc.
    """

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a bioprocess dataframe.

        Args:
            df: Raw dataframe

        Returns:
            Preprocessed dataframe
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()

        # Handle NaN values in state measurements by dropping those rows
        if 'CDW(g/L)' in processed_df.columns and 'Produktsol(g/L)' in processed_df.columns:
            # Only keep rows where both biomass and product measurements are available
            # This is for state measurements only
            state_df = processed_df[['feedtimer(h)', 'CDW(g/L)', 'Produktsol(g/L)']].dropna()

            # Mark which rows have state measurements
            processed_df['has_states'] = False
            for _, row in state_df.iterrows():
                # Find the corresponding row in processed_df
                mask = processed_df['feedtimer(h)'] == row['feedtimer(h)']
                processed_df.loc[mask, 'has_states'] = True

        # Handle NaN values in control inputs by forward-filling
        control_cols = [
            'Temp(°C)', 'Feed(L)', 'InductorMASS(mg)',
            'Inductor(yesno)', 'Base(L)', 'Reaktorvolumen(L)'
        ]

        # Forward-fill each control column individually
        for col in control_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].ffill()

        # Process inductor switch
        if 'Inductor(yesno)' in processed_df.columns:
            # Convert to binary 0/1
            processed_df['Iy/n'] = processed_df['Inductor(yesno)'].astype(int)

        return processed_df


class BioprocessExcelLoader(GenericExcelLoader):
    """
    Data loader for bioprocess experiments stored in Excel format.
    """

    def __init__(self):
        """Initialize bioprocess Excel loader with appropriate column mappings."""
        super().__init__(
            time_col='feedtimer(h)',
            state_cols=['CDW(g/L)', 'Produktsol(g/L)'],
            control_cols=[
                'Temp(°C)', 'Feed(L)', 'InductorMASS(mg)',
                'Inductor(yesno)', 'Base(L)', 'Reaktorvolumen(L)', 'Iy/n'
            ],
            run_id_col='RunID',
            preprocessing=BioprocessDataPreprocessor.preprocess_dataframe
        )

    def process_loaded_data(self, dataset: GenericExcelDataset) -> List[Dict[str, Any]]:
        """
        Process loaded dataset into format expected by bioprocess models.

        Args:
            dataset: Loaded dataset

        Returns:
            List of run dictionaries
        """
        run_dicts = []

        for run in dataset.runs:
            run_dict = {
                'run_id': run.id,
                'states_times': run.time_points,
                'controls_times': run.time_points,
                'X': run.state_values.get('CDW(g/L)', np.array([])),
                'P': run.state_values.get('Produktsol(g/L)', np.array([])),
                'temp': run.control_values.get('Temp(°C)', np.array([])),
                'feed': run.control_values.get('Feed(L)', np.array([])),
                'inductor_mass': run.control_values.get('InductorMASS(mg)', np.array([])),
                'inductor_switch': run.control_values.get('Iy/n', np.array([])),
                'base': run.control_values.get('Base(L)', np.array([])),
                'reactor_volume': run.control_values.get('Reaktorvolumen(L)', np.array([])),
                'norm_params': dataset.normalization_parameters
            }

            run_dicts.append(run_dict)

        return run_dicts


def load_bioprocess_data(train_file: str,
                         test_file: Optional[str] = None,
                         run_ids: Optional[List[str]] = None,
                         max_runs: int = 5) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Load bioprocess data from training and optional testing files.

    Args:
        train_file: Path to training data file
        test_file: Optional path to testing data file
        run_ids: Optional list of specific run IDs to load
        max_runs: Maximum number of runs to load if run_ids is None

    Returns:
        Tuple of (train_runs, test_runs)
    """
    # Create loader
    loader = BioprocessExcelLoader()

    # Load training data
    train_dataset = loader.load(train_file, run_ids=run_ids, max_runs=max_runs)
    train_runs = loader.process_loaded_data(train_dataset)

    # Load testing data if provided
    test_runs = None
    if test_file and os.path.exists(test_file):
        test_dataset = loader.load(test_file, run_ids=run_ids, max_runs=max_runs)
        test_runs = loader.process_loaded_data(test_dataset)

    return train_runs, test_runs