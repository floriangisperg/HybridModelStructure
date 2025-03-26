"""
Core data interfaces for hybrid modeling.

This module defines abstract base classes for data handling in the hybrid modeling framework.
These interfaces are domain-agnostic and can be implemented for different types of data.
"""

import abc
from typing import Dict, List, Any, Optional, Sequence, Union, Tuple
import numpy as np


class DataPoint(abc.ABC):
    """Abstract base class for a single data point."""

    @property
    @abc.abstractmethod
    def inputs(self) -> Dict[str, Any]:
        """Get input values for this data point."""
        pass

    @property
    @abc.abstractmethod
    def outputs(self) -> Dict[str, Any]:
        """Get output/target values for this data point."""
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for this data point (optional)."""
        return {}


class TimeSeriesDataPoint(DataPoint):
    """A data point in a time series."""

    @property
    @abc.abstractmethod
    def time(self) -> float:
        """Get time value for this data point."""
        pass


class Dataset(abc.ABC):
    """Abstract base class for a dataset."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get the number of data points."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx) -> Union[DataPoint, Sequence[DataPoint]]:
        """Get a data point or a batch of data points."""
        pass

    @property
    @abc.abstractmethod
    def input_names(self) -> List[str]:
        """Get the names of input features."""
        pass

    @property
    @abc.abstractmethod
    def output_names(self) -> List[str]:
        """Get the names of output features."""
        pass

    @property
    def has_time_dimension(self) -> bool:
        """Whether the dataset has a time dimension."""
        return False

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for the dataset."""
        return {}


class TimeSeriesDataset(Dataset):
    """A dataset of time series data."""

    @property
    def has_time_dimension(self) -> bool:
        """Time series datasets have a time dimension."""
        return True

    @abc.abstractmethod
    def get_time_points(self) -> np.ndarray:
        """Get all time points in the dataset."""
        pass

    @abc.abstractmethod
    def get_sequence(self, idx: int) -> Sequence[TimeSeriesDataPoint]:
        """Get a complete time series sequence."""
        pass

    @property
    @abc.abstractmethod
    def num_sequences(self) -> int:
        """Get the number of time series sequences."""
        pass


class ExperimentRun(abc.ABC):
    """Abstract base class for an experiment run."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Get the ID of this run."""
        pass

    @property
    @abc.abstractmethod
    def data(self) -> Dataset:
        """Get the dataset for this run."""
        pass

    @property
    @abc.abstractmethod
    def initial_conditions(self) -> Dict[str, Any]:
        """Get initial conditions for this run."""
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for this run."""
        return {}


class ExperimentDataset(abc.ABC):
    """A collection of experiment runs."""

    @abc.abstractmethod
    def get_run(self, run_id: str) -> ExperimentRun:
        """Get an experiment run by ID."""
        pass

    @property
    @abc.abstractmethod
    def runs(self) -> List[ExperimentRun]:
        """Get all experiment runs."""
        pass

    @property
    @abc.abstractmethod
    def num_runs(self) -> int:
        """Get the number of runs."""
        pass

    @property
    def normalization_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get normalization parameters for all features."""
        return {}


class DataLoader(abc.ABC):
    """Abstract base class for data loaders."""

    @abc.abstractmethod
    def load(self, source: Any) -> Union[Dataset, ExperimentDataset]:
        """
        Load data from a source.

        Args:
            source: Data source (file path, database connection, etc.)

        Returns:
            A dataset or experiment dataset
        """
        pass