"""
Core interfaces for parameter models.

This module defines abstract base classes for parameter models in the hybrid modeling framework.
These models predict parameters for mechanistic models based on inputs.
"""

import abc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import numpy as np


class ParameterModel(abc.ABC):
    """Abstract base class for parameter models."""

    @abc.abstractmethod
    def predict_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict parameters based on inputs.

        Args:
            inputs: Input values

        Returns:
            Dictionary of parameter values
        """
        pass

    @property
    @abc.abstractmethod
    def parameter_names(self) -> List[str]:
        """Get the names of parameters predicted by this model."""
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for the model."""
        return {}


class NeuralParameterModel(ParameterModel):
    """Abstract base class for neural network parameter models."""

    @abc.abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a forward pass of the neural network.

        Args:
            inputs: Input values

        Returns:
            Dictionary of parameter values
        """
        pass

    def predict_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict parameters using the neural network.

        Args:
            inputs: Input values

        Returns:
            Dictionary of parameter values
        """
        return self.forward(inputs)


class ConstantParameterModel(ParameterModel):
    """Model that returns constant parameter values."""

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize with constant parameter values.

        Args:
            parameters: Dictionary of parameter values
        """
        self._parameters = parameters

    def predict_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return constant parameter values.

        Args:
            inputs: Input values (not used)

        Returns:
            Dictionary of parameter values
        """
        return self._parameters

    @property
    def parameter_names(self) -> List[str]:
        """Get the names of parameters."""
        return list(self._parameters.keys())


class CompositeParameterModel(ParameterModel):
    """
    Composite model that combines multiple parameter models.

    This can be used to combine different types of parameter models,
    each predicting a subset of parameters.
    """

    def __init__(self, models: Dict[str, ParameterModel]):
        """
        Initialize with multiple parameter models.

        Args:
            models: Dictionary mapping parameter names to models
        """
        self._models = models

    def predict_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict parameters using all models.

        Args:
            inputs: Input values

        Returns:
            Dictionary of parameter values from all models
        """
        parameters = {}
        for name, model in self._models.items():
            parameters[name] = model.predict_parameters(inputs)
        return parameters

    @property
    def parameter_names(self) -> List[str]:
        """Get the names of parameters from all models."""
        names = []
        for model in self._models.values():
            names.extend(model.parameter_names)
        return names