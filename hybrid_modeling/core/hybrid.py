"""
Core interfaces for hybrid models.

This module defines abstract base classes for hybrid models that combine
mechanistic models with parameter models.
"""

import abc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import numpy as np

from hybrid_modeling.core.mechanistic import MechanisticModel
from hybrid_modeling.core.parameters import ParameterModel


class HybridModel(abc.ABC):
    """Abstract base class for hybrid models."""

    @property
    @abc.abstractmethod
    def mechanistic_model(self) -> MechanisticModel:
        """Get the mechanistic model component."""
        pass

    @property
    @abc.abstractmethod
    def parameter_model(self) -> ParameterModel:
        """Get the parameter model component."""
        pass

    @abc.abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a forward pass of the hybrid model.

        Args:
            inputs: Input values

        Returns:
            Dictionary of outputs
        """
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for the model."""
        return {}


class StandardHybridModel(HybridModel):
    """
    Standard hybrid model implementation.

    This combines a mechanistic model with a parameter model in the standard way:
    1. Predict parameters using the parameter model
    2. Run the mechanistic model with the predicted parameters
    """

    def __init__(self,
                 mechanistic_model: MechanisticModel,
                 parameter_model: ParameterModel):
        """
        Initialize the hybrid model.

        Args:
            mechanistic_model: Mechanistic model component
            parameter_model: Parameter model component
        """
        self._mechanistic_model = mechanistic_model
        self._parameter_model = parameter_model

    @property
    def mechanistic_model(self) -> MechanisticModel:
        """Get the mechanistic model component."""
        return self._mechanistic_model

    @property
    def parameter_model(self) -> ParameterModel:
        """Get the parameter model component."""
        return self._parameter_model

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a forward pass of the hybrid model.

        Args:
            inputs: Input values

        Returns:
            Dictionary of outputs
        """
        # Predict parameters
        parameters = self.parameter_model.predict_parameters(inputs)

        # Run mechanistic model
        outputs = self.mechanistic_model.forward(inputs, parameters)

        return outputs


class ResidualHybridModel(HybridModel):
    """
    Residual hybrid model implementation.

    This model computes:
    output = mechanistic_model(inputs, parameters) + residual_model(inputs)

    This can be useful when the mechanistic model captures the main trends,
    but there are systematic residuals that can be learned by a data-driven model.
    """

    def __init__(self,
                 mechanistic_model: MechanisticModel,
                 parameter_model: ParameterModel,
                 residual_model: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Initialize the residual hybrid model.

        Args:
            mechanistic_model: Mechanistic model component
            parameter_model: Parameter model component
            residual_model: Model for predicting residuals
        """
        self._mechanistic_model = mechanistic_model
        self._parameter_model = parameter_model
        self._residual_model = residual_model

    @property
    def mechanistic_model(self) -> MechanisticModel:
        """Get the mechanistic model component."""
        return self._mechanistic_model

    @property
    def parameter_model(self) -> ParameterModel:
        """Get the parameter model component."""
        return self._parameter_model

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a forward pass of the hybrid model.

        Args:
            inputs: Input values

        Returns:
            Dictionary of outputs
        """
        # Predict parameters
        parameters = self.parameter_model.predict_parameters(inputs)

        # Run mechanistic model
        mechanistic_outputs = self.mechanistic_model.forward(inputs, parameters)

        # Predict residuals
        residual_outputs = self._residual_model(inputs)

        # Combine outputs
        combined_outputs = {}
        for key in mechanistic_outputs:
            if key in residual_outputs:
                combined_outputs[key] = mechanistic_outputs[key] + residual_outputs[key]
            else:
                combined_outputs[key] = mechanistic_outputs[key]

        return combined_outputs