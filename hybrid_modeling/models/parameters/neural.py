"""
Neural network-based parameter models.

This module provides implementations of neural network models
for predicting parameters of mechanistic models.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Mapping
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from hybrid_modeling.core.parameters import NeuralParameterModel


class MLP(eqx.Module):
    """Multi-layer perceptron implementation using Equinox."""

    layers: List
    normalization: Optional[Dict[str, Dict[str, float]]] = None
    input_features: List[str]

    def __init__(self,
                input_features: List[str],
                hidden_dims: List[int],
                output_dim: int = 1,
                activation: str = 'relu',
                output_activation: Optional[str] = None,
                normalization: Optional[Dict[str, Dict[str, float]]] = None,
                key=None):
        """
        Initialize MLP.

        Args:
            input_features: List of input feature names
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name for hidden layers
            output_activation: Optional activation function name for output layer
            normalization: Optional normalization parameters for input features
            key: Random key for initialization
        """
        self.input_features = input_features
        self.normalization = normalization

        # Set up random key
        if key is None:
            key = jax.random.PRNGKey(0)

        # Build layers
        layers = []

        # Input dimension is number of features
        in_dim = len(input_features)

        # First layer: input -> first hidden
        layer_key, key = jax.random.split(key)
        layers.append(eqx.nn.Linear(in_dim, hidden_dims[0], key=layer_key))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            # Add activation
            if activation == 'relu':
                layers.append(jax.nn.relu)
            elif activation == 'tanh':
                layers.append(jnp.tanh)
            elif activation == 'sigmoid':
                layers.append(jax.nn.sigmoid)
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # Add linear layer
            layer_key, key = jax.random.split(key)
            layers.append(eqx.nn.Linear(hidden_dims[i], hidden_dims[i+1], key=layer_key))

        # Add activation after last hidden layer
        if activation == 'relu':
            layers.append(jax.nn.relu)
        elif activation == 'tanh':
            layers.append(jnp.tanh)
        elif activation == 'sigmoid':
            layers.append(jax.nn.sigmoid)

        # Output layer
        layer_key, key = jax.random.split(key)
        layers.append(eqx.nn.Linear(hidden_dims[-1], output_dim, key=layer_key))

        # Add output activation if specified
        if output_activation == 'relu':
            layers.append(jax.nn.relu)
        elif output_activation == 'tanh':
            layers.append(jnp.tanh)
        elif output_activation == 'sigmoid':
            layers.append(jax.nn.sigmoid)
        elif output_activation == 'softplus':
            layers.append(jax.nn.softplus)
        elif output_activation is not None:
            raise ValueError(f"Unsupported output activation: {output_activation}")

        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MLPParameterModel(NeuralParameterModel):
    """
    Neural parameter model using multi-layer perceptrons.

    This model uses separate MLPs for each parameter to be predicted.
    """

    def __init__(self,
                parameter_configs: Dict[str, Dict[str, Any]],
                normalization: Optional[Dict[str, Dict[str, float]]] = None,
                key=None):
        """
        Initialize MLP parameter model.

        Args:
            parameter_configs: Dictionary mapping parameter names to configs
            normalization: Optional normalization parameters for input features
            key: Random key for initialization
        """
        self._normalization = normalization
        self._parameter_configs = parameter_configs

        # Set up random key
        if key is None:
            key = jax.random.PRNGKey(0)

        # Create MLPs for each parameter
        self._networks = {}
        self._input_features = {}

        for param_name, config in parameter_configs.items():
            # Get config values
            input_features = config.get('input_features', [])
            hidden_dims = config.get('hidden_dims', [16, 16])
            activation = config.get('activation', 'relu')
            output_activation = config.get('output_activation', None)

            # Create network
            net_key, key = jax.random.split(key)
            network = MLP(
                input_features=input_features,
                hidden_dims=hidden_dims,
                output_dim=1,  # Each network predicts a single parameter
                activation=activation,
                output_activation=output_activation,
                normalization=normalization,
                key=net_key
            )

            self._networks[param_name] = network
            self._input_features[param_name] = input_features

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a forward pass to predict parameters.

        Args:
            inputs: Input values

        Returns:
            Dictionary of predicted parameter values
        """
        parameters = {}

        for param_name, network in self._networks.items():
            # Extract inputs for this parameter
            input_features = self._input_features[param_name]
            feature_values = []

            for feature in input_features:
                if feature in inputs:
                    value = inputs[feature]

                    # Ensure the value is a scalar to avoid shape issues
                    # If it's an array, take the first element (useful during ODE integration)
                    if isinstance(value, (np.ndarray, jnp.ndarray)) and value.size > 1:
                        # Use JAX-compatible indexing instead of float conversion
                        value = value.reshape(-1)[0]
                    # No conversion to float here - keep as JAX array

                    # Apply normalization if available
                    if self._normalization and feature in self._normalization:
                        mean = self._normalization[feature].get('mean', 0.0)
                        std = self._normalization[feature].get('std', 1.0)
                        value = (value - mean) / (std + 1e-8)

                    feature_values.append(value)
                else:
                    # Only raise error if default values aren't provided
                    raise ValueError(f"Missing input feature: {feature}")

            # Convert to JAX array
            x = jnp.array(feature_values)

            # Predict parameter
            pred = network(x)

            # Store parameter value - keep as JAX array
            if pred.size == 1:
                parameters[param_name] = pred[0]  # Keep as JAX array
            else:
                parameters[param_name] = pred

        return parameters

    @property
    def parameter_names(self) -> List[str]:
        """Get the names of parameters predicted by this model."""
        return list(self._networks.keys())