"""
General utility functions for the hybrid modeling framework.
"""

import os
import time
import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Callable, Tuple, Optional, Union
import equinox as eqx
import numpy as np


def setup_output_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create and return a timestamped output directory.

    Args:
        base_dir: Base directory for outputs
        experiment_name: Optional experiment name to prepend

    Returns:
        Path to the created output directory
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp

    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def save_model(model: Any, filepath: str) -> None:
    """
    Save a model to disk using equinox serialization.

    Args:
        model: The model to save
        filepath: Path where to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the model
    eqx.tree_serialise_leaves(filepath, model)
    print(f"Model saved to {filepath}")


def load_model(model_template: Any, filepath: str) -> Any:
    """
    Load a model from disk using equinox deserialization.

    Args:
        model_template: An instance of the model class with the same structure
        filepath: Path to the saved model

    Returns:
        The loaded model
    """
    loaded_model = eqx.tree_deserialise_leaves(filepath, model_template)
    print(f"Model loaded from {filepath}")
    return loaded_model


def get_control_at_time(t: float, times: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Get control input value at specific time t using nearest-time interpolation.

    Args:
        t: Time point
        times: Array of time points
        values: Array of values corresponding to time points

    Returns:
        Interpolated value at time t
    """
    idx = jnp.argmin(jnp.abs(times - t))
    return values[idx]


def calculate_derivative(t: float, times: jnp.ndarray, values: jnp.ndarray, dt: float = 0.01) -> float:
    """
    Calculate derivative at time t using forward differences.

    Args:
        t: Time point
        times: Array of time points
        values: Array of values corresponding to time points
        dt: Small time step for numerical derivative

    Returns:
        Approximate derivative at time t
    """
    current_value = get_control_at_time(t, times, values)
    future_value = get_control_at_time(t + dt, times, values)
    return (future_value - current_value) / dt


def partition_model(model: Any) -> Tuple[Any, Any]:
    """
    Partition a model into trainable and static parts.

    Args:
        model: The model to partition

    Returns:
        Tuple of (trainable_params, static_params)
    """
    return eqx.partition(model, eqx.is_array)


def activation_fn(name: str) -> Callable:
    """
    Get activation function by name.

    Args:
        name: Name of the activation function

    Returns:
        The corresponding JAX activation function
    """
    activations = {
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'tanh': jnp.tanh,
        'softplus': jax.nn.softplus,
        'elu': jax.nn.elu,
        'linear': lambda x: x,
    }

    if name not in activations:
        raise ValueError(f"Activation function '{name}' not recognized. "
                         f"Available options: {list(activations.keys())}")

    return activations[name]