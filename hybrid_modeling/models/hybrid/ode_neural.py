"""
ODE-Neural hybrid model implementations.

This module provides implementations of hybrid models that combine
ODE-based mechanistic models with neural network parameter models.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Mapping
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve

from hybrid_modeling.core.hybrid import StandardHybridModel
from hybrid_modeling.core.mechanistic import ODEModel
from hybrid_modeling.core.parameters import ParameterModel


def create_vector_field(mechanistic_model, parameter_model):
    """
    Create a JAX-compatible vector field function for Diffrax.

    This function is designed to be compatible with JAX's tracing.

    Args:
        mechanistic_model: The ODE mechanistic model
        parameter_model: The parameter model

    Returns:
        A vector field function
    """
    def vector_field(t, y, args):
        # Get inputs from args
        inputs = args.get('inputs', {}).copy()

        # Update inputs with current state
        state_names = getattr(mechanistic_model, 'state_names', ['X', 'P'])
        for i, name in enumerate(state_names):
            # Use JAX's type conversion
            inputs[name] = jnp.asarray(y[i], dtype=float)

        # Add time if not present
        if 't' not in inputs:
            inputs['t'] = jnp.asarray(t, dtype=float)

        # Predict parameters using neural network
        parameters = parameter_model.predict_parameters(inputs)

        # Call mechanistic model's system equations
        return mechanistic_model.system_equations(t, y, parameters, inputs)

    return vector_field


class ODENeuralHybridModel(StandardHybridModel):
    """
    Hybrid model combining an ODE mechanistic model with a neural parameter model.

    This model uses a neural network to predict parameters for an ODE model.
    The ODE model is then solved using these parameters to make predictions.
    """

    def __init__(self,
                ode_model: ODEModel,
                parameter_model: ParameterModel):
        """
        Initialize the hybrid model.

        Args:
            ode_model: ODE mechanistic model
            parameter_model: Neural parameter model
        """
        super().__init__(ode_model, parameter_model)
        # Create the vector field function
        self.vector_field_fn = create_vector_field(ode_model, parameter_model)

    def solve(self,
             initial_conditions: Dict[str, float],
             time_points: np.ndarray,
             inputs: Dict[str, Any],
             solver_options: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Solve the ODE system with neural-predicted parameters.

        Args:
            initial_conditions: Initial values for state variables
            time_points: Time points at which to evaluate the solution
            inputs: Additional inputs for the parameter model
            solver_options: Options for the ODE solver

        Returns:
            Dictionary mapping state variable names to solution arrays
        """
        # Default options
        solver_options = solver_options or {}

        # Get state names from mechanistic model
        state_names = getattr(self.mechanistic_model, 'state_names', ['X', 'P'])

        # Create initial state vector from numpy array
        # Convert to numpy first to avoid JAX tracing issues
        y0 = np.array([initial_conditions.get(name, 0.0) for name in state_names])

        # Time range - convert to standard floats for safety
        t0 = float(time_points[0])
        t1 = float(time_points[-1])

        # Prepare Diffrax components
        term = ODETerm(self.vector_field_fn)
        solver = solver_options.get('solver', Tsit5())
        saveat = SaveAt(ts=time_points)

        # Set up step size controller
        rtol = solver_options.get('rtol', 1e-3)
        atol = solver_options.get('atol', 1e-6)
        stepsize_controller = PIDController(rtol=rtol, atol=atol)

        try:
            # Convert initial conditions to jnp array right before solving
            jax_y0 = jnp.array(y0)

            # Solve the ODE
            sol = diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t1,
                dt0=solver_options.get('dt0', 0.01),
                y0=jax_y0,
                args={'inputs': inputs},
                saveat=saveat,
                max_steps=solver_options.get('max_steps', 50000),
                stepsize_controller=stepsize_controller
            )

            # Convert solution to dictionary
            solution = {}
            for i, name in enumerate(state_names):
                # Convert to numpy array for consistency
                solution[name] = np.array(sol.ys[:, i])

            return solution

        except Exception as e:
            # Provide more helpful error information
            error_msg = f"Error solving ODE system: {str(e)}"
            if "ConcretizationTypeError" in str(e):
                error_msg += ("\nThis is likely due to a JAX tracing error. "
                             "Make sure all operations in your vector field "
                             "function are JAX-compatible.")
            raise ValueError(error_msg)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a forward pass of the hybrid model.

        Args:
            inputs: Input values including initial_conditions and time_points

        Returns:
            Dictionary of outputs from the ODE solution
        """
        initial_conditions = inputs.get('initial_conditions', {})
        time_points = inputs.get('time_points', np.array([0.0, 1.0]))
        solver_options = inputs.get('solver_options', {})

        return self.solve(
            initial_conditions=initial_conditions,
            time_points=time_points,
            inputs=inputs,
            solver_options=solver_options
        )