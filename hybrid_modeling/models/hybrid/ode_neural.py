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

    This creates a generic vector field function that works with any
    mechanistic model following the standard interface.

    Args:
        mechanistic_model: The ODE mechanistic model
        parameter_model: The parameter model

    Returns:
        A vector field function
    """

    def vector_field(t, y, args):
        # Get inputs from args
        inputs = dict(args.get('inputs', {}))  # Make a copy using dict constructor

        # Get state names from the model or generate generic ones
        state_names = getattr(mechanistic_model, 'state_names',
                              [f"state_{i}" for i in range(len(y))])

        # Update inputs with current state values
        for i, name in enumerate(state_names):
            if i < len(y):  # Safety check
                inputs[name] = y[i]  # Keep as JAX array

        # Add time if not present
        if 't' not in inputs:
            inputs['t'] = t  # Keep as JAX scalar

        # Predict parameters using the parameter model
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
        JAX-compatible version.

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

        # Create initial state vector
        # Convert to jnp array directly without going through numpy
        y0 = jnp.array([initial_conditions.get(name, 0.0) for name in state_names])

        # Time range - convert to JAX values
        t0 = jnp.asarray(time_points[0])
        t1 = jnp.asarray(time_points[-1])

        # Prepare Diffrax components
        term = ODETerm(self.vector_field_fn)
        solver = solver_options.get('solver', Tsit5())
        saveat = SaveAt(ts=time_points)

        # Set up step size controller
        rtol = solver_options.get('rtol', 1e-4)  # Tighter tolerance
        atol = solver_options.get('atol', 1e-7)  # Tighter tolerance
        stepsize_controller = PIDController(rtol=rtol, atol=atol)

        # Use a much higher max_steps value to avoid the maximum steps error
        max_steps = solver_options.get('max_steps', 1000000)  # Increased from 50000

        try:
            # Solve the ODE - no need to convert y0 again
            sol = diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t1,
                dt0=solver_options.get('dt0', 0.01),
                y0=y0,
                args={'inputs': inputs},
                saveat=saveat,
                max_steps=max_steps,
                stepsize_controller=stepsize_controller
            )

            # Convert solution to dictionary
            solution = {}
            for i, name in enumerate(state_names):
                # Keep as JAX array - no conversion to numpy needed
                solution[name] = sol.ys[:, i]

            return solution

        except Exception as e:
            # Provide more helpful error information
            error_msg = f"Error solving ODE system: {str(e)}"

            print(f"DEBUG: Error in solve(): {error_msg}")
            print(f"DEBUG: Initial conditions: {initial_conditions}")
            print(f"DEBUG: State names: {state_names}")
            print(f"DEBUG: First few time points: {time_points[:5]}")

            if "maximum number of solver steps was reached" in str(e):
                error_msg += (
                    "\nThe ODE solver took too many steps. This often indicates a stiff system or unstable parameters. "
                    "Try increasing 'max_steps' even further, decreasing the time range, or using a stiff solver.")

            if "ConcretizationTypeError" in str(e):
                error_msg += ("\nThis is likely due to a JAX tracing error. "
                              "Make sure all operations in your vector field "
                              "function are JAX-compatible.")

                # For easier debugging, try printing some information about inputs
                print("DEBUG: Input keys:", list(inputs.keys()))
                print("DEBUG: Input types:", {k: type(v) for k, v in inputs.items()})

            raise ValueError(error_msg) from e

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