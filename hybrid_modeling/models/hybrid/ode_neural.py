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

from hybrid_modeling.core.hybrid import StandardHybridModel
from hybrid_modeling.core.mechanistic import ODEModel
from hybrid_modeling.core.parameters import ParameterModel


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

    def _custom_diffrax_system(self, t, y, args):
        """
        Custom ODE system for Diffrax with neural parameter prediction.

        This method is called by Diffrax during ODE integration. It:
        1. Takes the current state y and time t
        2. Updates the inputs with these values
        3. Predicts parameters using the neural network
        4. Calls the mechanistic model's system equations

        Args:
            t: Time point
            y: State vector
            args: Arguments dictionary containing inputs and other data

        Returns:
            Derivatives of the state variables
        """
        # Get inputs from args
        inputs = args.get('inputs', {}).copy()

        # Update inputs with current state
        state_names = getattr(self.mechanistic_model, 'state_names', ['X', 'P'])
        for i, name in enumerate(state_names):
            # Make sure we use a scalar value
            inputs[name] = float(y[i])

        # Add time if not present
        if 't' not in inputs:
            inputs['t'] = float(t)

        # Predict parameters using neural network
        try:
            parameters = self.parameter_model.predict_parameters(inputs)
        except Exception as e:
            # Provide more helpful error message
            raise ValueError(f"Error predicting parameters at t={t}, y={y}: {str(e)}")

        # Call mechanistic model's system equations
        return self.mechanistic_model.system_equations(t, y, parameters, inputs)

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

        # Create initial state vector
        y0 = np.array([initial_conditions.get(name, 0.0) for name in state_names])

        # Time range
        t0 = float(time_points[0])
        t1 = float(time_points[-1])

        # Prepare Diffrax components
        from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve

        # Prepare the system function
        term = ODETerm(self._custom_diffrax_system)

        # Choose solver
        solver = solver_options.get('solver', Tsit5())

        # Set up saving options
        saveat = SaveAt(ts=time_points)

        # Set up step size controller
        rtol = solver_options.get('rtol', 1e-3)
        atol = solver_options.get('atol', 1e-6)
        stepsize_controller = PIDController(rtol=rtol, atol=atol)

        # Solve the ODE
        args = {'inputs': inputs}

        try:
            sol = diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t1,
                dt0=solver_options.get('dt0', 0.01),
                y0=y0,
                args=args,
                saveat=saveat,
                max_steps=solver_options.get('max_steps', 50000),
                stepsize_controller=stepsize_controller
            )

            # Format the solution
            solution = {}
            for i, name in enumerate(state_names):
                solution[name] = sol.ys[:, i]

            return solution

        except Exception as e:
            # Provide a more helpful error message
            raise ValueError(f"Error solving ODE system: {str(e)}")

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