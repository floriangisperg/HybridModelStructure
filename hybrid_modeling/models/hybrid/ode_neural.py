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
        # Predict parameters
        parameters = self.parameter_model.predict_parameters(inputs)

        # Solve ODE
        return self.mechanistic_model.solve(
            initial_conditions=initial_conditions,
            time_points=time_points,
            parameters=parameters,
            inputs=inputs,
            solver_options=solver_options
        )

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


class TimeVaryingODENeuralModel(ODENeuralHybridModel):
    """
    Hybrid model with time-varying parameters.

    This model extends ODENeuralHybridModel to handle time-varying parameters.
    The neural network predicts parameters that can vary with time.
    """

    def __init__(self,
                 ode_model: ODEModel,
                 parameter_model: ParameterModel,
                 time_interpolation: str = 'linear'):
        """
        Initialize the hybrid model.

        Args:
            ode_model: ODE mechanistic model
            parameter_model: Neural parameter model
            time_interpolation: Method for interpolating parameters over time
                ('linear', 'nearest', 'cubic')
        """
        super().__init__(ode_model, parameter_model)
        self._time_interpolation = time_interpolation

    def predict_time_varying_parameters(self,
                                        t: float,
                                        inputs: Dict[str, Any],
                                        time_points: np.ndarray,
                                        precomputed_params: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Predict parameters for a specific time point.

        Args:
            t: Time point
            inputs: Additional inputs for the parameter model
            time_points: Time points for which parameters are precomputed
            precomputed_params: Optional precomputed parameters at time_points

        Returns:
            Dictionary of parameter values at time t
        """
        if precomputed_params is None:
            # Predict parameters at all time points
            all_params = {}
            for tp in time_points:
                inputs_at_t = inputs.copy()
                inputs_at_t['t'] = tp
                params_at_t = self.parameter_model.predict_parameters(inputs_at_t)

                for param, value in params_at_t.items():
                    if param not in all_params:
                        all_params[param] = []
                    all_params[param].append(value)

            # Convert to arrays
            for param in all_params:
                all_params[param] = np.array(all_params[param])
        else:
            all_params = precomputed_params

        # Interpolate to get parameters at time t
        interp_params = {}
        from scipy.interpolate import interp1d

        for param, values in all_params.items():
            kind = self._time_interpolation
            if len(time_points) < 4 and kind == 'cubic':
                kind = 'linear'  # Fallback if not enough points for cubic

            interp_fn = interp1d(time_points, values, kind=kind, bounds_error=False, fill_value="extrapolate")
            interp_params[param] = float(interp_fn(t))

        return interp_params

    def custom_system_equations(self,
                                t: float,
                                y: np.ndarray,
                                args: Dict[str, Any]) -> np.ndarray:
        """
        Custom ODE system with time-varying parameters.

        Args:
            t: Time point
            y: State vector
            args: Dictionary with inputs, time_points, and precomputed_params

        Returns:
            Derivatives of the state variables
        """
        inputs = args.get('inputs', {})
        time_points = args.get('time_points', np.array([0.0, 1.0]))
        precomputed_params = args.get('precomputed_params', None)

        # Predict parameters at time t
        parameters = self.predict_time_varying_parameters(
            t, inputs, time_points, precomputed_params
        )

        # Use the mechanistic model's system_equations
        return self.mechanistic_model.system_equations(t, y, parameters, inputs)

    def solve(self,
              initial_conditions: Dict[str, float],
              time_points: np.ndarray,
              inputs: Dict[str, Any],
              solver_options: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Solve the ODE system with time-varying neural-predicted parameters.

        Args:
            initial_conditions: Initial values for state variables
            time_points: Time points at which to evaluate the solution
            inputs: Additional inputs for the parameter model
            solver_options: Options for the ODE solver

        Returns:
            Dictionary mapping state variable names to solution arrays
        """
        # Precompute parameters at all time points
        precomputed_params = {}
        for tp in time_points:
            inputs_at_t = inputs.copy()
            inputs_at_t['t'] = tp
            params_at_t = self.parameter_model.predict_parameters(inputs_at_t)

            for param, value in params_at_t.items():
                if param not in precomputed_params:
                    precomputed_params[param] = []
                precomputed_params[param].append(value)

        # Convert to arrays
        for param in precomputed_params:
            precomputed_params[param] = np.array(precomputed_params[param])

        # Create custom model with time-varying parameters
        from hybrid_modeling.models.mechanistic.ode import ParameterizedODEModel

        custom_model = ParameterizedODEModel(
            state_names=getattr(self.mechanistic_model, 'state_names', []),
            equations_fn=lambda t, y, _, __: self.custom_system_equations(t, y, {
                'inputs': inputs,
                'time_points': time_points,
                'precomputed_params': precomputed_params
            })
        )

        # Solve ODE with custom model
        return custom_model.solve(
            initial_conditions=initial_conditions,
            time_points=time_points,
            parameters={},  # Parameters are handled internally
            inputs={},  # Inputs are handled internally
            solver_options=solver_options
        )