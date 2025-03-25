"""
Base classes for mechanistic models.
"""

import abc
from typing import Dict, List, Tuple, Optional, Any, Callable
import jax.numpy as jnp
import diffrax


class MechanisticModel(abc.ABC):
    """Abstract base class for mechanistic models."""

    @abc.abstractmethod
    def system_equations(self, t: float, y: jnp.ndarray, args: Dict[str, Any]) -> jnp.ndarray:
        """
        Define the ODE system.

        Args:
            t: Current time point
            y: Current state
            args: Additional arguments for the ODE system

        Returns:
            Derivatives of the system state
        """
        pass

    def solve(self,
              initial_conditions: jnp.ndarray,
              times: jnp.ndarray,
              args: Dict[str, Any],
              solver: Optional[diffrax.AbstractSolver] = None,
              rtol: float = 1e-3,
              atol: float = 1e-6) -> Dict[str, jnp.ndarray]:
        """
        Solve the ODE system.

        Args:
            initial_conditions: Initial values for all state variables
            times: Time points at which to evaluate the solution
            args: Additional arguments for the ODE system
            solver: ODE solver to use (default: Tsit5)
            rtol: Relative tolerance for the solver
            atol: Absolute tolerance for the solver

        Returns:
            Dictionary containing the solution
        """
        # Default to Tsit5 solver if none provided
        if solver is None:
            solver = diffrax.Tsit5()

        # Create term for the ODE system
        term = diffrax.ODETerm(self.system_equations)

        # Set up saving options
        saveat = diffrax.SaveAt(ts=times)

        # Set up step size controller
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        # Solve ODE
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=times[0],
            t1=times[-1],
            dt0=0.01,  # Initial time step
            y0=initial_conditions,
            args=args,
            saveat=saveat,
            max_steps=50000,
            stepsize_controller=stepsize_controller
        )

        # Return solution
        return {
            'times': times,
            'ys': sol.ys
        }


class ModelParameters(abc.ABC):
    """Abstract base class for model parameters."""

    @abc.abstractmethod
    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get model parameters based on inputs.

        Args:
            inputs: Input values

        Returns:
            Dictionary of parameter values
        """
        pass


class ConstantParameters(ModelParameters):
    """Model parameters with constant values."""

    def __init__(self, parameters: Dict[str, float]):
        """
        Initialize constant parameters.

        Args:
            parameters: Dictionary of parameter values
        """
        self.parameters = parameters

    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the constant parameter values.

        Args:
            inputs: Input values (not used)

        Returns:
            Dictionary of parameter values
        """
        return self.parameters


class DynamicParameters(ModelParameters):
    """Model parameters that depend on current conditions."""

    def __init__(self, parameter_functions: Dict[str, Callable]):
        """
        Initialize dynamic parameters.

        Args:
            parameter_functions: Dictionary mapping parameter names to functions
                that compute their values
        """
        self.parameter_functions = parameter_functions

    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute parameter values based on current inputs.

        Args:
            inputs: Input values

        Returns:
            Dictionary of computed parameter values
        """
        return {
            name: func(inputs) for name, func in self.parameter_functions.items()
        }