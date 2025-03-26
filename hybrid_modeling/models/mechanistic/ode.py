"""
Generic implementations of ODE-based mechanistic models.

This module provides base implementations for ODE-based models
that can be extended for specific applications.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Mapping
import numpy as np
import diffrax
import jax
import jax.numpy as jnp

from hybrid_modeling.core.mechanistic import ODEModel


class DiffraxODEModel(ODEModel):
    """
    ODE model implementation using Diffrax.

    This provides a general implementation for ODE models that can be
    solved using the Diffrax library with JAX.
    """

    def __init__(self,
                 state_names: List[str],
                 default_solver: Optional[diffrax.AbstractSolver] = None,
                 rtol: float = 1e-3,
                 atol: float = 1e-6):
        """
        Initialize the ODE model.

        Args:
            state_names: Names of state variables
            default_solver: Default ODE solver to use
            rtol: Relative tolerance for the solver
            atol: Absolute tolerance for the solver
        """
        self._state_names = state_names
        self._default_solver = default_solver or diffrax.Tsit5()
        self._rtol = rtol
        self._atol = atol

    @property
    def state_names(self) -> List[str]:
        """Get the names of state variables."""
        return self._state_names

    def system_equations(self,
                         t: float,
                         y: np.ndarray,
                         parameters: Dict[str, Any],
                         inputs: Dict[str, Any]) -> np.ndarray:
        """
        Define the ODE system. This should be implemented by subclasses.

        Args:
            t: Time point
            y: State vector
            parameters: Model parameters
            inputs: Additional inputs

        Returns:
            Derivatives of the state variables
        """
        raise NotImplementedError("Subclasses must implement system_equations")

    def _diffrax_system(self, t, y, args):
        """
        Wrapper for system_equations to use with Diffrax.

        Args:
            t: Time point
            y: State vector
            args: Dictionary of arguments including parameters and inputs

        Returns:
            Derivatives of the state variables
        """
        parameters = args.get('parameters', {})
        inputs = args.get('inputs', {})
        return self.system_equations(t, y, parameters, inputs)

    def solve(self,
              initial_conditions: Dict[str, float],
              time_points: np.ndarray,
              parameters: Dict[str, Any],
              inputs: Dict[str, Any],
              solver_options: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Solve the ODE system using Diffrax.

        Args:
            initial_conditions: Initial values for state variables
            time_points: Time points at which to evaluate the solution
            parameters: Model parameters
            inputs: Additional inputs
            solver_options: Options for the ODE solver

        Returns:
            Dictionary mapping state variable names to solution arrays
        """
        # Set default solver options
        solver_options = solver_options or {}

        # Convert initial conditions to array
        y0 = jnp.array([initial_conditions.get(name, 0.0) for name in self.state_names])

        # Get time range
        t0 = float(time_points[0])
        t1 = float(time_points[-1])

        # Create term for the ODE system
        term = diffrax.ODETerm(self._diffrax_system)

        # Get solver
        solver = solver_options.get('solver', self._default_solver)

        # Set up saving options
        saveat = diffrax.SaveAt(ts=time_points)

        # Set up step size controller
        rtol = solver_options.get('rtol', self._rtol)
        atol = solver_options.get('atol', self._atol)
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        # Solve ODE
        args = {
            'parameters': parameters,
            'inputs': inputs
        }

        sol = diffrax.diffeqsolve(
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

        # Convert solution to dictionary
        solution = {}
        for i, name in enumerate(self.state_names):
            solution[name] = sol.ys[:, i]

        return solution


class ParameterizedODEModel(DiffraxODEModel):
    """
    ODE model with user-defined equations.

    This allows defining an ODE model by providing a function for the system equations,
    without having to subclass DiffraxODEModel.
    """

    def __init__(self,
                 state_names: List[str],
                 equations_fn: Callable[[float, np.ndarray, Dict[str, Any], Dict[str, Any]], np.ndarray],
                 default_solver: Optional[diffrax.AbstractSolver] = None,
                 rtol: float = 1e-3,
                 atol: float = 1e-6):
        """
        Initialize the ODE model.

        Args:
            state_names: Names of state variables
            equations_fn: Function defining the system equations
            default_solver: Default ODE solver to use
            rtol: Relative tolerance for the solver
            atol: Absolute tolerance for the solver
        """
        super().__init__(state_names, default_solver, rtol, atol)
        self._equations_fn = equations_fn

    def system_equations(self,
                         t: float,
                         y: np.ndarray,
                         parameters: Dict[str, Any],
                         inputs: Dict[str, Any]) -> np.ndarray:
        """
        Define the ODE system using the provided function.

        Args:
            t: Time point
            y: State vector
            parameters: Model parameters
            inputs: Additional inputs

        Returns:
            Derivatives of the state variables
        """
        return self._equations_fn(t, y, parameters, inputs)