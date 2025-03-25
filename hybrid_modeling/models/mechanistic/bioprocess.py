"""
Bioprocess mechanistic model implementation.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from hybrid_modeling.models.mechanistic.base import MechanisticModel, ModelParameters
from hybrid_modeling.core.utils import get_control_at_time, calculate_derivative


class BioprocessParameters(ModelParameters):
    """Parameters for bioprocess model."""

    def __init__(self,
                 growth_rate_fn: Callable,
                 product_formation_fn: Callable):
        """
        Initialize bioprocess parameters.

        Args:
            growth_rate_fn: Function to compute growth rate (μ)
            product_formation_fn: Function to compute product formation rate (vp/x)
        """
        self.growth_rate_fn = growth_rate_fn
        self.product_formation_fn = product_formation_fn

    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute bioprocess parameters based on current inputs.

        Args:
            inputs: Input values

        Returns:
            Dictionary of parameter values
        """
        mu = self.growth_rate_fn(inputs)
        vpx = self.product_formation_fn(inputs)

        return {
            'mu': mu,
            'vpx': vpx
        }


class BioprocessModel(MechanisticModel):
    """
    Bioprocess mechanistic model with configurable parameters.

    This model describes the dynamics of biomass and product concentrations
    in a bioprocess, taking into account dilution effects from feed and base addition.
    """

    def __init__(self, parameters: Optional[ModelParameters] = None):
        """
        Initialize bioprocess model.

        Args:
            parameters: Model parameters object
        """
        self.parameters = parameters

    def system_equations(self, t: float, y: jnp.ndarray, args: Dict[str, Any]) -> jnp.ndarray:
        """
        Define the bioprocess ODE system.

        Args:
            t: Current time point
            y: Current state [X, P]
            args: Additional arguments for the ODE system

        Returns:
            Derivatives [dX/dt, dP/dt]
        """
        X, P = y

        # Create inputs dictionary with current state
        current_inputs = {
            'X': X,
            'P': P,
            't': t
        }

        # Add control inputs at current time
        control_features = [
            'temp', 'feed', 'inductor_mass', 'inductor_switch',
            'base', 'reactor_volume'
        ]

        controls_times = args['controls_times']

        for feature in control_features:
            if feature in args:
                current_inputs[feature] = get_control_at_time(t, controls_times, args[feature])

        # Get current volume
        current_volume = current_inputs['reactor_volume']

        # Calculate feed rate (volume change per hour)
        feed_rate = 0.0
        if 'feed' in args:
            feed_rate = calculate_derivative(t, controls_times, args['feed'])
            # Ensure non-negative feed rate (we can't remove feed)
            feed_rate = jnp.maximum(feed_rate, 0.0)

        # Calculate base rate (volume change per hour)
        base_rate = 0.0
        if 'base' in args:
            base_rate = calculate_derivative(t, controls_times, args['base'])
            # Ensure non-negative base rate (we can't remove base)
            base_rate = jnp.maximum(base_rate, 0.0)

        # Calculate total flow rate (L/h)
        total_flow_rate = feed_rate + base_rate

        # Calculate dilution rate (1/h)
        # Avoid division by zero
        dilution_rate = jnp.where(current_volume > 1e-6,
                                  total_flow_rate / current_volume,
                                  0.0)

        # Get parameters from the model parameters object
        # or use values provided in args
        if self.parameters is not None:
            params = self.parameters.get_parameters(current_inputs)
            mu = params['mu']
            vpx = params['vpx']
        else:
            # Use values provided in args
            if 'growth_fn' in args and 'product_fn' in args:
                mu = args['growth_fn'](current_inputs)
                vpx = args['product_fn'](current_inputs)
            else:
                # Default values if not provided
                mu = args.get('mu', 0.0)
                vpx = args.get('vpx', 0.0)

        # ODE system with dilution
        dXdt = mu * X - dilution_rate * X  # Biomass growth with dilution

        # Product formation with dilution
        # Note: inductor_switch can be 0 or 1, controlling whether product is formed
        dPdt = vpx * X * current_inputs['inductor_switch'] - dilution_rate * P

        return jnp.array([dXdt, dPdt])

    def solve_for_run(self, model: Any, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve ODE for a single run.

        Args:
            model: Model containing neural networks for parameters
            run_data: Data for a single run

        Returns:
            Dictionary containing solution and true values
        """
        # Extract data
        states_times = run_data['states_times']
        controls_times = run_data['controls_times']

        # Initial conditions
        X0 = run_data['X'][0]
        P0 = run_data['P'][0]
        y0 = jnp.array([X0, P0])

        # Arguments for ODE function
        args = {
            'controls_times': controls_times,
            'temp': run_data['temp'],
            'feed': run_data['feed'],
            'inductor_mass': run_data['inductor_mass'],
            'inductor_switch': run_data['inductor_switch'],
            'base': run_data['base'],
            'reactor_volume': run_data['reactor_volume']
        }

        # Add references to neural networks for parameter functions
        # This approach allows us to use networks from the hybrid model
        if hasattr(model, 'growth_nn') and hasattr(model, 'product_nn'):
            args['growth_fn'] = model.growth_nn
            args['product_fn'] = model.product_nn

        # Solve the system
        sol = self.solve(
            initial_conditions=y0,
            times=states_times,
            args=args
        )

        # Extract the solutions
        X_pred = sol['ys'][:, 0]
        P_pred = sol['ys'][:, 1]

        return {
            'times': states_times,
            'X_pred': X_pred,
            'P_pred': P_pred,
            'X_true': run_data['X'],
            'P_true': run_data['P']
        }