"""
Bioprocess-specific model implementations.

This module contains mechanistic and hybrid models for bioprocess modeling,
built on top of the generic hybrid modeling framework.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from hybrid_modeling.models.mechanistic.ode import DiffraxODEModel
from hybrid_modeling.models.parameters.neural import MLPParameterModel
from hybrid_modeling.models.hybrid.ode_neural import ODENeuralHybridModel


def get_control_at_time(t: float, times: np.ndarray, values: np.ndarray) -> float:
    """
    Get control input value at specific time t using nearest-time interpolation.

    Args:
        t: Time point
        times: Array of time points
        values: Array of values corresponding to time points

    Returns:
        Interpolated value at time t
    """
    idx = np.argmin(np.abs(times - t))
    return float(values[idx])


def calculate_derivative(t: float, times: np.ndarray, values: np.ndarray, dt: float = 0.01) -> float:
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


class BioprocessModel(DiffraxODEModel):
    """
    Bioprocess model for cell growth and product formation.

    This model describes the dynamics of biomass (X) and product (P) concentrations
    in a bioprocess, taking into account dilution effects from feed and base addition.
    """

    def __init__(self):
        """Initialize bioprocess model."""
        super().__init__(
            state_names=['X', 'P'],  # Biomass and product
            rtol=1e-3,
            atol=1e-6
        )

    def system_equations(self,
                         t: float,
                         y: np.ndarray,
                         parameters: Dict[str, Any],
                         inputs: Dict[str, Any]) -> np.ndarray:
        """
        Define the bioprocess ODE system.

        Args:
            t: Current time point
            y: Current state [X, P]
            parameters: Model parameters including growth rate (mu) and product formation rate (vpx)
            inputs: Additional inputs including control variables

        Returns:
            Derivatives [dX/dt, dP/dt]
        """
        X, P = y

        # Get parameters
        mu = parameters.get('mu', 0.0)  # Growth rate
        vpx = parameters.get('vpx', 0.0)  # Product formation rate

        # Get control inputs at current time
        controls_times = inputs.get('controls_times', np.array([0.0, 1.0]))

        current_inputs = {
            'X': X,
            'P': P,
            't': t
        }

        control_features = [
            'temp', 'feed', 'inductor_mass', 'inductor_switch',
            'base', 'reactor_volume'
        ]

        for feature in control_features:
            if feature in inputs:
                current_inputs[feature] = get_control_at_time(t, controls_times, inputs[feature])

        # Get current volume
        current_volume = current_inputs.get('reactor_volume', 1.0)

        # Calculate feed rate (volume change per hour)
        feed_rate = 0.0
        if 'feed' in inputs:
            feed_rate = calculate_derivative(t, controls_times, inputs['feed'])
            # Ensure non-negative feed rate (we can't remove feed)
            feed_rate = max(feed_rate, 0.0)

        # Calculate base rate (volume change per hour)
        base_rate = 0.0
        if 'base' in inputs:
            base_rate = calculate_derivative(t, controls_times, inputs['base'])
            # Ensure non-negative base rate (we can't remove base)
            base_rate = max(base_rate, 0.0)

        # Calculate total flow rate (L/h)
        total_flow_rate = feed_rate + base_rate

        # Calculate dilution rate (1/h)
        # Avoid division by zero
        dilution_rate = total_flow_rate / current_volume if current_volume > 1e-6 else 0.0

        # ODE system with dilution
        dXdt = mu * X - dilution_rate * X  # Biomass growth with dilution

        # Product formation with dilution
        # Note: inductor_switch can be 0 or 1, controlling whether product is formed
        inductor_switch = current_inputs.get('inductor_switch', 0.0)
        dPdt = vpx * X * inductor_switch - dilution_rate * P

        return np.array([dXdt, dPdt])

    def solve_for_run(self,
                      model: Any,
                      run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve ODE for a single run.

        Args:
            model: Hybrid model containing neural networks for parameters
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
        initial_conditions = {'X': X0, 'P': P0}

        # Set up inputs for hybrid model
        inputs = {
            'controls_times': controls_times,
            'temp': run_data['temp'],
            'feed': run_data['feed'],
            'inductor_mass': run_data['inductor_mass'],
            'inductor_switch': run_data['inductor_switch'],
            'base': run_data['base'],
            'reactor_volume': run_data['reactor_volume']
        }

        # If this is a hybrid model, it will use the parameter model
        if hasattr(model, 'solve'):
            solution = model.solve(
                initial_conditions=initial_conditions,
                time_points=states_times,
                inputs=inputs
            )
        else:
            # Otherwise, just use this mechanistic model directly
            # Assume parameters are provided from elsewhere
            parameters = {
                'mu': 0.1,  # Default values
                'vpx': 0.05
            }

            solution = self.solve(
                initial_conditions=initial_conditions,
                time_points=states_times,
                parameters=parameters,
                inputs=inputs
            )

        # Format the solution to include true values
        return {
            'times': states_times,
            'X_pred': solution['X'],
            'P_pred': solution['P'],
            'X_true': run_data['X'],
            'P_true': run_data['P']
        }


class BioprocessHybridModel:
    """
    Factory functions for bioprocess hybrid models.

    This class provides methods to create various hybrid bioprocess models
    by combining the BioprocessModel with different parameter estimation approaches.
    """

    @staticmethod
    def create_neural_hybrid(
            norm_params: Dict[str, Dict[str, float]],
            growth_inputs: List[str],
            product_inputs: List[str],
            growth_hidden_dims: List[int] = [16, 16],
            product_hidden_dims: Optional[List[int]] = None,
            key=None):
        """
        Create a bioprocess hybrid model with neural networks.

        Args:
            norm_params: Normalization parameters
            growth_inputs: List of input features for growth rate model
            product_inputs: List of input features for product formation model
            growth_hidden_dims: Hidden layer dimensions for growth network
            product_hidden_dims: Hidden layer dimensions for product network
            key: Random key for initialization

        Returns:
            A hybrid model for bioprocess modeling
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Use same hidden dimensions for product network if not specified
        if product_hidden_dims is None:
            product_hidden_dims = growth_hidden_dims

        # Create parameter model configuration
        parameter_configs = {
            'mu': {  # Growth rate
                'input_features': growth_inputs,
                'hidden_dims': growth_hidden_dims,
                'activation': 'relu',
                'output_activation': None  # Can be positive or negative
            },
            'vpx': {  # Product formation rate
                'input_features': product_inputs,
                'hidden_dims': product_hidden_dims,
                'activation': 'relu',
                'output_activation': 'softplus'  # Ensure non-negative
            }
        }

        # Create parameter model
        parameter_model = MLPParameterModel(
            parameter_configs=parameter_configs,
            normalization=norm_params,
            key=key
        )

        # Create mechanistic model
        mechanistic_model = BioprocessModel()

        # Create hybrid model
        hybrid_model = ODENeuralHybridModel(
            ode_model=mechanistic_model,
            parameter_model=parameter_model
        )

        return hybrid_model


def calculate_custom_loss(model: Any, runs: List[Dict[str, Any]], mechanistic_model: BioprocessModel) -> Tuple[
    float, Dict[str, float]]:
    """
    Calculate custom loss for bioprocess model.

    Args:
        model: Hybrid model
        runs: List of run data
        mechanistic_model: Bioprocess mechanistic model

    Returns:
        Tuple of (loss_value, loss_info)
    """
    total_loss = 0.0
    total_x_loss = 0.0
    total_p_loss = 0.0

    # Compute loss for each run
    for run_data in runs:
        # Solve ODE for this run
        sol = mechanistic_model.solve_for_run(model, run_data)

        # Extract predictions and true values
        X_pred = sol['X_pred']
        P_pred = sol['P_pred']
        X_true = sol['X_true']
        P_true = sol['P_true']

        # Compute MSE loss for X and P
        X_loss = np.mean(np.square(X_pred - X_true))
        P_loss = np.mean(np.square(P_pred - P_true))

        # Add to total loss
        run_loss = X_loss + P_loss
        total_loss += run_loss
        total_x_loss += X_loss
        total_p_loss += P_loss

    # Return average loss
    n_runs = len(runs)
    return total_loss / n_runs, {'x_loss': total_x_loss / n_runs, 'p_loss': total_p_loss / n_runs}