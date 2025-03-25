"""
Evaluation metrics and utilities for assessing model performance.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union


def calculate_metrics(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, float]:
    """
    Calculate error metrics for model evaluation.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary containing various error metrics
    """
    # Mean Squared Error
    mse = jnp.mean(jnp.square(y_pred - y_true))

    # Root Mean Squared Error
    rmse = jnp.sqrt(mse)

    # Mean Absolute Error
    mae = jnp.mean(jnp.abs(y_pred - y_true))

    # R² (coefficient of determination)
    ss_total = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
    ss_residual = jnp.sum(jnp.square(y_true - y_pred))

    # Handle edge case where ss_total is zero or very small
    r2 = jnp.where(ss_total > 1e-10,
                   1 - ss_residual / ss_total,
                   0.0)

    # Normalized RMSE as percentage of the mean
    mean_y_true = jnp.mean(jnp.abs(y_true))
    nrmse = jnp.where(mean_y_true > 1e-10,
                      rmse / mean_y_true * 100.0,
                      float('inf'))

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'nrmse': float(nrmse)
    }


def evaluate_runs(model: Any,
                  runs: List[Dict[str, Any]],
                  solve_fn: callable) -> Dict[str, Any]:
    """
    Evaluate model performance on multiple runs.

    Args:
        model: The model to evaluate
        runs: List of run data dictionaries
        solve_fn: Function to solve the ODE for a given run

    Returns:
        Dictionary containing overall and per-run metrics
    """
    # Initialize metrics storage
    evaluation = {
        'overall': {'X': {}, 'P': {}},
        'per_run': {}
    }

    # Collect all predictions for overall metrics
    all_x_true = []
    all_x_pred = []
    all_p_true = []
    all_p_pred = []

    # Process each run
    for run_data in runs:
        run_id = run_data['run_id']

        # Apply the model to this run
        sol = solve_fn(model, run_data)

        # Extract predictions and true values
        X_pred = sol['X_pred']
        P_pred = sol['P_pred']
        X_true = sol['X_true']
        P_true = sol['P_true']

        # Collect for overall metrics
        all_x_true.extend(X_true)
        all_x_pred.extend(X_pred)
        all_p_true.extend(P_true)
        all_p_pred.extend(P_pred)

        # Calculate run-specific metrics
        x_metrics = calculate_metrics(X_true, X_pred)
        p_metrics = calculate_metrics(P_true, P_pred)

        # Store metrics for this run
        evaluation['per_run'][run_id] = {
            'X': x_metrics,
            'P': p_metrics,
            'solution': sol
        }

    # Calculate overall metrics
    evaluation['overall']['X'] = calculate_metrics(jnp.array(all_x_true), jnp.array(all_x_pred))
    evaluation['overall']['P'] = calculate_metrics(jnp.array(all_p_true), jnp.array(all_p_pred))

    return evaluation


def print_metrics_summary(evaluation: Dict[str, Any]) -> None:
    """
    Print a summary of evaluation metrics.

    Args:
        evaluation: Evaluation metrics dictionary from evaluate_runs
    """
    print("\n=== OVERALL METRICS ===\n")

    # Biomass (X) metrics
    print("Biomass (X) Metrics:")
    print(f"  RMSE: {evaluation['overall']['X']['rmse']:.4f}")
    print(f"  MAE:  {evaluation['overall']['X']['mae']:.4f}")
    print(f"  R²:   {evaluation['overall']['X']['r2']:.4f}")
    print(f"  NRMSE: {evaluation['overall']['X']['nrmse']:.2f}%")

    # Product (P) metrics
    print("\nProduct (P) Metrics:")
    print(f"  RMSE: {evaluation['overall']['P']['rmse']:.4f}")
    print(f"  MAE:  {evaluation['overall']['P']['mae']:.4f}")
    print(f"  R²:   {evaluation['overall']['P']['r2']:.4f}")
    print(f"  NRMSE: {evaluation['overall']['P']['nrmse']:.2f}%")

    print("\n=== PER-RUN METRICS ===\n")

    # Print per-run metrics
    for run_id, metrics in evaluation['per_run'].items():
        print(f"Run {run_id}:")

        # X metrics
        print(f"  Biomass (X): R² = {metrics['X']['r2']:.4f}, RMSE = {metrics['X']['rmse']:.4f}")

        # P metrics
        print(f"  Product (P): R² = {metrics['P']['r2']:.4f}, RMSE = {metrics['P']['rmse']:.4f}")