"""
Visualization utilities for hybrid models.

This module provides general-purpose plotting functions for model training,
evaluation, and parameter analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union


def plot_training_history(history: Dict[str, List[float]],
                          output_path: Optional[str] = None,
                          show: bool = True,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot training history.

    Args:
        history: Dictionary containing training history
        output_path: Optional path to save the plot
        show: Whether to show the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Get available metrics
    metrics = [key for key in history.keys() if key not in ['epoch_time', 'time_elapsed']]

    # Total number of subplots
    n_plots = len(metrics)

    # Create subplots
    for i, metric in enumerate(metrics):
        plt.subplot(n_plots, 1, i + 1)
        plt.plot(history[metric], 'b-')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True)

    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.savefig(output_path)

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions(times: np.ndarray,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     title: str = 'Model Predictions',
                     xlabel: str = 'Time',
                     ylabel: str = 'Value',
                     metrics: Optional[Dict[str, float]] = None,
                     output_path: Optional[str] = None,
                     show: bool = True,
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot model predictions against true values.

    Args:
        times: Time points
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        metrics: Optional dictionary of metrics to display
        output_path: Optional path to save the plot
        show: Whether to show the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Plot true values
    plt.plot(times, y_true, 'bo-', label='Measured')

    # Plot predictions
    plt.plot(times, y_pred, 'r-', label='Predicted')

    # Add metrics to title if provided
    if metrics:
        title += ' - '
        for name, value in metrics.items():
            title += f'{name}: {value:.4f} '

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.savefig(output_path)

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_predictions(results: Dict[str, Dict[str, np.ndarray]],
                              output_names: List[str],
                              run_ids: Optional[List[str]] = None,
                              output_dir: Optional[str] = None,
                              show: bool = False,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot predictions for multiple runs and outputs.

    Args:
        results: Dictionary of evaluation results
        output_names: Names of outputs to plot
        run_ids: Optional list of run IDs to plot (if None, plot all)
        output_dir: Optional directory to save plots
        show: Whether to show plots
        figsize: Figure size
    """
    # Get run IDs if not provided
    if run_ids is None:
        run_ids = list(results['per_run'].keys())

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot each run and output
    for run_id in run_ids:
        if run_id not in results['per_run']:
            continue

        run_metrics = results['per_run'][run_id]
        solution = results['solutions'][run_id]

        for output in output_names:
            # Skip if output not available
            true_key = f"{output}_true"
            pred_key = f"{output}_pred"

            if true_key not in solution or pred_key not in solution:
                continue

            # Get data
            times = solution.get('times', np.arange(len(solution[true_key])))
            y_true = solution[true_key]
            y_pred = solution[pred_key]

            # Get metrics
            metrics = run_metrics.get(output, {})

            # Plot
            title = f'Run {run_id}: {output}'

            # Set output path if directory provided
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f'run_{run_id}_{output}.png')

            plot_predictions(
                times=times,
                y_true=y_true,
                y_pred=y_pred,
                title=title,
                xlabel='Time',
                ylabel=output,
                metrics=metrics,
                output_path=output_path,
                show=show,
                figsize=figsize
            )


def plot_residuals(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   title: str = 'Residuals',
                   output_path: Optional[str] = None,
                   show: bool = True,
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot residuals analysis.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        output_path: Optional path to save the plot
        show: Whether to show the plot
        figsize: Figure size
    """
    # Calculate residuals
    residuals = y_pred - y_true

    plt.figure(figsize=figsize)

    # Residuals vs. true values
    plt.subplot(2, 1, 1)
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'{title} vs. True Values')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.grid(True)

    # Residuals histogram
    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=30, alpha=0.6)
    plt.axvline(x=0, color='r', linestyle='-')
    plt.title('Residuals Histogram')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.savefig(output_path)

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_sensitivity(parameter_name: str,
                               parameter_values: np.ndarray,
                               outputs: Dict[str, np.ndarray],
                               title: Optional[str] = None,
                               xlabel: Optional[str] = None,
                               output_path: Optional[str] = None,
                               show: bool = True,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot sensitivity of outputs to a parameter.

    Args:
        parameter_name: Name of the parameter
        parameter_values: Array of parameter values
        outputs: Dictionary mapping output names to arrays of values
        title: Optional plot title
        xlabel: Optional x-axis label
        output_path: Optional path to save the plot
        show: Whether to show the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Plot each output
    for output_name, output_values in outputs.items():
        plt.plot(parameter_values, output_values, '-', label=output_name)

    # Set title and labels
    plt.title(title or f'Sensitivity to {parameter_name}')
    plt.xlabel(xlabel or parameter_name)
    plt.ylabel('Output')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.savefig(output_path)

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_sensitivity_2d(parameter1_name: str,
                                  parameter2_name: str,
                                  parameter1_values: np.ndarray,
                                  parameter2_values: np.ndarray,
                                  output_values: np.ndarray,
                                  output_name: str,
                                  title: Optional[str] = None,
                                  output_path: Optional[str] = None,
                                  show: bool = True,
                                  figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot 2D sensitivity of an output to two parameters.

    Args:
        parameter1_name: Name of the first parameter
        parameter2_name: Name of the second parameter
        parameter1_values: Array of first parameter values
        parameter2_values: Array of second parameter values
        output_values: 2D array of output values
        output_name: Name of the output
        title: Optional plot title
        output_path: Optional path to save the plot
        show: Whether to show the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Create mesh grid
    X, Y = np.meshgrid(parameter1_values, parameter2_values)

    # Create contour plot
    contour = plt.contourf(X, Y, output_values, 50, cmap='viridis')

    # Add colorbar
    plt.colorbar(label=output_name)

    # Set title and labels
    plt.title(title or f'Sensitivity of {output_name} to {parameter1_name} and {parameter2_name}')
    plt.xlabel(parameter1_name)
    plt.ylabel(parameter2_name)

    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.savefig(output_path)

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()


def create_evaluation_plots(results: Dict[str, Any],
                            output_names: List[str],
                            output_dir: str,
                            show: bool = False) -> None:
    """
    Create a comprehensive set of evaluation plots.

    Args:
        results: Dictionary of evaluation results
        output_names: Names of outputs to plot
        output_dir: Directory to save plots
        show: Whether to show plots
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create subdirectories
    predictions_dir = os.path.join(output_dir, 'predictions')
    residuals_dir = os.path.join(output_dir, 'residuals')

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    if not os.path.exists(residuals_dir):
        os.makedirs(residuals_dir)

    # Plot predictions for each run and output
    plot_multiple_predictions(
        results=results,
        output_names=output_names,
        output_dir=predictions_dir,
        show=show
    )

    # Plot residuals for each output
    for output in output_names:
        # Collect all true and predicted values
        all_true = []
        all_pred = []

        for run_id, run_solutions in results['solutions'].items():
            true_key = f"{output}_true"
            pred_key = f"{output}_pred"

            if true_key in run_solutions and pred_key in run_solutions:
                all_true.extend(run_solutions[true_key])
                all_pred.extend(run_solutions[pred_key])

        # Skip if no data
        if not all_true or not all_pred:
            continue

        # Convert to arrays
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        # Plot residuals
        output_path = os.path.join(residuals_dir, f'{output}_residuals.png')

        plot_residuals(
            y_true=all_true,
            y_pred=all_pred,
            title=f'{output} Residuals',
            output_path=output_path,
            show=show
        )