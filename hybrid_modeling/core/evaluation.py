"""
Core evaluation utilities for hybrid models.

This module provides domain-agnostic evaluation functionality that can be used
with any type of hybrid model, regardless of the application domain.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional, Union


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean(np.square(y_pred - y_true))

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_true))

    # R² (coefficient of determination)
    ss_total = np.sum(np.square(y_true - np.mean(y_true)))
    ss_residual = np.sum(np.square(y_true - y_pred))

    # Handle edge case where ss_total is close to zero
    if ss_total < 1e-10:
        r2 = 0.0
    else:
        r2 = 1 - ss_residual / ss_total

    # Normalized Root Mean Squared Error (as percentage of mean)
    mean_y_true = np.mean(np.abs(y_true))
    if mean_y_true < 1e-10:
        nrmse = float('inf')
    else:
        nrmse = (rmse / mean_y_true) * 100.0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'nrmse': float(nrmse)
    }


def evaluate_runs(model: Any,
                  runs: List[Dict[str, Any]],
                  solve_fn: Callable) -> Dict[str, Any]:
    """
    Evaluate model performance on multiple runs.

    Args:
        model: Model to evaluate
        runs: List of run data dictionaries
        solve_fn: Function to solve the model for a run

    Returns:
        Dictionary of evaluation results
    """
    # Initialize results
    results = {
        'overall': {},
        'per_run': {},
        'solutions': {}
    }

    # Process each run
    all_outputs = {}

    for run_data in runs:
        run_id = run_data.get('run_id', 'unknown')

        # Solve the model for this run
        solution = solve_fn(model, run_data)

        # Store the solution
        results['solutions'][run_id] = solution

        # Initialize per-run metrics
        results['per_run'][run_id] = {}

        # Compute metrics for each output
        for key in solution:
            # Skip time points
            if key == 'times':
                continue

            # Check if we have true values
            true_key = f"{key}_true"
            pred_key = f"{key}_pred"

            if true_key in solution and pred_key in solution:
                y_true = solution[true_key]
                y_pred = solution[pred_key]

                # Calculate metrics
                metrics = calculate_metrics(y_true, y_pred)

                # Store metrics
                results['per_run'][run_id][key] = metrics

                # Collect for overall metrics
                if key not in all_outputs:
                    all_outputs[key] = {'true': [], 'pred': []}

                all_outputs[key]['true'].extend(y_true)
                all_outputs[key]['pred'].extend(y_pred)

    # Calculate overall metrics
    for key, values in all_outputs.items():
        y_true = np.array(values['true'])
        y_pred = np.array(values['pred'])

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)

        # Store metrics
        results['overall'][key] = metrics

    return results


def print_metrics_summary(results: Dict[str, Any],
                          output_names: Optional[List[str]] = None) -> None:
    """
    Print a summary of evaluation metrics.

    Args:
        results: Evaluation results from evaluate_runs
        output_names: Optional list of output names to show. If None, show all.
    """
    # Get output names
    if output_names is None:
        output_names = list(results['overall'].keys())

    # Print overall metrics
    print("\n=== OVERALL METRICS ===\n")

    for output in output_names:
        if output in results['overall']:
            metrics = results['overall'][output]

            print(f"{output} Metrics:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            print(f"  NRMSE: {metrics['nrmse']:.2f}%")
            print()

    # Print per-run metrics
    print("=== PER-RUN METRICS ===\n")

    for run_id, run_metrics in results['per_run'].items():
        print(f"Run {run_id}:")

        for output in output_names:
            if output in run_metrics:
                metrics = run_metrics[output]
                print(f"  {output}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")

        print()


class ModelEvaluator:
    """
    General-purpose evaluator for hybrid models.

    This evaluator can be used with any model that follows the required interface,
    regardless of the specific application domain.
    """

    def __init__(self,
                 model: Any,
                 solve_fn: Callable):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            solve_fn: Function to solve the model for a run
        """
        self.model = model
        self.solve_fn = solve_fn

    def evaluate(self,
                 data: Union[Dict[str, Any], List[Dict[str, Any]]],
                 output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the model on data.

        Args:
            data: Data to evaluate on (run or list of runs)
            output_names: Optional list of output names to evaluate

        Returns:
            Evaluation results
        """
        # Convert single run to list
        if not isinstance(data, list):
            data = [data]

        # Evaluate the model
        results = evaluate_runs(self.model, data, self.solve_fn)

        # Print summary if output names provided
        if output_names:
            print_metrics_summary(results, output_names)

        return results


def cross_validate(model_factory: Callable,
                   runs: List[Dict[str, Any]],
                   solve_fn: Callable,
                   n_splits: int = 5,
                   shuffle: bool = True,
                   random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        model_factory: Function to create a new model instance
        runs: List of run data dictionaries
        solve_fn: Function to solve the model for a run
        n_splits: Number of cross-validation splits
        shuffle: Whether to shuffle the runs before splitting
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary of cross-validation results
    """
    import numpy as np

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle runs if requested
    if shuffle:
        indices = np.random.permutation(len(runs))
        shuffled_runs = [runs[i] for i in indices]
    else:
        shuffled_runs = runs

    # Split runs into folds
    fold_size = len(shuffled_runs) // n_splits
    folds = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else len(shuffled_runs)
        folds.append(shuffled_runs[start:end])

    # Perform cross-validation
    cv_results = {
        'folds': [],
        'overall': {}
    }

    all_metrics = {}

    for i in range(n_splits):
        # Create train and test sets
        test_fold = folds[i]
        train_folds = [fold for j, fold in enumerate(folds) if j != i]
        train_runs = [run for fold in train_folds for run in fold]

        # Create and train a new model
        model = model_factory()

        # Evaluate on test fold
        fold_results = evaluate_runs(model, test_fold, solve_fn)

        # Store fold results
        cv_results['folds'].append(fold_results)

        # Collect metrics for averaging
        for output, metrics in fold_results['overall'].items():
            if output not in all_metrics:
                all_metrics[output] = {metric: [] for metric in metrics}

            for metric, value in metrics.items():
                all_metrics[output][metric].append(value)

    # Calculate overall metrics
    for output, metrics in all_metrics.items():
        cv_results['overall'][output] = {}

        for metric, values in metrics.items():
            # Calculate mean and standard deviation
            mean = np.mean(values)
            std = np.std(values)

            cv_results['overall'][output][metric] = {
                'mean': float(mean),
                'std': float(std)
            }

    return cv_results


def print_cv_summary(cv_results: Dict[str, Any],
                     output_names: Optional[List[str]] = None) -> None:
    """
    Print a summary of cross-validation results.

    Args:
        cv_results: Cross-validation results from cross_validate
        output_names: Optional list of output names to show. If None, show all.
    """
    # Get output names
    if output_names is None:
        output_names = list(cv_results['overall'].keys())

    # Print overall metrics
    print("\n=== CROSS-VALIDATION RESULTS ===\n")

    for output in output_names:
        if output in cv_results['overall']:
            print(f"{output} Metrics:")

            for metric, stats in cv_results['overall'][output].items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

            print()