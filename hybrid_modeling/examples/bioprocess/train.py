"""
Training script for bioprocess hybrid models.

This script demonstrates how to train and evaluate bioprocess models
using the hybrid modeling framework.
"""

import os
import argparse
import jax
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Core framework imports
from hybrid_modeling.core.training import Trainer, OptimizerConfig, TrainingConfig
from hybrid_modeling.core.evaluation import evaluate_runs, print_metrics_summary
from hybrid_modeling.viz.plots import create_evaluation_plots, plot_training_history

# Bioprocess-specific imports
from hybrid_modeling.examples.bioprocess.data import load_bioprocess_data
from hybrid_modeling.examples.bioprocess.models import (
    BioprocessModel, BioprocessHybridModel, calculate_custom_loss
)


def train_bioprocess_model(args):
    """
    Train a bioprocess hybrid model.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with results
    """
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.train_file}...")
    train_runs, test_runs = load_bioprocess_data(
        train_file=args.train_file,
        test_file=args.test_file,
        run_ids=args.run_ids.split(",") if args.run_ids else None,
        max_runs=args.max_runs
    )

    print(f"Loaded {len(train_runs)} training runs")
    if test_runs:
        print(f"Loaded {len(test_runs)} testing runs")

    # Get normalization parameters from first run
    norm_params = train_runs[0]['norm_params']

    # Parse input features
    growth_inputs = args.growth_inputs.split(",") if args.growth_inputs else ['X', 'P', 'temp', 'feed', 'inductor_mass']
    product_inputs = args.product_inputs.split(",") if args.product_inputs else ['X', 'P', 'temp', 'feed', 'inductor_mass', 'inductor_switch']

    # Parse hidden dimensions
    growth_hidden_dims = [int(dim) for dim in args.growth_hidden_dims.split(",")]
    product_hidden_dims = [int(dim) for dim in args.product_hidden_dims.split(",")] if args.product_hidden_dims else None

    print(f"Growth inputs: {growth_inputs}")
    print(f"Product inputs: {product_inputs}")
    print(f"Growth hidden dimensions: {growth_hidden_dims}")
    print(f"Product hidden dimensions: {product_hidden_dims or growth_hidden_dims}")

    # Create model
    print("Creating hybrid model...")
    model = BioprocessHybridModel.create_neural_hybrid(
        norm_params=norm_params,
        growth_inputs=growth_inputs,
        product_inputs=product_inputs,
        growth_hidden_dims=growth_hidden_dims,
        product_hidden_dims=product_hidden_dims,
        key=key
    )

    # Create mechanistic model for ODE solving
    mechanistic_model = BioprocessModel()

    # Configure optimizer
    optimizer_config = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        lr_decay_steps=args.lr_decay_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta
    )

    # Configure training
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=len(train_runs),  # Train on all runs at once
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        checkpoint_freq=args.checkpoint_freq,
        verbose=True
    )

    # Define custom loss function
    def custom_loss_fn(model, runs):
        """Wrapper for the custom loss function."""
        return calculate_custom_loss(model, runs, mechanistic_model)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer_config=optimizer_config,
        training_config=training_config,
        loss_fn=custom_loss_fn
    )

    # Train model
    print(f"Training model for {args.num_epochs} epochs...")
    history, trained_model = trainer.train(train_runs)

    # Create plotting directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot training history
    plot_training_history(
        history=history,
        output_path=os.path.join(plots_dir, "training_history.png"),
        show=False
    )

    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_evaluation = evaluate_runs(
        model=trained_model,
        runs=train_runs,
        solve_fn=mechanistic_model.solve_for_run
    )

    print("\nTraining evaluation:")
    print_metrics_summary(train_evaluation)

    # Create evaluation plots for training data
    train_eval_dir = os.path.join(plots_dir, "train_evaluation")
    os.makedirs(train_eval_dir, exist_ok=True)

    create_evaluation_plots(
        results=train_evaluation,
        output_names=['X', 'P'],
        output_dir=train_eval_dir,
        show=False
    )

    # Evaluate on test data if available
    test_evaluation = None
    if test_runs:
        print("\nEvaluating on test data...")
        test_evaluation = evaluate_runs(
            model=trained_model,
            runs=test_runs,
            solve_fn=mechanistic_model.solve_for_run
        )

        print("\nTest evaluation:")
        print_metrics_summary(test_evaluation)

        # Create evaluation plots for test data
        test_eval_dir = os.path.join(plots_dir, "test_evaluation")
        os.makedirs(test_eval_dir, exist_ok=True)

        create_evaluation_plots(
            results=test_evaluation,
            output_names=['X', 'P'],
            output_dir=test_eval_dir,
            show=False
        )

    print(f"\nTraining and evaluation complete. Results saved to {args.output_dir}")

    return {
        'model': trained_model,
        'train_runs': train_runs,
        'test_runs': test_runs,
        'train_evaluation': train_evaluation,
        'test_evaluation': test_evaluation,
        'history': history,
        'output_dir': args.output_dir
    }


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bioprocess Hybrid Model Training")

    # Data arguments
    parser.add_argument("--train-file", type=str, required=True,
                      help="Path to training data file")
    parser.add_argument("--test-file", type=str, default=None,
                      help="Path to test data file")
    parser.add_argument("--run-ids", type=str, default=None,
                      help="Comma-separated list of run IDs to use")
    parser.add_argument("--max-runs", type=int, default=5,
                      help="Maximum number of runs to use if run-ids not specified")

    # Model arguments
    parser.add_argument("--growth-inputs", type=str, default="X,P,temp,feed,inductor_mass",
                      help="Comma-separated list of inputs for growth rate network")
    parser.add_argument("--product-inputs", type=str,
                      default="X,P,temp,feed,inductor_mass,inductor_switch",
                      help="Comma-separated list of inputs for product formation network")
    parser.add_argument("--growth-hidden-dims", type=str, default="32,32",
                      help="Comma-separated list of hidden layer dimensions for growth network")
    parser.add_argument("--product-hidden-dims", type=str, default=None,
                      help="Comma-separated list of hidden layer dimensions for product network")

    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=1000,
                      help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                      help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                      help="Weight decay factor")
    parser.add_argument("--lr-decay", type=float, default=0.95,
                      help="Learning rate decay factor")
    parser.add_argument("--lr-decay-steps", type=int, default=100,
                      help="Steps between learning rate decay")
    parser.add_argument("--early-stopping-patience", type=int, default=100,
                      help="Number of epochs to wait without improvement")
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4,
                      help="Minimum change to qualify as improvement")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                      help="Frequency of checkpoints (in epochs)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./results",
                      help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    results = train_bioprocess_model(args)