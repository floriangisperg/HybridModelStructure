"""
Training utilities for hybrid models.
"""

import os
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, List, Any, Callable, Tuple, Optional, Union
from functools import partial

from hybrid_modeling.core.config import OptimizerConfig, TrainingConfig
from hybrid_modeling.core.utils import save_model


class Trainer:
    """Generic trainer for hybrid models"""

    def __init__(self,
                 model: Any,
                 optimizer_config: OptimizerConfig,
                 training_config: TrainingConfig,
                 solve_ode_fn: Callable):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            optimizer_config: Configuration for the optimizer
            training_config: Configuration for training
            solve_ode_fn: Function to solve the ODE for a given run
        """
        self.model = model
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.solve_ode_fn = solve_ode_fn

        # Create output directory if needed
        if self.training_config.checkpoint_dir:
            os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)

        # Split model into trainable and static parts
        self.model_trainable, self.model_static = eqx.partition(model, eqx.is_array)

        # Set up optimizer
        self.optimizer = self._setup_optimizer()
        self.opt_state = self.optimizer.init(self.model_trainable)

        # Set up loss function
        self._setup_loss_fn()

    def _setup_optimizer(self) -> optax.GradientTransformation:
        """Set up the optimizer with learning rate schedule if needed."""
        if self.optimizer_config.lr_decay < 1.0:
            # Create an exponential decay schedule
            schedule_fn = optax.exponential_decay(
                init_value=self.optimizer_config.learning_rate,
                transition_steps=self.optimizer_config.lr_decay_steps,
                decay_rate=self.optimizer_config.lr_decay
            )
            return optax.adamw(schedule_fn, weight_decay=self.optimizer_config.weight_decay)
        else:
            return optax.adamw(
                self.optimizer_config.learning_rate,
                weight_decay=self.optimizer_config.weight_decay
            )

    def _setup_loss_fn(self) -> None:
        """Set up the loss function for training."""

        @partial(jax.jit, static_argnums=(1,))
        def loss_fn(trainable_params, static_params, runs):
            # Reconstitute the model
            full_model = eqx.combine(trainable_params, static_params)

            # Initialize total loss
            total_loss = 0.0
            total_x_loss = 0.0
            total_p_loss = 0.0

            # Compute loss for each run
            for run_data in runs:
                # Solve ODE for this run
                sol = self.solve_ode_fn(full_model, run_data)

                # Extract predictions and true values
                X_pred = sol['X_pred']
                P_pred = sol['P_pred']
                X_true = sol['X_true']
                P_true = sol['P_true']

                # Compute MSE loss for X and P with weights
                X_loss = jnp.mean(jnp.square(X_pred - X_true)) * self.optimizer_config.x_weight
                P_loss = jnp.mean(jnp.square(P_pred - P_true)) * self.optimizer_config.p_weight

                # Add to total loss
                run_loss = X_loss + P_loss
                total_loss += run_loss
                total_x_loss += X_loss
                total_p_loss += P_loss

            # Return average loss
            n_runs = len(runs)
            return total_loss / n_runs, (total_x_loss / n_runs, total_p_loss / n_runs)

        self.loss_fn = loss_fn

        # Define the update step
        @partial(jax.jit, static_argnums=(2,))
        def update_step(params, opt_state, model_static, runs):
            (loss_value, aux), grads = jax.value_and_grad(
                lambda p: loss_fn(p, model_static, runs), has_aux=True
            )(params)

            x_loss, p_loss = aux

            updates, opt_state_new = self.optimizer.update(grads, opt_state)
            params_new = optax.apply_updates(params, updates)

            return params_new, opt_state_new, loss_value, x_loss, p_loss

        self.update_step = update_step

    def train(self, runs: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Train the model on the provided runs.

        Args:
            runs: List of run data dictionaries

        Returns:
            Dictionary containing training history
        """
        # Setup for training
        params = self.model_trainable
        opt_state = self.opt_state

        # Setup for early stopping and model checkpointing
        best_loss = float('inf')
        best_params = params
        patience_counter = 0
        best_epoch = -1  # Track when we last found the best model

        # Training history
        history = {
            'total_loss': [],
            'x_loss': [],
            'p_loss': [],
            'learning_rates': [],
            'time_elapsed': []
        }

        start_time = time.time()

        # Training loop
        for epoch in range(self.training_config.num_epochs):
            epoch_start = time.time()

            # Update parameters
            params, opt_state, loss_value, x_loss, p_loss = self.update_step(
                params, opt_state, self.model_static, runs
            )

            # Record elapsed time
            elapsed = time.time() - start_time
            epoch_time = time.time() - epoch_start
            history['time_elapsed'].append(elapsed)

            # Record loss history
            history['total_loss'].append(float(loss_value))
            history['x_loss'].append(float(x_loss))
            history['p_loss'].append(float(p_loss))

            # Record learning rate if using a schedule
            if self.optimizer_config.lr_decay < 1.0:
                # Extract current learning rate
                if hasattr(self.optimizer, 'learning_rate'):
                    current_lr = self.optimizer.learning_rate
                else:
                    # Approximate for exponential decay
                    current_lr = self.optimizer_config.learning_rate * (
                            self.optimizer_config.lr_decay ** (epoch / self.optimizer_config.lr_decay_steps)
                    )
                history['learning_rates'].append(float(current_lr))
            else:
                history['learning_rates'].append(float(self.optimizer_config.learning_rate))

            # Print progress
            if self.training_config.verbose and (epoch % 20 == 0 or epoch == self.training_config.num_epochs - 1):
                print(
                    f"Epoch {epoch}/{self.training_config.num_epochs}, "
                    f"Loss: {loss_value:.4f} (X: {x_loss:.4f}, P: {p_loss:.4f}), "
                    f"Time: {elapsed:.2f}s, "
                    f"Epoch time: {epoch_time:.3f}s"
                )

            # Check if loss is below threshold for saving checkpoints
            below_threshold = (self.training_config.checkpoint_threshold is None or
                               loss_value < self.training_config.checkpoint_threshold)

            # Save regular checkpoint if specified (at specified frequency) and below threshold
            if (self.training_config.checkpoint_dir is not None and
                    below_threshold and
                    (
                            epoch % self.training_config.checkpoint_freq == 0 or epoch == self.training_config.num_epochs - 1)):
                checkpoint_model = eqx.combine(params, self.model_static)
                checkpoint_path = os.path.join(
                    self.training_config.checkpoint_dir,
                    f"model_epoch_{epoch}.eqx"
                )
                save_model(checkpoint_model, checkpoint_path)
                if self.training_config.verbose:
                    print(f"Checkpoint saved at epoch {epoch} (loss: {loss_value:.4f})")

            # Early stopping check
            if self.training_config.early_stopping_patience is not None:
                if loss_value < best_loss - self.training_config.early_stopping_delta:
                    # We found a better model
                    improvement = best_loss - loss_value
                    best_loss = loss_value
                    # Create a copy of the parameters
                    best_params = jax.tree_util.tree_map(
                        lambda x: x.copy() if hasattr(x, 'copy') else x,
                        params
                    )
                    patience_counter = 0

                    # Save best model checkpoint if using checkpointing and below threshold
                    if self.training_config.checkpoint_dir is not None and below_threshold:
                        # Only save and print if this is a significant improvement or first best model
                        if epoch - best_epoch >= 10 or best_epoch == -1 or improvement > 0.01 * best_loss:
                            best_model = eqx.combine(best_params, self.model_static)
                            best_path = os.path.join(self.training_config.checkpoint_dir, "best_model.eqx")
                            save_model(best_model, best_path)
                            if self.training_config.verbose:
                                print(
                                    f"New best model saved at epoch {epoch} "
                                    f"(loss: {loss_value:.4f}, improvement: {improvement:.4f})"
                                )
                            best_epoch = epoch
                else:
                    # No improvement
                    patience_counter += 1

                # Check if we should stop early
                if patience_counter >= self.training_config.early_stopping_patience:
                    if self.training_config.verbose:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                    # Use the best parameters
                    params = best_params
                    break

        # Reconstitute the model with the best parameters
        # (or final parameters if early stopping not used)
        if self.training_config.early_stopping_patience is not None:
            self.trained_model = eqx.combine(best_params, self.model_static)
            if self.training_config.verbose:
                print(f"Using best model from epoch {best_epoch} (loss: {best_loss:.4f})")
        else:
            self.trained_model = eqx.combine(params, self.model_static)

        return history


class TrainingCallback:
    """Base class for training callbacks"""

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch"""
        pass

    def on_training_end(self, logs: Dict[str, Any]) -> None:
        """Called at the end of training"""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Callback for early stopping"""

    def __init__(self, patience: int = 50, min_delta: float = 1e-4, monitor: str = 'total_loss'):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = -1

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """
        Check if training should be stopped.

        Args:
            epoch: Current epoch
            logs: Dict with training metrics

        Returns:
            True if training should stop, False otherwise
        """
        current = logs.get(self.monitor)
        if current is None:
            return False

        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.wait = 0
            self.best_epoch = epoch
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
            return False

    def on_training_end(self, logs: Dict[str, Any]) -> None:
        """Print message if stopped early"""
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")
            print(f"Best model was at epoch {self.best_epoch} with {self.monitor} = {self.best_value:.6f}")