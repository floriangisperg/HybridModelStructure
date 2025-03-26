"""
Core training utilities for hybrid models.

This module provides general-purpose training functionality that can be used
with any type of hybrid model, regardless of the application domain.
"""

import os
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, List, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """Configuration for optimization settings."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay: float = 1.0  # 1.0 means no decay
    lr_decay_steps: int = 100
    early_stopping_patience: Optional[int] = None
    early_stopping_delta: float = 1e-4


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""
    num_epochs: int = 1000
    batch_size: int = 32
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 50
    checkpoint_threshold: Optional[float] = None
    verbose: bool = True
    log_freq: int = 10
    save_best: bool = True
    callbacks: List[Any] = field(default_factory=list)


class Trainer:
    """
    General-purpose trainer for hybrid models.

    This trainer can be used with any model that follows the required interface,
    regardless of the specific application domain.
    """

    def __init__(self,
                 model: Any,
                 optimizer_config: OptimizerConfig,
                 training_config: TrainingConfig,
                 loss_fn: Optional[Callable] = None):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            optimizer_config: Configuration for the optimizer
            training_config: Configuration for training
            loss_fn: Optional custom loss function. If None, the model must have a loss method.
        """
        self.model = model
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.custom_loss_fn = loss_fn

        # Create output directory if needed
        if self.training_config.checkpoint_dir:
            os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)

        # Set up optimizer
        self.optimizer = self._setup_optimizer()

        # Split model into trainable and static parts
        self.trainable_params, self.static_params = eqx.partition(model, eqx.is_array)

        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.trainable_params)

    def _setup_optimizer(self) -> optax.GradientTransformation:
        """Set up the optimizer based on configuration."""
        # Create learning rate schedule if requested
        if self.optimizer_config.lr_decay < 1.0:
            schedule = optax.exponential_decay(
                init_value=self.optimizer_config.learning_rate,
                transition_steps=self.optimizer_config.lr_decay_steps,
                decay_rate=self.optimizer_config.lr_decay
            )
        else:
            schedule = self.optimizer_config.learning_rate

        # Create optimizer
        return optax.adamw(
            learning_rate=schedule,
            weight_decay=self.optimizer_config.weight_decay
        )

    def compute_loss(self, model: Any, data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Compute loss for a model on given data.

        Args:
            model: The model
            data: The data

        Returns:
            Tuple of (loss_value, loss_info)
        """
        if self.custom_loss_fn:
            # Use custom loss function
            return self.custom_loss_fn(model, data)

        # Use model's loss method if available
        if hasattr(model, 'loss'):
            return model.loss(data)

        raise ValueError("No loss function provided. Either specify a custom "
                         "loss function or use a model with a loss method.")

    def train_step(self, params, static_params, opt_state, data):
        """
        Perform a single training step.

        Args:
            params: Trainable parameters
            static_params: Static parameters
            opt_state: Optimizer state
            data: Training data

        Returns:
            Tuple of (updated_params, updated_opt_state, loss_value, loss_info)
        """

        # Define loss function for gradient calculation
        def loss_fn(p):
            # Combine parameters
            model = eqx.combine(p, static_params)

            # Compute loss
            loss_value, loss_info = self.compute_loss(model, data)

            return loss_value, loss_info

        # Compute loss and gradients
        (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss_value, loss_info

    def train(self, data: Any) -> Tuple[Dict[str, List[float]], Any]:
        """
        Train the model.

        Args:
            data: Training data

        Returns:
            Tuple of (training_history, trained_model)
        """
        # Get initial parameters and optimizer state
        params = self.trainable_params
        opt_state = self.opt_state
        static_params = self.static_params

        # Initialize history
        history = {
            'total_loss': [],
            'epoch_time': [],
            'time_elapsed': []
        }

        # Initialize early stopping
        best_loss = float('inf')
        best_params = params
        best_epoch = -1
        patience_counter = 0

        # Training loop
        start_time = time.time()
        for epoch in range(self.training_config.num_epochs):
            epoch_start = time.time()

            # Perform training step
            params, opt_state, loss_value, loss_info = self.train_step(
                params, static_params, opt_state, data
            )

            # Record time
            epoch_time = time.time() - epoch_start
            time_elapsed = time.time() - start_time

            # Record history
            history['total_loss'].append(float(loss_value))
            history['epoch_time'].append(float(epoch_time))
            history['time_elapsed'].append(float(time_elapsed))

            # Record additional loss info
            for key, value in loss_info.items():
                if key not in history:
                    history[key] = []
                history[key].append(float(value))

            # Print progress
            if (self.training_config.verbose and
                    (epoch % self.training_config.log_freq == 0 or
                     epoch == self.training_config.num_epochs - 1)):
                log_str = f"Epoch {epoch}/{self.training_config.num_epochs}"
                log_str += f", Loss: {loss_value:.6f}"

                # Add additional loss info
                for key, value in loss_info.items():
                    log_str += f", {key}: {float(value):.6f}"

                log_str += f", Time: {epoch_time:.3f}s"
                print(log_str)

            # Save checkpoint if requested
            if (self.training_config.checkpoint_dir and
                    epoch % self.training_config.checkpoint_freq == 0):
                checkpoint_model = eqx.combine(params, static_params)
                checkpoint_path = os.path.join(
                    self.training_config.checkpoint_dir,
                    f"model_epoch_{epoch}.eqx"
                )
                eqx.tree_serialise_leaves(checkpoint_path, checkpoint_model)

                if self.training_config.verbose:
                    print(f"Checkpoint saved at epoch {epoch}")

            # Early stopping if enabled
            if self.optimizer_config.early_stopping_patience is not None:
                if loss_value < best_loss - self.optimizer_config.early_stopping_delta:
                    # Found a better model
                    improvement = best_loss - loss_value
                    best_loss = loss_value
                    best_params = jax.tree_util.tree_map(
                        lambda x: x.copy() if hasattr(x, 'copy') else x, params
                    )
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model if requested
                    if (self.training_config.save_best and
                            self.training_config.checkpoint_dir):
                        best_model = eqx.combine(best_params, static_params)
                        best_path = os.path.join(
                            self.training_config.checkpoint_dir,
                            "best_model.eqx"
                        )
                        eqx.tree_serialise_leaves(best_path, best_model)

                        if self.training_config.verbose:
                            print(f"New best model saved at epoch {epoch} "
                                  f"(loss: {loss_value:.6f}, "
                                  f"improvement: {improvement:.6f})")
                else:
                    # No improvement
                    patience_counter += 1

                # Check if we should stop
                if patience_counter >= self.optimizer_config.early_stopping_patience:
                    if self.training_config.verbose:
                        print(f"Early stopping triggered at epoch {epoch}")

                    # Use best parameters
                    params = best_params
                    break

        # Create final model
        if (self.optimizer_config.early_stopping_patience is not None and
                best_epoch >= 0):
            # Use best model from early stopping
            trained_model = eqx.combine(best_params, static_params)

            if self.training_config.verbose:
                print(f"Using best model from epoch {best_epoch} "
                      f"(loss: {best_loss:.6f})")
        else:
            # Use final model
            trained_model = eqx.combine(params, static_params)

        return history, trained_model


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the end of an epoch."""
        pass


class EarlyStopping(Callback):
    """Callback for early stopping."""

    def __init__(self,
                 monitor: str = 'total_loss',
                 min_delta: float = 0.0,
                 patience: int = 0,
                 verbose: bool = False,
                 mode: str = 'min'):
        """
        Initialize early stopping callback.

        Args:
            monitor: Quantity to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement to stop training
            verbose: Whether to print messages
            mode: 'min' or 'max'
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode

        # Initialize state
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        """Reset state at the beginning of training."""
        self.wait = 0
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Check if training should stop at the end of an epoch."""
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        if self.mode == 'min':
            # Check if improved
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
        else:
            # Check if improved
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1

        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"Early stopping triggered at epoch {epoch}")

            # Stop training
            logs['stop_training'] = True

    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Print message at the end of training if stopped early."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped at epoch {self.stopped_epoch}")