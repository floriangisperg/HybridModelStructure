"""
Configuration module for hybrid models.

This module provides classes to manage configuration settings for models,
training, and evaluation in a consistent way.
"""

import dataclasses
from typing import List, Dict, Any, Optional, Union


@dataclasses.dataclass
class ModelConfig:
    """Base configuration for model architecture and parameters"""
    name: str
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [16, 16])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


@dataclasses.dataclass
class NeuralNetConfig(ModelConfig):
    """Configuration for neural network models"""
    input_features: List[str] = dataclasses.field(default_factory=list)
    output_dim: int = 1
    activation: str = "relu"
    output_activation: Optional[str] = None


@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for optimization settings"""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay: float = 1.0  # 1.0 means no decay
    lr_decay_steps: int = 100
    x_weight: float = 1.0  # Weight for biomass loss component
    p_weight: float = 1.0  # Weight for product loss component


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training procedure"""
    num_epochs: int = 1000
    batch_size: int = 32
    early_stopping_patience: Optional[int] = None
    early_stopping_delta: float = 1e-4
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 50
    checkpoint_threshold: Optional[float] = None
    verbose: bool = True