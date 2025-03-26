# Hybrid Modeling Framework

A truly general framework for hybrid mechanistic-neural network modeling that combines domain knowledge with data-driven approaches.

## Overview

This framework provides a flexible, modular, and extensible architecture for developing hybrid models that combine mechanistic knowledge with data-driven approaches. Key features include:

- **Domain-agnostic core**: Clean, abstract interfaces that work for any application domain
- **Flexible model structure**: Combine any mechanistic model with any parameter estimation approach
- **Comprehensive utilities**: Data loading, training, evaluation, and visualization
- **Example implementations**: Bioprocess modeling example with step-by-step instructions

## Installation

```bash
git clone https://github.com/yourusername/hybrid_modeling.git
cd hybrid_modeling
pip install -e .
```

## Core Concepts

The framework is built around these key abstractions:

1. **Mechanistic Models**: Domain-specific mathematical models (e.g., ODEs, PDEs, algebraic equations)
2. **Parameter Models**: Methods for estimating model parameters (e.g., neural networks, interpolation)
3. **Hybrid Models**: Combinations of mechanistic and parameter models
4. **Data Interfaces**: General-purpose data loading and handling

## Framework Structure

```
hybrid_modeling/
├── core/                  # Domain-agnostic core interfaces
│   ├── data.py            # Abstract data interfaces
│   ├── mechanistic.py     # Mechanistic model interfaces 
│   ├── parameters.py      # Parameter model interfaces
│   ├── hybrid.py          # Hybrid model interfaces
│   ├── training.py        # Generic training utilities
│   └── evaluation.py      # Generic evaluation utilities
├── models/                # Concrete model implementations
│   ├── mechanistic/       # Various mechanistic models
│   │   ├── ode.py         # Generic ODE models
│   │   └── bioprocess.py  # Bioprocess example
│   ├── parameters/        # Parameter estimation models
│   │   └── neural.py      # Neural network models
│   └── hybrid/            # Hybrid model implementations
│       └── ode_neural.py  # ODE + Neural Net hybrid
├── data/                  # Data utilities
│   └── loaders/           # Data loaders for different formats
│       ├── excel.py       # Excel data loader
│       └── csv.py         # CSV data loader
├── viz/                   # Visualization utilities
│   └── plots.py           # Plotting functions
└── examples/              # Example implementations
    └── bioprocess/        # Bioprocess example
        ├── data.py        # Data loading
        └── train.py       # Training script
```

## Quick Start

### A Simple ODE-Neural Network Hybrid Model

This example shows how to create a hybrid model that combines a simple ODE model with a neural network for parameter estimation:

```python
from hybrid_modeling.models.mechanistic.ode import ParameterizedODEModel
from hybrid_modeling.models.parameters.neural import MLPParameterModel
from hybrid_modeling.models.hybrid.ode_neural import ODENeuralHybridModel
import numpy as np
import jax

# 1. Define a simple ODE system (population growth)
def population_ode(t, y, parameters, inputs):
    # y[0] is population
    # parameters['growth_rate'] is the growth rate parameter
    return np.array([parameters['growth_rate'] * y[0]])

# 2. Create a mechanistic model
mechanistic_model = ParameterizedODEModel(
    state_names=['population'],
    equations_fn=population_ode
)

# 3. Define parameter model configuration
parameter_configs = {
    'growth_rate': {
        'input_features': ['temperature', 'nutrient_level'],
        'hidden_dims': [16, 16],
        'activation': 'relu',
        'output_activation': 'softplus'  # Ensure positive growth rate
    }
}

# 4. Create parameter model
key = jax.random.PRNGKey(0)
parameter_model = MLPParameterModel(
    parameter_configs=parameter_configs,
    key=key
)

# 5. Create hybrid model
hybrid_model = ODENeuralHybridModel(
    ode_model=mechanistic_model,
    parameter_model=parameter_model
)

# 6. Use the model to solve the ODE
solution = hybrid_model.solve(
    initial_conditions={'population': 100.0},
    time_points=np.linspace(0, 10, 100),
    inputs={
        'temperature': 25.0,
        'nutrient_level': 0.8
    }
)

# 7. Access the solution
population = solution['population']
```

### Using the Bioprocess Example

```python
from hybrid_modeling.examples.bioprocess.data import load_bioprocess_data
from hybrid_modeling.models.mechanistic.bioprocess import BioprocessHybridModel, BioprocessModel
from hybrid_modeling.core.training import Trainer, OptimizerConfig, TrainingConfig
from hybrid_modeling.core.evaluation import evaluate_runs
import jax

# 1. Load data
train_runs, test_runs = load_bioprocess_data(
    train_file='data/train_data.xlsx',
    test_file='data/test_data.xlsx'
)

# 2. Create model
key = jax.random.PRNGKey(0)
model = BioprocessHybridModel.create(
    norm_params=train_runs[0]['norm_params'],
    growth_inputs=['X', 'P', 'temp', 'feed'],
    product_inputs=['X', 'P', 'temp', 'feed', 'inductor_switch'],
    key=key
)

# 3. Configure training
optimizer_config = OptimizerConfig(
    learning_rate=1e-3,
    lr_decay=0.95,
    lr_decay_steps=100
)

training_config = TrainingConfig(
    num_epochs=1000,
    checkpoint_dir='checkpoints'
)

# 4. Create trainer
mechanistic_model = BioprocessModel()
trainer = Trainer(
    model=model,
    optimizer_config=optimizer_config,
    training_config=training_config,
    loss_fn=lambda model, runs: calculate_custom_loss(model, runs, mechanistic_model)
)

# 5. Train model
history, trained_model = trainer.train(train_runs)

# 6. Evaluate model
results = evaluate_runs(
    model=trained_model,
    runs=test_runs,
    solve_fn=mechanistic_model.solve_for_run
)
```

## Creating Your Own Models

### 1. Define a Mechanistic Model

```python
from hybrid_modeling.models.mechanistic.ode import DiffraxODEModel
import numpy as np

class MyODEModel(DiffraxODEModel):
    def __init__(self):
        super().__init__(
            state_names=['state1', 'state2'],
            rtol=1e-3,
            atol=1e-6
        )
    
    def system_equations(self, t, y, parameters, inputs):
        # Extract parameters and states
        param1 = parameters.get('param1', 0.0)
        param2 = parameters.get('param2', 0.0)
        
        state1, state2 = y
        
        # Define the ODE system
        dstate1_dt = param1 * state1 - param2 * state1 * state2
        dstate2_dt = param2 * state1 * state2 - param1 * state2
        
        return np.array([dstate1_dt, dstate2_dt])
```

### 2. Define a Parameter Model

```python
from hybrid_modeling.models.parameters.neural import MLPParameterModel

# Define parameter configurations
parameter_configs = {
    'param1': {
        'input_features': ['input1', 'input2'],
        'hidden_dims': [32, 32],
        'activation': 'relu'
    },
    'param2': {
        'input_features': ['input1', 'input2', 'input3'],
        'hidden_dims': [32, 32],
        'activation': 'relu',
        'output_activation': 'softplus'
    }
}

# Create parameter model
parameter_model = MLPParameterModel(
    parameter_configs=parameter_configs,
    key=jax.random.PRNGKey(0)
)
```

### 3. Create a Hybrid Model

```python
from hybrid_modeling.models.hybrid.ode_neural import ODENeuralHybridModel

# Create mechanistic model
mechanistic_model = MyODEModel()

# Create hybrid model
hybrid_model = ODENeuralHybridModel(
    ode_model=mechanistic_model,
    parameter_model=parameter_model
)
```

### 4. Load and Preprocess Data

```python
from hybrid_modeling.data.loaders.excel import GenericExcelLoader

# Define data loader
loader = GenericExcelLoader(
    time_col='Time',
    state_cols=['State1', 'State2'],
    control_cols=['Input1', 'Input2', 'Input3'],
    run_id_col='RunID'
)

# Load data
dataset = loader.load('data/experiments.xlsx')

# Get runs as dictionaries
runs = dataset.get_run_dicts()
```

### 5. Train the Model

```python
from hybrid_modeling.core.training import Trainer, OptimizerConfig, TrainingConfig

# Configure optimizer
optimizer_config = OptimizerConfig(
    learning_rate=1e-3,
    weight_decay=1e-4,
    lr_decay=0.95,
    lr_decay_steps=100,
    early_stopping_patience=50
)

# Configure training
training_config = TrainingConfig(
    num_epochs=1000,
    checkpoint_dir='checkpoints',
    checkpoint_freq=50
)

# Define custom loss function
def custom_loss_fn(model, runs):
    # Calculate loss for each run
    total_loss = 0.0
    auxiliary_info = {}
    
    for run in runs:
        # Make predictions using the model
        predictions = mechanistic_model.solve_for_run(model, run)
        
        # Calculate loss
        # ...
        
        # Add to total loss
        total_loss += run_loss
    
    return total_loss / len(runs), auxiliary_info

# Create trainer
trainer = Trainer(
    model=hybrid_model,
    optimizer_config=optimizer_config,
    training_config=training_config,
    loss_fn=custom_loss_fn
)

# Train model
history, trained_model = trainer.train(runs)
```

### 6. Evaluate and Visualize Results

```python
from hybrid_modeling.core.evaluation import evaluate_runs, print_metrics_summary
from hybrid_modeling.viz.plots import create_evaluation_plots

# Split data into training and testing sets
train_runs = runs[:len(runs)//2]
test_runs = runs[len(runs)//2:]

# Evaluate on test data
results = evaluate_runs(
    model=trained_model,
    runs=test_runs,
    solve_fn=mechanistic_model.solve_for_run
)

# Print summary
print_metrics_summary(results, output_names=['state1', 'state2'])

# Create plots
create_evaluation_plots(
    results=results,
    output_names=['state1', 'state2'],
    output_dir='results',
    show=True
)
```

## Extending the Framework

### Adding a New Mechanistic Model Type

To add a new type of mechanistic model (e.g., PDE, stochastic model), create a new subclass of `MechanisticModel` and implement the required methods:

```python
from hybrid_modeling.core.mechanistic import MechanisticModel

class PDEModel(MechanisticModel):
    def forward(self, inputs, parameters):
        # Implement PDE solver
        # ...
        return solution
```

### Adding a New Parameter Estimation Method

To add a new method for parameter estimation (e.g., Gaussian processes, random forests), create a new subclass of `ParameterModel`:

```python
from hybrid_modeling.core.parameters import ParameterModel

class GaussianProcessParameterModel(ParameterModel):
    def predict_parameters(self, inputs):
        # Implement GP inference
        # ...
        return parameters
    
    @property
    def parameter_names(self):
        # Return parameter names
        return ['param1', 'param2']
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.