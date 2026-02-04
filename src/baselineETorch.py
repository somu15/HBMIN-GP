"""
Baseline (unirradiated) elastic modulus model calibration script for graphite materials.

This script fits deterministic baseline elastic modulus models to experimental data
using PyTorch optimization. The baseline model captures temperature-dependent
Young's modulus behavior for unirradiated graphite.

Usage:
    python baselineETorch.py

The script will:
    1. Load elastic modulus baseline data from the specified graphite grade
    2. Fit a deterministic model using gradient descent
    3. Compute prediction statistics and error metrics
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import pyro.distributions as dist
from pyro.infer import Predictive
from torch.distributions.normal import Normal
import arviz as az

# Set project root and add to path
PROJECT_ROOT = Path("graphiteModels")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from codes.modelsHierarchical import BaselineElasticModulus
from codes.optimize import optimize
from codes.processData import (
    processDataIG110,
    processDataNBG18,
    processDataPCEA,
    processDataNBG17,
    processDataS2114
)

# Configure matplotlib for publication-quality plots
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams['axes.linewidth'] = 1.5
plt.rc('font', family='serif', size=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)

# Configuration constants
GRAPHITE_GRADE = "S2114"
ACTIVE_DIMS = [2]  # Temperature dimension
LEARNING_RATE = 0.001
NUM_EPOCHS = 99907
RESULTS_DIR = PROJECT_ROOT / "Results"


def load_elastic_modulus_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load baseline elastic modulus data for the specified graphite grade.

    Returns:
        Tuple of (train_x, train_y) where:
            - train_x: Input features (n_samples, n_features)
            - train_y: Target elastic modulus values (n_samples,)
    """
    data_processor = processDataS2114()
    train_x, train_y = data_processor.EBaselineData()
    return train_x, train_y


def fit_deterministic_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    active_dims: list,
    learning_rate: float = 0.001,
    epochs: int = 10000
) -> Tuple[torch.Tensor, list]:
    """
    Fit deterministic baseline elastic modulus model using gradient descent.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        active_dims: Active dimensions for GP kernel.
        learning_rate: Learning rate for optimization.
        epochs: Number of optimization epochs.

    Returns:
        Tuple of (minimized_params, param_history) where:
            - minimized_params: Optimized model parameters
            - param_history: History of parameters during optimization
    """
    # Initialize model
    dist1 = []
    model = BaselineElasticModulus(
        dist1, train_x, train_y,
        active_dims=active_dims,
        group_code=[]
    )

    # Set starting point for optimization
    starting_point = torch.tensor([0.01, 0.01, 0.01])

    # Optimize parameters
    optimizer = optimize(
        train_x=train_x,
        train_y=train_y,
        model_predict=model.determModel
    )
    minimized_params, list_of_params = optimizer.train(
        initial_parameters=starting_point,
        learning_rate=learning_rate,
        epochs=epochs
    )

    return minimized_params, list_of_params


def compute_prediction_statistics(
    model: BaselineElasticModulus,
    params: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute predictions and error statistics.

    Args:
        model: Trained baseline elastic modulus model.
        params: Model parameters.
        train_x: Training input features.
        train_y: Training target values.

    Returns:
        Tuple of (predictions, errors, frobenius_norm, std_error)
    """
    predictions = model.determModel(params, train_x)
    errors = predictions - train_y
    frobenius_norm = torch.norm(errors, p='fro')
    std_error = errors.std()

    return predictions, errors, frobenius_norm, std_error


def main():
    """Main execution function for baseline elastic modulus model calibration."""
    print("=" * 60)
    print("Baseline Elastic Modulus Model Calibration")
    print("=" * 60)
    print(f"Graphite Grade: {GRAPHITE_GRADE}")
    print(f"Active Dimensions: {ACTIVE_DIMS}")
    print()

    # Load data
    print("Loading elastic modulus baseline data...")
    train_x, train_y = load_elastic_modulus_data()
    print(f"Loaded {train_x.shape[0]} training samples")
    print()

    # Fit deterministic model
    print("Fitting deterministic model...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    minimized_params, list_of_params = fit_deterministic_model(
        train_x, train_y, ACTIVE_DIMS, LEARNING_RATE, NUM_EPOCHS
    )
    print("Optimization complete")
    print(f"Optimized parameters: {minimized_params}")
    print()

    # Compute statistics
    model = BaselineElasticModulus(
        [], train_x, train_y,
        active_dims=ACTIVE_DIMS,
        group_code=[]
    )
    predictions, errors, frobenius_norm, std_error = compute_prediction_statistics(
        model, minimized_params, train_x, train_y
    )
    print("Prediction Statistics:")
    print(f"  Frobenius norm of errors: {frobenius_norm.item():.6f}")
    print(f"  Standard deviation of errors: {std_error.item():.6f}")
    print()

    print("Calibration complete!")


if __name__ == "__main__":
    main()
