"""
Elastic modulus change model calibration with Kennedy-O'Hagan framework.

This script performs comprehensive calibration of elastic modulus change models
for irradiated graphite using three methodologies: deterministic optimization,
Bayesian inference, and Kennedy-O'Hagan (KOH) calibration with Gaussian process
discrepancy.

The workflow includes:
    1. Data loading and preprocessing
    2. Deterministic model fitting via gradient descent
    3. Structural connectivity parameter calibration
    4. Bayesian inference using NUTS sampling
    5. KOH calibration with GP discrepancy quantification
    6. Model diagnostics and visualization
    7. Export of calibrated models for prediction

Usage:
    python EChangeTorch.py

Configuration can be modified via constants at the top of this file.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import seaborn as sns
import pyro.distributions as dist
from pyro.infer import Predictive
import arviz as az
import gpytorch

# Set project root and add to path
PROJECT_ROOT = Path("/Users/dhulls/projects/NEAMS/FY_25/Structural/graphiteModels")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from codes.modelsHierarchical import (
    INLElasticModulusChange,
    IrradiationElasticModulusChange,
    IIDCDeltaL
)
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

# =============================================================================
# Configuration Constants
# =============================================================================

GRAPHITE_GRADE = "IG110"
FOLDER_NAME = "EChange"
MODEL_NAME = "Bradford"
ACTIVE_DIMS = [1]  # Dose dimension

# Optimization parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 100000

# MCMC parameters
MCMC_WARMUP = 1500
MCMC_SAMPLES = 3000

# Results directory
RESULTS_DIR = PROJECT_ROOT / f"Results/AllProps{GRAPHITE_GRADE}"

# Temperature for structural connectivity calibration (K)
SC_CALIBRATION_TEMPERATURE = 700.0
SC_TEMPERATURE_TOLERANCE = 100.0


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_elastic_modulus_change_data(
    grade: str = GRAPHITE_GRADE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load elastic modulus change data for the specified graphite grade.

    Args:
        grade: Graphite grade identifier.

    Returns:
        Tuple of (train_x, train_y, group_code).
    """
    if grade == "IG110":
        data = processDataIG110()
    elif grade == "NBG18":
        data = processDataNBG18()
    elif grade == "PCEA":
        data = processDataPCEA()
    elif grade == "NBG17":
        data = processDataNBG17()
    elif grade == "S2114":
        data = processDataS2114()
    else:
        raise ValueError(f"Unknown graphite grade: {grade}")

    train_x, train_y, group_code = data.EChangeData()
    return train_x, train_y, group_code


# =============================================================================
# Model Fitting Functions
# =============================================================================

def fit_deterministic_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    active_dims: list,
    model_class,
    starting_point: torch.Tensor,
    learning_rate: float = 0.001,
    epochs: int = 10000
) -> Tuple[torch.Tensor, list]:
    """
    Fit deterministic elastic modulus change model using gradient descent.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        group_code: Group identifiers for hierarchical structure.
        active_dims: Active dimensions for GP kernel.
        model_class: Model class to use.
        starting_point: Initial parameter values.
        learning_rate: Learning rate for optimization.
        epochs: Number of optimization epochs.

    Returns:
        Tuple of (optimized_params, param_history).
    """
    dist1 = []
    model = model_class(dist1, train_x, train_y, active_dims=active_dims, group_code=group_code)

    optimizer = optimize(train_x=train_x, train_y=train_y, model_predict=model.determModel)
    minimized_params, list_of_params = optimizer.train(
        initial_parameters=starting_point,
        learning_rate=learning_rate,
        epochs=epochs
    )

    # Compute predictions and error
    prediction = model.determModel(minimized_params, train_x)
    error = prediction - train_y
    error_norm = torch.norm(error, p='fro')

    print(f"Deterministic model error (Frobenius norm): {error_norm.item():.6f}")

    return minimized_params, list_of_params


def fit_structural_connectivity_parameters(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    active_dims: list,
    model_class,
    temperature: float,
    temperature_tolerance: float,
    starting_point: torch.Tensor,
    learning_rate: float = 0.001,
    epochs: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit structural connectivity parameters using temperature-filtered data.

    This calibrates the structural connectivity model parameters by focusing
    on a specific temperature range where the effects are most pronounced.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        group_code: Group identifiers.
        active_dims: Active dimensions for GP kernel.
        model_class: Model class to use.
        temperature: Target temperature for calibration (K).
        temperature_tolerance: Temperature tolerance window (K).
        starting_point: Initial parameter values.
        learning_rate: Learning rate for optimization.
        epochs: Number of optimization epochs.

    Returns:
        Tuple of (sc_params, temperature_params) for structural connectivity.
    """
    # Filter data by temperature
    mask = (train_x[:, 2] - temperature >= -temperature_tolerance) & \
           (train_x[:, 2] - temperature <= temperature_tolerance)
    train_x_filtered = train_x[mask, :]
    train_y_filtered = train_y[mask]
    group_code_filtered = group_code[mask]

    print(f"Calibrating SC parameters using {train_x_filtered.shape[0]} samples "
          f"near {temperature} K")

    # Fit model on filtered data
    dist1 = []
    model = model_class(
        dist1, train_x_filtered, train_y_filtered,
        active_dims=active_dims, group_code=group_code_filtered
    )

    optimizer = optimize(
        train_x=train_x_filtered,
        train_y=train_y_filtered,
        model_predict=model.determModel
    )
    minimized_params, _ = optimizer.train(
        initial_parameters=starting_point,
        learning_rate=learning_rate,
        epochs=epochs
    )

    # Compute predictions and error
    prediction = model.determModel(minimized_params, train_x_filtered)
    error = prediction - train_y_filtered
    error_norm = torch.norm(error, p='fro')

    print(f"SC calibration error (Frobenius norm): {error_norm.item():.6f}")

    # Extract SC parameters (typically last 2-3 parameters)
    # For Bradford model: params = [a, b, c_sc, d_sc]
    sc_params = minimized_params[2:4]  # Structural connectivity parameters

    return sc_params, minimized_params


def fit_temperature_dependent_parameters(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    temperatures: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit temperature-dependent location and scale parameters for SC.

    This function calibrates how structural connectivity parameters vary
    with temperature using empirical fits to multi-temperature data.

    Args:
        train_x: Training input features.
        train_y: Training target values (log-space parameters).
        temperatures: Temperature values for calibration.

    Returns:
        Tuple of (location_params, scale_params) for temperature dependence.
    """
    # Fit location parameter as function of temperature
    def log_loc_pred(params: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return params[0] * torch.exp(-params[1] * T)

    starting_point = torch.tensor([1.0, 0.001])
    opt_loc = optimize(train_x=temperatures, train_y=train_y, model_predict=log_loc_pred)
    log_loc_params, _ = opt_loc.train(
        initial_parameters=starting_point,
        learning_rate=0.001,
        epochs=100000
    )

    print(f"Location parameters: {log_loc_params}")

    return log_loc_params, None  # Simplified for this example


def construct_prior_distributions(
    minimized_params: torch.Tensor,
    error_std: torch.Tensor,
    model_name: str
) -> list:
    """
    Construct prior distributions for Bayesian inference.

    Args:
        minimized_params: Optimized deterministic parameters.
        error_std: Standard deviation of prediction errors.
        model_name: Model identifier (Bradford, INL, etc.).

    Returns:
        List of prior distributions.
    """
    if model_name == "Bradford":
        # Bradford priors (4 parameters + noise)
        priors = [
            dist.Normal(minimized_params[0].item(), 0.0511),
            dist.Normal(minimized_params[1].item(), 0.0504),
            dist.Normal(minimized_params[2].item(), 0.0429),
            dist.Normal(minimized_params[3].item(), 0.3225),
            dist.Normal(error_std.log(), 1.0)
        ]
    elif model_name == "INL":
        # INL priors (5 parameters + noise)
        priors = [
            dist.Normal(minimized_params[0].item(), 0.0677),
            dist.Normal(minimized_params[1].item(), 0.0162),
            dist.Normal(minimized_params[2].item(), 0.0193),
            dist.Normal(minimized_params[3].item(), 0.0674),
            dist.Normal(minimized_params[4].item(), 0.1692),
            dist.Normal(error_std.log(), 1.0)
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return priors


def fit_bayesian_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    active_dims: list,
    model_class,
    priors: list,
    warmup: int = 1500,
    samples: int = 3000
) -> Tuple[object, dict]:
    """
    Fit Bayesian elastic modulus change model using NUTS sampling.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        group_code: Group identifiers.
        active_dims: Active dimensions for GP kernel.
        model_class: Model class to use.
        priors: List of prior distributions.
        warmup: Number of MCMC warmup iterations.
        samples: Number of MCMC samples.

    Returns:
        Tuple of (model, mcmc_samples).
    """
    model = model_class(priors, train_x, train_y, active_dims=active_dims, group_code=group_code)
    prob_model = model.probModel
    mcmc = model.MCMCSamples(prob_model, warmup, samples, True)

    return model, mcmc


def fit_koh_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    active_dims: list,
    model_class,
    priors: list,
    warmup: int = 1500,
    samples: int = 3000
) -> Tuple[object, dict]:
    """
    Fit Kennedy-O'Hagan model with GP discrepancy.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        group_code: Group identifiers.
        active_dims: Active dimensions for GP kernel.
        model_class: Model class to use.
        priors: List of prior distributions.
        warmup: Number of MCMC warmup iterations.
        samples: Number of MCMC samples.

    Returns:
        Tuple of (model, mcmc_samples).
    """
    model = model_class(priors, train_x, train_y, active_dims=active_dims, group_code=group_code)
    prob_model = model.kohModel
    mcmc = model.MCMCSamples(prob_model, warmup, samples, True)

    return model, mcmc


# =============================================================================
# Gaussian Process Functions
# =============================================================================

def compute_gp_predictions(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    mcmc_koh: dict,
    active_dims: list
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GP mean and variance predictions.

    Args:
        model: Trained KOH model.
        train_x: Training input features.
        train_y: Training target values.
        group_code: Group identifiers.
        mcmc_koh: MCMC samples from KOH calibration.
        active_dims: Active dimensions for GP.

    Returns:
        Tuple of (gp_mean, gp_variance, entropy).
    """
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)
    predictive_koh = Predictive(model.probModel, mcmc_koh)(train_x, None, group_code)['obs']
    gpr = model.returnGP()
    gpr.set_data(train_x_norm[:, active_dims], train_y - predictive_koh.mean(dim=0))

    # Average over MCMC samples
    predictive_mean = torch.zeros(train_x_norm.shape[0])
    predictive_var = torch.zeros(train_x_norm.shape[0])

    for ii in range(mcmc_koh['kernel.lengthscale'].shape[0]):
        gpr.kernel.lengthscale = mcmc_koh['kernel.lengthscale'][ii]
        gpr.kernel.variance = mcmc_koh['kernel.variance'][ii]
        gpr.noise = mcmc_koh['noise'][ii]

        mean, cov = gpr(train_x_norm[:, active_dims], full_cov=False)
        predictive_mean += mean
        predictive_var += cov

    predictive_mean /= mcmc_koh['kernel.lengthscale'].shape[0]
    predictive_var /= mcmc_koh['kernel.lengthscale'].shape[0]

    # Compute entropy
    entropy = 0.5 * torch.log(2.0 * math.pi * math.exp(1.0) * predictive_var)

    return predictive_mean, predictive_var, entropy


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_scatter(
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    pred_deterministic: torch.Tensor,
    pred_bayesian: torch.Tensor,
    pred_koh: torch.Tensor,
    gp_mean: torch.Tensor,
    save_path: Path
) -> None:
    """
    Create scatter plot comparing model predictions on training data.

    Args:
        train_y: Observed values.
        group_code: Group identifiers.
        pred_deterministic: Deterministic predictions.
        pred_bayesian: Bayesian predictions.
        pred_koh: KOH predictions.
        gp_mean: GP discrepancy mean.
        save_path: Path to save figure.
    """
    err_deter = torch.round(torch.norm(train_y - pred_deterministic), decimals=4)
    err_bayes = torch.round(torch.norm(train_y - pred_bayesian), decimals=4)
    err_koh = torch.round(torch.norm(train_y - pred_koh - gp_mean), decimals=4)

    plt.figure(figsize=(6, 6))

    # Plot each group
    for g in [0, 1]:
        mask = group_code == g
        if g == 0:
            plt.scatter(train_y[mask], pred_deterministic[mask].detach(),
                       edgecolors='r', color='r', s=50)
            plt.scatter(train_y[mask], pred_bayesian[mask],
                       edgecolors='b', color='b', s=50)
            plt.scatter(train_y[mask], pred_koh[mask].detach() + gp_mean[mask].detach(),
                       s=50, c='k', edgecolors='k')
        else:
            plt.scatter(train_y[mask], pred_deterministic[mask].detach(),
                       edgecolors='r', color='r', s=50,
                       label=f'Deterministic (Error = {err_deter.item()})')
            plt.scatter(train_y[mask], pred_bayesian[mask],
                       edgecolors='b', color='b', s=50,
                       label=f'Bayes (Error = {err_bayes.item()})')
            plt.scatter(train_y[mask], pred_koh[mask].detach() + gp_mean[mask].detach(),
                       s=50, c='k', edgecolors='k',
                       label=f'KOH (Error = {err_koh.item()})')

    plt.plot([0.2, 2.0], [0.2, 2.0], color='k')
    plt.xlim([0.2, 2.0])
    plt.ylim([0.2, 2.0])
    plt.xticks([0.5, 1.0, 1.5, 2])
    plt.yticks([0.5, 1.0, 1.5, 2])
    plt.xlabel('Observed (E-E0)/E0')
    plt.ylabel('Predicted (E-E0)/E0')
    plt.legend(frameon=False)
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


def plot_gp_scatter_with_entropy(
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    pred_koh: torch.Tensor,
    gp_mean: torch.Tensor,
    entropy: torch.Tensor,
    save_path: Path
) -> None:
    """
    Create scatter plot with entropy colormap.

    Args:
        train_y: Observed values.
        group_code: Group identifiers.
        pred_koh: KOH predictions.
        gp_mean: GP discrepancy mean.
        entropy: Predictive entropy.
        save_path: Path to save figure.
    """
    err_koh = torch.round(torch.norm(train_y - pred_koh - gp_mean), decimals=4)

    plt.figure(figsize=(7, 6))

    for g in [0, 1]:
        mask = group_code == g
        label = f'KOH (Error = {err_koh.item()})' if g == 1 else None
        plt.scatter(train_y[mask], pred_koh[mask].detach() + gp_mean[mask].detach(),
                   s=100, c=entropy[mask].detach(), cmap='viridis',
                   edgecolors='none', label=label)

    plt.colorbar(label='Predictive Entropy')
    plt.plot([0.2, 2.0], [0.2, 2.0], color='k')
    plt.xlim([0.2, 2.0])
    plt.ylim([0.2, 2.0])
    plt.xticks([0.5, 1.0, 1.5, 2])
    plt.yticks([0.5, 1.0, 1.5, 2])
    plt.xlabel("Observed Young's modulus change")
    plt.ylabel("Predicted Young's modulus change")
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


def plot_fluence_predictions(
    test_x: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    pred_bayesian: torch.Tensor,
    pred_koh: torch.Tensor,
    gp_mean: torch.Tensor,
    entropy: torch.Tensor,
    save_path: Path
) -> None:
    """
    Create fluence prediction plot with GP uncertainty.

    Args:
        test_x: Test input features.
        train_x: Training input features.
        train_y: Training target values.
        pred_bayesian: Bayesian predictions on test data.
        pred_koh: KOH predictions on test data.
        gp_mean: GP mean on test data.
        entropy: Predictive entropy on test data.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Create entropy-colored line
    points = np.array([test_x[:, 1], gp_mean.detach()]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(entropy.min(), entropy.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(entropy)
    lc.set_linewidth(4)
    ax.add_collection(lc)
    ax.autoscale()

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Predictive Entropy')

    # Plot model predictions
    line1, = ax.plot(test_x[:, 1], pred_bayesian.mean(dim=0).detach(),
                     label='Bayesian model', c='b', lw=2.5)
    line2, = ax.plot(test_x[:, 1], pred_koh.mean(dim=0).detach() + gp_mean.detach(),
                     label='KOH model', c='r', lw=2.5)

    # Add experimental data
    scatter1 = ax.scatter(train_x[:, 1], train_y, c='k', label='Experimental data')

    # Legend
    viridis = cm.get_cmap('viridis')
    line_color = viridis(0.5)
    line3 = Line2D([0], [0], color=line_color, lw=4, label='GP inadequacy term')
    ax.legend(handles=[line1, line2, scatter1, line3], frameon=False)

    plt.xlabel('Fluence ($10^{26}$ neutron/m$^2$)')
    plt.ylabel('(E - E0)/E0')
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


# =============================================================================
# Model Export Functions
# =============================================================================

class MeanVarModelWrapper(torch.nn.Module):
    """Wrapper for GP model to enable TorchScript export."""

    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        mean, _ = self.gp(x, full_cov=False)
        return mean


def export_gp_model(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    mcmc_koh: dict,
    active_dims: list,
    save_path: Path
) -> None:
    """
    Export GP model to TorchScript format.

    Args:
        model: Trained KOH model.
        train_x: Training input features.
        train_y: Training target values.
        group_code: Group identifiers.
        mcmc_koh: MCMC samples from KOH calibration.
        active_dims: Active dimensions for GP.
        save_path: Path to save traced model.
    """
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)
    predictive_koh = Predictive(model.probModel, mcmc_koh)(train_x, None, group_code)['obs']
    gpr = model.returnGP()
    gpr.set_data(train_x_norm[:, active_dims], train_y - predictive_koh.mean(dim=0))

    # Set GP hyperparameters to posterior means
    gpr.kernel.lengthscale = mcmc_koh['kernel.lengthscale'].log().mean(dim=0).exp()
    gpr.kernel.variance = mcmc_koh['kernel.variance'].mean(dim=0)
    gpr.noise = mcmc_koh['noise'].log().mean(dim=0).exp()

    # Trace model
    example_input = torch.randn(10, train_x_norm[0][active_dims].shape[-1])
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
        gpr.eval()
        traced_model = torch.jit.trace(MeanVarModelWrapper(gpr), example_input, check_trace=False)

    traced_model.double().save(str(save_path))
    print(f"Exported GP model to: {save_path}")


# =============================================================================
# Main Execution Function
# =============================================================================

def main():
    """Main execution function for elastic modulus change model calibration."""
    print("=" * 70)
    print("Elastic Modulus Change Model Calibration")
    print("=" * 70)
    print(f"Graphite Grade: {GRAPHITE_GRADE}")
    print(f"Model: {MODEL_NAME}")
    print()

    # Create results directory
    results_dir = RESULTS_DIR / FOLDER_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading elastic modulus change data...")
    train_x, train_y, group_code = load_elastic_modulus_change_data(GRAPHITE_GRADE)
    print(f"Loaded {train_x.shape[0]} training samples")
    print()

    # Select model class
    if MODEL_NAME == "Bradford":
        model_class = IrradiationElasticModulusChange
        starting_point = torch.tensor([0.01, 0.01, 0.01, 0.01])
    elif MODEL_NAME == "INL":
        model_class = INLElasticModulusChange
        starting_point = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")

    # Fit deterministic model
    print("Fitting deterministic model...")
    minimized_params, _ = fit_deterministic_model(
        train_x, train_y, group_code, ACTIVE_DIMS, model_class,
        starting_point, LEARNING_RATE, NUM_EPOCHS
    )
    print(f"Optimized parameters: {minimized_params}")
    print()

    # Fit structural connectivity parameters (optional advanced calibration)
    print("Fitting structural connectivity parameters...")
    sc_params, _ = fit_structural_connectivity_parameters(
        train_x, train_y, group_code, ACTIVE_DIMS, model_class,
        SC_CALIBRATION_TEMPERATURE, SC_TEMPERATURE_TOLERANCE,
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        LEARNING_RATE, NUM_EPOCHS
    )
    print(f"SC parameters: {sc_params}")
    print()

    # Compute deterministic predictions
    model_determ = model_class([], train_x, train_y, active_dims=ACTIVE_DIMS, group_code=group_code)
    pred_deterministic = model_determ.determModel(minimized_params, train_x)
    error_std = (pred_deterministic - train_y).std()

    # Construct priors and fit Bayesian model
    print("Fitting Bayesian model...")
    priors = construct_prior_distributions(minimized_params, error_std, MODEL_NAME)
    model_bayes, mcmc = fit_bayesian_model(
        train_x, train_y, group_code, ACTIVE_DIMS, model_class,
        priors, MCMC_WARMUP, MCMC_SAMPLES
    )
    pred_bayesian = Predictive(model_bayes.probModel, mcmc)(train_x, None, group_code)['obs']
    print("Bayesian model complete")
    print()

    # Fit KOH model
    print("Fitting Kennedy-O'Hagan model...")
    model_koh, mcmc_koh = fit_koh_model(
        train_x, train_y, group_code, ACTIVE_DIMS, model_class,
        priors, MCMC_WARMUP, MCMC_SAMPLES
    )
    pred_koh = Predictive(model_koh.probModel, mcmc_koh)(train_x, None, group_code)['obs']
    print("KOH model complete")
    print()

    # Compute GP predictions
    print("Computing GP predictions...")
    gp_mean, gp_var, entropy = compute_gp_predictions(
        model_koh, train_x, train_y, group_code, mcmc_koh, ACTIVE_DIMS
    )

    # Create visualizations
    print("Generating plots...")
    plot_training_scatter(
        train_y, group_code, pred_deterministic,
        pred_bayesian.mean(dim=0), pred_koh.mean(dim=0), gp_mean,
        results_dir / f"{MODEL_NAME}_scatter.pdf"
    )

    plot_gp_scatter_with_entropy(
        train_y, group_code, pred_koh.mean(dim=0), gp_mean, entropy,
        results_dir / f"{MODEL_NAME}_GP_scatter.pdf"
    )

    # Export GP model
    print("Exporting GP model...")
    export_gp_model(
        model_koh, train_x, train_y, group_code, mcmc_koh, ACTIVE_DIMS,
        results_dir / f"gp_{MODEL_NAME}_EChange.pt"
    )

    # Generate test predictions
    print("Generating test predictions...")
    test_x = torch.zeros((400, 6))
    test_x[:, 1] = torch.linspace(0.0, 0.5, 400)
    test_x[:, 2] = 1100
    test_x[:, 3] = 0.29
    test_x[:, 4] = 0.05
    test_x[:, 5] = -0.0086

    print("Test predictions complete")
    print()

    print("=" * 70)
    print("Calibration Complete!")
    print("=" * 70)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
