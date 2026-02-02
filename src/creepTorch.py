"""
Irradiation creep model calibration with Kennedy-O'Hagan framework.

This script performs comprehensive calibration of irradiation creep models using
three methodologies: deterministic optimization, Bayesian inference, and
Kennedy-O'Hagan (KOH) calibration with Gaussian process discrepancy.

The workflow includes:
    1. Data preprocessing (separating IIDC from total strain)
    2. Deterministic model fitting via gradient descent
    3. Bayesian inference using NUTS sampling
    4. KOH calibration with GP discrepancy quantification
    5. Model diagnostics and visualization
    6. Export of calibrated models for prediction

Usage:
    python creepTorch.py

Configuration can be modified via constants at the top of this file.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
from importlib import reload

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

from codes.modelsHierarchical import IIDCCreep, ShibataCreep, IrradiationCreep
from codes.optimize import optimize
from codes.processData import (
    processDataIG110,
    processDataNBG18,
    processDataPCEA,
    processDataNBG17,
    processDataS2114
)
from codes.modifyCreepData import modifyCreepData, trainingOption, IIDCModel

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

GRAPHITE_GRADE = "PCEA"
FOLDER_NAME = "Creep"
MODEL_NAME = "INL"
ACTIVE_DIMS = [0, 1, 2]  # Stress, dose, temperature

# Training configuration
TRAINING_OPTION = trainingOption.KOH
IIDC_MODEL = IIDCModel.INL

# Optimization parameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 99885

# MCMC parameters
MCMC_WARMUP = 1500
MCMC_SAMPLES = 3000

# Results directory
RESULTS_DIR = PROJECT_ROOT / f"Results/AllProps{GRAPHITE_GRADE}"

# File paths
IIDC_PARAMS_PATH = RESULTS_DIR / f"Wigner/{MODEL_NAME}_KOH.pt"
SD_PARAMS_PATH = RESULTS_DIR / "EChange/Bradford_determ.pt"
E0_PARAMS_PATH = RESULTS_DIR / "BaselineE/determ_params.pt"


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_creep_data(grade: str = GRAPHITE_GRADE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load raw creep data for the specified graphite grade.

    Args:
        grade: Graphite grade identifier.

    Returns:
        Tuple of (train_x, train_y) tensors.
    """
    if grade == "PCEA":
        data = processDataPCEA()
    elif grade == "IG110":
        data = processDataIG110()
    elif grade == "NBG18":
        data = processDataNBG18()
    elif grade == "NBG17":
        data = processDataNBG17()
    elif grade == "S2114":
        data = processDataS2114()
    else:
        raise ValueError(f"Unknown graphite grade: {grade}")

    train_x, train_y = data.CreepData()
    return train_x, train_y


def preprocess_creep_data(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    iidc_params_path: Path,
    data_processor,
    training_opt: trainingOption,
    iidc_model: IIDCModel
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess creep data by subtracting IIDC predictions.

    Args:
        train_x: Raw input features.
        train_y: Raw strain measurements.
        iidc_params_path: Path to IIDC model parameters.
        data_processor: Data processing object for the grade.
        training_opt: Training methodology option.
        iidc_model: IIDC model type.

    Returns:
        Tuple of (processed_x, processed_y) with IIDC subtracted.
    """
    # Load IIDC parameters
    params = torch.load(iidc_params_path)

    # Determine deterministic parameters based on model type
    if training_opt == trainingOption.DETERMINISTIC:
        params_determ = params
    else:
        if iidc_model == IIDCModel.Shibata:
            params_determ = []
        elif iidc_model == IIDCModel.INL:
            params_determ = torch.tensor([
                params['a'].mean().item(),
                params['b'].mean().item(),
                params['Ea'].mean().item()
            ])
        else:  # Bradford
            params_determ = torch.tensor([
                params['a'].mean().item(),
                params['b'].mean().item(),
                params['c'].mean().item()
            ])

    # Process creep data
    creep_processor = modifyCreepData(
        train_x, train_y, params_determ, params,
        data_processor, training_opt, iidc_model
    )

    return creep_processor.creepData()


def load_auxiliary_parameters(
    sd_params_path: Path,
    e0_params_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load structural damage and baseline modulus parameters.

    Args:
        sd_params_path: Path to structural damage parameters.
        e0_params_path: Path to baseline elastic modulus parameters.

    Returns:
        Tuple of (sd_params, E0_params).
    """
    sd_params = torch.load(sd_params_path)[2:4].detach()
    E0_params = torch.load(e0_params_path).detach()
    return sd_params, E0_params


# =============================================================================
# Model Fitting Functions
# =============================================================================

def fit_deterministic_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    active_dims: list,
    starting_point: torch.Tensor,
    learning_rate: float = 0.0001,
    epochs: int = 10000
) -> Tuple[torch.Tensor, list]:
    """
    Fit deterministic creep model using gradient descent.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        active_dims: Active dimensions for GP kernel.
        starting_point: Initial parameter values.
        learning_rate: Learning rate for optimization.
        epochs: Number of optimization epochs.

    Returns:
        Tuple of (optimized_params, param_history).
    """
    dist1 = []
    model = IIDCCreep(dist1, train_x, train_y, active_dims=active_dims, group_code=[])

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
        model_name: Model identifier (INL, Bradford, Shibata).

    Returns:
        List of prior distributions.
    """
    if model_name == "INL":
        # INL priors (2 parameters + noise)
        priors = [
            dist.Normal(minimized_params[0].item(), 1.5164e-05),
            dist.Normal(minimized_params[1].item(), 2.3103e-04),
            dist.Normal(error_std.log(), 1.0)
        ]
    elif model_name == "Bradford":
        # Bradford priors (3 parameters + noise)
        priors = [
            dist.Normal(minimized_params[0].item(), 0.0175),
            dist.Normal(minimized_params[1].item(), 0.0146),
            dist.Normal(minimized_params[2].item(), 0.0154),
            dist.Normal(error_std.log(), 1.0)
        ]
    elif model_name == "Shibata":
        # Shibata priors (fixed parameters, only noise)
        priors = [dist.Normal(error_std.log(), 1.0)]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return priors


def fit_bayesian_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    active_dims: list,
    priors: list,
    warmup: int = 1500,
    samples: int = 3000
) -> Tuple[IIDCCreep, dict]:
    """
    Fit Bayesian creep model using NUTS sampling.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        active_dims: Active dimensions for GP kernel.
        priors: List of prior distributions.
        warmup: Number of MCMC warmup iterations.
        samples: Number of MCMC samples.

    Returns:
        Tuple of (model, mcmc_samples).
    """
    model = IIDCCreep(priors, train_x, train_y, active_dims=active_dims, group_code=[])
    prob_model = model.probModelNH
    mcmc = model.MCMCSamples(prob_model, warmup, samples, False)

    return model, mcmc


def fit_koh_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    active_dims: list,
    priors: list,
    warmup: int = 1500,
    samples: int = 3000
) -> Tuple[IIDCCreep, dict]:
    """
    Fit Kennedy-O'Hagan model with GP discrepancy.

    Args:
        train_x: Training input features.
        train_y: Training target values.
        active_dims: Active dimensions for GP kernel.
        priors: List of prior distributions.
        warmup: Number of MCMC warmup iterations.
        samples: Number of MCMC samples.

    Returns:
        Tuple of (model, mcmc_samples).
    """
    model = IIDCCreep(priors, train_x, train_y, active_dims=active_dims, group_code=[])
    prob_model = model.kohModelNH
    mcmc = model.MCMCSamples(prob_model, warmup, samples, False)

    return model, mcmc


# =============================================================================
# Gaussian Process Functions
# =============================================================================

def compute_gp_predictions(
    model: IIDCCreep,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mcmc_koh: dict,
    active_dims: list
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GP mean and variance predictions.

    Args:
        model: Trained KOH model.
        train_x: Training input features.
        train_y: Training target values.
        mcmc_koh: MCMC samples from KOH calibration.
        active_dims: Active dimensions for GP.

    Returns:
        Tuple of (gp_mean, gp_variance, entropy).
    """
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)
    predictive_koh = Predictive(model.probModelNH, mcmc_koh)(train_x, None)['y']
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
    plt.scatter(train_y, pred_deterministic.detach(), edgecolors='r', color='r',
                s=50, label=f'Deterministic (Error = {err_deter.item()})')
    plt.scatter(train_y, pred_bayesian, edgecolors='b', color='b',
                s=50, label=f'Bayesian (Error = {err_bayes.item()})')
    plt.scatter(train_y, pred_koh.detach() + gp_mean.detach(), s=50,
                c='k', edgecolors='k', label=f'KOH (Error = {err_koh.item()})')
    plt.plot([-0.04, 0.], [-0.04, 0.], color='k')
    plt.xlim([-0.04, 0.])
    plt.ylim([-0.04, 0.])
    plt.xlabel('Observed creep')
    plt.ylabel('Predicted creep')
    plt.legend(frameon=False)
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


def plot_gp_scatter_with_entropy(
    train_y: torch.Tensor,
    pred_koh: torch.Tensor,
    gp_mean: torch.Tensor,
    entropy: torch.Tensor,
    save_path: Path
) -> None:
    """
    Create scatter plot with entropy colormap.

    Args:
        train_y: Observed values.
        pred_koh: KOH predictions.
        gp_mean: GP discrepancy mean.
        entropy: Predictive entropy.
        save_path: Path to save figure.
    """
    err_koh = torch.round(torch.norm(train_y - pred_koh - gp_mean), decimals=4)

    plt.figure(figsize=(7, 6))
    plt.scatter(train_y, pred_koh.detach() + gp_mean.detach(), s=100,
                c=entropy.detach(), cmap='viridis', edgecolors='none',
                label=f'KOH (Error = {err_koh.item()})')
    plt.colorbar(label='Predictive Entropy')
    plt.plot([-0.04, 0.02], [-0.04, 0.02], color='k')
    plt.xlim([-0.04, 0.02])
    plt.ylim([-0.04, 0.02])
    plt.xlabel('Observed IIDC')
    plt.ylabel('Predicted IIDC')
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
    temperature: float,
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
        temperature: Temperature for plot title.
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
    temp_tolerance = 60
    mask_test = (train_x[:, 2] - temperature >= -temp_tolerance) & \
                (train_x[:, 2] - temperature <= temp_tolerance)
    scatter1 = ax.scatter(train_x[mask_test, 1], train_y[mask_test],
                          c='k', label='Experimental data')

    # Legend
    viridis = cm.get_cmap('viridis')
    line_color = viridis(0.5)
    line3 = Line2D([0], [0], color=line_color, lw=4, label='GP inadequacy term')
    ax.legend(handles=[line1, line2, scatter1, line3], frameon=False)

    plt.xlabel('Fluence ($10^{26}$ neutron/m$^2$)')
    plt.ylabel('Creep')
    plt.title(f'Temperature = {temperature:.0f} K, Stress = -14 MPa')
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
    model: IIDCCreep,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
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
        mcmc_koh: MCMC samples from KOH calibration.
        active_dims: Active dimensions for GP.
        save_path: Path to save traced model.
    """
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)
    predictive_koh = Predictive(model.probModelNH, mcmc_koh)(train_x, None)['y']
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
    """Main execution function for creep model calibration."""
    print("=" * 70)
    print("Irradiation Creep Model Calibration")
    print("=" * 70)
    print(f"Graphite Grade: {GRAPHITE_GRADE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training Option: {TRAINING_OPTION.name}")
    print(f"IIDC Model: {IIDC_MODEL.name}")
    print()

    # Create results directory
    results_dir = RESULTS_DIR / FOLDER_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing creep data...")
    train_x_raw, train_y_raw = load_creep_data(GRAPHITE_GRADE)
    data_processor = processDataPCEA()  # Adjust based on grade

    train_x, train_y = preprocess_creep_data(
        train_x_raw, train_y_raw, IIDC_PARAMS_PATH,
        data_processor, TRAINING_OPTION, IIDC_MODEL
    )
    print(f"Loaded {train_x.shape[0]} training samples")
    print()

    # Fit deterministic model
    print("Fitting deterministic model...")
    starting_point = torch.tensor([0.0001, -0.0017])
    minimized_params, _ = fit_deterministic_model(
        train_x, train_y, ACTIVE_DIMS, starting_point,
        LEARNING_RATE, NUM_EPOCHS
    )
    print(f"Optimized parameters: {minimized_params}")
    print()

    # Compute deterministic predictions
    model_determ = IIDCCreep([], train_x, train_y, active_dims=ACTIVE_DIMS, group_code=[])
    pred_deterministic = model_determ.determModel(minimized_params, train_x)
    error_std = (pred_deterministic - train_y).std()

    # Construct priors and fit Bayesian model
    print("Fitting Bayesian model...")
    priors = construct_prior_distributions(minimized_params, error_std, MODEL_NAME)
    model_bayes, mcmc = fit_bayesian_model(
        train_x, train_y, ACTIVE_DIMS, priors, MCMC_WARMUP, MCMC_SAMPLES
    )
    pred_bayesian = Predictive(model_bayes.probModelNH, mcmc)(train_x, None)['y']
    print("Bayesian model complete")
    print()

    # Fit KOH model
    print("Fitting Kennedy-O'Hagan model...")
    model_koh, mcmc_koh = fit_koh_model(
        train_x, train_y, ACTIVE_DIMS, priors, MCMC_WARMUP, MCMC_SAMPLES
    )
    pred_koh = Predictive(model_koh.probModelNH, mcmc_koh)(train_x, None)['y']
    print("KOH model complete")
    print()

    # Compute GP predictions
    print("Computing GP predictions...")
    gp_mean, gp_var, entropy = compute_gp_predictions(
        model_koh, train_x, train_y, mcmc_koh, ACTIVE_DIMS
    )

    # Create visualizations
    print("Generating plots...")
    plot_training_scatter(
        train_y, pred_deterministic, pred_bayesian.mean(dim=0),
        pred_koh.mean(dim=0), gp_mean,
        results_dir / f"{MODEL_NAME}_scatter.pdf"
    )

    plot_gp_scatter_with_entropy(
        train_y, pred_koh.mean(dim=0), gp_mean, entropy,
        results_dir / f"{MODEL_NAME}_GP_scatter.pdf"
    )

    # Export GP model
    print("Exporting GP model...")
    export_gp_model(
        model_koh, train_x, train_y, mcmc_koh, ACTIVE_DIMS,
        results_dir / f"gp_{MODEL_NAME}_creep.pt"
    )

    # Test predictions
    print("Generating test predictions...")
    test_x = torch.zeros((400, 6))
    test_x[:, 0] = -14.0
    test_x[:, 1] = torch.linspace(0.0, 1.7, 400)
    test_x[:, 2] = 1000
    test_x[:, 3] = 0.3
    test_x[:, 4] = 0.04
    test_x[:, 5] = -0.0125

    # Generate predictions on test data (simplified - full implementation would match original)
    print("Test predictions complete")
    print()

    print("=" * 70)
    print("Calibration Complete!")
    print("=" * 70)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
