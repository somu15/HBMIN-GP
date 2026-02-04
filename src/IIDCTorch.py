"""
Irradiation-Induced Dimensional Change (IIDC) Model Calibration.

This module implements comprehensive IIDC model calibration using deterministic,
Bayesian, and Kennedy-O'Hagan (KOH) calibration frameworks. It supports multiple
graphite grades and IIDC model formulations (INL, Shibata, Bradford).

Workflow:
    1. Load IIDC data for specified graphite grade
    2. Apply zero-fluence padding to enforce physical boundary conditions
    3. Fit deterministic model using optimization
    4. Construct Bayesian priors from deterministic results
    5. Fit Bayesian model using MCMC sampling
    6. Fit KOH model with GP discrepancy term
    7. Compute GP predictions with uncertainty quantification
    8. Generate training and testing visualizations
    9. Analyze key physical features (u-turn, crossover)
    10. Export calibrated GP model as TorchScript

Usage:
    python IIDCTorch.py

The script outputs:
    - Scatter plots comparing model predictions
    - Fluence evolution plots with predictive entropy
    - TorchScript GP models for deployment
    - Physical feature analysis (u-turn point, crossover point)
"""

from typing import Tuple, Optional
import math

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import seaborn as sns

import pyro.distributions as dist
from pyro.infer import Predictive
import gpytorch
import arviz as az

from codes.modelsHierarchical import (
    ShibataIIDCDeltaL,
    IIDCDeltaL,
    IrradiationStrain
)
from codes.optimize import optimize
from codes.processData import (
    processDataIG110,
    processDataNBG18,
    processDataPCEA,
    processDataNBG17,
    processDataS2114
)


# ============================================================================
# Configuration Constants
# ============================================================================

# Graphite grade and model configuration
GRAPHITE_GRADE = 'NBG18'
FOLDER_NAME = 'Wigner'
MODEL_NAME = 'INL'

# Model dimensions
ACTIVE_DIMS = [1, 2]  # Dose and temperature dimensions

# Log-normal distribution parameters for hierarchical model
LOG_LOC_PARAMS = torch.tensor([1.5630e+01, 3.6878e-03])
LOG_SCA_PARAMS = torch.tensor([1.1895, -0.0021])

# Zero-fluence padding configuration
N_PAD = 50  # Number of padding points

# Optimization settings
LEARNING_RATE = 0.001
EPOCHS = 100000

# MCMC sampling settings
NUM_WARMUP = 1500
NUM_SAMPLES = 3000

# Matplotlib configuration
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams['axes.linewidth'] = 1.5
plt.rc('font', family='serif', size=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_iidc_data(grade: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load IIDC data for specified graphite grade.

    Args:
        grade: Graphite grade identifier (e.g., 'NBG18', 'IG110', 'PCEA').

    Returns:
        Tuple containing:
            - train_x: Input features (stress, dose, temperature, etc.) [N, 6]
            - train_y: IIDC strain measurements [N]
            - group_code: Data source identifiers [N]

    Raises:
        ValueError: If grade is not recognized.
    """
    grade_processors = {
        'NBG18': processDataNBG18,
        'IG110': processDataIG110,
        'PCEA': processDataPCEA,
        'NBG17': processDataNBG17,
        'S2114': processDataS2114
    }

    if grade not in grade_processors:
        raise ValueError(
            f"Unknown grade: {grade}. "
            f"Available options: {list(grade_processors.keys())}"
        )

    data = grade_processors[grade]()
    train_x, train_y, group_code = data.IIDCData()

    return train_x, train_y, group_code


def apply_zero_fluence_padding(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    n_pad: int = N_PAD
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply zero-fluence padding to enforce physical boundary condition.

    Adds synthetic data points at zero fluence with zero strain to encourage
    the GP to predict zero dimensional change at zero irradiation dose.

    Args:
        train_x: Input features [N, 6].
        train_y: IIDC strain measurements [N].
        group_code: Data source identifiers [N].
        n_pad: Number of padding points to add.

    Returns:
        Tuple containing padded versions of (train_x, train_y, group_code).
    """
    # Create padding points
    train_x_pad = torch.zeros((n_pad, 6))
    train_x_pad[:, 1] = 1e-5  # Near-zero fluence
    train_x_pad[:, 2] = torch.linspace(
        torch.min(train_x[:, 2]),
        torch.max(train_x[:, 2]),
        steps=n_pad
    )
    train_x_pad[:, 3] = 1e-5  # Near-zero interconnectivity

    train_y_pad = torch.zeros(n_pad)
    group_code_pad = torch.zeros(n_pad).int()

    # Concatenate with original data
    train_x_padded = torch.vstack((train_x, train_x_pad))
    train_y_padded = torch.hstack((train_y, train_y_pad))
    group_code_padded = torch.hstack((group_code, group_code_pad))

    return train_x_padded, train_y_padded, group_code_padded


def load_agc_only_data(grade: str = 'IG110') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load only AGC data (group_code == 0) for specified grade.

    Args:
        grade: Graphite grade identifier.

    Returns:
        Tuple containing filtered (train_x, train_y, group_code).
    """
    train_x_full, train_y_full, group_code_full = load_iidc_data(grade)

    # Filter for AGC data only
    mask = (group_code_full == 0)
    train_x = train_x_full[mask]
    train_y = train_y_full[mask]
    group_code = torch.zeros(train_y.shape[0]).int()

    return train_x, train_y, group_code


# ============================================================================
# Model Fitting Functions
# ============================================================================

def fit_deterministic_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    active_dims: list,
    starting_point: torch.Tensor,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    log_loc_params: Optional[torch.Tensor] = None,
    log_sca_params: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, list, torch.Tensor]:
    """
    Fit deterministic IIDC model using gradient-based optimization.

    Args:
        train_x: Input features [N, 6].
        train_y: IIDC strain measurements [N].
        group_code: Data source identifiers [N].
        active_dims: Active dimensions for GP.
        starting_point: Initial parameter values.
        learning_rate: Optimization learning rate.
        epochs: Number of optimization epochs.
        log_loc_params: Log-normal location parameters (optional).
        log_sca_params: Log-normal scale parameters (optional).

    Returns:
        Tuple containing:
            - minimized_params: Optimized parameter values
            - param_history: List of parameters at each epoch
            - residuals: Model residuals (prediction - observation)
    """
    dist1 = []
    model = IIDCDeltaL(
        dist1, train_x, train_y,
        active_dims=active_dims,
        group_code=group_code,
        log_loc_params=log_loc_params,
        log_sca_params=log_sca_params
    )

    opt = optimize(
        train_x=train_x,
        train_y=train_y,
        model_predict=model.determModel
    )

    minimized_params, param_history = opt.train(
        initial_parameters=starting_point,
        learning_rate=learning_rate,
        epochs=epochs
    )

    iidc_prediction = model.determModel(minimized_params, train_x)
    residuals = iidc_prediction - train_y

    return minimized_params, param_history, residuals


def construct_inl_priors(
    minimized_params: torch.Tensor,
    residuals: torch.Tensor
) -> list:
    """
    Construct prior distributions for INL IIDC model.

    The INL model has three physical parameters (a, b, Ea) plus noise.

    Args:
        minimized_params: Optimized deterministic parameters.
        residuals: Model residuals for noise estimation.

    Returns:
        List of Pyro distribution objects for MCMC sampling.
    """
    return [
        dist.Normal(minimized_params[0].item(), 0.0026),
        dist.Normal(minimized_params[1].item(), 0.0005),
        dist.Normal(minimized_params[2].item(), 0.0165),
        dist.Normal(residuals.std().log(), 1.0)
    ]


def construct_bradford_priors(
    minimized_params: torch.Tensor,
    residuals: torch.Tensor
) -> list:
    """
    Construct prior distributions for Bradford physics-based IIDC model.

    Args:
        minimized_params: Optimized deterministic parameters.
        residuals: Model residuals for noise estimation.

    Returns:
        List of Pyro distribution objects for MCMC sampling.
    """
    return [
        dist.Normal(minimized_params[0].item(), 0.0972),
        dist.Normal(minimized_params[1].item(), 0.0001),
        dist.Normal(minimized_params[2].item(), 0.1023),
        dist.Normal(residuals.std().log(), 1.0)
    ]


def construct_shibata_priors(residuals: torch.Tensor) -> list:
    """
    Construct prior distributions for Shibata empirical IIDC model.

    The Shibata model uses fixed functional form with only noise parameter.

    Args:
        residuals: Model residuals for noise estimation.

    Returns:
        List of Pyro distribution objects for MCMC sampling.
    """
    return [dist.Normal(residuals.std().log(), 1.0)]


def fit_bayesian_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    group_code: torch.Tensor,
    active_dims: list,
    priors: list,
    num_warmup: int = NUM_WARMUP,
    num_samples: int = NUM_SAMPLES,
    log_loc_params: Optional[torch.Tensor] = None,
    log_sca_params: Optional[torch.Tensor] = None
) -> Tuple[IIDCDeltaL, dict]:
    """
    Fit Bayesian IIDC model using MCMC sampling.

    Args:
        train_x: Input features [N, 6].
        train_y: IIDC strain measurements [N].
        group_code: Data source identifiers [N].
        active_dims: Active dimensions for GP.
        priors: List of prior distributions.
        num_warmup: MCMC warmup iterations.
        num_samples: MCMC sampling iterations.
        log_loc_params: Log-normal location parameters (optional).
        log_sca_params: Log-normal scale parameters (optional).

    Returns:
        Tuple containing:
            - model: Fitted IIDCDeltaL model instance
            - mcmc: Dictionary of MCMC samples
    """
    model = IIDCDeltaL(
        priors, train_x, train_y,
        active_dims=active_dims,
        group_code=group_code,
        log_loc_params=log_loc_params,
        log_sca_params=log_sca_params
    )

    prob_iidc = model.probModel
    mcmc = model.MCMCSamples(prob_iidc, num_warmup, num_samples, True)

    return model, mcmc


def fit_koh_model(
    model: IIDCDeltaL,
    num_warmup: int = NUM_WARMUP,
    num_samples: int = NUM_SAMPLES
) -> dict:
    """
    Fit Kennedy-O'Hagan calibration model with GP discrepancy.

    Args:
        model: IIDCDeltaL model instance with priors.
        num_warmup: MCMC warmup iterations.
        num_samples: MCMC sampling iterations.

    Returns:
        Dictionary of KOH MCMC samples including GP hyperparameters.
    """
    prob_iidc = model.kohModel
    mcmc_koh = model.MCMCSamples(prob_iidc, num_warmup, num_samples, True)

    return mcmc_koh


def fit_bayesian_model_non_hierarchical(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    active_dims: list,
    priors: list,
    num_warmup: int = 1000,
    num_samples: int = 1000
) -> Tuple[IIDCDeltaL, dict]:
    """
    Fit non-hierarchical Bayesian IIDC model (single data source).

    Args:
        train_x: Input features [N, 6].
        train_y: IIDC strain measurements [N].
        active_dims: Active dimensions for GP.
        priors: List of prior distributions.
        num_warmup: MCMC warmup iterations.
        num_samples: MCMC sampling iterations.

    Returns:
        Tuple containing:
            - model: Fitted IIDCDeltaL model instance
            - mcmc: Dictionary of MCMC samples
    """
    group_code = torch.zeros(train_y.shape[0]).int()

    model = IIDCDeltaL(
        priors, train_x, train_y,
        active_dims=active_dims,
        group_code=group_code
    )

    prob_iidc = model.probModelNH
    mcmc = model.MCMCSamples(prob_iidc, num_warmup, num_samples, False)

    return model, mcmc


def fit_koh_model_non_hierarchical(
    model: IIDCDeltaL,
    num_warmup: int = 1000,
    num_samples: int = 1000
) -> dict:
    """
    Fit non-hierarchical KOH model (single data source).

    Args:
        model: IIDCDeltaL model instance with priors.
        num_warmup: MCMC warmup iterations.
        num_samples: MCMC sampling iterations.

    Returns:
        Dictionary of KOH MCMC samples.
    """
    prob_iidc = model.kohModelNH
    mcmc_koh = model.MCMCSamples(prob_iidc, num_warmup, num_samples, False)

    return mcmc_koh


# ============================================================================
# Gaussian Process Functions
# ============================================================================

def compute_gp_predictions(
    model: IIDCDeltaL,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mcmc_koh: dict,
    active_dims: list,
    group_code: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GP mean and variance predictions for KOH model.

    Averages predictions over MCMC samples of GP hyperparameters.

    Args:
        model: Fitted IIDCDeltaL model.
        train_x: Input features [N, 6].
        train_y: IIDC strain measurements [N].
        mcmc_koh: Dictionary of KOH MCMC samples.
        active_dims: Active dimensions for GP.
        group_code: Data source identifiers [N].

    Returns:
        Tuple containing:
            - gp_mean: Mean GP prediction [N]
            - gp_var: GP predictive variance [N]
            - entropy: Predictive entropy [N]
    """
    # Normalize features
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)

    # Get parametric predictions
    predictive_koh = Predictive(model.probModel, mcmc_koh)(
        train_x, None, group_code
    )['obs']

    # Initialize GP
    gpr = model.returnGP()
    gpr.set_data(
        train_x_norm[:, active_dims],
        train_y - predictive_koh.mean(dim=0)
    )

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

    # Set mean hyperparameters
    gpr.kernel.lengthscale = mcmc_koh['kernel.lengthscale'].log().mean(dim=0).exp()
    gpr.kernel.variance = mcmc_koh['kernel.variance'].mean(dim=0)
    gpr.noise = mcmc_koh['noise'].log().mean(dim=0).exp()

    mean, _ = gpr(train_x_norm[:, active_dims], full_cov=False)
    entropy = 0.5 * torch.log(2.0 * math.pi * math.exp(1.0) * predictive_var)

    return mean, predictive_var, entropy


def compute_gp_predictions_non_hierarchical(
    model: IIDCDeltaL,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mcmc_koh: dict,
    active_dims: list
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GP predictions for non-hierarchical model.

    Args:
        model: Fitted IIDCDeltaL model.
        train_x: Input features [N, 6].
        train_y: IIDC strain measurements [N].
        mcmc_koh: Dictionary of KOH MCMC samples.
        active_dims: Active dimensions for GP.

    Returns:
        Tuple containing (gp_mean, gp_var, entropy).
    """
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)

    predictive_koh = Predictive(model.probModelNH, mcmc_koh)(train_x, None)['y']

    gpr = model.returnGP()
    gpr.set_data(
        train_x_norm[:, active_dims],
        train_y - predictive_koh.mean(dim=0)
    )

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

    gpr.kernel.lengthscale = mcmc_koh['kernel.lengthscale'].log().mean(dim=0).exp()
    gpr.kernel.variance = mcmc_koh['kernel.variance'].mean(dim=0)
    gpr.noise = mcmc_koh['noise'].log().mean(dim=0).exp()

    mean, _ = gpr(train_x_norm[:, active_dims], full_cov=False)
    entropy = 0.5 * torch.log(2.0 * math.pi * math.exp(1.0) * predictive_var)

    return mean, predictive_var, entropy


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_scatter(
    train_y: torch.Tensor,
    iidc_pred_determ: torch.Tensor,
    iidc_pred_bayes: torch.Tensor,
    iidc_pred_koh: torch.Tensor,
    gp_mean: torch.Tensor,
    group_code: torch.Tensor,
    save_path: str
) -> None:
    """
    Create scatter plot comparing deterministic, Bayesian, and KOH predictions.

    Args:
        train_y: Observed IIDC values [N].
        iidc_pred_determ: Deterministic predictions [N].
        iidc_pred_bayes: Bayesian predictions [N].
        iidc_pred_koh: KOH parametric predictions [N].
        gp_mean: GP discrepancy predictions [N].
        group_code: Data source identifiers [N].
        save_path: Path to save figure.
    """
    err_deter = torch.round(torch.norm(train_y - iidc_pred_determ), decimals=4)
    err_bayes = torch.round(torch.norm(train_y - iidc_pred_bayes), decimals=4)
    err_koh = torch.round(torch.norm(train_y - iidc_pred_koh - gp_mean), decimals=4)

    plt.figure(figsize=(6, 6))

    # Deterministic predictions
    plt.scatter(
        train_y[group_code == 0], iidc_pred_determ[group_code == 0].detach(),
        edgecolors='r', color='r', s=50
    )
    plt.scatter(
        train_y[group_code == 1], iidc_pred_determ[group_code == 1].detach(),
        edgecolors='r', color='r', s=50,
        label=f'Deterministic (Error = {err_deter.item()})'
    )

    # Bayesian predictions
    plt.scatter(
        train_y[group_code == 0], iidc_pred_bayes[group_code == 0],
        edgecolors='b', color='b', s=50
    )
    plt.scatter(
        train_y[group_code == 1], iidc_pred_bayes[group_code == 1],
        edgecolors='b', color='b', s=50,
        label=f'Bayesian (Error = {err_bayes.item()})'
    )

    # KOH predictions
    plt.scatter(
        train_y[group_code == 0],
        iidc_pred_koh.detach()[group_code == 0] + gp_mean.detach()[group_code == 0],
        s=50, c='k', edgecolors='k'
    )
    plt.scatter(
        train_y[group_code == 1],
        iidc_pred_koh.detach()[group_code == 1] + gp_mean.detach()[group_code == 1],
        s=50, c='k', edgecolors='k',
        label=f'KOH (Error = {err_koh.item()})'
    )

    plt.plot([-0.02, 0.12], [-0.02, 0.12], color='k')
    plt.xlim([-0.02, 0.12])
    plt.ylim([-0.02, 0.12])
    plt.xlabel('Observed IIDC')
    plt.ylabel('Predicted IIDC')
    plt.legend(frameon=False)
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


def plot_gp_scatter_with_entropy(
    train_y: torch.Tensor,
    iidc_pred_koh: torch.Tensor,
    gp_mean: torch.Tensor,
    entropy: torch.Tensor,
    group_code: torch.Tensor,
    save_path: str
) -> None:
    """
    Create scatter plot with predictive entropy color-coding.

    Args:
        train_y: Observed IIDC values [N].
        iidc_pred_koh: KOH parametric predictions [N].
        gp_mean: GP discrepancy predictions [N].
        entropy: Predictive entropy [N].
        group_code: Data source identifiers [N].
        save_path: Path to save figure.
    """
    err_koh = torch.round(
        torch.norm(train_y - iidc_pred_koh - gp_mean),
        decimals=4
    )

    plt.figure(figsize=(7, 6))

    plt.scatter(
        train_y[group_code == 0],
        iidc_pred_koh.detach()[group_code == 0] + gp_mean.detach()[group_code == 0],
        s=100,
        c=entropy.detach()[group_code == 0],
        cmap='viridis',
        edgecolors='none'
    )
    plt.scatter(
        train_y[group_code == 1],
        iidc_pred_koh.detach()[group_code == 1] + gp_mean.detach()[group_code == 1],
        s=100,
        c=entropy.detach()[group_code == 1],
        cmap='viridis',
        edgecolors='none',
        label=f'KOH (Error = {err_koh.item()})'
    )

    plt.colorbar(label='Predictive Entropy')
    plt.plot([-0.02, 0.07], [-0.02, 0.07], color='k')
    plt.xlim([-0.02, 0.07])
    plt.ylim([-0.02, 0.07])
    plt.xlabel('Observed IIDC')
    plt.ylabel('Predicted IIDC')
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


def plot_fluence_predictions(
    test_x: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bayes_pred: torch.Tensor,
    koh_pred: torch.Tensor,
    gp_mean: torch.Tensor,
    entropy: torch.Tensor,
    temperature: float,
    temperature_tolerance: float,
    save_path: str,
    grade: str = 'IG-110'
) -> None:
    """
    Plot IIDC predictions vs fluence with entropy-colored GP term.

    Args:
        test_x: Test input features [M, 6].
        train_x: Training input features [N, 6].
        train_y: Training IIDC values [N].
        bayes_pred: Bayesian predictions [M].
        koh_pred: KOH parametric predictions [M].
        gp_mean: GP discrepancy predictions [M].
        entropy: Predictive entropy [M].
        temperature: Temperature for filtering training data (K).
        temperature_tolerance: Temperature tolerance for filtering (K).
        save_path: Path to save figure.
        grade: Graphite grade for plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot GP term with entropy coloring
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
    line1, = ax.plot(
        test_x[:, 1], bayes_pred.mean(dim=0).detach(),
        label='Bayesian model', c='b', lw=2.5
    )
    line2, = ax.plot(
        test_x[:, 1], koh_pred.mean(dim=0).detach() + gp_mean.detach(),
        label='KOH model', c='r', lw=2.5
    )

    # Plot training data at specified temperature
    mask_test = (
        (train_x[:, 2] - temperature >= -temperature_tolerance) &
        (train_x[:, 2] - temperature <= temperature_tolerance) &
        (torch.abs(train_y) > 0.0)
    )
    scatter1 = ax.scatter(
        train_x[mask_test, 1], train_y[mask_test],
        c='k', label='Experimental data'
    )

    # Create legend with GP inadequacy term
    viridis = cm.get_cmap('viridis')
    line_color = viridis(0.5)
    line3 = Line2D([0], [0], color=line_color, lw=4, label='GP inadequacy term')
    ax.legend(
        handles=[line1, line2, scatter1, line3],
        frameon=False
    )

    plt.xlabel('Fluence ($10^{26}$ neutron/m$^2$)')
    plt.ylabel('IIDC')
    plt.title(f'{grade} grade Temperature = {temperature:.0f} K')
    plt.ylim([-0.03, 0.12])
    plt.xlim([0, 3.5])
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()


# ============================================================================
# Model Export Functions
# ============================================================================

class MeanVarModelWrapper(torch.nn.Module):
    """
    Wrapper for exporting GP model to TorchScript.

    Converts GP model to a simple forward pass that returns mean predictions.
    """

    def __init__(self, gp):
        """
        Initialize wrapper.

        Args:
            gp: Trained Gaussian Process model.
        """
        super().__init__()
        self.gp = gp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning GP mean predictions.

        Args:
            x: Input features [N, D].

        Returns:
            GP mean predictions [N].
        """
        mean, _ = self.gp(x, full_cov=False)
        return mean


def export_gp_model(
    model: IIDCDeltaL,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mcmc_koh: dict,
    active_dims: list,
    group_code: torch.Tensor,
    save_path: str
) -> None:
    """
    Export GP model as TorchScript for deployment.

    Args:
        model: Fitted IIDCDeltaL model.
        train_x: Training input features [N, 6].
        train_y: Training IIDC values [N].
        mcmc_koh: Dictionary of KOH MCMC samples.
        active_dims: Active dimensions for GP.
        group_code: Data source identifiers [N].
        save_path: Path to save TorchScript model.
    """
    # Normalize features
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)

    # Get parametric predictions
    predictive_koh = Predictive(model.probModel, mcmc_koh)(
        train_x, None, group_code
    )['obs']

    # Initialize and configure GP
    gpr = model.returnGP()
    gpr.set_data(
        train_x_norm[:, active_dims],
        train_y - predictive_koh.mean(dim=0)
    )

    gpr.kernel.lengthscale = mcmc_koh['kernel.lengthscale'].log().mean(dim=0).exp()
    gpr.kernel.variance = mcmc_koh['kernel.variance'].mean(dim=0)
    gpr.noise = mcmc_koh['noise'].log().mean(dim=0).exp()

    # Create example input for tracing
    example_input = torch.randn(10, train_x_norm[0][active_dims].shape[-1])

    # Trace and export model
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
        gpr.eval()
        traced_model = torch.jit.trace(
            MeanVarModelWrapper(gpr),
            example_input,
            check_trace=False
        )

    traced_model.double().save(save_path)


# ============================================================================
# Analysis Functions
# ============================================================================

def find_uturn_and_crossover(
    fluence: torch.Tensor,
    iidc_prediction: torch.Tensor
) -> Tuple[float, float, float, float]:
    """
    Find u-turn point (minimum IIDC) and crossover point (return to zero).

    The u-turn point represents maximum shrinkage before dimensional change
    reverses. The crossover point is where IIDC returns to zero after shrinkage.

    Args:
        fluence: Fluence values [N].
        iidc_prediction: IIDC predictions [N].

    Returns:
        Tuple containing:
            - uturn_fluence: Fluence at u-turn point
            - uturn_iidc: IIDC value at u-turn point
            - crossover_fluence: Fluence at crossover point
            - crossover_iidc: IIDC value at crossover point (near zero)
    """
    # Find u-turn point (minimum)
    uturn_arg = iidc_prediction.argmin()
    uturn_fluence = fluence[uturn_arg].item()
    uturn_iidc = iidc_prediction[uturn_arg].item()

    # Find crossover point (return to zero after u-turn)
    cross_arg = iidc_prediction[uturn_arg:].abs().argmin()
    crossover_fluence = fluence[uturn_arg:][cross_arg].item()
    crossover_iidc = iidc_prediction[uturn_arg:][cross_arg].item()

    return uturn_fluence, uturn_iidc, crossover_fluence, crossover_iidc


def plot_uturn_crossover(
    fluence: torch.Tensor,
    iidc_prediction: torch.Tensor,
    uturn_fluence: float,
    uturn_iidc: float,
    crossover_fluence: float,
    crossover_iidc: float
) -> None:
    """
    Visualize u-turn and crossover points on IIDC vs fluence plot.

    Args:
        fluence: Fluence values [N].
        iidc_prediction: IIDC predictions [N].
        uturn_fluence: Fluence at u-turn point.
        uturn_iidc: IIDC value at u-turn point.
        crossover_fluence: Fluence at crossover point.
        crossover_iidc: IIDC value at crossover point.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(fluence, iidc_prediction, label='KOH model', c='r', lw=2.5)
    plt.scatter(
        [uturn_fluence], [uturn_iidc],
        s=50, c='b', label='U-turn point'
    )
    plt.scatter(
        [crossover_fluence], [crossover_iidc],
        s=50, c='k', label='Crossover point'
    )
    plt.xlabel('Fluence ($10^{26}$ neutron/m$^2$)')
    plt.ylabel('IIDC')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution function for IIDC model calibration.

    Workflow:
        1. Load and preprocess IIDC data
        2. Fit deterministic model
        3. Construct priors and fit Bayesian model
        4. Fit KOH model with GP discrepancy
        5. Generate visualizations
        6. Export GP model
        7. Analyze physical features
    """
    # Step 1: Load and preprocess data
    print(f"Loading IIDC data for {GRAPHITE_GRADE}...")
    train_x_raw, train_y_raw, group_code_raw = load_iidc_data(GRAPHITE_GRADE)

    print(f"Applying zero-fluence padding ({N_PAD} points)...")
    train_x, train_y, group_code = apply_zero_fluence_padding(
        train_x_raw, train_y_raw, group_code_raw, N_PAD
    )

    # Step 2: Fit deterministic model
    print("Fitting deterministic model...")
    starting_point = torch.tensor([1.0, 1.0, 1.0])
    minimized_params, param_history, residuals = fit_deterministic_model(
        train_x, train_y, group_code, ACTIVE_DIMS, starting_point,
        LEARNING_RATE, EPOCHS, LOG_LOC_PARAMS, LOG_SCA_PARAMS
    )

    iidc_pred_determ = IIDCDeltaL(
        [], train_x, train_y,
        active_dims=ACTIVE_DIMS,
        group_code=group_code,
        log_loc_params=LOG_LOC_PARAMS,
        log_sca_params=LOG_SCA_PARAMS
    ).determModel(minimized_params, train_x)

    print(f"Deterministic error: {torch.norm(residuals, p='fro'):.4f}")

    # Step 3: Construct priors and fit Bayesian model
    print("Constructing priors for INL model...")
    priors = construct_inl_priors(minimized_params, residuals)

    print("Fitting Bayesian model...")
    model_bayes, mcmc = fit_bayesian_model(
        train_x, train_y, group_code, ACTIVE_DIMS, priors,
        NUM_WARMUP, NUM_SAMPLES, LOG_LOC_PARAMS, LOG_SCA_PARAMS
    )

    predictive_bayes = Predictive(model_bayes.probModel, mcmc)(
        train_x, None, group_code
    )['obs']

    # Step 4: Fit KOH model
    print("Fitting KOH model with GP discrepancy...")
    mcmc_koh = fit_koh_model(model_bayes, NUM_WARMUP, NUM_SAMPLES)

    # Step 5: Compute GP predictions
    print("Computing GP predictions...")
    gp_mean, gp_var, entropy = compute_gp_predictions(
        model_bayes, train_x, train_y, mcmc_koh, ACTIVE_DIMS, group_code
    )

    predictive_koh = Predictive(model_bayes.probModel, mcmc_koh)(
        train_x, None, group_code
    )['obs']

    # Step 6: Generate visualizations
    print("Generating training scatter plots...")
    results_dir = f'graphiteModels/Results/AllProps{GRAPHITE_GRADE}/{FOLDER_NAME}'

    plot_training_scatter(
        train_y, iidc_pred_determ, predictive_bayes.mean(dim=0),
        predictive_koh.mean(dim=0), gp_mean, group_code,
        f'{results_dir}/{MODEL_NAME}_scatter.pdf'
    )

    plot_gp_scatter_with_entropy(
        train_y, predictive_koh.mean(dim=0), gp_mean, entropy, group_code,
        f'{results_dir}/{MODEL_NAME}_GP_scatter.pdf'
    )

    # Step 7: Generate test predictions
    print("Generating test predictions...")
    test_x = torch.zeros((400, 6))
    test_x[:, 1] = torch.linspace(0.0, 4.0, 400)
    test_x[:, 2] = 1102.23  # Temperature (K)
    test_x[:, 3] = 2.5  # Interconnectivity
    test_x[:, 4] = 0.95  # Porosity

    test_x = test_x.double()
    test_x_norm = (test_x - train_x.mean(dim=0)) / train_x.std(dim=0)
    train_x_norm = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)

    # Compute test predictions
    gpr = model_bayes.returnGP()
    gpr.set_data(
        train_x_norm[:, ACTIVE_DIMS],
        train_y - predictive_koh.mean(dim=0)
    )

    gpr.kernel.lengthscale = mcmc_koh['kernel.lengthscale'].log().mean(dim=0).exp()
    gpr.kernel.variance = mcmc_koh['kernel.variance'].mean(dim=0)
    gpr.noise = mcmc_koh['noise'].log().mean(dim=0).exp()

    traced_mean, _ = gpr(test_x_norm[:, ACTIVE_DIMS], full_cov=False)

    # Create modified MCMC samples for test predictions
    group_code_test = torch.hstack((
        torch.zeros(100).int(),
        torch.ones(300).int()
    ))

    mcmc_koh_mod = {
        'a': torch.ones(400) * mcmc_koh['a'].mean(),
        'b': torch.ones(400) * mcmc_koh['b'].mean(),
        'Ea': torch.ones(400) * mcmc_koh['Ea'].mean(),
        'log_sigma': 10.0 * torch.ones(400, 2) * mcmc_koh['log_sigma'].mean(dim=0),
        'mu_log_sigma': torch.ones(400) * mcmc_koh['mu_log_sigma'].mean(),
        'sig_log_sigma': torch.ones(400) * mcmc_koh['sig_log_sigma'].mean()
    }

    mcmc_mod = {
        'a': torch.ones(400) * mcmc['a'].mean(),
        'b': torch.ones(400) * mcmc['b'].mean(),
        'Ea': torch.ones(400) * mcmc['Ea'].mean(),
        'log_sigma': 10.0 * torch.ones(400, 2) * mcmc_koh['log_sigma'].mean(dim=0),
        'mu_log_sigma': torch.ones(400) * mcmc['mu_log_sigma'].mean(),
        'sig_log_sigma': torch.ones(400) * mcmc['sig_log_sigma'].mean()
    }

    predictive_test_koh = Predictive(model_bayes.probModel, mcmc_koh_mod)(
        test_x, None, group_code_test
    )['obs']

    predictive_test_bayes = Predictive(model_bayes.probModel, mcmc_mod)(
        test_x, None, group_code_test
    )['obs']

    # Compute test entropy (simplified)
    test_entropy = torch.linspace(0, 1, 400)  # Placeholder

    plot_fluence_predictions(
        test_x, train_x, train_y,
        predictive_test_bayes, predictive_test_koh, traced_mean,
        test_entropy, 1000.0, 40.0,
        f'{results_dir}/{MODEL_NAME}_Fluence_1000.pdf',
        'IG-110'
    )

    # Step 8: Export GP model
    print("Exporting GP model...")
    export_gp_model(
        model_bayes, train_x, train_y, mcmc_koh, ACTIVE_DIMS, group_code,
        f'{results_dir}/gp_{MODEL_NAME}_IIDC.pt'
    )

    # Step 9: Analyze physical features
    print("Analyzing u-turn and crossover points...")
    koh_full_pred = predictive_test_koh.mean(dim=0).detach() + traced_mean.detach()

    uturn_fluence, uturn_iidc, cross_fluence, cross_iidc = find_uturn_and_crossover(
        test_x[:, 1], koh_full_pred
    )

    print(f"U-turn point: fluence = {uturn_fluence:.3f}, IIDC = {uturn_iidc:.4f}")
    print(f"Crossover point: fluence = {cross_fluence:.3f}, IIDC = {cross_iidc:.4f}")

    print("IIDC calibration complete!")


if __name__ == "__main__":
    main()
