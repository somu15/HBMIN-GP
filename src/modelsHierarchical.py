"""
Hierarchical Bayesian models for graphite material behavior under irradiation.

This module provides a comprehensive framework for modeling various physical properties
of graphite materials subjected to neutron irradiation, including dimensional changes,
creep behavior, thermal expansion, elastic modulus, and thermal conductivity.

The models support both hierarchical and non-hierarchical Bayesian inference using
Pyro's probabilistic programming framework, with Gaussian process regression for
capturing complex dependencies.
"""

from typing import List, Tuple, Optional, Union
import math

import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.infer import Predictive
from torchquad import MonteCarlo, set_up_backend, Simpson
from gpytorch.means import Mean

# Physical constants
BOLTZMANN_CONSTANT_EV_K = 8.617333262e-5  # Boltzmann constant in eV/K
INTEGRATION_STEPS = 101  # Number of integration points for trapezoidal rule

class CustomKernel(gp.kernels.Kernel):
    """
    Custom Radial Basis Function (RBF) kernel with boundary constraints and noise.

    This kernel extends the standard RBF kernel by applying boundary constraints
    to enforce zero covariance when either input point has zero components, and
    includes observation noise on the diagonal.

    Attributes:
        input_dim (int): Dimensionality of the input space.
        variance (torch.Tensor): Signal variance parameter.
        lengthscale (torch.Tensor): Length scale parameters for each dimension.
        noise (torch.Tensor): Observation noise parameter.
    """

    def __init__(
        self,
        input_dim: int,
        variance: torch.Tensor = torch.tensor(1.0),
        lengthscale: Optional[torch.Tensor] = None,
        noise: torch.Tensor = torch.tensor(1.0)
    ) -> None:
        """
        Initialize the CustomKernel.

        Args:
            input_dim: Dimensionality of the input space.
            variance: Initial signal variance (default: 1.0).
            lengthscale: Initial length scale for each dimension (default: ones).
            noise: Initial observation noise (default: 1.0).
        """
        super(CustomKernel, self).__init__(input_dim)
        self.input_dim = input_dim
        self.variance = pyro.param("variance", variance, constraint=dist.constraints.positive)
        if lengthscale is None:
            lengthscale = torch.ones(input_dim)
        self.lengthscale = pyro.param("lengthscale", lengthscale, constraint=dist.constraints.positive)
        self.noise = pyro.param("noise", noise, constraint=dist.constraints.positive)

    def forward(
        self,
        X: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
        diag: bool = False
    ) -> torch.Tensor:
        """
        Compute the kernel matrix between input points.

        Args:
            X: First set of input points, shape (n, input_dim).
            X2: Second set of input points, shape (m, input_dim). If None, X2 = X.
            diag: If True, return only the diagonal of the kernel matrix.

        Returns:
            Kernel matrix of shape (n, m) or (n,) if diag=True.
        """
        if X2 is None:
            X2 = X
        if diag:
            return self.variance * torch.ones(X.shape[0])

        # Compute squared distance with automatic relevance determination (ARD)
        sqdist = torch.zeros(X.shape[0], X2.shape[0])
        for i in range(self.input_dim):
            sqdist += (X[:, i].reshape(-1, 1) - X2[:, i].reshape(1, -1)) ** 2 / self.lengthscale[i] ** 2
        K = self.variance * torch.exp(-0.5 * sqdist)

        # Apply boundary constraint: zero covariance if any component is zero
        X_zero_constraint = torch.prod((X != 0).float(), dim=1).reshape(-1, 1)
        X2_zero_constraint = torch.prod((X2 != 0).float(), dim=1).reshape(1, -1)
        K *= X_zero_constraint * X2_zero_constraint

        # Add observation noise to the diagonal
        if X2 is X:
            K += self.noise * torch.eye(X.shape[0])

        return K

class Base:
    """
    Base class for hierarchical Bayesian models with Gaussian process regression.

    This abstract base class provides the foundational structure for implementing
    hierarchical Bayesian models combined with Gaussian process regression for
    modeling graphite material properties under irradiation.

    Attributes:
        distributions (List): Prior distributions for model parameters.
        train_x (torch.Tensor): Training input features.
        train_y (torch.Tensor): Training target values.
        active_dims (List[int]): Indices of active dimensions for GP kernel.
        group_code (torch.Tensor): Group identifiers for hierarchical modeling.
        train_x_norm (torch.Tensor): Standardized training inputs.
        kernel (gp.kernels.Kernel): Gaussian process kernel.
        gpr (gp.models.GPRegression): Gaussian process regression model.
        mcmcSamples (dict): Stored MCMC posterior samples.
    """

    def __init__(
        self,
        distributions: List,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        active_dims: List[int],
        group_code: torch.Tensor
    ) -> None:
        """
        Initialize the base model.

        Args:
            distributions: List of prior distributions for model parameters.
            train_x: Training input features, shape (n_samples, n_features).
            train_y: Training target values, shape (n_samples,).
            active_dims: Indices of dimensions to use in the GP kernel.
            group_code: Group identifiers for hierarchical structure, shape (n_samples,).
        """
        self.distributions = distributions
        self.train_x = train_x
        self.active_dims = active_dims
        self.train_x_norm = (self.train_x - self.train_x.mean(dim=0)) / self.train_x.std(dim=0)
        self.train_y = train_y
        self.group_code = group_code

        # Initialize RBF kernel for Gaussian process
        self.kernel = gp.kernels.RBF(
            input_dim=len(self.active_dims),
            variance=torch.tensor(1.0),
            lengthscale=torch.ones(len(self.active_dims))
        )

        # Initialize Gaussian process regression model
        self.gpr = gp.models.GPRegression(
            self.train_x_norm[:, self.active_dims],
            torch.zeros(self.train_x.shape[0]),
            self.kernel,
            noise=torch.tensor(0.1)
        )

        # Set priors for GP hyperparameters
        self.gpr.kernel.set_prior(
            "lengthscale",
            dist.LogNormal(torch.zeros(len(self.active_dims)), torch.ones(len(self.active_dims)))
        )
        self.gpr.kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.gpr.set_prior("noise", dist.Uniform(5e-7, 0.5))

    def kohModel(self, x: torch.Tensor, y: torch.Tensor, group_code: torch.Tensor) -> None:
        """
        Kennedy-O'Hagan (KOH) hierarchical model with GP discrepancy.

        Combines parametric model predictions with Gaussian process discrepancy
        and hierarchical observation noise.

        Args:
            x: Input features, shape (n_samples, n_features).
            y: Observed target values, shape (n_samples,).
            group_code: Group identifiers, shape (n_samples,).
        """
        m, log_sigma = self.hierarchicalModel(x, group_code)
        self.gpr.set_data(self.train_x_norm[:, self.active_dims], None)
        f, f_var = self.gpr.model()
        with pyro.plate("data", len(group_code)):
            pyro.sample(
                "obs",
                dist.Normal(m + f, (f_var + torch.exp(log_sigma[group_code])**2).sqrt()),
                obs=y
            )

    def kohModelNH(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Kennedy-O'Hagan (KOH) non-hierarchical model with GP discrepancy.

        Non-hierarchical version with single observation noise parameter.

        Args:
            x: Input features, shape (n_samples, n_features).
            y: Observed target values, shape (n_samples,).

        Returns:
            Sample from the observation distribution.
        """
        m = self.model(x)
        self.gpr.set_data(self.train_x_norm[:, self.active_dims], None)
        f, f_var = self.gpr.model()
        log_sigma = pyro.sample("log_sigma", self.distributions[-1])
        return pyro.sample("y", dist.Normal(m + f, (f_var + torch.exp(log_sigma)**2).sqrt()), obs=y)

    def model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-hierarchical parametric model (to be implemented by subclasses).

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        raise NotImplementedError("Subclasses must implement the model method")

    def probModel(self, x: torch.Tensor, y: torch.Tensor, group_code: torch.Tensor) -> None:
        """
        Probabilistic hierarchical model without GP discrepancy.

        Args:
            x: Input features, shape (n_samples, n_features).
            y: Observed target values, shape (n_samples,).
            group_code: Group identifiers, shape (n_samples,).
        """
        m, log_sigma = self.hierarchicalModel(x, group_code)
        with pyro.plate("data", len(group_code)):
            pyro.sample("obs", dist.Normal(m, torch.exp(log_sigma[group_code])), obs=y)

    def probModelNH(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Probabilistic non-hierarchical model without GP discrepancy.

        Args:
            x: Input features, shape (n_samples, n_features).
            y: Observed target values, shape (n_samples,).

        Returns:
            Sample from the observation distribution.
        """
        m = self.model(x)
        log_sigma = pyro.sample("log_sigma", self.distributions[-1])
        return pyro.sample("y", dist.Normal(m, torch.exp(log_sigma)), obs=y)

    def returnGP(self) -> gp.models.GPRegression:
        """
        Return the Gaussian process regression model.

        Returns:
            The GP regression model instance.
        """
        return self.gpr

    def MCMCSamples(
        self,
        model: callable,
        warmup_steps: int,
        num_samples: int,
        HFlag: bool,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        step_size: float = 1e-3
    ) -> dict:
        """
        Generate MCMC samples using No-U-Turn Sampler (NUTS).

        Args:
            model: Probabilistic model to sample from.
            warmup_steps: Number of warmup iterations.
            num_samples: Number of posterior samples to generate.
            HFlag: If True, use hierarchical model with group codes.
            adapt_step_size: Whether to adapt the NUTS step size.
            adapt_mass_matrix: Whether to adapt the mass matrix.
            target_accept_prob: Target acceptance probability for NUTS.
            max_tree_depth: Maximum tree depth for NUTS.
            step_size: Initial step size.

        Returns:
            Dictionary of posterior samples for each parameter.
        """
        nuts_kernel = NUTS(
            model,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            step_size=step_size
        )
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)

        if HFlag:
            mcmc_run.run(self.train_x, self.train_y, self.group_code)
        else:
            mcmc_run.run(self.train_x, self.train_y)

        self.mcmcSamples = mcmc_run.get_samples()
        return self.mcmcSamples

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic model prediction with fixed parameters (to be implemented by subclasses).

        Args:
            params: Model parameters.
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        raise NotImplementedError("Subclasses must implement the determModel method")
    def unirradiatedModulus(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate unirradiated elastic modulus as a function of temperature.

        Empirical quadratic relationship for baseline elastic modulus.

        Args:
            temperature: Temperature values in appropriate units.

        Returns:
            Unirradiated elastic modulus values.
        """
        return 9.98 - 8.0e-4 * temperature + 2.0e-6 * torch.square(temperature)

    def epsilonV(self, volume_change: torch.Tensor) -> torch.Tensor:
        """
        Compute volumetric strain from volume change.

        Args:
            volume_change: Fractional volume change.

        Returns:
            Volumetric strain (epsilon_V).
        """
        return volume_change

    def structuralConnectivity(
        self,
        dose: torch.Tensor,
        mean_dose: torch.Tensor,
        std_dose: torch.Tensor,
        flag: int = 0
    ) -> torch.Tensor:
        """
        Compute structural connectivity function based on dose statistics.

        Uses error function (erf) to model the evolution of structural connectivity
        as a function of dose, characterized by mean and standard deviation.

        Args:
            dose: Dose values, shape depends on flag.
            mean_dose: Mean dose parameter, shape (n_samples,) or scalar.
            std_dose: Standard deviation of dose, shape (n_samples,) or scalar.
            flag: Shape handling flag (0 for batched integration, 1 for direct computation).

        Returns:
            Structural connectivity values between 0 and 1.
        """
        if flag == 0:
            # For integration over dose: dose shape (n_samples, n_integration_points)
            sc = 0.5 * (1.0 + torch.erf(
                (dose - mean_dose.unsqueeze(-1)) / (std_dose.unsqueeze(-1) * math.sqrt(2))
            ))
        else:
            # For direct computation: dose shape (n_samples,)
            sc = 0.5 * (1.0 + torch.erf((dose - mean_dose) / (std_dose * math.sqrt(2))))
        return sc

    def structuralDamage(
        self,
        params: Union[List, torch.Tensor],
        dose: torch.Tensor,
        mean_dose: torch.Tensor,
        std_dose: torch.Tensor,
        volume_change: torch.Tensor,
        flag: int
    ) -> torch.Tensor:
        """
        Compute structural damage factor combining connectivity and volumetric strain effects.

        The structural damage factor accounts for both the evolution of structural
        connectivity with dose and the influence of volumetric strain.

        Args:
            params: Model parameters [connectivity_coeff, strain_coeff].
            dose: Dose values.
            mean_dose: Mean dose for connectivity calculation.
            std_dose: Standard deviation of dose for connectivity calculation.
            volume_change: Volumetric strain.
            flag: Shape handling flag for connectivity computation.

        Returns:
            Structural damage factor.
        """
        sc_value = self.structuralConnectivity(dose, mean_dose, std_dose, flag)

        if flag == 0:
            epsilon = self.epsilonV(volume_change).unsqueeze(-1)
        else:
            epsilon = self.epsilonV(volume_change)

        return (1.0 + params[0] * sc_value) * torch.exp(-params[1] * epsilon)

class IIDCDeltaL(Base):
    """
    IIDC (Irradiation-Induced Dimensional Change) model for axial strain.

    Models dimensional change using a quadratic dose term and a temperature-dependent
    recovery term with activation energy.

    Model equation:
        ΔL/L = a·γ² - b·exp(-Ea/(k_B·T))·γ

    where γ is dose, T is temperature, Ea is activation energy, and k_B is Boltzmann constant.
    """

    def hierarchicalModel(
        self,
        x: torch.Tensor,
        group_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical model with group-level observation noise.

        Args:
            x: Input features [stress, dose, temperature, ...], shape (n_samples, n_features).
            group_code: Group identifiers, shape (n_samples,).

        Returns:
            Tuple of (predictions, log_sigma) where:
                - predictions: shape (n_samples,)
                - log_sigma: shape (n_groups,)
        """
        n_groups = len(group_code.unique())

        # Hierarchical prior for observation noise
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))

        # Sample population-level parameters
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        Ea = pyro.sample("Ea", self.distributions[2])

        # Sample group-level noise parameters
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))

        # Extract features
        gamma = x[:, 1]
        temperature = x[:, 2]

        # Compute model predictions
        m = a * gamma ** 2 - b * torch.exp(-Ea / (BOLTZMANN_CONSTANT_EV_K * temperature)) * gamma
        return m, log_sigma

    def model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-hierarchical model.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        Ea = pyro.sample("Ea", self.distributions[2])

        gamma = x[:, 1]
        temperature = x[:, 2]

        m = a * gamma ** 2 - b * torch.exp(-Ea / (BOLTZMANN_CONSTANT_EV_K * temperature)) * gamma
        return m

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic model with fixed parameters.

        Args:
            params: Model parameters [a, b, Ea].
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        a, b, Ea = params[0], params[1], params[2]

        gamma = x[:, 1]
        temperature = x[:, 2]

        m = a * gamma ** 2 - b * torch.exp(-Ea / (BOLTZMANN_CONSTANT_EV_K * temperature)) * gamma
        return m
    
class ShibataIIDCDeltaL(Base):
    """
    Shibata empirical model for dimensional change.

    Implements temperature-dependent piecewise linear model with quadratic dose
    dependence based on Shibata's empirical correlations for graphite dimensional change.

    Temperature ranges:
        - T < 400°C
        - 400°C ≤ T < 600°C
        - 600°C ≤ T < 800°C
        - T ≥ 800°C
    """

    def hierarchicalModel(
        self,
        x: torch.Tensor,
        group_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical Shibata model with temperature-dependent coefficients.

        Args:
            x: Input features [stress, dose, temperature, ...], shape (n_samples, n_features).
            group_code: Group identifiers, shape (n_samples,).

        Returns:
            Tuple of (predictions, log_sigma).
        """
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))

        gamma = x[:, 1]
        T = x[:, 2] - 273.15
        result = self._compute_shibata_strain(gamma, T)

        return result, log_sigma

    def model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-hierarchical Shibata model.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        gamma = x[:, 1]
        T = x[:, 2] - 273.15
        return self._compute_shibata_strain(gamma, T)

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic Shibata model (parameters not used).

        Args:
            params: Model parameters (not used in this empirical model).
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        gamma = x[:, 1]
        T = x[:, 2] - 273.15
        return self._compute_shibata_strain(gamma, T)

    def _compute_shibata_strain(self, gamma: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute dimensional change using Shibata's piecewise temperature model.

        Args:
            gamma: Neutron dose values.
            T: Temperature in Celsius.

        Returns:
            Dimensional change (ΔL/L).
        """
        result = torch.zeros_like(gamma)

        condition1 = T < 400.0
        condition2 = (T >= 400.0) & (T < 600.0)
        condition3 = (T >= 600.0) & (T < 800.0)
        condition4 = T >= 800.0

        # T < 400°C
        result[condition1] = (0.279 * gamma[condition1] ** 2 - 1.64 * gamma[condition1]) / 100.0

        # 400°C ≤ T < 600°C (linear interpolation)
        result[condition2] = (
            ((0.279 * gamma[condition2] ** 2 - 1.64 * gamma[condition2]) * (600.0 - T[condition2]) +
             (0.450 * gamma[condition2] ** 2 - 1.86 * gamma[condition2]) * (T[condition2] - 400.0)) / 200.0
        ) / 100.0

        # 600°C ≤ T < 800°C (linear interpolation)
        result[condition3] = (
            ((0.450 * gamma[condition3] ** 2 - 1.86 * gamma[condition3]) * (800.0 - T[condition3]) +
             (0.821 * gamma[condition3] ** 2 - 2.19 * gamma[condition3]) * (T[condition3] - 600.0)) / 200.0
        ) / 100.0

        # T ≥ 800°C
        result[condition4] = (0.821 * gamma[condition4] ** 2 - 2.19 * gamma[condition4]) / 100.0

        return result

class BradfordPhysicsDeltaL(Base):
    """
    Bradford physics-based model for dimensional change with structural connectivity.

    Integrates dose-dependent strain with structural connectivity effects using
    an integral formulation over the dose history.

    Model incorporates:
        - Saturation behavior: (1 - exp(-b·dose))
        - Structural connectivity effects
        - Dose history integration
    """

    def func(
        self,
        dosage: torch.Tensor,
        X: torch.Tensor,
        params: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute integrand for dimensional change at given dosage.

        Args:
            dosage: Dose values for integration, shape (n_samples, n_points).
            X: Input features including mean and std dose.
            params: Model parameters [a, b, c].

        Returns:
            Integrand values, shape (n_samples, n_points).
        """
        Sc = 0.5 * (1.0 + torch.erf(
            (dosage - X[:, 2].unsqueeze(-1)) / (X[:, 3].unsqueeze(-1) * math.sqrt(2))
        ))
        pred = params[0] * (1.0 - torch.exp(-params[1] * dosage)) * (1.0 + params[2] * Sc)
        return pred
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
        dosages = dosages.unsqueeze(0).repeat(x.shape[0], 1) * x[:, 0].unsqueeze(-1)
        integrands = self.func(dosages, x, [a, b, c])
        preds = torch.trapezoid(integrands)*dosages[:,1]
        return preds, log_sigma
    def model(self, x1):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
        dosages = dosages.unsqueeze(0).repeat(x1.shape[0], 1) * x1[:, 0].unsqueeze(-1)
        integrands = self.func(dosages, x1, [a, b, c])
        preds = torch.trapezoid(integrands)*dosages[:,1]
        return preds
    def determModel(self, params, x1):
        dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
        dosages = dosages.unsqueeze(0).repeat(x1.shape[0], 1) * x1[:, 0].unsqueeze(-1)
        integrands = self.func(dosages, x1, params)
        preds = torch.trapezoid(integrands)*dosages[:,1]
        return preds

class IIDCCreep(Base):
    """
    IIDC model for irradiation creep behavior.

    Combines elastic strain with temperature-dependent irradiation creep using
    an empirical Young's modulus formulation.

    Model equation:
        ε_creep = ε_elastic + a·exp(-b·T)·σ·γ

    where:
        ε_elastic = σ/E(T)
        E(T) is temperature-dependent Young's modulus
        σ is stress, γ is dose, T is temperature
    """

    def hierarchicalModel(
        self,
        x: torch.Tensor,
        group_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical creep model with group-level noise.

        Args:
            x: Input features [stress, dose, temperature, ...], shape (n_samples, n_features).
            group_code: Group identifiers, shape (n_samples,).

        Returns:
            Tuple of (predictions, log_sigma).
        """
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))

        y_modulus = 4.5680e-02 + 1.9506e-02 * x[:, 2] - 1.0055e-05 * x[:, 2] ** 2
        m = 1e-3 * x[:, 0] / y_modulus + a * torch.exp(-b * x[:, 2]) * x[:, 0] * x[:, 1]
        return m, log_sigma

    def model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-hierarchical creep model.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        y_modulus = 4.5680e-02 + 1.9506e-02 * x[:, 2] - 1.0055e-05 * x[:, 2] ** 2
        m = 1e-3 * x[:, 0] / y_modulus + a * torch.exp(-b * x[:, 2]) * x[:, 0] * x[:, 1]
        return m

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic creep model with fixed parameters.

        Args:
            params: Model parameters [a, b].
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        a, b = params[0], params[1]
        y_modulus = 4.5680e-02 + 1.9506e-02 * x[:, 2] - 1.0055e-05 * x[:, 2] ** 2
        m = 1e-3 * x[:, 0] / y_modulus + a * torch.exp(-b * x[:, 2]) * x[:, 0] * x[:, 1]
        return m

    def cteCreep(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute creep with constant Young's modulus (for CTE calculations).

        Args:
            params: Model parameters [a, b].
            x: Input features, shape (n_samples, n_features).

        Returns:
            Creep strain predictions, shape (n_samples,).
        """
        a, b = params[0], params[1]
        y_modulus = 9.3730  # Constant modulus
        m = 1e-3 * x[:, 0] / y_modulus + a * torch.exp(-b * x[:, 2]) * x[:, 0] * x[:, 1]
        return m

class ShibataCreep(Base):
    """
    Shibata empirical model for irradiation creep.

    Uses fixed empirical coefficients from Shibata's correlations for
    Japanese graphite grades.

    Model equation:
        ε_creep = -σ/(1000·E(T)) + a·exp(b·T)·σ·γ

    where E(T) is temperature-dependent Young's modulus.

    Note:
        Parameters a and b are fixed empirical constants, not fitted.
    """

    def hierarchicalModel(self, x: torch.Tensor, group_code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = 7.163e-4
        b = 1.2e-3
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        temp = x[:,2] - 273.15
        yModulus = 9.98 - 8e-4 * temp + 2e-6 * temp * temp
        m = -1e-3 * x[:,0] / yModulus + a * torch.exp(b * temp) * x[:,0] * x[:,1]
        return m, log_sigma
    def model(self, x):
        a = 7.163e-4
        b = 1.2e-3
        temp = x[:,2] - 273.15
        yModulus = 9.98 - 8e-4 * temp + 2e-6 * temp * temp
        m = -1e-3 * x[:,0] / yModulus + a * torch.exp(b * temp) * x[:,0] * x[:,1]
        return m
    def determModel(self, params, x):
        a = 7.163e-4
        b = 1.2e-3
        temp = x[:,2] - 273.15
        yModulus = 9.98 - 8e-4 * temp + 2e-6 * temp * temp
        m = -1e-3 * x[:,0] / yModulus + a * torch.exp(b * temp) * x[:,0] * x[:,1]
        return m

# class BradfordPhysicsCTE(Base):
#     def func(self, dosage, cte_0, X, params):
#         Sc = 0.5 * (1.0 + torch.erf((dosage - X[:, 2].unsqueeze(-1)) / (X[:, 3].unsqueeze(-1) * math.sqrt(2)) ))
#         pred = (1.0 - params[0] * Sc) * (1 + params[1] * (1.0 - torch.exp(- params[2] * dosage))) * cte_0 * params[3]  # Params[3] is the creep term.
#         return pred
#     def model(self, x1):
#         a = pyro.sample("a", self.distributions[0])
#         b = pyro.sample("b", self.distributions[1])
#         c = pyro.sample("c", self.distributions[2])
#         d = pyro.sample("d", self.distributions[3])
#         dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
#         dosages = dosages.unsqueeze(0).repeat(x1.shape[0], 1) * x1[:, 0].unsqueeze(-1)
#         cte_0 = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
#         cte_0 = dosages.unsqueeze(0).repeat(x1.shape[0], 1) * x1[:, 1].unsqueeze(-1)
#         integrands = self.func(dosages, x1, cte_0, [a, b, c, d])
#         preds = torch.trapezoid(integrands)*dosages[:,1]
#         return preds
#     def determModel(self, params, x1):
#         dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
#         dosages = dosages.unsqueeze(0).repeat(x1.shape[0], 1) * x1[:, 0].unsqueeze(-1)
#         cte_0 = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
#         cte_0 = dosages.unsqueeze(0).repeat(x1.shape[0], 1) * x1[:, 1].unsqueeze(-1)
#         integrands = self.func(dosages, cte_0, x1, params)
#         preds = torch.trapezoid(integrands)*dosages[:,1]
#         return preds

class ShibataCTE(Base):
    """
    Shibata empirical model for coefficient of thermal expansion under irradiation.

    Complex piecewise model with:
        - Temperature-dependent behavior (4 temperature regimes)
        - Dose-dependent behavior with turnaround dose
        - Cubic polynomial for pre-turnaround regime
        - Linear for post-turnaround regime

    The model captures the characteristic CTE evolution including initial increase,
    peak, and subsequent decrease with increasing dose.

    Temperature ranges:
        - T < 400°C
        - 400°C ≤ T < 600°C
        - 600°C ≤ T < 800°C
        - T ≥ 800°C
    """

    def hierarchicalModel(self, x: torch.Tensor, group_code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma_ta= 1.0
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        gamma_ta = 1.0
        gamma = x[:, 1]
        T = x[:, 2] - 273.15
        result = torch.zeros_like(gamma)

        coeffs_f1 = {
            'T1': [1.33e-1, -6.83e-1, 8.0e-1, 1],
            'T2': [2.64e-1, -9.93e-1, 7.96e-1, 1],
            'T3': [6.28e-1, -1.57, -1.036e-1, 1]
        }

        coeffs_f2 = {
            'T1': [-1.03e-1, 1.07],
            'T2': [-1.34e-1, 1.01],
            'T3': [-1.64e-1, 8.96e-1],
        }

        condition1 = (T < 400.0) & (gamma <= gamma_ta)
        condition2 = (T >= 400.0) & (T < 600.0) & (gamma <= gamma_ta)
        condition3 = (T >= 600.0) & (T < 800.0) & (gamma <= gamma_ta)
        condition4 = (T >= 800.0) & (gamma <= gamma_ta)
        condition5 = (T < 400.0) & (gamma > gamma_ta)
        condition6 = (T >= 400.0) & (T < 600.0) & (gamma > gamma_ta)
        condition7 = (T >= 600.0) & (T < 800.0) & (gamma > gamma_ta)
        condition8 = (T >= 800.0) & (gamma > gamma_ta)

        f1 = lambda gamma, a, b, c, d: a * gamma ** 3 + b * gamma ** 2 + c * gamma + d
        f2 = lambda gamma, a, b: a * gamma + b

        result[condition1] = f1(gamma[condition1], *coeffs_f1['T1'])
        result[condition2] = (f1(gamma[condition2], *coeffs_f1['T2']) * (600.0 - T[condition2]) + f1(gamma[condition2], *coeffs_f1['T1']) * (T[condition2] - 400.0)) / 200.00
        result[condition3] = (f1(gamma[condition3], *coeffs_f1['T3']) * (800.0 - T[condition3]) + f1(gamma[condition3], *coeffs_f1['T2']) * (T[condition3] - 600.0)) / 200.00
        result[condition4] = f1(gamma[condition4], *coeffs_f1['T3'])
        result[condition5] = f2(gamma[condition5], *coeffs_f2['T1'])
        result[condition6] = (f2(gamma[condition6], *coeffs_f2['T2']) * (600.0 - T[condition6]) + f2(gamma[condition6], *coeffs_f2['T1']) * (T[condition6] - 400.0)) / 200.00
        result[condition7] = (f2(gamma[condition7], *coeffs_f2['T3']) * (800.0 - T[condition7]) + f2(gamma[condition7], *coeffs_f2['T2']) * (T[condition7] - 600.0)) / 200.00
        result[condition8] = f2(gamma[condition8], *coeffs_f2['T3'])
        return result - 1.0, log_sigma
    def model(self, x): # We will have to check what the Turnaround dose is.
        gamma_ta = 1.0
        gamma = x[:, 1]
        T = x[:, 2] - 273.15
        result = torch.zeros_like(gamma)

        coeffs_f1 = {
            'T1': [1.33e-1, -6.83e-1, 8.0e-1, 1],
            'T2': [2.64e-1, -9.93e-1, 7.96e-1, 1],
            'T3': [6.28e-1, -1.57, -1.036e-1, 1]
        }

        coeffs_f2 = {
            'T1': [-1.03e-1, 1.07],
            'T2': [-1.34e-1, 1.01],
            'T3': [-1.64e-1, 8.96e-1],
        }

        condition1 = (T < 400.0) & (gamma <= gamma_ta)
        condition2 = (T >= 400.0) & (T < 600.0) & (gamma <= gamma_ta)
        condition3 = (T >= 600.0) & (T < 800.0) & (gamma <= gamma_ta)
        condition4 = (T >= 800.0) & (gamma <= gamma_ta)
        condition5 = (T < 400.0) & (gamma > gamma_ta)
        condition6 = (T >= 400.0) & (T < 600.0) & (gamma > gamma_ta)
        condition7 = (T >= 600.0) & (T < 800.0) & (gamma > gamma_ta)
        condition8 = (T >= 800.0) & (gamma > gamma_ta)

        f1 = lambda gamma, a, b, c, d: a * gamma ** 3 + b * gamma ** 2 + c * gamma + d
        f2 = lambda gamma, a, b: a * gamma + b

        result[condition1] = f1(gamma[condition1], *coeffs_f1['T1'])
        result[condition2] = (f1(gamma[condition2], *coeffs_f1['T2']) * (600.0 - T[condition2]) + f1(gamma[condition2], *coeffs_f1['T1']) * (T[condition2] - 400.0)) / 200.00
        result[condition3] = (f1(gamma[condition3], *coeffs_f1['T3']) * (800.0 - T[condition3]) + f1(gamma[condition3], *coeffs_f1['T2']) * (T[condition3] - 600.0)) / 200.00
        result[condition4] = f1(gamma[condition4], *coeffs_f1['T3'])
        result[condition5] = f2(gamma[condition5], *coeffs_f2['T1'])
        result[condition6] = (f2(gamma[condition6], *coeffs_f2['T2']) * (600.0 - T[condition6]) + f2(gamma[condition6], *coeffs_f2['T1']) * (T[condition6] - 400.0)) / 200.00
        result[condition7] = (f2(gamma[condition7], *coeffs_f2['T3']) * (800.0 - T[condition7]) + f2(gamma[condition7], *coeffs_f2['T2']) * (T[condition7] - 600.0)) / 200.00
        result[condition8] = f2(gamma[condition8], *coeffs_f2['T3'])
        return result - 1.0
    def determModel(self, params, x): # We will have to check what the Turnaround dose is.
        gamma_ta = 1.0
        gamma = x[:, 1]
        T = x[:, 2] - 273.15
        result = torch.zeros_like(gamma)

        coeffs_f1 = {
            'T1': [1.33e-1, -6.83e-1, 8.0e-1, 1],
            'T2': [2.64e-1, -9.93e-1, 7.96e-1, 1],
            'T3': [6.28e-1, -1.57, -1.036e-1, 1]
        }

        coeffs_f2 = {
            'T1': [-1.03e-1, 1.07],
            'T2': [-1.34e-1, 1.01],
            'T3': [-1.64e-1, 8.96e-1],
        }

        condition1 = (T < 400.0) & (gamma <= gamma_ta)
        condition2 = (T >= 400.0) & (T < 600.0) & (gamma <= gamma_ta)
        condition3 = (T >= 600.0) & (T < 800.0) & (gamma <= gamma_ta)
        condition4 = (T >= 800.0) & (gamma <= gamma_ta)
        condition5 = (T < 400.0) & (gamma > gamma_ta)
        condition6 = (T >= 400.0) & (T < 600.0) & (gamma > gamma_ta)
        condition7 = (T >= 600.0) & (T < 800.0) & (gamma > gamma_ta)
        condition8 = (T >= 800.0) & (gamma > gamma_ta)

        f1 = lambda gamma, a, b, c, d: a * gamma ** 3 + b * gamma ** 2 + c * gamma + d
        f2 = lambda gamma, a, b: a * gamma + b

        result[condition1] = f1(gamma[condition1], *coeffs_f1['T1'])
        result[condition2] = (f1(gamma[condition2], *coeffs_f1['T2']) * (600.0 - T[condition2]) + f1(gamma[condition2], *coeffs_f1['T1']) * (T[condition2] - 400.0)) / 200.00
        result[condition3] = (f1(gamma[condition3], *coeffs_f1['T3']) * (800.0 - T[condition3]) + f1(gamma[condition3], *coeffs_f1['T2']) * (T[condition3] - 600.0)) / 200.00
        result[condition4] = f1(gamma[condition4], *coeffs_f1['T3'])
        result[condition5] = f2(gamma[condition5], *coeffs_f2['T1'])
        result[condition6] = (f2(gamma[condition6], *coeffs_f2['T2']) * (600.0 - T[condition6]) + f2(gamma[condition6], *coeffs_f2['T1']) * (T[condition6] - 400.0)) / 200.00
        result[condition7] = (f2(gamma[condition7], *coeffs_f2['T3']) * (800.0 - T[condition7]) + f2(gamma[condition7], *coeffs_f2['T2']) * (T[condition7] - 600.0)) / 200.00
        result[condition8] = f2(gamma[condition8], *coeffs_f2['T3'])

        return result - 1.0


# =============================================================================
# Advanced Material Property Models (INL and Physics-Based)
# =============================================================================

class BaselineConductivity(Base):
    """
    Baseline (unirradiated) thermal conductivity model.

    Models temperature-dependent thermal conductivity for unirradiated graphite
    using a quadratic relationship.

    Model equation:
        k0 = a + b·T + c·T²
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        preds = params[0] + params[1] * Temperature + params[2] * Temperature**2
        return preds
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        preds = a + b * Temperature + c * Temperature**2
        return preds
    
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change= x[:, 5]         # Volume change
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        
        preds = a + b * Temperature + c * Temperature**2
        return preds, log_sigma

class INLConductivityChange(Base):
    """
    INL empirical model for thermal conductivity change under irradiation.

    Captures conductivity degradation with dose using exponential saturation
    and recovery terms.

    Model equation:
        k/k0 = 1 / [1 + (1 - exp(-a·dose)) + exp(-b/dose)·dose]

    The model accounts for:
        - Initial rapid decrease
        - Saturation at moderate doses
        - Potential recovery effects at high doses
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        inv_preds = 1.0 + (1.0-torch.exp(-params[0] * Dose))+ torch.exp(-params[1] / Dose) * Dose
        return 1.0/inv_preds
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        inv_preds = 1.0 + (1.0-torch.exp(-a * Dose))+ torch.exp(-b / Dose) * Dose
        return 1.0/inv_preds
    
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change= x[:, 5]         # Volume change
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        
        inv_preds = 1.0 + (1.0-torch.exp(-a * Dose))+ torch.exp(-b / Dose) * Dose
        return 1.0/inv_preds, log_sigma

class INLElasticModulusChange(Base):
    """
    INL empirical model for elastic modulus change with regime switching.

    Uses a smooth transition (sigmoid) between two regimes:
        - Low-dose: Linear behavior
        - High-dose: Quadratic behavior with offset

    Model equation:
        E_change = w1·(a·dose) + w2·(b·(dose + c)² + d)

    where w1 and w2 are sigmoid weights that sum to 1, with transition
    point controlled by parameter e.
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic elastic modulus change with regime switching.

        Args:
            params: Model parameters [a, b, c, d, e] where:
                - a: low-dose linear coefficient
                - b: high-dose quadratic coefficient
                - c: dose offset for high-dose regime
                - d: constant offset for high-dose regime
                - e: transition dose between regimes
            x: Input features, shape (n_samples, n_features).

        Returns:
            Elastic modulus change, shape (n_samples,).
        """
        dose = x[:, 1]

        # Smooth transition between regimes using sigmoid
        w2 = 1.0 / (1.0 + torch.exp(-(dose - params[4])))
        w1 = 1.0 - w2

        preds = w1 * (params[0] * dose) + w2 * (params[1] * (dose + params[2]) ** 2 + params[3])
        return preds
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1]) 
        c = pyro.sample("c", self.distributions[2])
        d = pyro.sample("d", self.distributions[3])
        e = pyro.sample("e", self.distributions[4])
        f = pyro.sample("f", self.distributions[5])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        w2 = 1.0 / (1.0 + torch.exp(-(Dose - e)))
        w1 = 1.0 - w2
        preds = w1 * (a * Dose) + w2 * (b * (Dose + c)**2 + d)
        return preds
    
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        d = pyro.sample("d", self.distributions[3])
        e = pyro.sample("e", self.distributions[4])
        f = pyro.sample("f", self.distributions[5])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change= x[:, 5]         # Volume change
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        
        w2 = 1.0 / (1.0 + torch.exp(-(Dose - e)))
        w1 = 1.0 - w2
        preds = w1 * (a * Dose) + w2 * (b * (Dose + c)**2 + d)
        return preds, log_sigma
    
class INLCTEChange(Base):
    """
    INL empirical model for coefficient of thermal expansion change under irradiation.

    Models CTE change using exponential saturation behavior with dose.

    Model equation:
        ΔCTE = a + b·(1 - exp(-c·dose))

    The model captures:
        - Initial CTE offset (a)
        - Saturation amplitude (b)
        - Rate of change with dose (c)
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        # preds = params[0] * torch.tanh((Dose - params[1]) / (params[2] * Dose + params[3]))
        preds = params[0] + params[1] * (1.0 - torch.exp(-params[2] * Dose))
        return preds
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        # d = pyro.sample("d", self.distributions[3])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        # preds = a * torch.tanh((Dose - b) / (c * Dose + d))
        preds = a + b * (1.0 - torch.exp(-c * Dose))
        return preds
    
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        # d = pyro.sample("d", self.distributions[3])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change= x[:, 5]         # Volume change
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        
        # preds = a * torch.tanh((Dose - b) / (c * Dose + d))
        preds = a + b * (1.0 - torch.exp(-c * Dose))
        return preds, log_sigma

class IrradiationElasticModulusChange(Base):
    """
    Physics-based model for elastic modulus change under irradiation.

    Combines dose-dependent modulus evolution with structural damage effects.

    Model structure:
        E/E0 = [1 + (a - 1)·(1 - exp(-b·dose))] × SD(dose)

    where SD is the structural damage factor accounting for connectivity
    and volumetric strain effects.

    The model captures:
        - Initial modulus increase with dose (a > 1)
        - Saturation behavior at high doses
        - Structural damage-induced degradation
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic elastic modulus change model.

        Args:
            params: Model parameters [a, b, c_sd, d_sd] where:
                - a, b: dose evolution parameters
                - c_sd, d_sd: structural damage parameters
            x: Input features [stress, dose, temperature, mean_dose, std_dose, volume_change].

        Returns:
            Relative elastic modulus change (E/E0), shape (n_samples,).
        """
        dose = x[:, 1]
        mean_dose = x[:, 3]
        std_dose = x[:, 4]
        volume_change = x[:, 5]

        structural_damage = self.structuralDamage(
            params[2:4], dose, mean_dose, std_dose, volume_change, flag=1
        )
        return (1.0 + (params[0] - 1.0) * (1.0 - torch.exp(-params[1] * dose))) * structural_damage
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        d = pyro.sample("d", self.distributions[3])
        # e = pyro.sample("e", self.distributions[4])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        # E0 = self.unirradiatedModulus(Temperature)

        structural_damage = self.structuralDamage([c,d], Dose, MeanDose, StdDose, volume_change, flag = 1)
        return (1.0 + (a - 1.0) * (1.0 - torch.exp(-b * Dose))) * structural_damage
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        d = pyro.sample("d", self.distributions[3])
        # e = pyro.sample("e", self.distributions[4])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change= x[:, 5]         # Volume change
        # E0 = self.unirradiatedModulus(Temperature)
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        
        structural_damage = self.structuralDamage([c,d], Dose, MeanDose, StdDose, volume_change, flag = 1)
        preds = (1.0 + (a - 1.0) * (1.0 - torch.exp(-b * Dose))) * structural_damage
        return preds, log_sigma

class IrradiationStrain(Base):
    """
    Physics-based model for irradiation-induced dimensional strain.

    Integrates instantaneous strain rate over dose history, accounting for:
        - Saturation behavior with dose
        - Structural connectivity evolution
        - Dose history effects

    Model equation:
        ε = ∫[a·(1 - exp(-b·γ))·(1 + c·Sc(γ))] dγ

    where Sc is the structural connectivity function.
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic irradiation strain model with dose history integration.

        Args:
            params: Model parameters [a, b, c] where:
                - a: maximum strain rate
                - b: saturation rate parameter
                - c: structural connectivity coupling strength
            x: Input features [stress, dose, temperature, mean_dose, std_dose, volume_change].

        Returns:
            Cumulative dimensional strain, shape (n_samples,).
        """
        dose = x[:, 1]
        mean_dose = x[:, 3]
        std_dose = x[:, 4]

        # Create dose array for integration
        dosages = torch.linspace(0, 1, steps=INTEGRATION_STEPS)
        dosages = dosages.unsqueeze(0).repeat(dose.shape[0], 1) * dose.unsqueeze(-1)

        # Compute structural connectivity and handle NaN values
        SC_value = self.structuralConnectivity(dosages, mean_dose, std_dose)
        SC_value[torch.isnan(SC_value)] = 1.0

        # Integrate strain rate over dose history
        integrands = params[0] * (1.0 - torch.exp(-params[1] * dosages)) * (1.0 + params[2] * SC_value)
        preds = torch.trapezoid(integrands) * dosages[:, 1]

        return preds
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change

        dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
        dosages = dosages.unsqueeze(0).repeat(Dose.shape[0], 1) * Dose.unsqueeze(-1)
        SC_value = self.structuralConnectivity(dosages, MeanDose, StdDose)
        SC_value[torch.isnan(SC_value)] = 1.0
        integrands = a * (1.0 - torch.exp(-b * dosages)) * (1.0 + c * SC_value)
        preds = torch.trapezoid(integrands)*dosages[:,1]

        return preds
    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))
        
        dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
        dosages = dosages.unsqueeze(0).repeat(Dose.shape[0], 1) * Dose.unsqueeze(-1)
        SC_value = self.structuralConnectivity(dosages, MeanDose, StdDose)
        SC_value[torch.isnan(SC_value)] = 1.0
        integrands = a * (1.0 - torch.exp(-b * dosages)) * (1.0 + c * SC_value)
        preds = torch.trapezoid(integrands)*dosages[:,1]

        return preds, log_sigma


class IrradiationCreep(Base):
    """
    Physics-based irradiation creep model with structural damage.

    Models irradiation creep as the sum of primary and secondary creep components,
    accounting for structural damage evolution and temperature-dependent elastic modulus.

    Components:
        - Primary creep: Dose-rate dependent with exponential dose term
        - Secondary creep: Steady-state creep proportional to stress and dose

    Both components are modified by structural damage factor.

    Attributes:
        sd_params (List): Structural damage model parameters.
        E0_params (List): Baseline elastic modulus parameters [a, b, c] for E0 = a + b·T + c·T².
    """

    def __init__(
        self,
        distributions: List,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        active_dims: List[int],
        group_code: torch.Tensor,
        sd_params: List[float],
        E0_params: List[float]
    ) -> None:
        """
        Initialize irradiation creep model.

        Args:
            distributions: Prior distributions for model parameters.
            train_x: Training input features.
            train_y: Training target values.
            active_dims: Active dimensions for GP kernel.
            group_code: Group identifiers.
            sd_params: Structural damage parameters.
            E0_params: Baseline elastic modulus parameters.
        """
        super().__init__(distributions, train_x, train_y, active_dims, group_code)
        self.sd_params = sd_params
        self.E0_params = E0_params
    def split(
        self,
        params: Union[List, torch.Tensor],
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute primary and secondary creep components separately.

        Useful for analyzing individual contributions to total creep strain.

        Args:
            params: Model parameters [a_primary, b_primary, a_secondary].
            x: Input features [stress, dose, temperature, mean_dose, std_dose, volume_change].

        Returns:
            Tuple of (primary_creep, secondary_creep).
        """
        dose = x[:, 1]
        temperature = x[:, 2]
        stress = x[:, 0]
        mean_dose = x[:, 3]
        std_dose = x[:, 4]
        volume_change = x[:, 5]

        E0 = self.E0_params[0] + self.E0_params[1] * temperature + self.E0_params[2] * temperature ** 2

        # Create dose array for integration
        dosages = torch.linspace(0, 1, steps=INTEGRATION_STEPS)
        dosages = dosages.unsqueeze(0).repeat(dose.shape[0], 1) * dose.unsqueeze(-1)
        structural_damage = self.structuralDamage(
            self.sd_params, dosages, mean_dose, std_dose, volume_change, flag=0
        )

        # Primary creep (dose-rate dependent)
        integrand = (params[0] * stress.unsqueeze(-1) * torch.exp(params[1] * dosages) /
                     (structural_damage * E0.unsqueeze(-1)))
        primary = torch.trapz(integrand) * dosages[:, 1]

        # Secondary creep (steady-state)
        integrand = params[2] * stress.unsqueeze(-1) / (structural_damage * E0.unsqueeze(-1))
        secondary = torch.trapz(integrand) * dosages[:, 1]

        return primary, secondary
    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic irradiation creep model with fixed parameters.

        Computes total creep as sum of primary and secondary components using
        structural damage evolution from prior calibration.

        Args:
            params: Model parameters [a_primary, b_primary, a_secondary].
            x: Input features [stress, dose, temperature, mean_dose, std_dose, volume_change].

        Returns:
            Total creep strain, shape (n_samples,).
        """
        dose = x[:, 1]
        temperature = x[:, 2]
        stress = x[:, 0]
        mean_dose = x[:, 3]
        std_dose = x[:, 4]
        volume_change = x[:, 5]

        E0 = self.E0_params[0] + self.E0_params[1] * temperature + self.E0_params[2] * temperature ** 2

        # Create dose array for integration
        dosages = torch.linspace(0, 1, steps=INTEGRATION_STEPS)
        dosages = dosages.unsqueeze(0).repeat(dose.shape[0], 1) * dose.unsqueeze(-1)
        structural_damage = self.structuralDamage(
            self.sd_params, dosages, mean_dose, std_dose, volume_change, flag=0
        )

        # Primary creep (dose-rate dependent)
        integrand = (params[0] * stress.unsqueeze(-1) * torch.exp(params[1] * dosages) /
                     (structural_damage * E0.unsqueeze(-1)))
        primary = torch.trapz(integrand) * dosages[:, 1]

        # Secondary creep (steady-state)
        integrand = params[2] * stress.unsqueeze(-1) / (structural_damage * E0.unsqueeze(-1))
        secondary = torch.trapz(integrand) * dosages[:, 1]

        return primary + secondary
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        E0 = self.E0_params[0] + self.E0_params[1] * Temperature + self.E0_params[2] * Temperature**2
        # E0 = self.unirradiatedModulus()
        dosages = torch.linspace(0, 1, steps=101)  # Assuming 101 points for integration
        dosages = dosages.unsqueeze(0).repeat(Dose.shape[0], 1) * Dose.unsqueeze(-1)
        structural_damage = self.structuralDamage(self.sd_params, dosages, MeanDose, StdDose, volume_change, flag = 0)

        # Primary creep computation
        integrand = (a * Stress.unsqueeze(-1) * torch.exp(b * dosages) /
                    (structural_damage * E0.unsqueeze(-1)))
        primary = torch.trapz(integrand)*dosages[:,1]

        # Secondary creep computation
        integrand = c * Stress.unsqueeze(-1) / (structural_damage * E0.unsqueeze(-1))
        secondary = torch.trapz(integrand)*dosages[:,1]

        return primary + secondary

    def hierarchicalModel(
        self,
        x: torch.Tensor,
        group_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical irradiation creep model with group-level noise.

        Args:
            x: Input features [stress, dose, temperature, mean_dose, std_dose, volume_change].
            group_code: Group identifiers, shape (n_samples,).

        Returns:
            Tuple of (predictions, log_sigma).
        """
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))

        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])

        dose = x[:, 1]
        temperature = x[:, 2]
        stress = x[:, 0]
        mean_dose = x[:, 3]
        std_dose = x[:, 4]
        volume_change = x[:, 5]

        E0 = self.E0_params[0] + self.E0_params[1] * temperature + self.E0_params[2] * temperature ** 2

        # Create dose array for integration
        dosages = torch.linspace(0, 1, steps=INTEGRATION_STEPS)
        dosages = dosages.unsqueeze(0).repeat(dose.shape[0], 1) * dose.unsqueeze(-1)
        structural_damage = self.structuralDamage(
            self.sd_params, dosages, mean_dose, std_dose, volume_change, flag=0
        )

        # Primary creep (dose-rate dependent)
        integrand = (a * stress.unsqueeze(-1) * torch.exp(b * dosages) /
                     (structural_damage * E0.unsqueeze(-1)))
        primary = torch.trapz(integrand) * dosages[:, 1]

        # Secondary creep (steady-state)
        integrand = c * stress.unsqueeze(-1) / (structural_damage * E0.unsqueeze(-1))
        secondary = torch.trapz(integrand) * dosages[:, 1]

        preds = primary + secondary
        return preds, log_sigma

class IrradiationCTEChange(Base):
    """
    Model for coefficient of thermal expansion (CTE) change under irradiation.

    Accounts for:
        - Structural connectivity degradation
        - Dose-dependent dimensional change
        - Creep-induced dimensional effects

    Model structure:
        CTE_change = (1 - a·Sc) × (1 + b·(1 - exp(-c·dose))) × (1 + d·creep) - 1

    Attributes:
        creep (torch.Tensor): Pre-computed creep values for the training data.
    """

    def __init__(
        self,
        distributions: List,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        active_dims: List[int],
        group_code: torch.Tensor,
        creep: torch.Tensor
    ) -> None:
        """
        Initialize CTE change model with pre-computed creep.

        Args:
            distributions: Prior distributions for model parameters.
            train_x: Training input features.
            train_y: Training target values.
            active_dims: Active dimensions for GP kernel.
            group_code: Group identifiers.
            creep: Pre-computed creep values, shape (n_samples,).
        """
        super().__init__(distributions, train_x, train_y, active_dims, group_code)
        self.creep = creep

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the coefficient of thermal expansion (CTE) under irradiation.
        Expects params to have at least 5 elements.
        The creep contributions are currently set to zero. If needed,
        you can instantiate an IrradiationCreep with self.X and compute them.
        """
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        SC_value = self.structuralConnectivity(Dose, MeanDose, StdDose, flag = 1)

        # CTE0 = params[2] * torch.square(Temperature) + params[1] * Temperature + params[0]
        cte = ((1.0 - params[0] * SC_value) *
                (1.0 + params[1] * (1.0 - torch.exp(-params[2] * Dose))) *
                (1.0 + params[3] * self.creep)) - 1.0
        return cte
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        d = pyro.sample("d", self.distributions[3])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        SC_value = self.structuralConnectivity(Dose, MeanDose, StdDose, flag = 1)

        # CTE0 = c * torch.square(Temperature) + b * Temperature + a
        cte = ((1.0 - a * SC_value) *
                (1.0 + b * (1.0 - torch.exp(-c * Dose))) *
                (1.0 + d * self.creep)) - 1.0
        return cte

    def hierarchicalModel(self, x, group_code):
        n_groups = len(group_code.unique())
        mu_log_sigma = pyro.sample("mu_log_sigma", dist.Normal(self.distributions[-1].loc.item(), 1.0))
        sig_log_sigma = pyro.sample("sig_log_sigma", dist.InverseGamma(1.0, 1.0))
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        d = pyro.sample("d", self.distributions[3])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        SC_value = self.structuralConnectivity(Dose, MeanDose, StdDose, flag = 1)
        with pyro.plate("plate_i", n_groups):
            log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, sig_log_sigma))

        # CTE0 = c * torch.square(Temperature) + b * Temperature + a
        cte = ((1.0 - a * SC_value) *
                (1.0 + b * (1.0 - torch.exp(-c * Dose))) *
                (1.0 + d * self.creep)) - 1.0
        return cte, log_sigma

class BaselineCTE(Base):
    """
    Baseline (unirradiated) coefficient of thermal expansion model.

    Models temperature-dependent CTE for unirradiated graphite using
    a scaled quadratic relationship.

    Model equation:
        CTE = a·(b + c·T + d·T²)
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        
        cte0 = params[0] * (params[1] + params[2] * Temperature + params[3] * Temperature**2)

        # cte0 = params[0] + params[1] * Temperature + params[2] * torch.square(Temperature)
        return cte0
    
    def model(self, x):
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        Dose = x[:, 1]           # Dose (tensor)
        Temperature = x[:, 2]    # Temperature (tensor)
        Stress = x[:, 0]         # Stress (tensor)
        MeanDose = x[:, 3]         # Mean dosage
        StdDose = x[:, 4]         # Std dosage
        volume_change = x[:, 5]         # Volume change
        
        cte0 = a + b * Temperature + c * torch.square(Temperature)
        return cte0


class BaselineElasticModulus(Base):
    """
    Baseline (unirradiated) elastic modulus model.

    Models temperature-dependent elastic modulus for unirradiated graphite
    using a quadratic relationship.

    Model equation:
        E0 = a + b·T + c·T²
    """

    def determModel(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic baseline elastic modulus model.

        Args:
            params: Model parameters [a, b, c] for quadratic temperature dependence.
            x: Input features, shape (n_samples, n_features).

        Returns:
            Baseline elastic modulus, shape (n_samples,).
        """
        temperature = x[:, 2]
        modulus = params[0] + params[1] * temperature + params[2] * torch.square(temperature)
        return modulus

    def model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-hierarchical baseline elastic modulus model.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Model predictions, shape (n_samples,).
        """
        a = pyro.sample("a", self.distributions[0])
        b = pyro.sample("b", self.distributions[1])
        c = pyro.sample("c", self.distributions[2])
        temperature = x[:, 2]
        modulus = a + b * temperature + c * torch.square(temperature)
        return modulus
