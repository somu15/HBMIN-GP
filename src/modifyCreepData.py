"""
Creep data modification utilities for separating dimensional change from creep strain.

This module provides classes and enumerations for processing creep data by
subtracting predicted dimensional change (IIDC) from total strain measurements.
This separation is crucial for accurate creep model calibration.

Classes:
    trainingOption: Enum for training methodology (deterministic, Bayesian, KOH)
    IIDCModel: Enum for dimensional change model type (INL, Shibata, Bradford)
    modifyCreepData: Main class for creep data processing
"""

from enum import Enum
from typing import Union

import torch
import numpy
import warnings
from pyro.infer import Predictive

from codes.modelsHierarchical import (
    ShibataIIDCDeltaL,
    IIDCDeltaL,
    IrradiationStrain
)
from codes.processData import (
    processDataIG110,
    processDataNBG18,
    processDataPCEA,
    processDataNBG17,
    processDataS2114
)


class trainingOption(Enum):
    """Training methodology options for creep data processing."""
    DETERMINISTIC = 1
    BAYESIAN = 2
    KOH = 3  # Kennedy-O'Hagan calibration


class IIDCModel(Enum):
    """Irradiation-Induced Dimensional Change (IIDC) model options."""
    INL = 1        # Idaho National Laboratory model
    Shibata = 2    # Shibata empirical model
    Bradford = 3   # Bradford physics-based model


class modifyCreepData:
    """
    Process creep data by subtracting dimensional change predictions.

    This class separates total strain into creep and dimensional change components
    by subtracting IIDC model predictions from measured total strain. Supports
    multiple training methodologies (deterministic, Bayesian, KOH) and IIDC models.

    Attributes:
        train_x (torch.Tensor): Input features for creep data.
        train_y (torch.Tensor): Processed creep strain (total strain - IIDC prediction).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y_full: torch.Tensor,
        params: Union[torch.Tensor, dict],
        params_dict: dict,
        dataFunc,
        training: trainingOption,
        model: IIDCModel
    ) -> None:
        """
        Initialize creep data processor.

        Args:
            train_x: Input features (stress, dose, temperature, etc.).
            train_y_full: Full strain measurements (IIDC + creep).
            params: IIDC model parameters (tensor for deterministic, dict for Bayesian/KOH).
            params_dict: Dictionary of MCMC samples for KOH calibration.
            dataFunc: Data processing function for the graphite grade.
            training: Training methodology to use.
            model: IIDC model type to use.

        Raises:
            ValueError: If training or model option is invalid.
        """
        # Validate inputs
        if not isinstance(training, trainingOption):
            raise ValueError(
                "training option must be either DETERMINISTIC, BAYESIAN, or KOH."
            )

        if not isinstance(model, IIDCModel):
            raise ValueError(
                "model option must be either INL, Shibata, or Bradford."
            )

        self.train_x = train_x

        # Select appropriate IIDC model
        if model == IIDCModel.INL:
            predictive_model = IIDCDeltaL
        elif model == IIDCModel.Shibata:
            predictive_model = ShibataIIDCDeltaL
        elif model == IIDCModel.Bradford:
            predictive_model = IrradiationStrain
        else:
            raise ValueError(f"Unknown model type: {model}")

        # Configure model dimensions
        active_dims = [1, 2]  # Dose and temperature dimensions
        group_code = torch.zeros(train_x.shape[0]).int()

        # Process data based on training methodology
        if training == trainingOption.DETERMINISTIC:
            self.train_y = self._process_deterministic(
                predictive_model, params, train_x, train_y_full,
                active_dims, group_code
            )
        elif training == trainingOption.BAYESIAN:
            self.train_y = self._process_bayesian(
                predictive_model, params, train_x, train_y_full,
                active_dims, group_code
            )
        elif training == trainingOption.KOH:
            self.train_y = self._process_koh(
                predictive_model, params, params_dict, train_x,
                train_y_full, dataFunc, active_dims, group_code
            )
        else:
            raise ValueError(f"Unknown training option: {training}")

    def _process_deterministic(
        self,
        model_class,
        params: torch.Tensor,
        train_x: torch.Tensor,
        train_y_full: torch.Tensor,
        active_dims: list,
        group_code: torch.Tensor
    ) -> torch.Tensor:
        """
        Process data using deterministic IIDC model.

        Args:
            model_class: IIDC model class.
            params: Deterministic model parameters.
            train_x: Input features.
            train_y_full: Full strain measurements.
            active_dims: Active dimensions for GP.
            group_code: Group identifiers.

        Returns:
            Creep strain (total strain - IIDC prediction).
        """
        model_defn = model_class(
            [], train_x, train_y_full,
            active_dims=active_dims,
            group_code=group_code
        )
        iidc_prediction = model_defn.determModel(params, train_x)
        train_y_creep = train_y_full - iidc_prediction.detach()
        return train_y_creep

    def _process_bayesian(
        self,
        model_class,
        params: torch.Tensor,
        train_x: torch.Tensor,
        train_y_full: torch.Tensor,
        active_dims: list,
        group_code: torch.Tensor
    ) -> torch.Tensor:
        """
        Process data using Bayesian IIDC model (mean prediction).

        Args:
            model_class: IIDC model class.
            params: Deterministic parameters for prediction.
            train_x: Input features.
            train_y_full: Full strain measurements.
            active_dims: Active dimensions for GP.
            group_code: Group identifiers.

        Returns:
            Creep strain (total strain - mean IIDC prediction).
        """
        model_defn = model_class(
            [], train_x, train_y_full,
            active_dims=active_dims,
            group_code=group_code
        )
        iidc_prediction = model_defn.determModel(params, train_x)
        train_y_creep = train_y_full - iidc_prediction.mean(dim=0).detach()
        return train_y_creep

    def _process_koh(
        self,
        model_class,
        params: torch.Tensor,
        params_dict: dict,
        train_x: torch.Tensor,
        train_y_full: torch.Tensor,
        dataFunc,
        active_dims: list,
        group_code: torch.Tensor
    ) -> torch.Tensor:
        """
        Process data using Kennedy-O'Hagan calibration with GP discrepancy.

        Args:
            model_class: IIDC model class.
            params: Deterministic parameters for prediction.
            params_dict: Dictionary of MCMC samples for GP hyperparameters.
            train_x: Input features for creep data.
            train_y_full: Full strain measurements.
            dataFunc: Data processing function for IIDC calibration data.
            active_dims: Active dimensions for GP.
            group_code: Group identifiers.

        Returns:
            Creep strain (total strain - IIDC prediction - GP discrepancy).
        """
        # Initialize model and get predictions
        model_defn = model_class(
            [], train_x, train_y_full,
            active_dims=active_dims,
            group_code=group_code
        )
        predictive_koh = model_defn.determModel(params, train_x)
        gpr = model_defn.returnGP()

        # Load IIDC calibration data for GP training
        data = dataFunc
        train_x_iidc, train_y_iidc, _ = data.IIDCData()

        # Normalize features (zero out stress dimension for IIDC)
        train_x_norm = (train_x - train_x_iidc.mean(dim=0)) / train_x_iidc.std(dim=0)
        train_x_norm[:, 0] = 0.0

        train_x_iidc_norm = (train_x_iidc - train_x_iidc.mean(dim=0)) / train_x_iidc.std(dim=0)
        train_x_iidc_norm[:, 0] = 0.0

        # Compute IIDC predictions and fit GP to residuals
        predictive_koh_iidc = model_defn.determModel(params, train_x_iidc)
        gpr.set_data(train_x_iidc_norm, train_y_iidc - predictive_koh_iidc)

        # Set GP hyperparameters from MCMC samples
        gpr.kernel.lengthscale = params_dict['kernel.lengthscale'].log().mean(dim=0).exp()
        gpr.kernel.variance = params_dict['kernel.variance'].mean(dim=0)
        gpr.noise = params_dict['noise'].log().mean(dim=0).exp()

        # Predict GP discrepancy for creep data points
        mean, _ = gpr(train_x_norm)

        # Subtract both parametric prediction and GP discrepancy
        train_y_creep = train_y_full - predictive_koh.detach() - mean
        return train_y_creep

    def creepData(self) -> tuple:
        """
        Return processed creep data.

        Returns:
            Tuple of (train_x, train_y) where train_y is the isolated creep strain.
        """
        return self.train_x, self.train_y
