# HBMIN-GP

Hierarchical Bayesian Models with Gaussian Processes for Graphite Material Properties Under Irradiation

## Overview

HBMIN-GP is a comprehensive probabilistic modeling framework for predicting the evolution of graphite material properties under neutron irradiation. The framework combines hierarchical Bayesian inference with Gaussian process regression to model key physical properties including dimensional change, creep, thermal expansion, and elastic modulus.

## Key Features

- **Multiple Property Models**: Supports modeling of dimensional change (IIDC), irradiation creep, coefficient of thermal expansion (CTE), and elastic modulus (E)
- **Multiple Model Types**: Implements empirical (Shibata), semi-empirical (INL), and physics-based (Bradford) models
- **Bayesian Calibration**: Three calibration methodologies:
  - Deterministic optimization (gradient descent)
  - Bayesian inference (MCMC with NUTS sampler)
  - Kennedy-O'Hagan (KOH) calibration with Gaussian process discrepancy
- **Hierarchical Structure**: Supports hierarchical models for multiple data sources with group-level parameters
- **Uncertainty Quantification**: Full probabilistic predictions with predictive entropy quantification
- **Model Export**: TorchScript export for deployment of calibrated GP models
- **Multiple Graphite Grades**: Supports IG110, NBG18, PCEA, NBG17, S2114 graphite grades

## Installation

### Requirements

```
torch
pyro-ppl
gpytorch
numpy
pandas
matplotlib
seaborn
arviz
scikit-learn
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd HBMIN-GP

# Install dependencies (recommended: use a virtual environment)
pip install torch pyro-ppl gpytorch numpy pandas matplotlib seaborn arviz scikit-learn
```

## Project Structure

```
src/
├── modelsHierarchical.py     # Core model library with all property models
├── baselineCTETorch.py        # Baseline CTE model calibration
├── baselineETorch.py          # Baseline elastic modulus calibration
├── modifyCreepData.py         # Creep data preprocessing utilities
├── creepTorch.py              # Irradiation creep model calibration
├── CTEChangeTorch.py          # CTE change model calibration
├── EChangeTorch.py            # Elastic modulus change calibration
└── IIDCTorch.py               # IIDC model calibration
```

## Core Models

### 1. Dimensional Change Models

**Irradiation-Induced Dimensional Change (IIDC)**

- **INL Model**: Quadratic dose term with temperature-dependent recovery
  - Equation: `ΔL/L = a·γ² - b·exp(-Ea/(k_B·T))·γ`
  - Parameters: a (dose coefficient), b (recovery coefficient), Ea (activation energy)

- **Shibata Model**: Empirical piecewise temperature-dependent model
  - Four temperature regimes with cubic/linear dose dependence
  - Fixed empirical coefficients from Japanese graphite data

- **Bradford Model**: Physics-based with structural connectivity
  - Integrates strain rate over dose history with saturation behavior
  - Includes structural connectivity effects and dose history

### 2. Irradiation Creep Models

- **INL Creep Model**: Elastic strain plus temperature-dependent irradiation creep
  - Equation: `ε_creep = ε_elastic + a·exp(-b·T)·σ·γ`

- **Shibata Creep Model**: Fixed empirical coefficients
  - Based on graphite correlation data provided in reference Shibata 2010

- **Bradford Creep Model**: Physics-based with primary and secondary creep
  - Primary: Dose-rate dependent with exponential dose term
  - Secondary: Steady-state creep proportional to stress and dose
  - Includes structural damage evolution

### 3. CTE Change Models

- **INL CTE Model**: Exponential saturation with dose
  - Equation: `ΔCTE = a + b·(1 - exp(-c·dose))`

- **Shibata CTE Model**: Complex piecewise model with turnaround dose
  - Pre-turnaround: Cubic polynomial
  - Post-turnaround: Linear decline

- **Bradford CTE Model**: Structural connectivity and dimensional coupling
  - Accounts for connectivity degradation, dose effects, and creep

### 4. Elastic Modulus Models

- **INL E Model**: Regime-switching with sigmoid transition
  - Low-dose: Linear behavior
  - High-dose: Quadratic behavior

- **Bradford E Model**: Dose evolution with structural damage
  - Equation: `E/E0 = [1 + (a-1)·(1-exp(-b·dose))] × SD(dose)`

## Usage

### Basic Workflow

1. **Baseline Model Calibration** (for unirradiated properties):

```python
python src/baselineETorch.py  # Elastic modulus baseline
python src/baselineCTETorch.py  # CTE baseline
```

2. **IIDC Model Calibration**:

```python
python src/IIDCTorch.py
```

Configuration in the script:
- Set `GRAPHITE_GRADE` (e.g., 'NBG18', 'IG110', 'PCEA')
- Set `MODEL_NAME` ('INL', 'Shibata', or 'Bradford')
- Adjust MCMC parameters if needed

3. **Creep Model Calibration**:

```python
python src/creepTorch.py
```

Note: Requires IIDC parameters from step 2

4. **Property Change Models**:

```python
python src/EChangeTorch.py      # Elastic modulus change
python src/CTEChangeTorch.py    # CTE change
```

### Output Files

Each calibration script generates:
- **Parameter files**: `.pt` files with optimized/sampled parameters
- **Diagnostic plots**: Scatter plots comparing predictions to observations
- **GP models**: TorchScript models for deployment (`gp_*.pt`)
- **Entropy plots**: Uncertainty visualization
- **Fluence plots**: Property evolution vs. neutron fluence

## Calibration Methodologies

### 1. Deterministic Optimization

Gradient-based optimization using PyTorch:
- Fast convergence
- Point estimates
- No uncertainty quantification

### 2. Bayesian Inference

MCMC sampling with NUTS:
- Full posterior distributions
- Uncertainty quantification
- Requires good priors (from deterministic optimization)

### 3. Kennedy-O'Hagan (KOH) Calibration

Combines parametric model with GP discrepancy:
- `y = m(x, θ) + δ(x) + ε`
- m(x, θ): Parametric physics model
- δ(x): GP discrepancy term (model inadequacy)
- ε: Observation noise

Benefits:
- Captures model inadequacy
- Improved predictions
- Quantifies epistemic uncertainty

## Model Classes

### Base Class

All models inherit from `Base` class which provides:
- Gaussian process regression interface
- MCMC sampling with NUTS
- Hierarchical and non-hierarchical model support
- Helper functions for structural connectivity, damage, etc.

### Key Methods

```python
model.determModel(params, x)           # Deterministic prediction
model.probModel(x, y, group_code)      # Bayesian model (hierarchical)
model.probModelNH(x, y)                # Bayesian model (non-hierarchical)
model.kohModel(x, y, group_code)       # KOH model (hierarchical)
model.kohModelNH(x, y)                 # KOH model (non-hierarchical)
model.MCMCSamples(...)                 # Run MCMC sampling
```

## Input Features

Models expect input tensors with shape `[N, 6]`:
- `x[:, 0]`: Stress (MPa)
- `x[:, 1]`: Neutron dose/fluence (10^26 n/m²)
- `x[:, 2]`: Temperature (K)
- `x[:, 3]`: Mean dose (for structural connectivity)
- `x[:, 4]`: Std dose (for structural connectivity)
- `x[:, 5]`: Volume change (for structural damage)

## Graphite Grades

Supported grades with data processing classes:
- **IG110**: Fine-grained isotropic graphite
- **NBG18**: Nuclear grade graphite
- **PCEA**: Extruded nuclear graphite
- **NBG17**: Nuclear grade graphite
- **S2114**: Specialty nuclear graphite

## Physical Constants

- Boltzmann constant: `8.617333262e-5 eV/K`
- Integration points (trapezoidal rule): 101 steps

## Visualization

The framework generates publication-quality plots including:
- Training scatter plots (observed vs. predicted)
- GP scatter plots with entropy coloring
- Fluence evolution plots with experimental data overlay
- Uncertainty bands and predictive entropy

## Advanced Features

### Hierarchical Models

Support for multiple data sources with group-level parameters:
- Group-specific observation noise
- Population-level hyperpriors
- Information sharing across groups

### Structural Connectivity

Models graphite microstructure evolution:
- Error function-based connectivity: `Sc = 0.5·(1 + erf((γ-μ)/(σ√2)))`
- Affects elastic modulus, CTE, and dimensional change

### Structural Damage

Combines connectivity and volumetric strain effects:
- `SD = (1 + c·Sc)·exp(-d·ε_V)`
- Used in elastic modulus and creep models

### Zero-Fluence Padding

Enforces physical boundary condition (zero property change at zero dose):
- Adds synthetic data points at near-zero fluence
- Improves GP extrapolation behavior

## Configuration

Key parameters (found at top of each script):

```python
GRAPHITE_GRADE = "NBG18"          # Grade identifier
MODEL_NAME = "INL"                # Model type
ACTIVE_DIMS = [1, 2]              # GP dimensions (dose, temperature)
LEARNING_RATE = 0.001             # Optimization learning rate
NUM_EPOCHS = 100000               # Optimization epochs
MCMC_WARMUP = 1500                # MCMC warmup iterations
MCMC_SAMPLES = 3000               # MCMC sampling iterations
```

## Model Export and Deployment

Trained models can be exported as TorchScript:

```python
# GP model export
export_gp_model(model, train_x, train_y, mcmc_koh, active_dims,
                group_code, "gp_model.pt")

# Load and use
model = torch.jit.load("gp_model.pt")
predictions = model(test_x_normalized)
```

## References

This framework implements models and methodologies from:
- Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B.
- Shibata, T., Sawa, K., Eto, M., Kunimoto, E., Shiozawa, S., Oku, T., & Maruyama, T. (2010). Draft of standard for graphite core components in high temperature gas-cooled reactor.
- Bajpai, P., Prithivirajan, V., Munday, L. B., Singh, G., & Spencer, B. W. (2024). Development of Graphite Thermal and Mechanical Modeling Capabilities in Grizzly (No. INL/RPT-24-78905-Rev000). Idaho National Laboratory (INL), Idaho Falls, ID (United States).
- Bradford, M. R., & Steer, A. G. (2008). A structurally-based model of irradiated graphite properties. Journal of Nuclear Materials, 381(1-2), 137-144.
- Dhulipala, S. L., Bajpai, P., Singh, G., & Spencer, B. W. (2025). Bayesian calibration of irradiated graphite property models under high temperatures. npj Materials Degradation.

## License

See LICENSE file for details.

## Contributing

This is a research codebase for nuclear graphite property modeling. For questions or collaboration opportunities, please contact the repository maintainers. (som.dhulipala@inl.gov)

## Citation

If you use this code in your research, please cite:
```
Dhulipala, S. L., Bajpai, P., Singh, G., & Spencer, B. W. (2025). Bayesian calibration of irradiated graphite property models under high temperatures. npj Materials Degradation.
```