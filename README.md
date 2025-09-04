# Univariate Lindley Snow Model

A clean, modular implementation of a univariate Lindley random walk model for snow depth simulation with realistic seasonal dynamics.

## Model Description

The model simulates daily snow depth using a Lindley random walk with temperature-dependent daily changes:

- **Daily Changes**: `C_t ~ N(b0 + b1 * temp_t, σ²)`
- **Snow Evolution**: `X_t = max(X_{t-1} + C_t, 0)`

Where:
- `b0`: Baseline daily change (intercept)
- `b1`: Temperature coefficient (typically negative for melting)
- `σ`: Daily variability (standard deviation)
- `temp_t`: Daily temperature

### Key Features

- **Realistic Initial Conditions**: Simulations start in mid-summer (July 15th) where x₀=0 is physically meaningful
- **Seasonal Progression**: Natural progression from summer → fall → winter → spring
- **Temperature-Driven Dynamics**: Direct relationship between temperature and snow accumulation/melting
- **Statistical Rigor**: Maximum likelihood estimation with confidence intervals

## Project Structure

```
├── models/
│   └── UnivariateSnowModel.py     # Core snow model implementation
├── estimation/
│   └── ParameterEstimation.py     # MLE estimation and convergence analysis
├── synthetic_data/
│   ├── temp.py                    # Temperature data generation utilities
│   └── temperature_data_*/        # Timestamped synthetic temperature datasets
├── simulation/
│   └── run_simulation.py          # Main simulation orchestrator
├── v1/                           # Legacy code (archived)
└── README.md                     # This file
```

## Quick Start

### Generate Temperature Data

First, generate synthetic temperature data with organized timestamped output:

```bash
cd synthetic_data
python temp.py
```

This creates a timestamped directory (e.g., `temperature_data_20250904_145121/`) containing:
- `temperature_data_1920_2019.csv`: 100 years of synthetic daily temperatures
- `temperature_visualization.png`: Temperature time series and seasonal plots

### Running a Complete Simulation

```bash
cd simulation  
python run_simulation.py --years 10 --n_sims 50 --seed 42
```

This will:
1. Automatically find and load the most recent temperature data
2. Run a single simulation starting from mid-summer (realistic x₀=0)
3. Perform convergence study with 50 simulations
4. Generate comprehensive plots and analysis
5. Save all results to a timestamped output directory

### Command Line Options

- `--n_sims N`: Number of simulations for convergence study (default: 100)
- `--years Y`: Number of years to simulate (default: all available data)
- `--seed S`: Random seed for reproducibility (default: 42)
- `--no_convergence`: Skip convergence study for faster execution

### Using Individual Modules

```python
import numpy as np
from models.UnivariateSnowModel import UnivariateSnowModel
from estimation.ParameterEstimation import estimate_parameters

# Load temperature data (or generate synthetic data)
temperatures = np.random.normal(5, 15, 365*5)  # 5 years of daily temps

# Create model with realistic parameters
model = UnivariateSnowModel(b0=0.5, b1=-0.15, sigma=1.2)

# Simulate snow depths (automatically starts from summer)
changes, depths = model.simulate(temperatures, x0=0)

# Estimate parameters from simulated data
estimates, success = estimate_parameters(depths, changes, temperatures)
print(f"Estimated: b0={estimates[0]:.3f}, b1={estimates[1]:.3f}, σ={estimates[2]:.3f}")
```

## Simulation Behavior

### Seasonal Realism
- **Summer Start (Day 0-60)**: High temperatures (20-35°C), snow depth remains 0
- **Fall Transition (Day 60-150)**: Cooling temperatures, first snow accumulation
- **Winter Peak (Day 150-240)**: Cold temperatures (-20 to 0°C), maximum snow depths
- **Spring Melt (Day 240-365)**: Warming temperatures, snow melting back to 0

### Physical Constraints
- Non-negative snow depths enforced by Lindley recursion: `X_t = max(X_{t-1} + C_t, 0)`
- Realistic initial conditions: x₀=0 starting in mid-summer
- Temperature-dependent accumulation and melting processes

## Model Parameters

- **b0** (Baseline): Typical values around 0.1 to 1.0 cm/day
  - Positive: Net accumulation tendency
  - Negative: Net melting tendency

- **b1** (Temperature coefficient): Typically negative (-0.05 to -0.3)
  - Represents temperature sensitivity of snow processes
  - Each 1°C increase causes `|b1|` cm more daily melting

- **σ** (Variability): Typical values 0.5 to 2.0 cm
  - Daily standard deviation of changes
  - Captures weather variability and model uncertainty

## Output Files

Each simulation creates a timestamped directory containing:

- `plots/simulation_results.png`: Main simulation visualizations
- `plots/convergence_analysis.png`: Parameter estimation convergence plots
- `data/simulation_data.csv`: Time series of temperatures, depths, and changes
- `data/parameter_estimates.csv`: Estimated parameters with confidence intervals
- `data/convergence_study.csv`: All parameter estimates from convergence study
- `simulation_summary.txt`: Comprehensive text summary

## Model Features

### Core Capabilities
- ✅ Realistic snow accumulation/melting physics with summer start
- ✅ Maximum likelihood parameter estimation
- ✅ Confidence interval calculation
- ✅ Convergence analysis over multiple simulations
- ✅ Comprehensive visualization and reporting
- ✅ Organized data management with timestamped outputs

### Key Advantages
- **Physical realism**: Non-negative snow constraint via Lindley process
- **Seasonal authenticity**: Starts simulations in summer when x₀=0 makes sense
- **Temperature sensitivity**: Direct relationship between temperature and snow changes
- **Statistical rigor**: MLE estimation with proper uncertainty quantification
- **Computational efficiency**: Vectorized operations and optimized algorithms
- **Modular design**: Clean separation of concerns for maintainability
- **Data organization**: Automatic timestamped data discovery and organized outputs

## Data Management

### Temperature Data Generation
The `synthetic_data/temp.py` script generates:
- 100 years (1920-2019) of synthetic daily temperatures
- Realistic seasonal patterns with daily variability
- Organized output in timestamped directories

### Automatic Data Discovery
The simulation runner automatically:
- Finds the most recent temperature data in `synthetic_data/`
- Loads data from timestamped subdirectories
- Ensures reproducible workflows

## Dependencies

- `numpy`: Numerical computing
- `scipy`: Optimization and statistical functions
- `matplotlib`: Plotting and visualization
- `pandas`: Data manipulation and I/O

## Installation

```bash
pip install numpy scipy matplotlib pandas
```

## Example Output

### Summer Start Progression
```
Running simulation starting from summer (day 196) with 730 days (2.0 years)

Summer start (first 10 days):
Day 0-9: Temps 22-34°C, Snow depth: 0.0 cm (realistic summer conditions)

First snow accumulation (day 150-170):
Day 150-170: Temps -5 to -21°C, Snow depth: 72-130 cm (fall transition)

Peak winter (day 200-220):
Day 200-220: Temps -7 to -20°C, Snow depth: 216-276 cm (winter maximum)
```

### Parameter Recovery
```
Parameter Estimation Summary:
  b0: True=0.500, Estimated=0.490 (Bias: -0.010)
  b1: True=-0.150, Estimated=-0.150 (Bias: 0.000)
  σ:  True=1.200, Estimated=1.182 (Bias: -0.018)
  
Convergence Rate: 98.0% (49/50 simulations)
```

### Snow Statistics
```
Snow Simulation (2 years starting from summer):
  Maximum depth: 328.9 cm
  Mean depth: 140.1 cm  
  Days with snow: 603 (82.6%)
  Days without snow: 127 (17.4%)
```

## Author

Joshua King  
Snow Climate Modeling Project  
Created: September 4, 2025

## Updates Log

- **v2.0** (Sept 4, 2025): 
  - Realistic summer start (x₀=0 begins July 15th)
  - Organized timestamped data management
  - Automatic temperature data discovery
  - Improved seasonal visualization
  - Comprehensive modular architecture
