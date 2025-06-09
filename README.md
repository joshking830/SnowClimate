# Lindley Random Walk Snow Depth Simulation

A computational framework for simulating and analyzing snow depth dynamics using the Lindley Random Walk model. This project implements Monte Carlo simulation studies to estimate parameters of a stochastic snow accumulation model with seasonal variation and long-term climate trends.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Scientific Background](#scientific-background)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository implements a sophisticated snow depth modeling system based on the Lindley Random Walk process. The model captures:

- **Seasonal snow dynamics** with periodic accumulation and melt patterns
- **Long-term climate trends** through parameter estimation
- **Stochastic variability** in daily snow depth changes
- **Storage balance equations** for realistic snow depth evolution

### Key Applications
- Climate change impact assessment
- Snow water equivalent forecasting  
- Hydrological modeling validation
- Statistical method development for environmental data

## Features

### Core Capabilities
-  **Lindley Random Walk Implementation**: Complete model with seasonal components
-  **Monte Carlo Simulation Studies**: Large-scale parameter estimation validation
-  **Performance Optimization**: Vectorized algorithms with 10-50x speedup
-  **Comprehensive Visualization**: Time series plots, parameter analysis, statistical summaries
-  **Reproducible Research**: Seed control and organized output structure

### Model Components
- Periodic mean function with cosine seasonal variation
- Linear climate trend incorporation
- Maximum likelihood parameter estimation
- Success rate tracking and convergence analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- NumPy, SciPy, Pandas, Matplotlib

### Setup
```bash
# Clone the repository
git clone https://github.com/joshking830/SnowClimate.git
cd SnowClimate

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy pandas matplotlib argparse
```

## Quick Start

### Run a Test Simulation
```bash
# Single test simulation (fast)
python run_optimized.py --test

# View results
python visualize.py --folder [generated_folder_name]
```

### Compare Original vs Optimized Performance
```bash
# Performance comparison study
python run_comparison.py --test

# View comparison results
python visualize.py --folder sim_comparisons/[comparison_folder]/optimized
```

## Usage

### Production Simulations

#### Fast Optimized Runs (Recommended)
```bash
# Standard study (100 simulations, 100 years each)
python run_optimized.py --full

# Custom parameters
python run_optimized.py --n_sims 1000 --n_years 50

# Large-scale study
python run_optimized.py --n_sims 10000
```

#### Detailed Analysis Runs
```bash
# Full verbose output with all simulation data
python run_simulation.py --full

# Custom detailed study
python run_simulation.py --n_sims 100 --n_years 200
```

### Performance Benchmarking
```bash
# Compare estimation methods
python run_comparison.py --n_sims 50

# List previous comparisons
python run_comparison.py --list
```

### Visualization
```bash
# Complete visualization suite
python visualize.py --folder [results_folder] --plot all

# Specific plot types
python visualize.py --folder [results_folder] --plot params
python visualize.py --folder [results_folder] --plot series --sim 5

# Interactive mode
python visualize.py --folder [results_folder] --interactive
```

## File Structure

### Core Implementation
```
├── LindleySnowModel.py          # Core model implementation
├── OptimizedEstimations.py      # Vectorized parameter estimation  
├── run_optimized.py            # Fast production simulations
├── run_simulation.py           # Detailed analysis runs
├── run_comparison.py           # Performance comparison studies
└── visualize.py                # Comprehensive plotting suite
```

### AI Assistance Disclosure
- **Significant AI Assistance**: `run_optimized.py`, `run_comparison.py`, `visualize.py`
- **Minimal AI Assistance**: `LindleySnowModel.py`, `OptimizedEstimations.py`, (documentation only)

### Generated Output Structure
```
results_folder/
├── parameter_estimates/
│   ├── all_estimates.csv       # All simulation results
│   └── valid_estimates.csv     # Successful estimations only
├── time_series/                # Simulation data for visualization
├── summary/
│   ├── summary_statistics.txt  # Statistical analysis
│   └── configuration.txt       # Run parameters
└── visuals/                    # Generated plots (if created)
```

## Scientific Background

### Mathematical Model

The Lindley Random Walk for snow depth follows:
```
X_t = max(X_{t-1} + C_t, 0)
```

Where:
- `X_t`: Snow depth at time t
- `C_t`: Daily change in snow depth
- `C_t ~ N(μ_t, σ²)` with periodic mean:

```
μ_t = α₀ + α₁ cos(2π(ν - τ₁)/P) + α₂t
```

### Parameters
- `α₀`: Baseline daily change level
- `α₁`: Seasonal variation amplitude  
- `τ₁`: Phase shift (day of minimum change)
- `σ`: Daily variability standard deviation
- `α₂`: Long-term linear trend

### Literature
- Woody, J. et al. "A storage model approach to the assessment of snow depth trends"
- Virden, C. "Snow Trends and a Lindley Random Walk"

## Performance

### Optimization Features
- **Vectorized Calculations**: NumPy-based operations for 10-50x speedup
- **SIMD Processing**: Bulk operations instead of loops
- **Memory Efficient**: Optimized data structures for large studies
- **Parallel-Friendly**: Algorithm design supports future parallelization

### Benchmarks
- **1,000 simulations**: ~25 minutes (optimized) vs ~12 hours (original)
- **10,000 simulations**: ~4 hours (optimized) vs ~5 days (original)
- **Success Rate**: Typically 100% parameter estimation convergence

### Scaling Projections
Use `run_comparison.py` to generate custom performance projections for your hardware.

## Example Output

### Parameter Estimation Results
```
Parameter  | True       | Mean       | Std        | Bias      
------------------------------------------------------------
alpha0     | -0.600000  | -0.601234  | 0.045678   | -0.001234
alpha1     |  1.000000  |  0.998765  | 0.034567   | -0.001235
tau1       |274.000000  |273.856789  | 12.345678  | -0.143211
sigma      |  1.000000  |  1.002345  | 0.023456   |  0.002345
alpha2     | -0.000005  | -0.000005  | 0.000012   |  0.000000
```

### Visualization Examples
- Time series plots showing seasonal snow accumulation/melt cycles
- Parameter convergence analysis across simulations
- Statistical distribution plots for estimation quality assessment
- Winter season validation with realistic snow depth patterns

## Contributing

### Development Guidelines
1. Maintain clear separation between original and AI-assisted code
2. Include comprehensive docstrings for all functions
3. Follow existing code style and organization
4. Add appropriate AI assistance disclosures to new files

### Testing
Run comparison studies to validate any changes:
```bash
python run_comparison.py --test
```

## License

This project is available under the MIT License. See LICENSE file for details.

## Author

**Joshua King**  
Date: June 9, 2025

## 🔗 Keywords

Snow modeling, Lindley process, Monte Carlo simulation, Climate analysis, Parameter estimation, Stochastic processes, Environmental statistics, Climate change, Hydrology, Snow water equivalent
