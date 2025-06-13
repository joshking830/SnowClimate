#!/usr/bin/env python3
"""
Lindley Snow Model Implementation with Periodic Variance

Core implementation of the Lindley Random Walk model for snow depth simulation
with periodic variance in addition to periodic mean.

Mathematical Model:
- Mean: μ_t = α₀ + α₁ * cos(2π(ν - τ₁)/P) + α₂ * t
- Standard deviation: σ_t = β₀ + β₁ * cos(2π(ν - τ₂)/P) + β₂ * t
- Changes: C_t ~ N(μ_t, σ²_t)
- Snow depths: X_t = max(X_{t-1} + C_t, 0)

Where:
- ν = day of year (1-365)
- t = total day number from start
- P = 365 (period)

Author: Joshua King
Date: Jun 12, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

class LindleyPeriodicVarModel:
    def __init__(self, alpha0=-0.6, alpha1=1.0, tau1=274, alpha2=-0.000005,
                 beta0=1.0, beta1=0.5, tau2=274, beta2=0.0):
        """
        Initialize the Lindley Snow Model with periodic variance
        
        Mean parameters:
        - alpha0: baseline level of the periodic mean
        - alpha1: amplitude of the seasonal variation in mean
        - tau1: phase shift for mean (day of year for minimum)
        - alpha2: linear trend coefficient for mean
        
        Variance parameters:
        - beta0: baseline level of the periodic standard deviation
        - beta1: amplitude of the seasonal variation in standard deviation
        - tau2: phase shift for variance (day of year for minimum variance)
        - beta2: linear trend coefficient for standard deviation
        """
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.tau1 = tau1
        self.alpha2 = alpha2
        
        self.beta0 = beta0
        self.beta1 = beta1
        self.tau2 = tau2
        self.beta2 = beta2
        
        self.P = 365  # Period (days in a year)
    
    def periodic_mean(self, day_of_year, year):
        """
        Calculate the periodic mean μ_t for a given day and year
        
        μ_t = α₀ + α₁ * cos(2π(ν - τ₁)/P) + α₂ * t
        
        Returns: (μ_t, t)
        """
        t = (year - 1) * self.P + day_of_year
        seasonal_component = self.alpha1 * np.cos(2 * np.pi * (day_of_year - self.tau1) / self.P)
        mu_t = self.alpha0 + seasonal_component + self.alpha2 * t
        return mu_t, t
    
    def periodic_std_piecewise(self, day_of_year, year):
        """
        Calculate periodic standard deviation with two distinct high-variance periods
        
        Pattern:
        - High variance period 1: days tau2 to tau2+30 
        - Low variance: days tau2+30 to tau2+120
        - High variance period 2: days tau2+120 to tau2+150
        - Low variance: days tau2+150 to tau2+365 (wrapping around)
        
        Returns: (σ_t, t)
        """
        t = (year - 1) * self.P + day_of_year
        
        # Normalize day_of_year to handle year wrapping
        normalized_day = ((day_of_year - self.tau2) % self.P)
        
        # Define the periods (can be customized)
        high_period_1_start = 0       # Start at tau2
        high_period_1_end = 30        # 30 days of high variance
        low_period_1_end = 120        # 90 days of low variance  
        high_period_2_start = 120     # Start second high period
        high_period_2_end = 150       # 30 days of high variance
        # Remaining days (150-365) are low variance
        
        # Define high and low variance levels
        high_std = self.beta0 + self.beta1     # beta0 + beta1 = high variance
        low_std = self.beta0 - self.beta1     # beta0 - beta1 = low variance
        
        # Determine which period we're in
        if high_period_1_start <= normalized_day < high_period_1_end:
            # High variance period 1
            sigma_t = high_std
        elif high_period_1_end <= normalized_day < low_period_1_end:
            # Low variance period 1  
            sigma_t = low_std
        elif high_period_2_start <= normalized_day < high_period_2_end:
            # High variance period 2
            sigma_t = high_std
        else:
            # Low variance period 2 (remainder of year)
            sigma_t = low_std
        
        # Add linear trend
        sigma_t += self.beta2 * t
        
        # Ensure positive
        sigma_t = max(sigma_t, 0.01)
        
        return sigma_t, t
    
    def periodic_std(self, day_of_year, year):
        """
        Calculate the periodic standard deviation σ_t for a given day and year
        
        Choose between smooth cosine pattern or piecewise pattern
        """
        # Use piecewise function for two distinct high-variance periods
        return self.periodic_std_piecewise(day_of_year, year)
        
        # Alternative: use smooth cosine (comment out the line above and uncomment below)
        # t = (year - 1) * self.P + day_of_year
        # seasonal_component = self.beta1 * np.cos(2 * np.pi * (day_of_year - self.tau2) / self.P)
        # sigma_t = self.beta0 + seasonal_component + self.beta2 * t
        # sigma_t = max(sigma_t, 0.01)
        # return sigma_t, t
    
    def periodic_changes(self, n_years):
        """
        Generate the daily changes C_t process with periodic variance
        
        C_t ~ N(μ_t, σ²_t) where both μ_t and σ_t vary periodically
        """
        total_days = n_years * self.P
        changes = np.zeros(total_days)
        
        for year in range(1, n_years + 1):
            for day in range(1, self.P + 1):
                idx = (year - 1) * self.P + (day - 1)  # Convert to 0-based indexing
                
                # Get periodic mean and standard deviation
                mu_t, t = self.periodic_mean(day, year)
                sigma_t, _ = self.periodic_std(day, year)
                
                # Generate random change with time-varying variance
                epsilon_t = np.random.normal(0, sigma_t)
                changes[idx] = mu_t + epsilon_t
        
        return changes
    
    def storage_balance_equation(self, changes, x0=0):
        """
        Apply the Lindley recursion to generate snow depths
        
        X_t = max(X_{t-1} + C_t, 0)
        """
        n = len(changes)
        depths = np.zeros(n + 1)
        depths[0] = x0
        
        for t in range(n):
            depths[t + 1] = max(depths[t] + changes[t], 0)
        
        return depths[1:]  # Return X_1, X_2, ..., X_n
    
    def simulate_snow_depths(self, n_years, x0=0):
        """
        Complete simulation: generate changes and apply Lindley recursion
        """
        changes = self.periodic_changes(n_years)
        depths = self.storage_balance_equation(changes, x0)
        return changes, depths
    
    def get_theoretical_variance_profile(self, year=1):
        """
        Get the theoretical variance profile for a given year
        Useful for visualization and validation
        """
        days = np.arange(1, 366)
        variances = []
        
        for day in days:
            sigma_t, _ = self.periodic_std(day, year)
            variances.append(sigma_t**2)
        
        return days, np.array(variances)
    
    def get_theoretical_mean_profile(self, year=1):
        """
        Get the theoretical mean profile for a given year
        Useful for visualization and validation
        """
        days = np.arange(1, 366)
        means = []
        
        for day in days:
            mu_t, _ = self.periodic_mean(day, year)
            means.append(mu_t)
        
        return days, np.array(means)

def log_likelihood_periodic_var(params, depths, changes):
    """
    OPTIMIZED: Calculate the log-likelihood for the periodic variance model using vectorized operations
    
    This is more complex than the constant variance case because
    each observation has a different variance σ²_t
    
    L(Θ|X) = Π_{t=1}^{NP} [1_{X_t>0} * f_{C_t}(C_t|μ_t,σ²_t) + 1_{X_t=0} * F_{C_t}(-X_{t-1}|μ_t,σ²_t)]
    
    OPTIMIZATIONS:
    1. Vectorized calculations instead of loops
    2. Pre-compute arrays for seasonal components (both mean and variance)
    3. Use boolean indexing for conditional operations
    4. Minimize redundant calculations
    """
    if len(params) == 8:
        alpha0, alpha1, tau1, alpha2, beta0, beta1, tau2, beta2 = params
    else:
        # Fallback to constant variance if only 5 parameters
        alpha0, alpha1, tau1, sigma_const, alpha2 = params
        beta0, beta1, tau2, beta2 = sigma_const, 0, 274, 0
    
    # Ensure positive variance parameters
    if beta0 <= 0:
        return -np.inf
    
    P = 365
    n_total = len(depths)
    
    # PRE-COMPUTE: Convert array indices to year/day arrays (vectorized)
    t_indices = np.arange(n_total)
    years = t_indices // P + 1
    days_of_year = t_indices % P + 1
    total_days = (years - 1) * P + days_of_year
    
    # PRE-COMPUTE: Seasonal components for MEAN (vectorized)
    seasonal_mean = alpha1 * np.cos(2 * np.pi * (days_of_year - tau1) / P)
    mu_t_array = alpha0 + seasonal_mean + alpha2 * total_days
    
    # PRE-COMPUTE: Seasonal components for VARIANCE with TWO CYCLES (vectorized)
    seasonal_std = beta1 * np.cos(4 * np.pi * (days_of_year - tau2) / P)  # 4π for two cycles
    sigma_t_array = beta0 + seasonal_std + beta2 * total_days
    
    # ENSURE: All standard deviations are positive (vectorized)
    sigma_t_array = np.maximum(sigma_t_array, 0.01)
    
    # SPLIT: Identify positive depth vs zero depth cases (boolean indexing)
    positive_depths = depths > 0
    zero_depths = ~positive_depths
    
    # CASE 1: X_t > 0 - use density f_{C_t}(C_t|μ_t,σ_t) (vectorized)
    log_lik_positive = 0.0
    if np.any(positive_depths):
        changes_pos = changes[positive_depths]
        mu_t_pos = mu_t_array[positive_depths]
        sigma_t_pos = sigma_t_array[positive_depths]
        log_lik_positive = np.sum(norm.logpdf(changes_pos, mu_t_pos, sigma_t_pos))
    
    # CASE 2: X_t = 0 - use CDF F_{C_t}(-X_{t-1}|μ_t,σ_t) (vectorized)
    log_lik_zero = 0.0
    if np.any(zero_depths):
        zero_indices = np.where(zero_depths)[0]
        
        # Get previous depths for zero cases (vectorized approach)
        x_prev_array = np.zeros(len(zero_indices))
        valid_prev = zero_indices > 0  # Boolean mask for valid previous indices
        if np.any(valid_prev):
            prev_indices = zero_indices[valid_prev] - 1
            x_prev_array[valid_prev] = depths[prev_indices]
        # x_prev_array[~valid_prev] remains 0 (correct for t=0 case)
        
        mu_t_zero = mu_t_array[zero_depths]
        sigma_t_zero = sigma_t_array[zero_depths]
        log_lik_zero = np.sum(norm.logcdf(-x_prev_array, mu_t_zero, sigma_t_zero))
    
    return log_lik_positive + log_lik_zero

def estimate_parameters_periodic_var(depths, changes, initial_guess=None):
    """
    OPTIMIZED: Estimate parameters for the periodic variance model using maximum likelihood
    
    Parameters to estimate: [α₀, α₁, τ₁, α₂, β₀, β₁, τ₂, β₂]
    
    OPTIMIZATIONS:
    1. Uses optimized vectorized log-likelihood function
    2. Smart initial guess based on data characteristics
    3. Adaptive bounds based on data properties
    4. High-precision optimizer settings
    """
    if initial_guess is None:
        # SMART: Data-driven initial guess
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # Estimate seasonal patterns from data
        n_days = len(changes)
        if n_days >= 365:
            # Try to estimate seasonal amplitude from first year
            first_year_changes = changes[:365]
            seasonal_est_mean = np.std(first_year_changes) * 0.5
            seasonal_est_std = std_change * 0.3  # Variance should vary less than mean
        else:
            seasonal_est_mean = std_change * 0.5
            seasonal_est_std = std_change * 0.3
        
        initial_guess = [
            mean_change,          # alpha0: empirical mean
            seasonal_est_mean,    # alpha1: estimated seasonal amplitude for mean
            355,                  # tau1: December (winter accumulation)
            0.0,                  # alpha2: no initial trend assumption
            std_change,           # beta0: empirical standard deviation
            seasonal_est_std,     # beta1: estimated seasonal amplitude for std
            200,                  # tau2: July (summer variability peak)
            0.0                   # beta2: no trend in variance
        ]
    
    # ADAPTIVE: Bounds based on data characteristics
    change_range = np.max(changes) - np.min(changes)
    std_change = np.std(changes)
    
    bounds = [
        # Mean parameters
        (np.min(changes) - 1, np.max(changes) + 1),    # alpha0: based on data range
        (0.1, change_range),                            # alpha1: positive, up to data range
        (1, 365),                                       # tau1: valid calendar days
        (-0.001, 0.001),                                # alpha2: small trend coefficient
        
        # Variance parameters
        (0.01, std_change * 3),                         # beta0: positive, reasonable range
        (-std_change * 2, std_change * 2),              # beta1: can be negative (variance decreases)
        (1, 365),                                       # tau2: valid calendar days
        (-0.001, 0.001)                                 # beta2: small trend in variance
    ]
    
    # OPTIMIZE: High-precision settings due to vectorization speedup
    # Due to massive benefits from vectorization (10-50x speedup) we can afford to 
    # prioritize precision over speed in the optimizer settings
    result = minimize(
        lambda params: -log_likelihood_periodic_var(params, depths, changes),
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'ftol': 1e-12,      # Function tolerance (very high precision)
            'gtol': 1e-10,      # Gradient tolerance (very tight)
            'maxiter': 20000,   # More iterations for complex 8-parameter model
            'disp': False       # Don't display convergence messages
        }
    )
    
    return result.x, result.success