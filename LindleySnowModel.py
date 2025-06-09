#!/usr/bin/env python3
"""
Lindley Snow Model Implementation

Core implementation of the Lindley Random Walk model for snow depth simulation
based on 'A storage model approach to the assessment of snow depth trends' 
(Jonathan Woody et al.) and 'Snow Trends and a Lindley Random Walk' 
(Caroline Virden).

AI Assistance Disclosure:
This file contains original implementation with AI assistance limited to code 
comments, documentation formatting, and minor structural improvements.
All mathematical implementations and algorithmic logic are original work.

Author: Joshua King
Date: Jun 9, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

# Step 1: Define the model parameters and functions
class LindleySnowModel:
    def __init__(self, alpha0=-0.6, alpha1=1.0, tau1=274, sigma=1.0, alpha2=-0.000005):
        """
        Initialize the Lindley Snow Model with parameters from the paper
        
        Parameters:
        - alpha0: baseline level of the periodic mean
        - alpha1: amplitude of the seasonal variation
        - tau1: phase shift (day of year for minimum)
        - sigma: standard deviation of random noise
        - alpha2: linear trend coefficient (negative indicates decreasing trend)
        """
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.tau1 = tau1
        self.sigma = sigma
        self.alpha2 = alpha2
        self.P = 365  # Period (days in a year)
    
    def periodic_mean(self, day_of_year, year):
        """
        Calculate the periodic mean μ_ν for a given day and year and return t
        
        μ_ν = α₀ + α₁ * cos(2π(ν - τ₁)/P) + α₂ * t
        where t is the total day number from start
        
        Returns μ_ν, t
        """
        t = (year - 1) * self.P + day_of_year
        seasonal_component = self.alpha1 * np.cos(2 * np.pi * (day_of_year - self.tau1) / self.P)
        return self.alpha0 + seasonal_component + self.alpha2 * t, t
    
    def periodic_changes(self, n_years):
        """
        Generate the daily changes C_t process
        
        C_t = μ_ν + α₂t + ε_t where ε_t ~ N(0, σ²)
        """
        total_days = n_years * self.P
        changes = np.zeros(total_days) # Initialize 0's
        
        for year in range(1, n_years + 1):
            for day in range(1, self.P + 1):
                idx = (year - 1) * self.P + (day - 1) # Convert to 0-based indexing
                mu_nu, t = self.periodic_mean(day, year)
                alpha2_t = self.alpha2 * t
                epsilon_t = np.random.normal(0, self.sigma)
                changes[idx] = mu_nu + alpha2_t + epsilon_t
        
        return changes
    
    def storage_balance_equation(self, changes, x0=0):
        """
        Apply the Lindley recursion to generate snow depths
        
        X_t = max(X_{t-1} + C_t, 0)
        """
        n = len(changes)
        depths = np.zeros(n + 1) # Initialize snow depths
        depths[0] = x0 # Define start snow depth
        
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

# Step 2: Likelihood function for parameter estimation
def log_likelihood(params, depths, changes):
    """
    Calculate the log-likelihood for the observed data
    
    L(Θ|X) = Π_{t=1}^{NP} [1_{X_t>0} * f_{C_t}(C_t) + 1_{X_t=0} * F_{C_t}(-X_{t-1})]
    """
    alpha0, alpha1, tau1, sigma, alpha2 = params
    
    if sigma <= 0: # Log-Likelihood standard deviation must be positive
        return -np.inf
    
    log_lik = 0 # Initialize with 0
    P = 365
    n_total = len(depths)
    
    for t in range(n_total): # Loop here could be optimized for performance
        # Convert array indexing back to year/day
        year = t // P + 1
        day_of_year = t % P + 1
        
        # Calculate μ_t
        total_day = (year - 1) * P + day_of_year
        seasonal_comp = alpha1 * np.cos(2 * np.pi * (day_of_year - tau1) / P)
        mu_t = alpha0 + seasonal_comp + alpha2 * total_day
        
        if depths[t] > 0: # If branch within loop can be optimized to boolean indexing
            # X_t > 0: use density f_{C_t}(C_t)
            log_lik += norm.logpdf(changes[t], mu_t, sigma) # Log of Normal Gaussian Density
                                                            # better for computer computation
        else:
            # X_t = 0: use CDF F_{C_t}(-X_{t-1})
            if t == 0:
                x_prev = 0
            else:
                x_prev = depths[t-1]
            log_lik += norm.logcdf(-x_prev, mu_t, sigma) # Log of Gaussian Cumulative Distribution
    
    return log_lik

def estimate_parameters(depths, changes, initial_guess=None):
    """
    Estimate parameters using maximum likelihood estimation
    """
    if initial_guess is None:
        initial_guess = [-0.6, 1.0, 274, 1.0, -0.000005]
    
    # Define bounds to ensure sigma > 0
    bounds = [(-5, 5), (-5, 5), (1, 365), (0.1, 10), (-0.001, 0.001)]
    
    """
    Parameter bounds reasoning:
    
    α₀ ∈ (-5, 5): Baseline level of daily changes
        • Negative = overall melting tendency, Positive = accumulation tendency
        • α₀ = -5 would mean losing 5cm/day on average (extreme melting)
        • α₀ = +5 would mean gaining 5cm/day on average (extreme accumulation)
        • True value -0.6 represents modest overall melting trend
    
    α₁ ∈ (-5, 5): Seasonal amplitude
        • Controls peak-to-trough seasonal variation in daily changes
        • α₁ = 1.0 means ±1cm variation from baseline between seasons
        • α₁ = 5 would mean 10cm swing between winter and summer (extreme)
        • Negative α₁ would flip seasons (winter=melting, summer=snow) - unusual but allowed
    
    τ₁ ∈ (1, 365): Phase shift (day of year for minimum change)
        • Must be valid calendar day (Jan 1 = 1, Dec 31 = 365)
        • τ₁ = 150 ≈ May 30th, meaning peak melting in late spring
        • Physically reasonable: maximum snow loss in late spring/early summer
    
    σ ∈ (0.1, 10): Standard deviation of daily noise
        • Must be positive for normal distribution to be defined
        • σ = 0.1 = very predictable daily changes (minimal weather variability)
        • σ = 10 = extremely chaotic weather (daily changes could be ±20cm)
        • True value 1.0 means ~68% of days within ±1cm of seasonal trend
    
    α₂ ∈ (-0.001, 0.001): Linear climate trend per day
        • Captures long-term climate change effects
        • α₂ = -0.000005 means -0.000005 cm/day decline
        • Over 100 years: -0.000005 × 365 × 100 = -0.18 cm total (gradual)
        • α₂ = ±0.001 over 100 years = ±36.5 cm (major climate shift)
        • Range allows for various climate scenarios while staying realistic
    
    These bounds are conservative "safety nets" - wide enough to avoid excluding
    the true solution while preventing the optimizer from wandering into 
    physically implausible parameter space.
    """
    
    # Maximum Likelihood Estimation using scipy.optimize.minimize
    # This is the core of parameter estimation - finding values that make our data most likely
    
    result = minimize(
        # Objective function: we minimize the NEGATIVE log-likelihood - max(f(x)) = min(-f(x))
        lambda params: -log_likelihood(params, depths, changes),
        
        # Starting point for optimization search - searches for better parameter values
        initial_guess,
        
        # L-BFGS-B: Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds
        # - Quasi-Newton optimization algorithm
        # - Works well with smooth likelihood functions
        # - Efficiently handles bound constraints (e.g., σ > 0)
        # - Good balance of speed and robustness
        method='L-BFGS-B',
        
        # Parameter bounds: [(min₁,max₁), (min₂,max₂), ...]
        # Keeps optimizer within physically meaningful parameter space
        bounds=bounds
    )
    
    # Return the results:
    # result.x: Best parameter estimates [α₀, α₁, τ₁, σ, α₂] found by optimizer
    # result.success: Boolean indicating if optimization converged successfully
    #                 False means don't trust the parameter estimates
    return result.x, result.success