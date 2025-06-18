#!/usr/bin/env python3
"""
Simple Snow Series Plot Generator

Used to simulate and visually verify parameters

Runs one 100-year simulation and creates a 4-panel plot:
1. Full 100-year C_t series
2. Last 10 years X_t series  
3. One year X_t (Oct to Oct)
4. One year C_t (Oct to Oct)

Usage: python parameter_test_perVar.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from LindleySnowModel import LindleySnowModel
from LindleyPeriodicVarModel import LindleyPeriodicVarModel

def main():
    # Setup
    os.makedirs("parameter_test", exist_ok=True)
    np.random.seed(42)
    
    # True parameters - EXACTLY as in your original
    true_params = {'alpha0': -0.7, 'alpha1': 1.4, 'tau1': 1, 'sigma': 0.8, 'alpha2': -0.000005}
    
    # True parameters for periodic variance model (same mean, add variance)
    true_params_perVar = {
        'alpha0': -0.55, 'alpha1': 1.2, 'tau1': 355, 'alpha2': -0.000005,
        'beta0': 0.5, 'beta1': 0.3, 'tau2': 200, 'beta2': 0.0
    }
    
    #true_params_perVar = {
    #    'alpha0': -0.55, 'alpha1': 1.2, 'tau1': 355, 'alpha2': -0.000005,
    #   'beta0': 0.5, 'beta1': 0.3, 'tau2': 200, 'beta2': 0.0
    #
    
    # Generate 100 years of data - BOTH models
    model = LindleySnowModel(**true_params)
    changes, depths = model.simulate_snow_depths(100)
    
    model_perVar = LindleyPeriodicVarModel(**true_params_perVar)
    changes_perVar, depths_perVar = model_perVar.simulate_snow_depths(100)
    
    # Create time arrays
    total_days = len(changes)
    days_since_start = np.arange(total_days)
    years = 1900 + days_since_start / 365.0
    
    # Create 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Theoretical variance profiles for the SAME timeframe as plots 3&4
    # Day 258 to day 273 (same as the one-year plots)
    variance_days = np.arange(1, 274)  # 273 days to match plots 3&4
    
    # Get standard deviation profile for the same timeframe
    std_profile_year = []
    for i, day in enumerate(variance_days):
        # Convert to actual day of year for the time period
        # Starting from Sep 15 (day 258) and going for 273 days
        actual_day_of_year = ((258 + i - 1) % 365) + 1
        year = 90  # Same year as plots 3&4
        sigma_t, _ = model_perVar.periodic_std(actual_day_of_year, year)
        std_profile_year.append(sigma_t)
    
    # Constant standard deviation baseline for comparison
    constant_sigma = true_params['sigma']  # This should be 0.8
    constant_sigma_line = np.full(len(variance_days), constant_sigma)
    
    # Force the y-axis to show the full range so we can see both lines
    axes[0, 0].set_ylim(0, max(1.0, max(std_profile_year) + 0.1, constant_sigma + 0.1))
    
    axes[0, 0].plot(variance_days, constant_sigma_line, 'k-', linewidth=2, label=f'Constant Variance (σ={constant_sigma})')
    axes[0, 0].plot(variance_days, std_profile_year, 'r-', linewidth=2, label='Periodic Variance')
    axes[0, 0].set_xlabel('Day of Year')
    axes[0, 0].set_ylabel('Standard Deviation σ_t')
    axes[0, 0].set_title('Theoretical Standard Deviation Profiles (Oct 1989 - Jun 1990)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Add same month markers as plots 3&4
    month_starts = [17, 48, 78, 109, 139, 170, 200, 231, 262]
    month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    axes[0, 0].set_xticks(month_starts)
    axes[0, 0].set_xticklabels(month_names, rotation=45)
    
    # Plot 2: Last 10 years X_t series
    last_10_years = 10 * 365
    changes_10yr = changes[-last_10_years:]
    depths_10yr = depths[-last_10_years:]
    years_10yr = years[-last_10_years:]
    
    changes_10yr_perVar = changes_perVar[-last_10_years:]
    depths_10yr_perVar = depths_perVar[-last_10_years:]
    
    axes[0, 1].plot(years_10yr, depths_10yr, 'b-', linewidth=0.5, label='Constant Variance')
    axes[0, 1].plot(years_10yr, depths_10yr_perVar, 'r-', linewidth=0.5, alpha=0.7, label='Periodic Variance')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('X_t')
    axes[0, 1].set_title('Last 10 Years - Snow Depths X_t')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].legend()
    
    # Plot 3: One year X_t (Oct to Oct)
    # Start from day 274 of year 90 (1990) to day 273 of year 91 (1991)
    year_start = 89 * 365 + 258  # Day 258 of (Sep 15) year 90 (0-indexed)
    year_end = year_start + 273   # To day 166 (June 15) of next year
    
    depths_1yr = depths[year_start:year_end]
    depths_1yr_perVar = depths_perVar[year_start:year_end]
    days_in_year = np.arange(1, len(depths_1yr) + 1)
    
    axes[1, 0].plot(days_in_year, depths_1yr, 'b-', linewidth=1, label='Constant Variance')
    axes[1, 0].plot(days_in_year, depths_1yr_perVar, 'r-', linewidth=1, alpha=0.7, label='Periodic Variance')
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('X_t')
    axes[1, 0].set_title('One Year Snow Depths (Oct 1989 - Oct 1990)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].legend()
    
    # Add month markers
    month_starts = [17, 48, 78, 109, 139, 170, 200, 231, 262]
    month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    axes[1, 0].set_xticks(month_starts)
    axes[1, 0].set_xticklabels(month_names, rotation=45)
    
    # Plot 4: One year C_t (Oct to Oct)
    changes_1yr = changes[year_start:year_end]
    changes_1yr_perVar = changes_perVar[year_start:year_end]
    
    axes[1, 1].plot(days_in_year, changes_1yr, 'k-', linewidth=1, label='Constant Variance')
    axes[1, 1].plot(days_in_year, changes_1yr_perVar, 'r-', linewidth=1, alpha=0.7, label='Periodic Variance')
    axes[1, 1].set_xlabel('Day of Year')
    axes[1, 1].set_ylabel('C_t')
    axes[1, 1].set_title('One Year Changes (Oct 1989 - Oct 1990)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-4, 4)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].legend()
    
    # Add month markers
    axes[1, 1].set_xticks(month_starts)
    axes[1, 1].set_xticklabels(month_names, rotation=45)
    
    # Add parameter values at the top - two lines with matching colors
    param_text_constant = f"Constant Variance: α₀ = {true_params['alpha0']}, α₁ = {true_params['alpha1']}, τ₁ = {true_params['tau1']}, σ = {true_params['sigma']}, α₂ = {true_params['alpha2']}"
    param_text_periodic = f"Periodic Variance: α₀ = {true_params_perVar['alpha0']}, α₁ = {true_params_perVar['alpha1']}, τ₁ = {true_params_perVar['tau1']}, α₂ = {true_params_perVar['alpha2']}, β₀ = {true_params_perVar['beta0']}, β₁ = {true_params_perVar['beta1']}, τ₂ = {true_params_perVar['tau2']}, β₂ = {true_params_perVar['beta2']}"
    
    # Add two-line title with color-coded text
    fig.text(0.5, 0.98, param_text_constant, fontsize=10, ha='center', va='top', color='black')
    fig.text(0.5, 0.95, param_text_periodic, fontsize=10, ha='center', va='top', color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for the parameter title
    plt.savefig("parameter_test/snow_series_perVar_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()