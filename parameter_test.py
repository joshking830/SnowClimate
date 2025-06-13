#!/usr/bin/env python3
"""
Simple Snow Series Plot Generator

Used to simulate and visually verify parameters

Runs one 100-year simulation and creates a 4-panel plot:
1. Full 100-year C_t series
2. Last 10 years X_t series  
3. One year X_t (Oct to Oct)
4. One year C_t (Oct to Oct)

Usage: python simple_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from LindleySnowModel import LindleySnowModel

def main():
    # Setup
    os.makedirs("parameter_test", exist_ok=True)
    np.random.seed(42)
    
    # True parameters 'alpha0': -0.6, 'alpha1': 1.0, 'tau1': 274, 'sigma': 1.0, 'alpha2': -0.000005
    true_params = {'alpha0': -0.7, 'alpha1': 1.4, 'tau1': 1, 'sigma': 0.8, 'alpha2': -0.000005}
    
    # Generate 100 years of data
    model = LindleySnowModel(**true_params)
    changes, depths = model.simulate_snow_depths(100)
    
    # Create time arrays
    total_days = len(changes)
    days_since_start = np.arange(total_days)
    years = 1900 + days_since_start / 365.0
    
    # Create 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Full 100-year C_t series
    axes[0, 0].plot(years, changes, 'k-', linewidth=0.3)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('C_t')
    axes[0, 0].set_title('100-Year Changes C_t Series')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-6, 6)
    
    # Plot 2: Last 10 years X_t series
    last_10_years = 10 * 365
    changes_10yr = changes[-last_10_years:]
    depths_10yr = depths[-last_10_years:]
    years_10yr = years[-last_10_years:]
    
    axes[0, 1].plot(years_10yr, depths_10yr, 'b-', linewidth=0.5)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('X_t')
    axes[0, 1].set_title('Last 10 Years - Snow Depths X_t')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)
    
    # Plot 3: One year X_t (Oct to Oct)
    # Start from day 274 of year 90 (1990) to day 273 of year 91 (1991)
    year_start = 89 * 365 + 258  # Day 258 of (Sep 15) year 90 (0-indexed)
    year_end = year_start + 273   # To day 166 (June 15) of next year
    
    depths_1yr = depths[year_start:year_end]
    days_in_year = np.arange(1, len(depths_1yr) + 1)
    
    axes[1, 0].plot(days_in_year, depths_1yr, 'b-', linewidth=1)
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('X_t')
    axes[1, 0].set_title('One Year Snow Depths (Oct 1989 - Oct 1990)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)
    
    # Add month markers
    month_starts = [17, 48, 78, 109, 139, 170, 200, 231, 262]
    month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    axes[1, 0].set_xticks(month_starts)
    axes[1, 0].set_xticklabels(month_names, rotation=45)
    
    # Plot 4: One year C_t (Oct to Oct)
    changes_1yr = changes[year_start:year_end]
    
    axes[1, 1].plot(days_in_year, changes_1yr, 'k-', linewidth=1)
    axes[1, 1].set_xlabel('Day of Year')
    axes[1, 1].set_ylabel('C_t')
    axes[1, 1].set_title('One Year Changes (Oct 1989 - Oct 1990)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-4, 4)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Add month markers
    axes[1, 1].set_xticks(month_starts)
    axes[1, 1].set_xticklabels(month_names, rotation=45)
    
    # Add parameter values at the top
    param_text = f"α₀ = {true_params['alpha0']}, α₁ = {true_params['alpha1']}, τ₁ = {true_params['tau1']}, σ = {true_params['sigma']}, α₂ = {true_params['alpha2']}"
    fig.suptitle(param_text, fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for the parameter title
    plt.savefig("parameter_test/snow_series_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()