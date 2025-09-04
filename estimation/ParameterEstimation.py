#!/usr/bin/env python3
"""
Parameter Estimation for Univariate Lindley Snow Model

This module handles all parameter estimation tasks including:
- Maximum likelihood estimation
- Convergence analysis over multiple simulations  
- Confidence interval calculation
- Parameter estimation validation and diagnostics

Author: Joshua King
Date: September 4, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd


def log_likelihood_univariate(params, depths, changes, temperatures):
    """
    Calculate the log-likelihood for the univariate model
    
    For the univariate model: C_t ~ N(b0 + b1 * temp_t, σ²)
    Uses vectorized operations for improved performance.
    
    Parameters:
    -----------
    params : array-like
        [b0, b1, sigma] - model parameters
    depths : array-like
        Snow depths at each time step
    changes : array-like
        Daily changes in snow depth
    temperatures : array-like
        Daily temperature values
        
    Returns:
    --------
    float
        Log-likelihood value
    """
    b0, b1, sigma = params
    
    if sigma <= 0:
        return -np.inf
    
    # Vectorized computation of expected changes
    mu_t = b0 + b1 * temperatures
    
    # Split into positive and zero depth cases
    positive_depths = depths > 0
    zero_depths = ~positive_depths
    
    log_lik = 0.0
    
    # Case 1: X_t > 0 - use density f_{C_t}(C_t)
    if np.any(positive_depths):
        changes_pos = changes[positive_depths]
        mu_t_pos = mu_t[positive_depths]
        log_lik += np.sum(norm.logpdf(changes_pos, mu_t_pos, sigma))
    
    # Case 2: X_t = 0 - use CDF F_{C_t}(-X_{t-1})
    if np.any(zero_depths):
        zero_indices = np.where(zero_depths)[0]
        
        # Get previous depths efficiently
        x_prev = np.zeros(len(zero_indices))
        for i, idx in enumerate(zero_indices):
            if idx > 0:
                x_prev[i] = depths[idx - 1]
        
        mu_t_zero = mu_t[zero_depths]
        log_lik += np.sum(norm.logcdf(-x_prev, mu_t_zero, sigma))
    
    return log_lik


def estimate_parameters(depths, changes, temperatures, initial_guess=None, verbose=False):
    """
    Estimate parameters using maximum likelihood estimation
    
    Parameters:
    -----------
    depths : array-like
        Snow depths at each time step
    changes : array-like
        Daily changes in snow depth
    temperatures : array-like
        Daily temperature values
    initial_guess : array-like, optional
        Initial parameter guess [b0, b1, sigma]
    verbose : bool, default=False
        Print estimation progress
        
    Returns:
    --------
    tuple
        (estimates, success) - parameter estimates and convergence flag
    """
    if verbose:
        print("  Starting parameter estimation...")
    
    if initial_guess is None:
        # Smart initial guess based on data
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # Linear regression estimate for initial b1
        temp_mean = np.mean(temperatures)
        temp_centered = temperatures - temp_mean
        change_centered = changes - mean_change
        
        # Use numpy's efficient dot product
        numerator = np.dot(temp_centered, change_centered)
        denominator = np.dot(temp_centered, temp_centered)
        
        b1_est = numerator / denominator if denominator != 0 else -0.1
        b0_est = mean_change - b1_est * temp_mean
        
        initial_guess = [b0_est, b1_est, std_change]
        
        if verbose:
            print(f"  Initial guess: b0={b0_est:.4f}, b1={b1_est:.4f}, sigma={std_change:.4f}")
    
    # Set bounds based on data characteristics
    temp_range = np.max(temperatures) - np.min(temperatures)
    change_range = np.max(changes) - np.min(changes)
    
    bounds = [
        (np.min(changes) - 2, np.max(changes) + 2),  # b0
        (-change_range/temp_range, change_range/temp_range),  # b1
        (0.01, 3 * np.std(changes))  # sigma (positive)
    ]
    
    if verbose:
        print("  Running optimization...")
    
    # Maximum likelihood estimation
    result = minimize(
        lambda params: -log_likelihood_univariate(params, depths, changes, temperatures),
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if verbose:
        print(f"  Optimization {'converged' if result.success else 'failed'}")
        if result.success:
            print(f"  Final estimates: b0={result.x[0]:.4f}, b1={result.x[1]:.4f}, sigma={result.x[2]:.4f}")
    
    return result.x, result.success


def calculate_confidence_intervals(depths, changes, temperatures, estimates, alpha=0.05):
    """
    Calculate confidence intervals using the Hessian matrix
    
    Parameters:
    -----------
    depths : array-like
        Snow depths at each time step
    changes : array-like
        Daily changes in snow depth
    temperatures : array-like
        Daily temperature values
    estimates : array-like
        Parameter estimates [b0, b1, sigma]
    alpha : float, default=0.05
        Significance level (0.05 for 95% confidence intervals)
        
    Returns:
    --------
    tuple
        (lower_bounds, upper_bounds, std_errors) or (None, None, None) if fails
    """
    # Calculate Hessian matrix numerically
    def neg_log_lik(params):
        return -log_likelihood_univariate(params, depths, changes, temperatures)
    
    # Use finite differences to approximate Hessian
    eps = 1e-5
    n_params = len(estimates)
    hessian = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            params_pp = estimates.copy()
            params_pm = estimates.copy()
            params_mp = estimates.copy()
            params_mm = estimates.copy()
            
            params_pp[i] += eps
            params_pp[j] += eps
            
            params_pm[i] += eps
            params_pm[j] -= eps
            
            params_mp[i] -= eps
            params_mp[j] += eps
            
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            hessian[i, j] = (neg_log_lik(params_pp) - neg_log_lik(params_pm) - 
                           neg_log_lik(params_mp) + neg_log_lik(params_mm)) / (4 * eps**2)
    
    try:
        # Invert Hessian to get covariance matrix
        cov_matrix = np.linalg.inv(hessian)
        
        # Standard errors are square root of diagonal elements
        std_errors = np.sqrt(np.diag(cov_matrix))
        
        # Calculate confidence intervals
        z_alpha = norm.ppf(1 - alpha/2)
        
        lower_bounds = estimates - z_alpha * std_errors
        upper_bounds = estimates + z_alpha * std_errors
        
        return lower_bounds, upper_bounds, std_errors
    except:
        # If Hessian inversion fails, return None
        return None, None, None


def run_convergence_study(model, temperatures, true_params, n_simulations=100, 
                         random_seed=None, verbose=True):
    """
    Run convergence study to validate parameter estimation
    
    Parameters:
    -----------
    model : UnivariateSnowModel
        The snow model to use for simulations
    temperatures : array-like
        Temperature data for simulations
    true_params : array-like
        True parameter values [b0, b1, sigma]
    n_simulations : int, default=100
        Number of simulations to run
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information
        
    Returns:
    --------
    dict
        Results dictionary with estimates and convergence information
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if verbose:
        print(f"Running convergence study with {n_simulations} simulations...")
        print(f"True parameters: b0={true_params[0]:.3f}, b1={true_params[1]:.3f}, sigma={true_params[2]:.3f}")
    
    # Initialize storage
    all_estimates = []
    all_convergence = []
    
    for sim in range(n_simulations):
        if verbose and (sim + 1) % 20 == 0:
            print(f"  Simulation {sim + 1}/{n_simulations} ({100*(sim+1)/n_simulations:.0f}%)")
        
        # Simulate data using the model
        changes, depths = model.simulate(temperatures, x0=0)
        
        # Estimate parameters
        estimates, success = estimate_parameters(depths, changes, temperatures)
        
        all_estimates.append(estimates)
        all_convergence.append(success)
    
    # Convert to arrays
    all_estimates = np.array(all_estimates)
    all_convergence = np.array(all_convergence)
    
    # Filter successful estimations
    successful_estimates = all_estimates[all_convergence]
    
    # Calculate statistics
    results = {
        'true_params': true_params,
        'param_names': ['b0', 'b1', 'sigma'],
        'all_estimates': all_estimates,
        'all_convergence': all_convergence,
        'successful_estimates': successful_estimates,
        'convergence_rate': np.mean(all_convergence),
        'n_successful': np.sum(all_convergence),
        'n_total': n_simulations,
    }
    
    if len(successful_estimates) > 0:
        results['mean_estimates'] = np.mean(successful_estimates, axis=0)
        results['std_estimates'] = np.std(successful_estimates, axis=0)
        results['bias'] = results['mean_estimates'] - true_params
        results['rmse'] = np.sqrt(np.mean((successful_estimates - true_params)**2, axis=0))
    
    if verbose:
        print(f"\nConvergence study completed:")
        print(f"  Convergence rate: {results['convergence_rate']:.1%}")
        if len(successful_estimates) > 0:
            print(f"  Mean estimates: b0={results['mean_estimates'][0]:.4f}, "
                  f"b1={results['mean_estimates'][1]:.4f}, sigma={results['mean_estimates'][2]:.4f}")
            print(f"  Bias: b0={results['bias'][0]:.4f}, "
                  f"b1={results['bias'][1]:.4f}, sigma={results['bias'][2]:.4f}")
    
    return results


def plot_convergence_analysis(results, save_path=None):
    """
    Create convergence analysis plots
    
    Parameters:
    -----------
    results : dict
        Results from run_convergence_study
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if results['n_successful'] == 0:
        print("No successful estimations to plot!")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    true_params = results['true_params']
    successful_estimates = results['successful_estimates']
    param_names = results['param_names']
    
    # Plot 1-3: Parameter convergence over simulations
    for i, param_name in enumerate(param_names):
        ax = axes[0, i]
        
        # True parameter as horizontal line
        ax.axhline(y=true_params[i], color='red', linestyle='--', linewidth=2, 
                  label=f'True {param_name}={true_params[i]:.3f}')
        
        # Estimated parameters
        ax.plot(successful_estimates[:, i], color='blue', alpha=0.7, linewidth=1,
               label=f'Estimates')
        
        # Running mean
        running_mean = np.cumsum(successful_estimates[:, i]) / np.arange(1, len(successful_estimates) + 1)
        ax.plot(running_mean, color='green', linewidth=2, label='Running Mean')
        
        ax.set_title(f'Parameter {param_name} Convergence')
        ax.set_xlabel('Simulation Number')
        ax.set_ylabel(f'{param_name} Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Parameter distribution histograms
    for i, param_name in enumerate(param_names):
        ax = axes[1, i]
        
        # Histogram of estimates
        ax.hist(successful_estimates[:, i], bins=30, density=True, alpha=0.7, 
               color='blue', edgecolor='black', label='Estimates')
        
        # True parameter line
        ax.axvline(x=true_params[i], color='red', linestyle='--', linewidth=2,
                  label=f'True {param_name}')
        
        # Mean estimate line
        mean_est = results['mean_estimates'][i]
        ax.axvline(x=mean_est, color='green', linestyle='-', linewidth=2,
                  label=f'Mean Est.')
        
        ax.set_title(f'{param_name} Distribution')
        ax.set_xlabel(f'{param_name} Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        bias = results['bias'][i]
        rmse = results['rmse'][i]
        stats_text = f'Bias: {bias:.4f}\nRMSE: {rmse:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=9)
    
    plt.suptitle(f'Parameter Estimation Convergence Analysis\n'
                f'Convergence Rate: {results["convergence_rate"]:.1%} '
                f'({results["n_successful"]}/{results["n_total"]} simulations)',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plot saved: {save_path}")
    
    return fig


def print_estimation_summary(results, confidence_intervals=None):
    """
    Print a comprehensive parameter estimation summary
    
    Parameters:
    -----------
    results : dict
        Results from run_convergence_study
    confidence_intervals : tuple, optional
        (lower_bounds, upper_bounds, std_errors) from calculate_confidence_intervals
    """
    print("\n" + "="*70)
    print("PARAMETER ESTIMATION SUMMARY")
    print("="*70)
    
    print(f"\nSimulation Details:")
    print(f"  Total simulations: {results['n_total']}")
    print(f"  Successful estimations: {results['n_successful']}")
    print(f"  Convergence rate: {results['convergence_rate']:.1%}")
    
    if results['n_successful'] > 0:
        print(f"\nParameter Estimates:")
        print(f"{'Parameter':<10} {'True':<10} {'Mean Est':<10} {'Std Dev':<10} {'Bias':<10} {'RMSE':<10}")
        print("-" * 70)
        
        for i, param in enumerate(results['param_names']):
            true_val = results['true_params'][i]
            mean_est = results['mean_estimates'][i]
            std_est = results['std_estimates'][i]
            bias = results['bias'][i]
            rmse = results['rmse'][i]
            
            print(f"{param:<10} {true_val:<10.4f} {mean_est:<10.4f} {std_est:<10.4f} "
                  f"{bias:<10.4f} {rmse:<10.4f}")
        
        if confidence_intervals is not None:
            lower, upper, std_errors = confidence_intervals
            if lower is not None:
                print(f"\n95% Confidence Intervals (single simulation):")
                print(f"{'Parameter':<10} {'Lower':<10} {'Upper':<10} {'Width':<10} {'Std Error':<10}")
                print("-" * 60)
                
                for i, param in enumerate(results['param_names']):
                    width = upper[i] - lower[i]
                    print(f"{param:<10} {lower[i]:<10.4f} {upper[i]:<10.4f} "
                          f"{width:<10.4f} {std_errors[i]:<10.4f}")
    
    print("\nModel Interpretation:")
    if results['n_successful'] > 0:
        b0_est = results['mean_estimates'][0]
        b1_est = results['mean_estimates'][1]
        sigma_est = results['mean_estimates'][2]
        
        print(f"  b0 = {b0_est:.4f}: Baseline daily change in snow depth")
        if b0_est > 0:
            print(f"    → Positive baseline suggests net accumulation tendency")
        else:
            print(f"    → Negative baseline suggests net melting tendency")
        
        print(f"  b1 = {b1_est:.4f}: Temperature coefficient")
        if b1_est < 0:
            print(f"    → Negative coefficient: higher temperatures increase melting")
            print(f"    → Each 1°C increase causes {-b1_est:.3f} cm more daily melting")
        else:
            print(f"    → Positive coefficient: higher temperatures increase accumulation")
        
        print(f"  sigma = {sigma_est:.4f}: Daily variability (standard deviation)")
        print(f"    → ~68% of daily changes within ±{sigma_est:.2f} cm of expected value")
