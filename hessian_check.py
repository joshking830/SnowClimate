#!/usr/bin/env python3
"""
Simple Hessian Check for Lindley Snow Model

Runs one simulation, gets parameter estimates and standard errors,
calculates full Hessian and inverse Hessian matrices.

Creates hessiancheck/inverse_hessian_matrix.csv

Verifies sqrt of diagonals are the standard errors of the parameter estimates.

Usage: python hessian_check.py
"""

import numpy as np
import pandas as pd
import os
from LindleySnowModel import LindleySnowModel
from OptimizedEstimations import log_likelihood_optimized, estimate_parameters_optimized

def numerical_hessian(func, x, h=1e-5):
    """Calculate numerical Hessian matrix using finite differences"""
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: f''(x) = [f(x+h) - 2f(x) + f(x-h)] / h²
                x_plus = x.copy(); x_plus[i] += h
                x_minus = x.copy(); x_minus[i] -= h
                hessian[i, j] = (func(x_plus) - 2*func(x) + func(x_minus)) / (h**2)
            else:
                # Off-diagonal: mixed partial derivatives
                x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
    
    return hessian

def main():
    # Setup
    os.makedirs("hessiancheck", exist_ok=True)
    np.random.seed(42)
    
    # True parameters
    true_params = {'alpha0': -0.6, 'alpha1': 1.0, 'tau1': 274, 'sigma': 1.0, 'alpha2': -0.000005}
    param_names = ['alpha0', 'alpha1', 'tau1', 'sigma', 'alpha2']
    
    # Generate data
    model = LindleySnowModel(**true_params)
    changes, depths = model.simulate_snow_depths(100)
    
    # Estimate parameters
    estimated_params, success = estimate_parameters_optimized(depths, changes)
    
    # Calculate Hessian
    def neg_log_likelihood(params):
        return -log_likelihood_optimized(params, depths, changes)
    
    hessian_matrix = numerical_hessian(neg_log_likelihood, estimated_params)
    inv_hessian = np.linalg.inv(hessian_matrix)
    standard_errors = np.sqrt(np.diag(inv_hessian))
    
    # Save results
    # Parameter estimates with standard errors
    results_df = pd.DataFrame({
        'parameter': param_names,
        'true_value': [true_params[p] for p in param_names],
        'estimated_value': estimated_params,
        'standard_error': standard_errors
    })
    results_df.to_csv("hessiancheck/parameter_estimates.csv", index=False)
    
    # Hessian matrix
    hessian_df = pd.DataFrame(hessian_matrix, columns=param_names, index=param_names)
    hessian_df.to_csv("hessiancheck/hessian_matrix.csv")
    
    # Inverse Hessian matrix  
    inv_hessian_df = pd.DataFrame(inv_hessian, columns=param_names, index=param_names)
    inv_hessian_df.to_csv("hessiancheck/inverse_hessian_matrix.csv")

if __name__ == "__main__":
    main()