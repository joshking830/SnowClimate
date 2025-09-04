#!/usr/bin/env python3
"""
Univariate Lindley Snow Model

A simplified univariate implementation of the Lindley random walk model for snow depth simulation.
The model uses temperature as the sole covariate with the structure:

Daily changes: C_t = b0 + b1 * temp_t + ε_t where ε_t ~ N(0, σ²)
Snow depths:   X_t = max(X_{t-1} + C_t, 0)

This file contains ONLY the core model implementation.

Author: Joshua King
Date: September 4, 2025
"""

import numpy as np


class UnivariateSnowModel:
    """
    Univariate Lindley Snow Model
    
    Implements the Lindley random walk for snow depth simulation using 
    temperature as the primary covariate.
    """
    
    def __init__(self, b0=-0.4, b1=-.3, sigma=1.0):
        """
        Initialize the Univariate Lindley Snow Model
        
        Parameters:
        -----------
        b0 : float, default=-0.5
            Baseline level of daily change (intercept parameter)
        b1 : float, default=-0.3  
            Temperature coefficient (negative = more melting with higher temps)
        sigma : float, default=1.0
            Standard deviation of random noise
        """
        self.b0 = b0
        self.b1 = b1
        self.sigma = sigma
    
    def daily_changes(self, temperatures):
        """
        Generate daily changes C_t based on temperature
        
        Implements: C_t = b0 + b1 * temp_t + ε_t where ε_t ~ N(0, σ²)
        
        Parameters:
        -----------
        temperatures : array-like
            Daily temperature values
            
        Returns:
        --------
        numpy.ndarray
            Daily changes in snow depth
        """
        n_days = len(temperatures)
        
        # Generate random noise: ε_t ~ N(0, σ²)
        epsilon = np.random.normal(0, self.sigma, size=n_days)
        
        # Calculate daily changes: C_t = b0 + b1 * temp_t + ε_t
        changes = self.b0 + self.b1 * temperatures + epsilon
        
        return changes
    
    def lindley_recursion(self, changes, x0=0):
        """
        Apply the Lindley recursion to generate snow depths
        
        Implements: X_t = max(X_{t-1} + C_t, 0)
        
        Parameters:
        -----------
        changes : array-like
            Daily changes in snow depth
        x0 : float, default=0
            Initial snow depth
            
        Returns:
        --------
        numpy.ndarray
            Snow depths at each time step
        """
        n = len(changes)
        depths = np.zeros(n + 1)
        depths[0] = x0
        
        # Apply Lindley recursion: X_t = max(X_{t-1} + C_t, 0)
        for t in range(n):
            depths[t + 1] = max(depths[t] + changes[t], 0)
        
        # Return X_1, X_2, ..., X_n (exclude initial value)
        return depths[1:]
    
    def simulate(self, temperatures, x0=0, random_seed=None):
        """
        Complete simulation: generate changes and apply Lindley recursion
        
        Parameters:
        -----------
        temperatures : array-like
            Daily temperature values
        x0 : float, default=0
            Initial snow depth
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        tuple of numpy.ndarray
            (daily_changes, snow_depths)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Generate daily changes based on temperature
        changes = self.daily_changes(temperatures)
        
        # Apply Lindley recursion to get snow depths
        depths = self.lindley_recursion(changes, x0)
        
        return changes, depths
    
    def __repr__(self):
        """String representation of the model"""
        return f"UnivariateSnowModel(b0={self.b0:.3f}, b1={self.b1:.3f}, sigma={self.sigma:.3f})"
