#!/usr/bin/env python3
"""
Lindley Snow Model - Optimized Only Runner

This script runs simulations using only the optimized estimation functions
with monkey-patching for actual reporting and simulation.

AI Assistance Disclosure:
This file was developed with assistance from generative AI (Claude) for code structure 
optimization, error handling, and implementation of performance features.
The core mathematical algorithms and model logic remain based on original research.

Usage:
- python run_optimized.py --test
- python run_optimized.py --n_sims 100
- python run_optimized.py --n_sims 1000 --n_years 50
- python run_optimized.py --n_sims 100 --mute

Author: Joshua King
Date: June 9, 2025
"""

import numpy as np
import argparse
import time
import sys
import os
import pandas as pd

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:  # 24 hours * 3600 seconds/hour
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def create_results_folder(n_simulations, n_years):
    """
    Create organized folder structure for optimized results
    
    Returns:
    - results_folder: Main results folder
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_folder = f"lindley_optimized_{n_simulations}sims_{n_years}years_{timestamp}"
    
    # Create the main directory and subdirectories
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(os.path.join(results_folder, "time_series"), exist_ok=True)
    os.makedirs(os.path.join(results_folder, "parameter_estimates"), exist_ok=True)
    os.makedirs(os.path.join(results_folder, "summary"), exist_ok=True)

    return results_folder

def run_optimized_simulation(n_simulations=1, n_years=100, seed=42, mute=False):
    """
    Run optimized simulation study using monkey-patching to match run_comparison.py behavior
    """
    # Import the run_simulation module
    from run_simulation import run_simulation_study
    
    # Create results folder
    results_folder = create_results_folder(n_simulations, n_years)
    
    print("RUNNING OPTIMIZED SIMULATION")
    print("="*50)
    
    # Set seed to match run_comparison.py behavior
    np.random.seed(seed)
    
    # Import and monkey-patch (exactly like run_comparison.py)
    from OptimizedEstimations import log_likelihood_optimized, estimate_parameters_optimized
    import LindleySnowModel
    
    original_log_likelihood = LindleySnowModel.log_likelihood
    original_estimate_parameters = LindleySnowModel.estimate_parameters
    
    LindleySnowModel.log_likelihood = log_likelihood_optimized
    LindleySnowModel.estimate_parameters = estimate_parameters_optimized
    
    start_time = time.time()
    results = run_simulation_study(
        n_simulations=n_simulations,
        n_years=n_years,
        save_results=True,
        results_folder=results_folder,
        mute=mute
    )
    total_time = time.time() - start_time
    
    # Restore original functions
    LindleySnowModel.log_likelihood = original_log_likelihood
    LindleySnowModel.estimate_parameters = original_estimate_parameters
    
    # =================================================================
    # RESULTS SUMMARY
    # =================================================================
    if not mute:
        print("\n" + "="*60)
        print("OPTIMIZED SIMULATION RESULTS")
        print("="*60)
    
    valid_results = results['valid_results']
    success_rate = len(valid_results) / n_simulations * 100
    
    if not mute:
        print(f"Total simulations:         {n_simulations}")
        print(f"Successful estimates:      {len(valid_results)}")
        print(f"Success rate:              {success_rate:.1f}%")
        print(f"Total runtime:             {format_duration(total_time)}")
        
        if len(valid_results) > 0:
            avg_time_per_sim = total_time / n_simulations
            print(f"Average time per sim:      {format_duration(avg_time_per_sim)}")
            
            # Runtime projections
            print(f"\nRUNTIME PROJECTIONS:")
            for target_sims in [100, 1000, 10000]:
                projected_time = avg_time_per_sim * target_sims
                print(f"{target_sims:>6,} simulations: {format_duration(projected_time)}")
        
        # Parameter summary
        print(f"\nPARAMETER ESTIMATION SUMMARY:")
        print(f"{'Parameter':<10} | {'True':<10} | {'Mean':<10} | {'Std':<10} | {'Bias':<10}")
        print("-" * 60)
        
        true_params = results['true_params']
        summary_stats = results['summary_stats']
        
        for param in results['param_names']:
            stats = summary_stats[param]
            print(f"{param:<10} | {stats['true']:<10.6f} | {stats['mean']:<10.6f} | {stats['std']:<10.6f} | {stats['bias']:<10.6f}")
    
    # =================================================================
    # SAVE SUMMARY FILE
    # =================================================================
    summary_file = os.path.join(results_folder, "optimized_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("LINDLEY SNOW MODEL OPTIMIZED RESULTS\n")
        f.write("=" * 45 + "\n")
        f.write(f"Run date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Simulations: {n_simulations}\n")
        f.write(f"Years per simulation: {n_years}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Estimation method: OPTIMIZED (vectorized)\n")
        f.write(f"Mute mode: {mute}\n")
        f.write("\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total simulations:      {n_simulations}\n")
        f.write(f"Successful estimates:   {len(valid_results)}\n")
        f.write(f"Success rate:           {success_rate:.1f}%\n")
        f.write(f"Total runtime:          {format_duration(total_time)}\n")
        if len(valid_results) > 0:
            f.write(f"Avg time per sim:       {format_duration(total_time / n_simulations)}\n")
        f.write("\n")
        
        f.write("PARAMETER ESTIMATES\n")
        f.write("-" * 25 + "\n")
        f.write(f"{'Parameter':<10} | {'True':<10} | {'Mean':<10} | {'Std':<10} | {'Bias':<10}\n")
        f.write("-" * 60 + "\n")
        
        summary_stats = results['summary_stats']
        for param in results['param_names']:
            stats = summary_stats[param]
            f.write(f"{param:<10} | {stats['true']:<10.6f} | {stats['mean']:<10.6f} | {stats['std']:<10.6f} | {stats['bias']:<10.6f}\n")
        
        f.write(f"\nRUNTIME PROJECTIONS\n")
        f.write("-" * 20 + "\n")
        if len(valid_results) > 0:
            avg_time_per_sim = total_time / n_simulations
            for target_sims in [100, 1000, 10000]:
                projected_time = avg_time_per_sim * target_sims
                f.write(f"{target_sims:>6,} simulations: {format_duration(projected_time)}\n")
        
        f.write(f"\nVISUALIZATION\n")
        f.write("-" * 15 + "\n")
        f.write(f"python visualize.py --folder {results_folder}\n")
    
    # Final output
    if not mute:
        print(f"\nRESULTS SAVED:")
        print(f"Main folder:     {results_folder}/")
        print(f"Summary file:    {summary_file}")
        print(f"\nVISUALIZE WITH:")
        print(f"python visualize.py --folder {results_folder}")
    else:
        # In mute mode, still show essential completion info
        print(f"\n✓ Optimized simulation complete: {n_simulations} sims, {success_rate:.1f}% success, {format_duration(total_time)}")
        print(f"  \nResults saved to: {results_folder}/")
    
    return {
        'results': results,
        'total_time': total_time,
        'results_folder': results_folder,
        'summary_file': summary_file
    }

def list_optimized_results():
    """List all available optimized result folders"""
    pattern = "lindley_optimized_*"
    import glob
    folders = glob.glob(pattern)
    
    if not folders:
        print("No optimized result folders found matching pattern 'lindley_optimized_*'")
        return
    
    print("Available optimized results:")
    print("-" * 50)
    
    for folder in sorted(folders, reverse=True):  # Most recent first
        summary_file = os.path.join(folder, "optimized_summary.txt")
        
        info = []
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    for line in f:
                        if "Simulations:" in line:
                            n_sims = line.split(":")[1].strip()
                            info.append(f"{n_sims} sims")
                        elif "Success rate:" in line:
                            success = line.split(":")[1].strip()
                            info.append(f"{success} success")
                        elif "Total runtime:" in line:
                            runtime = line.split(":")[1].strip()
                            info.append(f"{runtime}")
            except Exception:
                info.append("Error reading summary")
        
        info_str = " | ".join(info) if info else "No summary info"
        print(f"  📁 {folder:<40} ({info_str})")
    
    print(f"\nTo visualize results:")
    print(f"python visualize.py --folder <folder_name>")

def main():
    parser = argparse.ArgumentParser(description='Run Optimized Lindley Snow Model Simulations')
    
    # Simulation options
    parser.add_argument('--test', action='store_true',
                       help='Run single test simulation (1 sim, 100 years)')
    parser.add_argument('--n_sims', type=int, default=None,
                       help='Number of simulations to run')
    parser.add_argument('--n_years', type=int, default=100,
                       help='Years per simulation (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--list', action='store_true',
                       help='List available optimized results')
    parser.add_argument('--mute', action='store_true',
                       help='Show only progress bar and summary instead of detailed output')
    
    args = parser.parse_args()
    
    # List existing results
    if args.list:
        list_optimized_results()
        return
    
    # Determine simulation parameters
    if args.test:
        n_simulations = 1
        n_years = 100
        if not args.mute:
            print("RUNNING TEST SIMULATION (OPTIMIZED)")
            print("=" * 50)
    elif args.n_sims is not None:
        n_simulations = args.n_sims
        n_years = args.n_years
        if not args.mute:
            print(f"RUNNING CUSTOM SIMULATION (OPTIMIZED)")
            print("=" * 50)
    else:
        print("Error: Must specify --test, --n_sims, or --list")
        parser.print_help()
        return
    
    print(f"\nConfiguration:")
    print(f"• Number of simulations: {n_simulations}")
    print(f"• Years per simulation: {n_years}")
    print(f"• Random seed: {args.seed}")
    print(f"• Estimation method: OPTIMIZED (vectorized)")
    print(f"• Mute mode: {args.mute}")
    print()
    
    # Confirm for large runs (unless muted)
    if n_simulations >= 100 and not args.mute:
        response = input(f"This will run {n_simulations} optimized simulations. Continue? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Simulation cancelled.")
            return
    
    # Run optimized simulation
    if not args.mute:
        print("Starting optimized simulation...")
    
    results = run_optimized_simulation(
        n_simulations=n_simulations,
        n_years=n_years,
        seed=args.seed,
        mute=args.mute
    )
    
    return results

if __name__ == "__main__":
    results = main()