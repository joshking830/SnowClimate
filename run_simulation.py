#!/usr/bin/env python3
"""
Lindley Random Walk Simulation Runner

This file runs simulations using the original estimation functions and saves all data to files.
Uses LindleySnowModel.py functions for parameter estimation.

AI Assistance Disclosure:
This file was developed with assistance from generative AI (Claude) for code structure 
optimization, error handling, and implementation of performance features.
The core mathematical algorithms and model logic remain based on original research.

Usage:
- Test run: python run_simulation.py --test
- Full study: python run_simulation.py --full
- Custom: python run_simulation.py --n_sims 50 --n_years 50

Author: Joshua King
Date: Jun 9, 2025
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os
import time
from datetime import datetime, timedelta
from LindleySnowModel import LindleySnowModel, estimate_parameters

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"
    
def update_progress_bar(current, total, bar_length=50, prefix="Progress"):
    """Update progress bar in place"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = progress * 100
    print(f'\r{prefix}|{bar}| {percent:6.1f}% ({current}/{total})', end='', flush=True)

def run_simulation_study(n_simulations=100, n_years=100, save_results=True, results_folder=None, model_type="original", mute=False):
    """
    Run the complete simulation study as described in the paper:
    
    Process:
    1. Simulate one 100-year series using Lindley recursion
    2. Obtain parameter estimates from this 100-year series
    3. Store the parameter estimates and the C_t/X_t datasets
    4. Repeat this entire process 100 times
    
    Parameters:
    - n_simulations: Number of simulations to run
    - n_years: Years per simulation  
    - save_results: Whether to save results to files
    - results_folder: Folder name (if None, creates descriptive folder)
    - model_type: Type of model ("original", "optimized", "vectorized")
    - mute: If True, show only progress bar instead of detailed output
    
    Result:
    - 100 different 100-year time series (C_t and X_t)
    - 100 parameter estimates (one from each 100-year series)
    - Summary statistics showing convergence to true parameters
    - All results saved to files in organized folder structure
    """
    
    # Create results folder with descriptive naming
    if save_results:
        if results_folder is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_folder = f"lindley_{model_type}_{n_simulations}sims_{n_years}years_{timestamp}"
        
        # Create main results folder
        os.makedirs(results_folder, exist_ok=True)
        
        # Create subfolders
        os.makedirs(os.path.join(results_folder, "time_series"), exist_ok=True)
        os.makedirs(os.path.join(results_folder, "parameter_estimates"), exist_ok=True)
        os.makedirs(os.path.join(results_folder, "summary"), exist_ok=True)
        
        if not mute:
            print(f"Results will be saved to: {results_folder}/")
            print()
    
    # True parameters from Table 1
    true_params = {
        'alpha0': -0.6,
        'alpha1': 1.0,
        'tau1': 274, # Set tau to october
        'sigma': 1.0,
        'alpha2': -0.000005
    }
    
    model = LindleySnowModel(**true_params)
    param_names = ['alpha0', 'alpha1', 'tau1', 'sigma', 'alpha2']
    
    # Store results
    estimated_params = []
    success_flags = []
    all_series_data = []  # Store all 100-year C_t and X_t series
    runtime_data = []     # Store runtime for each simulation
    
    if not mute:
        print(f"\nRunning {n_simulations} simulations, each with {n_years} years...")
    
    # Progress tracking
    if mute:
        update_progress_bar(0, n_simulations, prefix="")
    
    for sim in range(n_simulations):
        # Start simulation timer
        sim_start_time = time.time()
        
        if not mute:
            print(f"Simulation {sim + 1}/{n_simulations}:")
            print("  Generating data...", end="", flush=True)
        
        # Step 1: Generate one complete 100-year series (C_t and X_t)
        changes, depths = model.simulate_snow_depths(n_years)
        
        if not mute:
            print(" ✓ Complete")
        
        # Step 2: Store this series for later analysis
        all_series_data.append({
            'changes': changes.copy(),
            'depths': depths.copy(),
            'simulation': sim + 1,
            'n_years': n_years,
            'n_days': len(changes)
        })
        
        # Step 3: Estimate parameters from this complete 100-year series
        if not mute:
            print("  Estimating parameters...", end="", flush=True)
        
        try:
            params_est, success = estimate_parameters(depths, changes)
            estimated_params.append(params_est)
            success_flags.append(success)
            
            if not success:
                print(" ✗ FAILED")
                print(f"    Warning: Optimization failed for simulation {sim + 1}")
            else:
                if not mute:
                    print(" ✓ Complete")
        except Exception as e:
            print(" ✗ ERROR")
            print(f"    Error in simulation {sim + 1}: {e}")
            # Add NaN values to maintain array structure
            estimated_params.append([np.nan] * 5)
            success_flags.append(False)
        
        # Save individual simulation results if requested
        if save_results:
            if not mute:
                print("  Saving data...", end="", flush=True)
            
            # Save time series data
            # Save C_t series
            ct_df = pd.DataFrame({'day': range(1, len(changes) + 1), 'ct': changes})
            ct_df.to_csv(os.path.join(results_folder, "time_series", f"ct_simulation_{sim+1:03d}.csv"), index=False)
            
            # Save X_t series  
            xt_df = pd.DataFrame({'day': range(1, len(depths) + 1), 'xt': depths})
            xt_df.to_csv(os.path.join(results_folder, "time_series", f"xt_simulation_{sim+1:03d}.csv"), index=False)
            
            if not mute:
                print(" ✓ Complete")
        
        # Calculate simulation runtime
        sim_runtime = time.time() - sim_start_time
        
        if not mute:
            print(f"  → Simulation runtime: {format_duration(sim_runtime)}")
            # Overall simulation progress
            overall_progress = (sim + 1) / n_simulations * 100
            print(f"  → Overall progress: {overall_progress:.1f}% ({sim + 1}/{n_simulations} simulations)")
            print()  # Extra line between simulations for readability
        
        # Store runtime data
        runtime_data.append({
            'simulation': sim + 1,
            'runtime_seconds': sim_runtime,
            'success': success_flags[-1]
        })
        
        # Update progress bar in mute mode
        if mute:
            update_progress_bar(sim + 1, n_simulations, prefix="")
    
    # Finish progress bar
    if mute:
        print()  # New line after progress bar
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(estimated_params, columns=param_names)
    results_df['simulation'] = list(range(1, n_simulations + 1))
    
    # Convert runtime data to DataFrame
    runtime_df = pd.DataFrame(runtime_data)
    
    # Remove failed optimizations for summary statistics
    valid_results = results_df.dropna(subset=param_names)
    
    if not mute:
        print("\nSIMULATION COMPLETE!")
        print("="*50)
        
        print(f"Total simulations: {n_simulations}")
        print(f"Successful parameter estimates: {len(valid_results)}")
        print(f"Success rate: {sum(success_flags)}/{len(success_flags)} = {100*sum(success_flags)/len(success_flags):.1f}%")
        
        # Add simple runtime summary with debug info
        successful_runtimes = runtime_df[runtime_df['success']]['runtime_seconds']

        if len(successful_runtimes) > 0:
            print(f"\nRuntime summary:")
            print(f"Average time per simulation: {format_duration(successful_runtimes.mean())}")
            print(f"Total study time: {format_duration(successful_runtimes.sum())}")

            # Runtime projections
            avg_time_per_sim = successful_runtimes.mean()
            print(f"\nRuntime Projections:")
            print(f"Estimated time for 100 simulations: {format_duration(avg_time_per_sim * 100)}")
            print(f"Estimated time for 1,000 simulations: {format_duration(avg_time_per_sim * 1000)}")
            print(f"Estimated time for 10,000 simulations: {format_duration(avg_time_per_sim * 10000)}")
            print(f"Estimated time for 100,000 simulations: {format_duration(avg_time_per_sim * 100000)}")

            # Show confidence intervals
            std_time_per_sim = successful_runtimes.std()
            if len(successful_runtimes) > 1:
                print(f"\nRange estimate (±1 std dev):")
                print(f"  100 sims: {format_duration((avg_time_per_sim - std_time_per_sim) * 100)} to {format_duration((avg_time_per_sim + std_time_per_sim) * 100)}")
                print(f"  1,000 sims: {format_duration((avg_time_per_sim - std_time_per_sim) * 1000)} to {format_duration((avg_time_per_sim + std_time_per_sim) * 1000)}")
                print(f"  10,000 sims: {format_duration((avg_time_per_sim - std_time_per_sim) * 10000)} to {format_duration((avg_time_per_sim + std_time_per_sim) * 10000)}")
                print(f"  100,000 sims: {format_duration((avg_time_per_sim - std_time_per_sim) * 100000)} to {format_duration((avg_time_per_sim + std_time_per_sim) * 100000)}")
        else:
            print(f"\nNo successful simulations found - cannot calculate runtime projections")
            print(f"Check the success flags and runtime data above")
        
    # Calculate and display summary statistics (matching paper's Table 2)
    if not mute:
        print("\nPARAMETER ESTIMATION RESULTS (Summary Statistics)")
        print("="*90)
        print("\nFormat: True Value | Mean Estimate | Std Deviation | Bias")
        print("-"*90)
    else:
        print("Parameter Estimation Results (Summary Statistics)")
        print("="*50)
        print("\nFormat: True Value | Mean Estimate | Std Deviation | Bias")
        print("-"*50)
    
    summary_stats = {}
    for param in param_names:
        true_val = true_params[param]
        if len(valid_results) > 0:
            mean_est = valid_results[param].mean()
            std_est = valid_results[param].std()
            bias = mean_est - true_val
        else:
            mean_est = std_est = bias = np.nan
        
        summary_stats[param] = {
            'true': true_val,
            'mean': mean_est,
            'std': std_est,
            'bias': bias
        }
        
        print(f"{param:<12} | {true_val:>12.8f} | {mean_est:>13.8f} | {std_est:>13.8f} | {bias:>12.8f}")
    
    if not mute:
        # Show first few individual estimates for verification
        print(f"\nSample of Individual Parameter Estimates (First 5 simulations):")
        print("="*80)
        print(f"{'Sim':<4} | {'Success':<8} | {'alpha0':<10} | {'alpha1':<10} | {'tau1':<8} | {'sigma':<10} | {'alpha2':<12}")
        print("-"*80)
        
        for i in range(min(5, len(results_df))):
            sim_num = results_df.iloc[i]['simulation']
            success = success_flags[i]
            if success and not results_df.iloc[i][param_names].isna().any():
                params = results_df.iloc[i][param_names].values
                print(f"{sim_num:<4} | {'True':<8} | {params[0]:<10.4f} | {params[1]:<10.4f} | {params[2]:<8.1f} | {params[3]:<10.4f} | {params[4]:<12.8f}")
            else:
                print(f"{sim_num:<4} | {'False':<8} | {'NaN':<10} | {'NaN':<10} | {'NaN':<8} | {'NaN':<10} | {'NaN':<12}")
        
        if len(results_df) > 10:
            print(f"... (showing 5 of {len(results_df)} total simulations)")
    
    # Save final results
    if save_results:
        if not mute:
            print(f"\nSaving final results to {results_folder}/...")
        
        # Save parameter estimates
        results_df.to_csv(os.path.join(results_folder, "parameter_estimates", "all_estimates.csv"), index=False)
        valid_results.to_csv(os.path.join(results_folder, "parameter_estimates", "valid_estimates.csv"), index=False)
        
        # Save runtime data
        runtime_df.to_csv(os.path.join(results_folder, "runtime_data.csv"), index=False)
        
        # Get runtime summary for summary file
        successful_runtimes = runtime_df[runtime_df['success']]['runtime_seconds']
        
        # Save summary statistics
        summary_file = os.path.join(results_folder, "summary", "summary_statistics.txt")
        with open(summary_file, 'w') as f:
            f.write("LINDLEY RANDOM WALK SIMULATION STUDY RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total simulations: {n_simulations}\n")
            f.write(f"Years per simulation: {n_years}\n")
            f.write(f"Successful estimates: {len(valid_results)}\n")
            f.write(f"Success rate: {100 * len(valid_results) / n_simulations:.1f}%\n")
            if len(successful_runtimes) > 0:
                f.write(f"Average runtime per simulation: {format_duration(successful_runtimes.mean())}\n")
                f.write(f"Total study time: {format_duration(successful_runtimes.sum())}\n")
            f.write("\n")
            
            f.write("PARAMETER ESTIMATION RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Parameter':<12} | {'True Value':<12} | {'Mean Estimate':<15} | {'Std Error':<12} | {'Bias':<12}\n")
            f.write("-" * 70 + "\n")
            for param in param_names:
                stats = summary_stats[param]
                f.write(f"{param:<12} | {stats['true']:<12.8f} | {stats['mean']:<15.8f} | {stats['std']:<12.8f} | {stats['bias']:<12.8f}\n")
        
        # Save configuration info
        config_file = os.path.join(results_folder, "summary", "configuration.txt")
        with open(config_file, 'w') as f:
            f.write("SIMULATION CONFIGURATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"Number of simulations: {n_simulations}\n")
            f.write(f"Years per simulation: {n_years}\n")
            f.write(f"Total time series: {len(all_series_data)}\n")
            f.write(f"Days per series: {n_years * 365}\n")
            f.write(f"Mute mode: {mute}\n")
            f.write("\nTRUE PARAMETERS (Table 1)\n")
            f.write("-" * 25 + "\n")
            for param, value in true_params.items():
                f.write(f"{param}: {value}\n")
        
        if not mute:
            print(f"✓ Results saved to: {results_folder}/")
    
    return {
        'results_df': results_df,
        'valid_results': valid_results,
        'success_flags': success_flags,
        'all_series_data': all_series_data,
        'runtime_df': runtime_df,
        'summary_stats': summary_stats,
        'true_params': true_params,
        'param_names': param_names,
        'n_simulations': n_simulations,
        'n_years': n_years,
        'results_folder': results_folder if save_results else None
    }


def main():
    parser = argparse.ArgumentParser(description='Run Lindley Random Walk Simulations')
    
    # Simulation parameters
    parser.add_argument('--test', action='store_true', 
                       help='Run test simulation (1 simulation, 100 years)')
    parser.add_argument('--full', action='store_true',
                       help='Run full study (100 simulations, 100 years)')
    parser.add_argument('--n_sims', type=int, default=None,
                       help='Number of simulations to run')
    parser.add_argument('--n_years', type=int, default=100,
                       help='Years per simulation (default: 100)')
    
    # Output options
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Custom output folder name')
    parser.add_argument('--model_type', type=str, default='original',
                       choices=['original', 'optimized', 'vectorized'],
                       help='Model type for folder naming (default: original)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_save', action='store_true',
                       help='Run without saving files (testing only)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine simulation parameters
    if args.test:
        n_simulations = 1
        n_years = 100
        print("\nRUNNING TEST SIMULATION")
        print("=" * 50)
    elif args.full:
        n_simulations = 100
        n_years = 100
        print("\nRUNNING FULL SIMULATION STUDY")
        print("=" * 50)
    elif args.n_sims is not None:
        n_simulations = args.n_sims
        n_years = args.n_years
        print(f"\nRUNNING CUSTOM SIMULATION")
        print("=" * 50)
    else:
        print("Error: Must specify --test, --full, or --n_sims")
        parser.print_help()
        return
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"• Number of simulations: {n_simulations}")
    print(f"• Years per simulation: {n_years}")
    print(f"• Random seed: {args.seed}")
    print(f"• Save results: {not args.no_save}")
    if args.output_folder:
        print(f"• Custom output folder: {args.output_folder}")
    print()
    
    # Confirm for large runs
    if n_simulations >= 50:
        response = input(f"This will run {n_simulations} simulations. Continue? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Simulation cancelled.")
            return
    
    # Run simulation
    print("Starting simulation...")
    results = run_simulation_study(
        n_simulations=n_simulations,
        n_years=n_years,
        save_results=not args.no_save,
        results_folder=args.output_folder,
        model_type=args.model_type
    )
    
    if not args.no_save and results['results_folder']:
        print(f"\nResults saved to: {results['results_folder']}/")
        print(f"To visualize results, run:")
        print(f"  python visualize.py --folder {results['results_folder']}")
        
        # Create a quick reference file
        ref_file = os.path.join(results['results_folder'], "run_info.txt")
        with open(ref_file, 'w') as f:
            f.write("SIMULATION RUN INFORMATION\n")
            f.write("=" * 30 + "\n")
            f.write(f"Command: python run_simulation.py")
            if args.test:
                f.write(" --test")
            elif args.full:
                f.write(" --full")
            else:
                f.write(f" --n_sims {n_simulations} --n_years {n_years}")
            f.write(f" --seed {args.seed}\n")
            f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Simulations: {n_simulations}\n")
            f.write(f"Years: {n_years}\n")
            f.write(f"Success rate: {100 * len(results['valid_results']) / results['n_simulations']:.1f}%\n")
            f.write(f"\nTo visualize:\n")
            f.write(f"python visualize.py --folder {results['results_folder']}\n")
    else:
        print(f"\nResults not saved (--no_save flag used)")
    
    return results

if __name__ == "__main__":
    results = main()