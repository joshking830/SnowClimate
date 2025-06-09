#!/usr/bin/env python3
"""
Enhanced Lindley Snow Model Comparison Runner

This script runs the simulation study two ways:
1. Original estimation functions
2. Optimized estimation functions (monkey-patched)

Usage:
- python run_comparison.py --test
- python run_comparison.py --n_sims 5
- python run_comparison.py --n_sims 20
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

def create_comparison_folder_structure(n_simulations, n_years):
    """
    Create organized folder structure for comparison results
    
    Returns:
    - comparison_root: Main comparison folder
    - original_folder: Subfolder for original results
    - optimized_folder: Subfolder for optimized results
    """
    # Create main comparison folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_root = f"sim_comparisons/comparison_{n_simulations}sims_{n_years}years_{timestamp}"
    
    # Create the main directory and subdirectories
    os.makedirs(comparison_root, exist_ok=True)
    
    # Create subfolders for each method
    original_folder = os.path.join(comparison_root, "original")
    optimized_folder = os.path.join(comparison_root, "optimized")
    
    # Create the method-specific folders and their subdirectories
    for folder in [original_folder, optimized_folder]:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "time_series"), exist_ok=True)
        os.makedirs(os.path.join(folder, "parameter_estimates"), exist_ok=True)
        os.makedirs(os.path.join(folder, "summary"), exist_ok=True)
    
    print(f"Created comparison folder structure:")
    print(f"  📁 {comparison_root}/")
    print(f"    📁 original/")
    print(f"    📁 optimized/")
    print()
    
    return comparison_root, original_folder, optimized_folder

def run_comparison_test(n_simulations=1, n_years=100):
    """
    Run both versions: original and optimized
    """
    # Import the run_simulation module
    from run_simulation import run_simulation_study
    
    # Create organized folder structure
    comparison_root, original_folder, optimized_folder = create_comparison_folder_structure(n_simulations, n_years)
    
    all_results = {}
    all_times = {}
    
    # =================================================================
    # 1. ORIGINAL VERSION
    # =================================================================
    print("\nRUNNING ORIGINAL VERSION")
    print("="*40)
    
    # Set same seed for all runs
    np.random.seed(42)
    
    start_time = time.time()
    original_results = run_simulation_study(
        n_simulations=n_simulations,
        n_years=n_years,
        save_results=True,
        results_folder=original_folder
    )
    original_time = time.time() - start_time
    
    all_results['original'] = original_results
    all_times['original'] = original_time
    print(f"\n✓ Original version complete: {format_duration(original_time)}")
    print()
    
    # =================================================================
    # 2. OPTIMIZED VERSION (Sequential)
    # =================================================================
    print("\nRUNNING OPTIMIZED VERSION (Sequential)")
    print("="*40)
    print("Patching estimation functions with optimized versions...")
    
    # Reset seed
    np.random.seed(42)
    
    # Import and monkey-patch
    from OptimizedEstimations import log_likelihood_optimized, estimate_parameters_optimized
    import LindleySnowModel
    
    original_log_likelihood = LindleySnowModel.log_likelihood
    original_estimate_parameters = LindleySnowModel.estimate_parameters
    
    LindleySnowModel.log_likelihood = log_likelihood_optimized
    LindleySnowModel.estimate_parameters = estimate_parameters_optimized
    
    start_time = time.time()
    optimized_results = run_simulation_study(
        n_simulations=n_simulations,
        n_years=n_years,
        save_results=True,
        results_folder=optimized_folder
    )
    optimized_time = time.time() - start_time
    
    # Restore original functions
    LindleySnowModel.log_likelihood = original_log_likelihood
    LindleySnowModel.estimate_parameters = original_estimate_parameters
    
    all_results['optimized'] = optimized_results
    all_times['optimized'] = optimized_time
    print(f"\n✓ Optimized version complete: {format_duration(optimized_time)}")
    print()
    
    # =================================================================
    # COMPARISON RESULTS
    # =================================================================
    print("="*80)
    print("COMPARISON RESULTS (Original vs Optimized)")
    print("="*80)
    
    # Timing comparison
    print(f"TIMING COMPARISON:")
    print(f"Original total time:       {format_duration(all_times['original'])}")
    print(f"Optimized total time:      {format_duration(all_times['optimized'])}")
    
    # Calculate speedups
    opt_speedup = all_times['original'] / all_times['optimized'] if all_times['optimized'] > 0 else 0
    
    print(f"\nSPEEDUP FACTORS:")
    print(f"Optimized vs Original:     {opt_speedup:.2f}x")
    
    # Success rates
    print(f"\nSUCCESS RATE COMPARISON:")
    orig_success = len(all_results['original']['valid_results'])
    opt_success = len(all_results['optimized']['valid_results'])
    
    print(f"Original success:          {orig_success}/{n_simulations} = {orig_success/n_simulations*100:.1f}%")
    print(f"Optimized success:         {opt_success}/{n_simulations} = {opt_success/n_simulations*100:.1f}%")
    
    # Time projections for larger studies
    if opt_speedup > 1:
        print(f"\nPROJECTED TIME SAVINGS FOR LARGER STUDIES:")
        
        # Calculate time per simulation for each method
        original_time_per_sim = all_times['original'] / n_simulations
        opt_time_per_sim = all_times['optimized'] / n_simulations
        
        # Show projections and savings
        for target_sims in [100, 1000, 10000]:
            original_projected = original_time_per_sim * target_sims
            opt_projected = opt_time_per_sim * target_sims
            savings = original_projected - opt_projected
            
            print(f"{target_sims:>6,} simulations: {format_duration(savings)} saved")
    
    # =================================================================
    # SAVE COMPARISON SUMMARY
    # =================================================================
    summary_file = os.path.join(comparison_root, "comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("LINDLEY SNOW MODEL COMPARISON RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Comparison date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Simulations: {n_simulations}\n")
        f.write(f"Years per simulation: {n_years}\n")
        f.write(f"Random seed: 42 (same for both methods)\n")
        f.write("\n")
        
        f.write("TIMING RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original total time:    {format_duration(all_times['original'])}\n")
        f.write(f"Optimized total time:   {format_duration(all_times['optimized'])}\n")
        f.write(f"Speedup factor:         {opt_speedup:.2f}x\n")
        f.write("\n")
        
        f.write("SUCCESS RATES\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original success:       {orig_success}/{n_simulations} = {orig_success/n_simulations*100:.1f}%\n")
        f.write(f"Optimized success:      {opt_success}/{n_simulations} = {opt_success/n_simulations*100:.1f}%\n")
        f.write("\n")
        
        if opt_speedup > 1:
            f.write("PROJECTED TIME SAVINGS\n")
            f.write("-" * 25 + "\n")
            original_time_per_sim = all_times['original'] / n_simulations
            opt_time_per_sim = all_times['optimized'] / n_simulations
            
            for target_sims in [100, 1000, 10000]:
                original_projected = original_time_per_sim * target_sims
                opt_projected = opt_time_per_sim * target_sims
                savings = original_projected - opt_projected
                f.write(f"{target_sims:>6,} simulations: {format_duration(savings)} saved\n")
        
        f.write("\nFOLDER STRUCTURE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original results:       {original_folder}\n")
        f.write(f"Optimized results:      {optimized_folder}\n")
        f.write("\nVISUALIZATION COMMANDS\n")
        f.write("-" * 25 + "\n")
        f.write(f"python visualize.py --folder {original_folder}\n")
        f.write(f"python visualize.py --folder {optimized_folder}\n")
    
    # Results folders
    print(f"\nRESULTS ORGANIZATION:")
    print(f"Main comparison folder: {comparison_root}")
    print(f"├── original/           {original_folder}")
    print(f"├── optimized/          {optimized_folder}")
    print(f"└── comparison_summary.txt")
    
    print(f"\nVISUALIZATION COMMANDS:")
    print(f"python visualize.py --folder {original_folder}")
    print(f"python visualize.py --folder {optimized_folder}")
    
    print(f"\nCOMPARISON SUMMARY SAVED:")
    print(f"{summary_file}")
    
    return {
        'results': all_results,
        'times': all_times,
        'speedups': {
            'optimized_vs_original': opt_speedup
        },
        'comparison_root': comparison_root,
        'original_folder': original_folder,
        'optimized_folder': optimized_folder
    }

def list_comparison_results():
    """List all available comparison result folders"""
    if not os.path.exists("sim_comparisons"):
        print("No sim_comparisons folder found.")
        return
    
    comparison_folders = [f for f in os.listdir("sim_comparisons") 
                         if f.startswith("comparison_") and os.path.isdir(os.path.join("sim_comparisons", f))]
    
    if not comparison_folders:
        print("No comparison results found in sim_comparisons/")
        return
    
    print("Available comparison results:")
    print("-" * 50)
    
    for folder in sorted(comparison_folders, reverse=True):  # Most recent first
        folder_path = os.path.join("sim_comparisons", folder)
        summary_file = os.path.join(folder_path, "comparison_summary.txt")
        
        info = []
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    for line in f:
                        if "Simulations:" in line:
                            n_sims = line.split(":")[1].strip()
                            info.append(f"{n_sims} sims")
                        elif "Speedup factor:" in line:
                            speedup = line.split(":")[1].strip()
                            info.append(f"{speedup} speedup")
            except Exception:
                info.append("Error reading summary")
        
        info_str = " | ".join(info) if info else "No summary info"
        print(f"  📁 {folder:<35} ({info_str})")
        
        # Show subfolders
        original_path = os.path.join(folder_path, "original")
        optimized_path = os.path.join(folder_path, "optimized")
        if os.path.exists(original_path) and os.path.exists(optimized_path):
            print(f"     ├── original/")
            print(f"     └── optimized/")
    
    print(f"\nTo visualize comparison results:")
    print(f"python visualize.py --folder sim_comparisons/<comparison_folder>/original")
    print(f"python visualize.py --folder sim_comparisons/<comparison_folder>/optimized")

def main():
    parser = argparse.ArgumentParser(description='Compare Original vs Optimized Lindley Snow Model')
    
    # Test options
    parser.add_argument('--test', action='store_true',
                       help='Run single test simulation (1 sim, 100 years)')
    parser.add_argument('--n_sims', type=int, default=None,
                       help='Number of simulations to run')
    parser.add_argument('--n_years', type=int, default=100,
                       help='Years per simulation (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--list', action='store_true',
                       help='List available comparison results')
    
    args = parser.parse_args()
    
    # List existing comparisons
    if args.list:
        list_comparison_results()
        return
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine simulation parameters
    if args.test:
        n_simulations = 1
        n_years = 100
        print("=" * 60)
        print("\nRUNNING SINGLE TEST COMPARISON (Original vs Optimized)")
        print("=" * 60)
    elif args.n_sims is not None:
        n_simulations = args.n_sims
        n_years = args.n_years
        print(f"\nRUNNING CUSTOM COMPARISON (Original vs Optimized)")
        print("=" * 60)
    else:
        print("Error: Must specify --test, --n_sims, or --list")
        parser.print_help()
        return
    
    print(f"\nConfiguration:")
    print(f"• Number of simulations: {n_simulations}")
    print(f"• Years per simulation: {n_years}")
    print(f"• Random seed: {args.seed}")
    print(f"• Results will be organized in: sim_comparisons/")
    print()
    
    # Warning for large runs
    if n_simulations >= 20:
        total_sims = n_simulations * 2  # Two methods
        response = input(f"This will run {total_sims} total simulations ({n_simulations} × 2 methods). Continue? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Comparison cancelled.")
            return
    
    # Run comparison test
    print("Starting two-way comparison test...")
    results = run_comparison_test(
        n_simulations=n_simulations,
        n_years=n_years
    )
    
    return results

if __name__ == "__main__":
    results = main()