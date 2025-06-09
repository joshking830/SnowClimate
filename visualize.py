#!/usr/bin/env python3
"""
Lindley Random Walk Visualization Tool

This file loads saved simulation results and creates various visualizations.
Use this after running simulations to explore results without re-running sims.

AI Assistance Disclosure:
This file combines original visualization implementations with AI-assisted features.

Original Work (Joshua King):
- Core plotting functions: visualize_series_data(), create_parameter_boxplots()
- Statistical visualization logic: winter_validation_plot(), parameter analysis
- Mathematical plot layouts and scientific data presentation
- Domain-specific visualization choices for snow depth analysis

AI-Assisted Components:
- Command-line interface and argument parsing (--plot, --interactive modes)
- File I/O management and folder organization features  
- Code structure, error handling, and documentation formatting
- Advanced CLI features (--list, --show options, batch processing)

All scientific content, statistical interpretations, and domain-specific
visualization decisions are original work.

Author: Joshua King
Date: June 9, 2025

Usage:
- python visualize.py --folder lindley_results_20241205_143022
- python visualize.py --folder my_results --sim 5 --plot series
- python visualize.py --list  # List available result folders
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob
from pathlib import Path

def load_simulation_results(results_folder):
    """
    Load simulation results from saved files
    
    Returns a dictionary matching the structure from run_simulation_study()
    """
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"Results folder not found: {results_folder}")
    
    print(f"Loading results from: {results_folder}")
    
    # Load parameter estimates
    param_file = os.path.join(results_folder, "parameter_estimates", "all_estimates.csv")
    valid_file = os.path.join(results_folder, "parameter_estimates", "valid_estimates.csv")
    
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter estimates file not found: {param_file}")
    
    results_df = pd.read_csv(param_file)
    valid_results = pd.read_csv(valid_file) if os.path.exists(valid_file) else results_df.dropna()
    
    # Load configuration
    config_file = os.path.join(results_folder, "summary", "configuration.txt")
    n_simulations = len(results_df)
    n_years = 100  # default
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if line.startswith("Number of simulations:"):
                    n_simulations = int(line.split(":")[1].strip())
                elif line.startswith("Years per simulation:"):
                    n_years = int(line.split(":")[1].strip())
    
    # Load time series data
    time_series_folder = os.path.join(results_folder, "time_series")
    all_series_data = []
    
    if os.path.exists(time_series_folder):
        print(f"Loading time series data...")
        
        for sim in range(1, n_simulations + 1):
            ct_file = os.path.join(time_series_folder, f"ct_simulation_{sim:03d}.csv")
            xt_file = os.path.join(time_series_folder, f"xt_simulation_{sim:03d}.csv")
            
            if os.path.exists(ct_file) and os.path.exists(xt_file):
                ct_data = pd.read_csv(ct_file)
                xt_data = pd.read_csv(xt_file)
                
                all_series_data.append({
                    'changes': ct_data['ct'].values,
                    'depths': xt_data['xt'].values,
                    'simulation': sim,
                    'n_years': n_years,
                    'n_days': len(ct_data)
                })
            else:
                print(f"Warning: Missing data files for simulation {sim}")
                all_series_data.append(None)
    
    # True parameters (from paper)
    true_params = {
        'alpha0': -0.6,
        'alpha1': 1.0,
        'tau1': 274,
        'sigma': 1.0,
        'alpha2': -0.000005
    }
    
    param_names = ['alpha0', 'alpha1', 'tau1', 'sigma', 'alpha2']
    
    # Calculate summary stats
    summary_stats = {}
    for param in param_names:
        true_val = true_params[param]
        if len(valid_results) > 0 and param in valid_results.columns:
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
    
    print(f"✓ Loaded {len(results_df)} simulations")
    print(f"✓ Found {len(valid_results)} valid parameter estimates")
    print(f"✓ Loaded {len([x for x in all_series_data if x is not None])} time series")
    
    return {
        'results_df': results_df,
        'valid_results': valid_results,
        'success_flags': [not pd.isna(results_df.iloc[i]['alpha0']) for i in range(len(results_df))],
        'all_series_data': all_series_data,
        'summary_stats': summary_stats,
        'true_params': true_params,
        'param_names': param_names,
        'n_simulations': n_simulations,
        'n_years': n_years,
        'results_folder': results_folder
    }

def list_available_results():
    """List all available result folders in current directory"""
    pattern = "lindley_results_*"
    folders = glob.glob(pattern)
    
    if not folders:
        print("No result folders found matching pattern 'lindley_results_*'")
        return
    
    print("Available result folders:")
    print("-" * 40)
    
    for folder in sorted(folders):
        # Try to get basic info
        config_file = os.path.join(folder, "summary", "configuration.txt")
        param_file = os.path.join(folder, "parameter_estimates", "all_estimates.csv")
        
        info = []
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        if "Simulations:" in line:
                            n_sims = line.split(":")[1].strip()
                            info.append(f"{n_sims} sims")
                        elif "Years:" in line:
                            n_years = line.split(":")[1].strip()
                            info.append(f"{n_years} years")
                        elif "Success rate:" in line:
                            success = line.split(":")[1].strip()
                            info.append(f"{success} success")
            except Exception as e:
                info.append("Error reading config")
        
        info_str = " | ".join(info) if info else "No config info"
        print(f"  {folder:<30} ({info_str})")
    
    print(f"\nTo visualize: python visualize.py --folder <folder_name>")

def setup_visuals_folder(results_folder):
    """Create visuals folder and return path"""
    visuals_folder = os.path.join(results_folder, "visuals")
    os.makedirs(visuals_folder, exist_ok=True)
    return visuals_folder

def visualize_series_data(simulation_results, simulation_number=1, start_year=1900, visuals_folder=None):
    """
    Visualize the C_t and X_t data from a specific simulation series
    """
    if simulation_number < 1 or simulation_number > len(simulation_results['all_series_data']):
        print(f"Error: simulation_number must be between 1 and {len(simulation_results['all_series_data'])}")
        return
    
    series_data = simulation_results['all_series_data'][simulation_number - 1]
    if series_data is None:
        print(f"Error: No data available for simulation {simulation_number}")
        return
    
    changes = series_data['changes']
    depths = series_data['depths']
    
    # Create time arrays
    total_days = len(changes)
    days_since_start = np.arange(total_days)
    years = start_year + days_since_start / 365.0
    
    print(f"Creating series visualization for simulation {simulation_number}")
    print(f"Data: {total_days} days ({total_days/365:.1f} years)")
    print(f"Time range: {years[0]:.1f} to {years[-1]:.1f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Full C_t time series
    axes[0, 0].plot(years, changes, 'k-', linewidth=0.3)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('C_t')
    axes[0, 0].set_title(f'Observed Changes C_t (Simulation {simulation_number})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-6, 6)
    
    # Last 10 year plots
    last_10_years = 10 * 365
    if len(changes) >= last_10_years:
        changes_subset = changes[-last_10_years:]
        years_subset = years[-last_10_years:]
        depths_subset = depths[-last_10_years:]
            
        # Plot 2: Last 10 years X_t time series
        axes[0, 1].plot(years_subset, depths_subset, 'b-', linewidth=0.5)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('X_t')
        axes[0, 1].set_title(f'Lindley Recursion - Snow Depths (Simulation {simulation_number})')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(bottom=0)
    
        # Plot 3: Last 10 years of C_t
        axes[1, 0].plot(years_subset, changes_subset, 'k-', linewidth=0.5)
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('C_t')
        axes[1, 0].set_title('Observed Changes C_t from 1990 to 2000 (Simulation {simulation_number})')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(-4, 4)
    
    # Plot 4: Single year of C_t (last year)
    if len(changes) >= 365:
        changes_last_year = changes[-365:]
        days_in_year = np.arange(1, 366)
        
        axes[1, 1].plot(days_in_year, changes_last_year, 'k-', linewidth=0.5)
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('C_t')
        axes[1, 1].set_title('Observed Changes C_t for 2000 (Simulation {simulation_number})')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-4, 4)
        
        # Add month markers
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1, 1].set_xticks(month_starts[::2])
        axes[1, 1].set_xticklabels(month_names[::2], rotation=45)
    
    plt.tight_layout()
    
    # Save instead of show
    if visuals_folder:
        filename = os.path.join(visuals_folder, f"series_simulation_{simulation_number:03d}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved series plot: {filename}")
    else:
        plt.show()
    
    plt.close()  # Close figure to free memory

def create_parameter_boxplots(simulation_results, visuals_folder=None):
    """
    Create box plots of parameter estimates across all simulations
    """
    valid_results = simulation_results['valid_results']
    true_params = simulation_results['true_params']
    param_names = simulation_results['param_names']
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        if i < len(axes):
            # Create box plot
            box_data = valid_results[param].dropna()
            axes[i].boxplot(box_data, patch_artist=True)
            
            # Add true value line
            true_val = true_params[param]
            axes[i].axhline(true_val, color='red', linestyle='--', linewidth=2,
                          label=f'True: {true_val:.6f}')
            
            # Add mean line
            mean_val = box_data.mean()
            axes[i].axhline(mean_val, color='blue', linestyle='-', linewidth=2,
                          label=f'Mean: {mean_val:.6f}')
            
            axes[i].set_title(f'Parameter Estimates: {param}')
            axes[i].set_ylabel('Estimate Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(param_names) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    
    # Save instead of show
    if visuals_folder:
        filename = os.path.join(visuals_folder, "parameter_boxplots.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved parameter boxplots: {filename}")
    else:
        plt.show()
    
    plt.close()  # Close figure to free memory
    
    print(f"Box plots show distribution of estimates across {len(valid_results)} successful simulations")

def winter_validation_plot(simulation_results, simulation_number=1, visuals_folder=None):
    """
    Simple validation plots: Oct to Jul, showing C_t and X_t for 1990
    """
    if simulation_number < 1 or simulation_number > len(simulation_results['all_series_data']):
        print(f"Error: simulation_number must be between 1 and {len(simulation_results['all_series_data'])}")
        return
    
    series_data = simulation_results['all_series_data'][simulation_number - 1]
    if series_data is None:
        print(f"Error: No data available for simulation {simulation_number}")
        return
    
    changes = series_data['changes']
    depths = series_data['depths']
    
    # 1990 is year 91 in simulation (starting from 1900)
    year_start = 90 * 365
    
    if len(changes) <= year_start + 365 + 213:
        print(f"Error: Not enough data for 1990 validation plot")
        return
    
    # Get full year data for 1990
    year_changes = changes[year_start + 274: year_start + 365 + 213]
    year_depths = depths[year_start + 274: year_start + 365 + 213]
    
    # Month labels and positions
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_positions = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273]  # Removed the extra position
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: C_t
    ax1.plot(year_changes, 'k-', linewidth=1)
    ax1.set_title(f'C_t: Oct 1989 to Jul 1990 (Simulation {simulation_number})')
    ax1.set_ylabel('C_t')
    ax1.set_xticks(month_positions)
    ax1.set_xticklabels(months, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: X_t  
    ax2.plot(year_depths, 'b-', linewidth=1)
    ax2.set_title(f'X_t: Oct 1989 to Jul 1990 (Simulation {simulation_number})')
    ax2.set_ylabel('X_t')
    ax2.set_xticks(month_positions)
    ax2.set_xticklabels(months, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save instead of show
    if visuals_folder:
        filename = os.path.join(visuals_folder, f"winter_validation_simulation_{simulation_number:03d}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved winter validation plot: {filename}")
    else:
        plt.show()
    
    plt.close()  # Close figure to free memory

def create_summary_plots(simulation_results, visuals_folder=None):
    """
    Create summary plots showing convergence and distribution statistics
    """
    valid_results = simulation_results['valid_results']
    param_names = simulation_results['param_names']
    true_params = simulation_results['true_params']
    
    if len(valid_results) == 0:
        print("No valid results to create summary plots")
        return
    
    # Create convergence plot showing parameter estimates over simulation runs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        if i < len(axes):
            # Plot parameter estimates over simulation runs
            sim_numbers = valid_results['simulation'].values
            param_values = valid_results[param].values
            true_val = true_params[param]
            
            axes[i].scatter(sim_numbers, param_values, alpha=0.6, s=20)
            axes[i].axhline(true_val, color='red', linestyle='--', linewidth=2, 
                          label=f'True Value: {true_val:.6f}')
            
            # Add rolling mean
            if len(param_values) >= 10:
                rolling_mean = pd.Series(param_values).rolling(window=10, min_periods=1).mean()
                axes[i].plot(sim_numbers, rolling_mean, 'b-', linewidth=2, 
                           label='Rolling Mean (10 sims)')
            
            axes[i].set_xlabel('Simulation Number')
            axes[i].set_ylabel(f'{param} Estimate')
            axes[i].set_title(f'Parameter {param} Convergence')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(param_names) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    
    # Save convergence plot
    if visuals_folder:
        filename = os.path.join(visuals_folder, "parameter_convergence.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved parameter convergence plot: {filename}")
    else:
        plt.show()
    
    plt.close()

def print_summary_stats(simulation_results, visuals_folder=None):
    """Print summary statistics and optionally save to file"""
    summary_stats = simulation_results['summary_stats']
    param_names = simulation_results['param_names']
    
    summary_text = []
    summary_text.append("=" * 90)
    summary_text.append("PARAMETER ESTIMATION RESULTS (Summary Statistics)")
    summary_text.append("=" * 90)
    summary_text.append("Format: True Value | Mean Estimate | Std Deviation | Bias")
    summary_text.append("-" * 90)
    
    for param in param_names:
        stats = summary_stats[param]
        line = f"{param:<12} | {stats['true']:>12.8f} | {stats['mean']:>13.8f} | {stats['std']:>13.8f} | {stats['bias']:>12.8f}"
        summary_text.append(line)
    
    # Print to console
    print("\n" + "\n".join(summary_text))
    
    # Save to file if visuals folder provided
    if visuals_folder:
        summary_file = os.path.join(visuals_folder, "summary_statistics.txt")
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_text))
        print(f"✓ Saved summary statistics: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Lindley Random Walk Simulation Results')
    
    # Data loading
    parser.add_argument('--folder', type=str, help='Results folder to load')
    parser.add_argument('--list', action='store_true', help='List available result folders')
    
    # Visualization options
    parser.add_argument('--plot', type=str, choices=['series', 'params', 'winter', 'summary', 'convergence', 'all'],
                       default='all', help='Type of plot to create')
    parser.add_argument('--sim', type=int, default=1, 
                       help='Simulation number for series/winter plots (default: 1)')
    parser.add_argument('--start_year', type=int, default=1900,
                       help='Start year for time axis (default: 1900)')
    
    # Output options
    parser.add_argument('--show', action='store_true',
                       help='Show plots in windows instead of saving to files')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved images (default: 300)')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode - prompt for plot choices')
    
    args = parser.parse_args()
    
    # List available folders
    if args.list:
        list_available_results()
        return
    
    # Load results
    if not args.folder:
        print("Error: Must specify --folder or use --list to see available folders")
        parser.print_help()
        return
    
    try:
        simulation_results = load_simulation_results(args.folder)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Setup visuals folder (unless showing plots)
    visuals_folder = None
    if not args.show:
        visuals_folder = setup_visuals_folder(simulation_results['results_folder'])
        print(f"Saving visualizations to: {visuals_folder}/")
        print()
    
    # Interactive mode
    if args.interactive:
        while True:
            print("\nVisualization Options:")
            print("1. Series plots (C_t and X_t time series)")
            print("2. Parameter box plots")
            print("3. Winter validation plot")
            print("4. Summary statistics")
            print("5. Parameter convergence plots")
            print("6. All plots")
            print("7. Quit")
            
            choice = input("Choose option (1-7): ").strip()
            
            if choice == '1':
                sim_num = input(f"Simulation number (1-{simulation_results['n_simulations']}): ")
                try:
                    sim_num = int(sim_num)
                    visualize_series_data(simulation_results, sim_num, args.start_year, visuals_folder)
                except ValueError:
                    print("Invalid simulation number")
            elif choice == '2':
                create_parameter_boxplots(simulation_results, visuals_folder)
            elif choice == '3':
                sim_num = input(f"Simulation number (1-{simulation_results['n_simulations']}): ")
                try:
                    sim_num = int(sim_num)
                    winter_validation_plot(simulation_results, sim_num, visuals_folder)
                except ValueError:
                    print("Invalid simulation number")
            elif choice == '4':
                print_summary_stats(simulation_results, visuals_folder)
            elif choice == '5':
                create_summary_plots(simulation_results, visuals_folder)
            elif choice == '6':
                print_summary_stats(simulation_results, visuals_folder)
                create_parameter_boxplots(simulation_results, visuals_folder)
                create_summary_plots(simulation_results, visuals_folder)
                visualize_series_data(simulation_results, args.sim, args.start_year, visuals_folder)
                winter_validation_plot(simulation_results, args.sim, visuals_folder)
            elif choice == '7':
                break
            else:
                print("Invalid choice")
    
    # Non-interactive mode
    else:
        if args.plot in ['summary', 'all']:
            print_summary_stats(simulation_results, visuals_folder)
        
        if args.plot in ['params', 'all']:
            create_parameter_boxplots(simulation_results, visuals_folder)
        
        if args.plot in ['convergence', 'all']:
            create_summary_plots(simulation_results, visuals_folder)
        
        if args.plot in ['series', 'all']:
            visualize_series_data(simulation_results, args.sim, args.start_year, visuals_folder)
        
        if args.plot in ['winter', 'all']:
            winter_validation_plot(simulation_results, args.sim, visuals_folder)
    
    # Final summary
    if visuals_folder and not args.show:
        print(f"\n✓ All visualizations saved to: {visuals_folder}/")
        
        # List saved files
        png_files = glob.glob(os.path.join(visuals_folder, "*.png"))
        txt_files = glob.glob(os.path.join(visuals_folder, "*.txt"))
        
        if png_files or txt_files:
            print(f"Created {len(png_files)} image files and {len(txt_files)} text files:")
            for file in sorted(png_files + txt_files):
                filename = os.path.basename(file)
                print(f"  • {filename}")

if __name__ == "__main__":
    main()