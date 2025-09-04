#!/usr/bin/env python3
"""
Simulation Runner for Univariate Lindley Snow Model

This script coordinates all aspects of the snow model analysis:
- Data loading and preparation
- Model simulation with real temperature data
- Parameter estimation and validation
- Convergence analysis over multiple simulations
- Results visualization and summary reporting

Usage:
    python run_simulation.py [--n_sims N] [--years Y] [--seed S]

Author: Joshua King
Date: September 4, 2025
"""

import sys
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the project modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.UnivariateSnowModel import UnivariateSnowModel
from estimation.ParameterEstimation import (
    estimate_parameters, 
    calculate_confidence_intervals,
    run_convergence_study,
    plot_convergence_analysis,
    print_estimation_summary
)


def load_temperature_data(data_path=None):
    """
    Load and prepare temperature data for simulation
    
    Parameters:
    -----------
    data_path : str, optional
        Specific path to temperature data CSV file. If None, will search for 
        the most recent temperature data in synthetic_data folder
        
    Returns:
    --------
    pandas.DataFrame
        Temperature data with proper date indexing
    """
    try:
        if data_path is None:
            # Search for the most recent temperature data folder
            synthetic_data_dir = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data')
            
            # Find all temperature data directories
            temp_dirs = []
            if os.path.exists(synthetic_data_dir):
                for item in os.listdir(synthetic_data_dir):
                    if item.startswith('temperature_data_') and os.path.isdir(os.path.join(synthetic_data_dir, item)):
                        temp_dirs.append(item)
            
            if not temp_dirs:
                raise FileNotFoundError("No temperature data directories found in synthetic_data folder. "
                                      "Please run temp.py to generate temperature data first.")
            
            # Sort to get the most recent (latest timestamp)
            temp_dirs.sort(reverse=True)
            latest_dir = temp_dirs[0]
            
            # Look for CSV file in the latest directory
            latest_dir_path = os.path.join(synthetic_data_dir, latest_dir)
            csv_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {latest_dir_path}")
            
            # Use the first CSV file found (should be the temperature data)
            data_path = os.path.join(latest_dir_path, csv_files[0])
            print(f"Using most recent temperature data: {latest_dir}")
            
        else:
            # Use the provided path
            full_path = os.path.join(os.path.dirname(__file__), data_path)
            if not os.path.exists(full_path):
                # Try other common locations
                alt_paths = [
                    os.path.join(os.path.dirname(__file__), '..', 'temperature_data_1920_2019.csv'),
                    os.path.join(os.path.dirname(__file__), '..', 'data', 'temperature_data_1920_2019.csv')
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        data_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Temperature data not found at: {data_path}")
            else:
                data_path = full_path
            
        print(f"Loading temperature data from: {data_path}")
        data = pd.read_csv(data_path)
        
        # Convert date column to datetime if needed
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        
        print(f"Loaded {len(data)} temperature observations")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Temperature range: {data['temperature'].min():.1f}°C to {data['temperature'].max():.1f}°C")
        
        return data
        
    except Exception as e:
        print(f"Error loading temperature data: {e}")
        raise


def create_output_directory(base_name="simulation"):
    """
    Create timestamped output directory for results
    
    Parameters:
    -----------
    base_name : str
        Base name for the output directory
        
    Returns:
    --------
    str
        Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_name}_{timestamp}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    return output_dir


def run_single_simulation(model, temperatures, n_years=None, x0=0):
    """
    Run a single snow simulation and estimate parameters
    
    Parameters:
    -----------
    model : UnivariateSnowModel
        The snow model to use
    temperatures : array-like
        Temperature data
    n_years : int, optional
        Number of years to simulate (if None, use all data)
    x0 : float, default=0
        Initial snow depth
        
    Returns:
    --------
    dict
        Simulation results including snow data and parameter estimates
    """
    # Start simulation in mid-summer (around July 15th = day 196 of year)
    # This way x0=0 makes sense as snow depth should be minimal in summer
    summer_start_day = 196  # July 15th (31+28+31+30+31+30+15 = 196)
    
    if n_years is not None:
        # Calculate total days needed including the summer offset
        total_days_needed = n_years * 365 + summer_start_day
        if total_days_needed > len(temperatures):
            print(f"Warning: Requested {n_years} years + summer offset requires {total_days_needed} days, but only {len(temperatures)} available.")
            print(f"Using all available data starting from summer day {summer_start_day}")
            temp_subset = temperatures[summer_start_day:]
        else:
            temp_subset = temperatures[summer_start_day:summer_start_day + n_years * 365]
    else:
        # Use all data starting from summer
        if len(temperatures) > summer_start_day:
            temp_subset = temperatures[summer_start_day:]
        else:
            print(f"Warning: Not enough data to start from summer day {summer_start_day}. Starting from beginning.")
            temp_subset = temperatures
    
    print(f"Running simulation starting from summer (day {summer_start_day}) with {len(temp_subset)} days ({len(temp_subset)/365:.1f} years)")
    
    # Simulate snow data starting from summer with x0=0 (realistic for summer)
    changes, depths = model.simulate(temp_subset, x0=x0)
    
    # Estimate parameters
    print("Estimating parameters...")
    estimates, success = estimate_parameters(depths, changes, temp_subset, verbose=True)
    
    # Calculate confidence intervals
    lower, upper, std_errors = calculate_confidence_intervals(
        depths, changes, temp_subset, estimates
    )
    
    return {
        'temperatures': temp_subset,
        'changes': changes,
        'depths': depths,
        'estimates': estimates,
        'estimation_success': success,
        'confidence_intervals': (lower, upper, std_errors),
        'model_params': [model.b0, model.b1, model.sigma]
    }


def plot_simulation_results(results, output_dir):
    """
    Create comprehensive plots of simulation results
    
    Parameters:
    -----------
    results : dict
        Results from run_single_simulation
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    temperatures = results['temperatures']
    depths = results['depths']
    changes = results['changes']
    n_days = len(depths)
    
    # Create date array starting from summer (July 15th)
    import datetime
    start_date = datetime.date(2000, 7, 15)  # Start from July 15th to match simulation start
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]
    
    # Define seasons based on month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    season_colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Fall': 'orange'}
    
    # Plot 1 (Top Left): Last 10 years of snow depths
    ax1 = axes[0, 0]
    n_years = n_days // 365
    if n_years >= 10:
        # Show last 10 years
        start_idx = max(0, n_days - 10*365)
        last_10_years_depths = depths[start_idx:]
        last_10_years_dates = dates[start_idx:]
        
        # Convert to years for x-axis
        years = [(d - dates[0]).days / 365.25 for d in last_10_years_dates]
        
        ax1.plot(years, last_10_years_depths, color='blue', alpha=0.7, linewidth=1)
        ax1.set_title('Snow Depth - Last 10 Years')
        ax1.set_xlabel('Years from Start')
        ax1.set_ylabel('Snow Depth (cm)')
    else:
        # Show all available data
        years = [(d - dates[0]).days / 365.25 for d in dates]
        ax1.plot(years, depths, color='blue', alpha=0.7, linewidth=1)
        ax1.set_title(f'Snow Depth - All {n_years} Years')
        ax1.set_xlabel('Years from Start')
        ax1.set_ylabel('Snow Depth (cm)')
    
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Max: {np.max(depths):.1f} cm\n'
    stats_text += f'Mean: {np.mean(depths):.1f} cm\n'
    stats_text += f'Snow days: {np.sum(depths > 0)} ({100*np.sum(depths > 0)/len(depths):.1f}%)'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.8), fontsize=9)
    
    # Plot 2 (Top Right): One year cycle (October to September)
    ax2 = axes[0, 1]
    
    # Find the most recent complete water year (Oct-Sep)
    water_year_data = []
    water_year_days = []
    
    # Look for a complete water year in the data
    for start_year in range(len(dates) // 365):
        start_idx = start_year * 365
        if start_idx + 365 <= len(dates):
            # Get data for this year
            year_dates = dates[start_idx:start_idx + 365]
            year_depths = depths[start_idx:start_idx + 365]
            
            # Reorder to start from October (month 10)
            reordered_depths = []
            reordered_days = []
            
            for i, date in enumerate(year_dates):
                if date.month >= 10:  # Oct, Nov, Dec
                    day_of_water_year = (date - datetime.date(date.year, 10, 1)).days + 1
                else:  # Jan-Sep of next year
                    day_of_water_year = (date - datetime.date(date.year-1, 10, 1)).days + 1
                
                reordered_days.append(day_of_water_year)
                reordered_depths.append(year_depths[i])
            
            if len(reordered_depths) > 0:
                water_year_data = reordered_depths
                water_year_days = reordered_days
                break
    
    if water_year_data:
        # Sort by water year day
        sorted_data = sorted(zip(water_year_days, water_year_data))
        sorted_days, sorted_depths = zip(*sorted_data)
        
        ax2.plot(sorted_days, sorted_depths, color='blue', linewidth=2)
        ax2.set_title('Annual Snow Cycle (Oct-Sep)')
        ax2.set_xlabel('Day of Water Year')
        ax2.set_ylabel('Snow Depth (cm)')
        
        # Add month labels
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        ax2.set_xticks(month_starts[::2])  # Every other month
        ax2.set_xticklabels(month_labels[::2])
    else:
        # Fallback: show first year available
        if len(depths) >= 365:
            year_depths = depths[:365]
            days_in_year = np.arange(1, 366)
            ax2.plot(days_in_year, year_depths, color='blue', linewidth=2)
            ax2.set_title('First Year Snow Depths')
            ax2.set_xlabel('Day of Year')
            ax2.set_ylabel('Snow Depth (cm)')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor annual cycle', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Annual Snow Cycle (Oct-Sep)')
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3 (Bottom Left): Changes vs Temperature colored by season
    ax3 = axes[1, 0]
    
    # Assign seasons to each data point
    seasons = [get_season(date.month) for date in dates]
    
    # Plot by season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_mask = [s == season for s in seasons]
        season_temps = temperatures[season_mask]
        season_changes = changes[season_mask]
        
        if len(season_temps) > 0:
            # Sample data if too many points
            if len(season_temps) > 1000:
                sample_idx = np.random.choice(len(season_temps), 1000, replace=False)
                season_temps = season_temps[sample_idx]
                season_changes = season_changes[sample_idx]
            
            ax3.scatter(season_temps, season_changes, alpha=0.6, s=2, 
                       color=season_colors[season], label=season)
    
    # Add fitted line
    temp_range = np.linspace(np.min(temperatures), np.max(temperatures), 100)
    fitted_changes = results['estimates'][0] + results['estimates'][1] * temp_range
    ax3.plot(temp_range, fitted_changes, 'black', linewidth=2, linestyle='--',
            label=f'Fitted: {results["estimates"][0]:.3f} + {results["estimates"][1]:.3f}T')
    
    ax3.set_title('Daily Changes vs Temperature by Season')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Daily Change (cm)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4 (Bottom Right): Days with/without snow by month
    ax4 = axes[1, 1]
    
    # Calculate snow days and no-snow days for each month
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    snow_days_by_month = []
    no_snow_days_by_month = []
    
    for month_num in range(1, 13):
        month_mask = [date.month == month_num for date in dates]
        month_depths = depths[month_mask]
        
        if len(month_depths) > 0:
            snow_days = np.sum(month_depths > 0)
            no_snow_days = np.sum(month_depths == 0)
            snow_days_by_month.append(snow_days)
            no_snow_days_by_month.append(no_snow_days)
        else:
            snow_days_by_month.append(0)
            no_snow_days_by_month.append(0)
    
    # Create stacked bar chart
    x_pos = np.arange(len(months))
    ax4.bar(x_pos, snow_days_by_month, label='Days with Snow', color='lightblue', alpha=0.8)
    ax4.bar(x_pos, no_snow_days_by_month, bottom=snow_days_by_month, 
           label='Days without Snow', color='lightcoral', alpha=0.8)
    
    ax4.set_title('Snow Days by Month')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Number of Days')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(months)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (snow, no_snow) in enumerate(zip(snow_days_by_month, no_snow_days_by_month)):
        total = snow + no_snow
        if total > 0:
            snow_pct = 100 * snow / total
            ax4.text(i, snow / 2, f'{snow_pct:.0f}%', ha='center', va='center', 
                    fontsize=8, fontweight='bold')
    
    plt.suptitle('Univariate Lindley Snow Model - Simulation Results', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'plots', 'simulation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Simulation plots saved: {plot_path}")


def save_results(results, convergence_results, output_dir):
    """
    Save all results to files
    
    Parameters:
    -----------
    results : dict
        Single simulation results
    convergence_results : dict
        Convergence study results
    output_dir : str
        Directory to save results
    """
    # Save single simulation data
    sim_data = pd.DataFrame({
        'day': np.arange(len(results['depths'])),
        'temperature': results['temperatures'],
        'snow_depth': results['depths'],
        'daily_change': results['changes']
    })
    
    sim_path = os.path.join(output_dir, 'data', 'simulation_data.csv')
    sim_data.to_csv(sim_path, index=False)
    
    # Save parameter estimates
    param_data = pd.DataFrame({
        'parameter': ['b0', 'b1', 'sigma'],
        'true_value': results['model_params'],
        'estimated_value': results['estimates']
    })
    
    if results['confidence_intervals'][0] is not None:
        lower, upper, std_errors = results['confidence_intervals']
        param_data['lower_ci'] = lower
        param_data['upper_ci'] = upper
        param_data['std_error'] = std_errors
    
    param_path = os.path.join(output_dir, 'data', 'parameter_estimates.csv')
    param_data.to_csv(param_path, index=False)
    
    # Save convergence study results
    if convergence_results is not None:
        conv_data = pd.DataFrame(convergence_results['all_estimates'], 
                                columns=['b0_est', 'b1_est', 'sigma_est'])
        conv_data['converged'] = convergence_results['all_convergence']
        
        conv_path = os.path.join(output_dir, 'data', 'convergence_study.csv')
        conv_data.to_csv(conv_path, index=False)
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'simulation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("UNIVARIATE LINDLEY SNOW MODEL - SIMULATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Length: {len(results['temperatures'])} days ({len(results['temperatures'])/365:.1f} years)\n")
        f.write(f"Temperature Range: {np.min(results['temperatures']):.1f}°C to {np.max(results['temperatures']):.1f}°C\n\n")
        
        f.write("Model Parameters (True):\n")
        f.write(f"  b0 = {results['model_params'][0]:.4f}\n")
        f.write(f"  b1 = {results['model_params'][1]:.4f}\n")
        f.write(f"  sigma = {results['model_params'][2]:.4f}\n\n")
        
        f.write("Parameter Estimates:\n")
        f.write(f"  b0 = {results['estimates'][0]:.4f}\n")
        f.write(f"  b1 = {results['estimates'][1]:.4f}\n")
        f.write(f"  sigma = {results['estimates'][2]:.4f}\n")
        f.write(f"  Estimation Success: {results['estimation_success']}\n\n")
        
        if results['confidence_intervals'][0] is not None:
            lower, upper, std_errors = results['confidence_intervals']
            f.write("95% Confidence Intervals:\n")
            for i, param in enumerate(['b0', 'b1', 'sigma']):
                f.write(f"  {param}: [{lower[i]:.4f}, {upper[i]:.4f}] (SE: {std_errors[i]:.4f})\n")
            f.write("\n")
        
        f.write("Snow Simulation Statistics:\n")
        f.write(f"  Maximum depth: {np.max(results['depths']):.1f} cm\n")
        f.write(f"  Mean depth: {np.mean(results['depths']):.1f} cm\n")
        f.write(f"  Days with snow: {np.sum(results['depths'] > 0)} ({100*np.sum(results['depths'] > 0)/len(results['depths']):.1f}%)\n")
        f.write(f"  Days without snow: {np.sum(results['depths'] == 0)} ({100*np.sum(results['depths'] == 0)/len(results['depths']):.1f}%)\n\n")
        
        if convergence_results is not None:
            f.write("Convergence Study Results:\n")
            f.write(f"  Total simulations: {convergence_results['n_total']}\n")
            f.write(f"  Successful estimations: {convergence_results['n_successful']}\n")
            f.write(f"  Convergence rate: {convergence_results['convergence_rate']:.1%}\n")
            
            if convergence_results['n_successful'] > 0:
                f.write(f"  Mean estimates: b0={convergence_results['mean_estimates'][0]:.4f}, "
                       f"b1={convergence_results['mean_estimates'][1]:.4f}, "
                       f"sigma={convergence_results['mean_estimates'][2]:.4f}\n")
                f.write(f"  Bias: b0={convergence_results['bias'][0]:.4f}, "
                       f"b1={convergence_results['bias'][1]:.4f}, "
                       f"sigma={convergence_results['bias'][2]:.4f}\n")
                f.write(f"  RMSE: b0={convergence_results['rmse'][0]:.4f}, "
                       f"b1={convergence_results['rmse'][1]:.4f}, "
                       f"sigma={convergence_results['rmse'][2]:.4f}\n")
    
    print(f"Results saved to: {output_dir}")
    print(f"  - Simulation data: {sim_path}")
    print(f"  - Parameter estimates: {param_path}")
    if convergence_results is not None:
        print(f"  - Convergence study: {conv_path}")
    print(f"  - Summary report: {summary_path}")


def main():
    """Main simulation runner"""
    parser = argparse.ArgumentParser(description='Run Univariate Lindley Snow Model Simulation')
    parser.add_argument('--n_sims', type=int, default=100, 
                       help='Number of simulations for convergence study (default: 100)')
    parser.add_argument('--years', type=int, default=None,
                       help='Number of years to simulate (default: all available data)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_convergence', action='store_true',
                       help='Skip convergence study (faster execution)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("UNIVARIATE LINDLEY SNOW MODEL SIMULATION")
    print("="*70)
    
    # Set random seed
    np.random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    try:
        # Load temperature data
        temp_data = load_temperature_data()
        temperatures = temp_data['temperature'].values
        
        # Create snow model with realistic parameters
        print("\nInitializing snow model...")
        model = UnivariateSnowModel(b0=-.04, b1=-.02, sigma=1)
        print(f"Model parameters: b0={model.b0}, b1={model.b1}, sigma={model.sigma}")
        
        # Create output directory
        output_dir = create_output_directory("univariate_simulation")
        print(f"Output directory: {output_dir}")
        
        # Run single simulation
        print("\n" + "-"*50)
        print("RUNNING SINGLE SIMULATION")
        print("-"*50)
        
        results = run_single_simulation(model, temperatures, n_years=args.years)
        
        # Print single simulation results
        print(f"\nSingle Simulation Results:")
        print(f"  Parameter estimation {'succeeded' if results['estimation_success'] else 'failed'}")
        if results['estimation_success']:
            true_params = results['model_params']
            estimates = results['estimates']
            print(f"  True parameters:      b0={true_params[0]:.4f}, b1={true_params[1]:.4f}, sigma={true_params[2]:.4f}")
            print(f"  Estimated parameters: b0={estimates[0]:.4f}, b1={estimates[1]:.4f}, sigma={estimates[2]:.4f}")
            
            bias = estimates - np.array(true_params)
            print(f"  Bias:                 b0={bias[0]:.4f}, b1={bias[1]:.4f}, sigma={bias[2]:.4f}")
        
        # Plot single simulation results
        plot_simulation_results(results, output_dir)
        
        # Run convergence study
        convergence_results = None
        if not args.no_convergence:
            print("\n" + "-"*50)
            print("RUNNING CONVERGENCE STUDY")
            print("-"*50)
            
            # Use subset of temperature data for convergence study
            temp_subset = temperatures[:10*365] if len(temperatures) > 10*365 else temperatures
            
            convergence_results = run_convergence_study(
                model, temp_subset, 
                true_params=results['model_params'],
                n_simulations=args.n_sims,
                random_seed=args.seed
            )
            
            # Plot convergence analysis
            conv_fig = plot_convergence_analysis(
                convergence_results, 
                save_path=os.path.join(output_dir, 'plots', 'convergence_analysis.png')
            )
            if conv_fig:
                plt.show()
            
            # Print detailed summary
            print_estimation_summary(convergence_results, results['confidence_intervals'])
        
        # Save all results
        print("\n" + "-"*50)
        print("SAVING RESULTS")
        print("-"*50)
        
        save_results(results, convergence_results, output_dir)
        
        print(f"\nSimulation completed successfully!")
        print(f"All results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
