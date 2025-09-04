#!/usr/bin/env python3
"""
Temperature Simulation Module - FIXED VERSION

Simulates realistic temperature data using periodic mean with seasonal components.

Author: Joshua King
Date: Jun 26, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta

class TemperatureSimulator:
    def __init__(self, alpha0=5.0, alpha1=20.0, 
                 tau1=196, sigma=4.0, alpha2=0.00006):
        """
        Initialize temperature simulator with realistic parameters
        
        Parameters:
        - alpha0: baseline average annual temperature (°C)
        - alpha1: amplitude of seasonal variation (°C)
        - tau1: phase shift (day of year for maximum temperature; typically Jul 15, 196th day)
        - sigma: standard deviation of daily temperature noise (°C)
        - alpha2: long-term warming trend per day (°C/day)
        """
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.tau1 = tau1
        self.sigma = sigma
        self.alpha2 = alpha2
        self.P = 365 # Period (days in a year)
    
    def periodic_temperature_mean(self, day_of_year, year):
        """
        Calculate the expected temperature for a given day and year
        
        mu_t = alpha0 + alpha1 * cos(2π(day - tau1)/365) + alpha2 * t
        
        Returns: (expected_temp, total_day_number)
        """
        t = (year - 1) * self.P + day_of_year
        
        seasonal_component = self.alpha1 * np.cos(2 * np.pi * (day_of_year - self.tau1) / self.P)
        
        mu_t = self.alpha0 + seasonal_component + self.alpha2 * t
        
        return mu_t, t
    
    def simulate_temperatures(self, n_years, start_year=2000, random_seed=None):
        """
        Generate daily temperature time series
        
        Returns:
        - temperatures: array of daily temperatures
        - dates: array of corresponding dates
        - metadata: dictionary with simulation parameters
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            
        total_days = n_years * self.P
        temperatures = np.zeros(total_days)
        
        for year in range(1, n_years + 1):
            for day in range(1, self.P + 1):
                idx = (year - 1) * self.P + (day - 1) # Convert to 0-based indexing
                mu_t, t = self.periodic_temperature_mean(day, year)
                epsilon_t = np.random.normal(0, self.sigma)
                temperatures[idx] = mu_t + epsilon_t

        # Get dates for visualization (not a part of actual simulation)
        months = [31, 28, 31, 30, 31, 30,
                  31, 31, 30, 31, 30, 31]  # Days in each month (non-leap year)

        dates = []
        year = start_year
        month = 1
        day = 1

        for _ in range(total_days):
            dates.append(f"{year:04d}-{month:02d}-{day:02d}")

            day += 1
            if day > months[month - 1]:
                day = 1
                month += 1
                if month > 12:
                    month = 1
                    year += 1
        
        dates = pd.to_datetime(dates) # Set to datetime for easy extraction
        
        # Metadata for analysis
        metadata = {
            'n_years': n_years,
            'start_year': start_year,
            'alpha0': self.alpha0,
            'alpha1': self.alpha1,
            'tau1': self.tau1,
            'sigma': self.sigma,
            'alpha2': self.alpha2,
            'total_warming': self.alpha2 * total_days,
            'annual_warming': self.alpha2 * self.P
        }
        
        return temperatures, dates, metadata
    
    def visualize_temperatures(self, temperatures, dates, metadata, temps_f=False,
                               save_plot=False, filename=None):
        """
        Create comprehensive temperature visualization with monthly min/max, full time series,
        trend analysis, and distribution
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temperature Simulation Analysis', fontsize=16, fontweight='bold')
        
        # Convert to numpy arrays for easier manipulation
        temps = np.array(temperatures)
        n_years = metadata['n_years']
        
        temp_scale = "(°C)"
        
        # Convert to Fahrenheit
        if temps_f == True:
            temp_scale = "(°F)"
            temps = temps * 1.8 + 32
        
        # Plot 1 (Top Left): Monthly Min/Max Temperature Pattern
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Calculate monthly statistics across all years
        monthly_mins = []
        monthly_maxs = []
        
        for month in range(1, 13):
            month_temps = []
            for year in range(n_years):
                # Get all days in this month for this year
                year_start_date = dates[year * 365]
                for day_idx in range(year * 365, min((year + 1) * 365, len(dates))):
                    if dates[day_idx].month == month:
                        month_temps.append(temps[day_idx])
            
            if month_temps:
                monthly_mins.append(np.min(month_temps))
                monthly_maxs.append(np.max(month_temps))
            else:
                monthly_mins.append(0)
                monthly_maxs.append(0)
        
        # Create monthly plot with min/max bands
        month_indices = np.arange(1, 13)
        
        # Fill between min and max
        axes[0, 0].fill_between(month_indices, monthly_mins, monthly_maxs, 
                               alpha=0.3, color='#E6E6FA', label='Temperature Range')
        
        # Plot min and max lines
        axes[0, 0].plot(month_indices, monthly_maxs, ls='-', linewidth=2, color='red', alpha=0.6,
                       marker='o', markersize=4, label='Monthly High')
        axes[0, 0].plot(month_indices, monthly_mins, ls='-', linewidth=2, color='blue', alpha=0.6,
                       marker='o', markersize=4, label='Monthly Low')
        
        # Add temperature labels on the points
        for i, (month_min, month_max) in enumerate(zip(monthly_mins, monthly_maxs)):
            axes[0, 0].text(i+1, month_max + 1, f'{month_max:.0f}°', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            axes[0, 0].text(i+1, month_min - 1, f'{month_min:.0f}°', 
                           ha='center', va='top', fontsize=9, fontweight='bold')
        
        axes[0, 0].set_title('Monthly Temperature Range')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel(f'Temperature {temp_scale}')
        axes[0, 0].set_xticks(month_indices)
        axes[0, 0].set_xticklabels(month_names)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2 (Top Right): Last 5 Years Time Series with Theoretical Line
        # Determine the range for last 5 years
        years_to_show = min(5, n_years)  # Show last 5 years or all if less than 5
        start_index = max(0, len(temps) - (years_to_show * 365))
        end_index = len(temps)
        
        # Slice data for last 5 years
        last_temps = temps[start_index:end_index]
        start_year = metadata['start_year']
        last_years_start = start_year + (n_years - years_to_show)
        years_float_last = np.array([last_years_start + (i / 365.0) for i in range(len(last_temps))])
        
        # Plot actual temperatures for last 5 years
        axes[0, 1].plot(years_float_last, last_temps, ls='-',
                        color="blue", linewidth=0.8, alpha=0.6, label='Simulated')
        
        # Plot theoretical line for last 5 years (no noise, no trend)
        theoretical_temps_last = []
        for i in range(len(last_temps)):
            day_of_year = ((start_index + i) % 365) + 1
            # Calculate baseline temperature without trend
            seasonal_component = self.alpha1 * np.cos(
                2 * np.pi * (day_of_year - self.tau1) / self.P
            )
            expected_temp = self.alpha0 + seasonal_component
            theoretical_temps_last.append(expected_temp)
        
        if temps_f == True:
            theoretical_temps_last = np.array(theoretical_temps_last) * 1.8 + 32
        
        axes[0, 1].plot(years_float_last, theoretical_temps_last, ls='-',
                        color='red', linewidth=1.5, alpha=0.8, label='Theoretical (no noise/trend)')
        
        axes[0, 1].set_title(f'Temperature Time Series - Last {years_to_show} Years')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel(f'Temperature {temp_scale}')
        
        # Set x-axis to show years without decimals for the last period
        year_ticks = np.arange(last_years_start, start_year + n_years + 1, 1)  # Show ~5 ticks for 5 years
        axes[0, 1].set_xticks(year_ticks)
        axes[0, 1].set_xticklabels([f'{int(year)}' for year in year_ticks])
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3 (Bottom Left): Temperature Distribution Change by Month (Ridgeline Style)
        if n_years >= 10:
            
            # Determine periods for comparison
            first_10_years = min(10, n_years)
            last_10_years = min(10, n_years)
            
            # Get indices for first and last periods
            first_period_end = first_10_years * 365
            last_period_start = max(0, len(temps) - (last_10_years * 365))
            
            first_temps = temps[:first_period_end]
            last_temps = temps[last_period_start:]
            
            # Get corresponding dates for period identification
            first_dates = dates[:first_period_end]
            last_dates = dates[last_period_start:]
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Create ridgeline plot
            from scipy.stats import gaussian_kde
            
            # Set up the plot - months on y-axis, temperature on x-axis
            y_positions = np.arange(12)
            month_height = 0.8  # Height allocated for each month's distributions
            
            # Determine overall temperature range
            all_temps = list(first_temps) + list(last_temps)
            temp_min, temp_max = min(all_temps), max(all_temps)
            temp_range = np.linspace(temp_min - 5, temp_max + 5, 200)
            
            # For each month, create overlapping density curves
            for month_idx in range(12):
                month_num = month_idx + 1
                y_base = month_idx  # Base y-position for this month
                
                # Get temperatures for this month from first period
                first_month_temps = []
                for i, date in enumerate(first_dates):
                    if date.month == month_num:
                        first_month_temps.append(first_temps[i])
                
                # Get temperatures for this month from last period  
                last_month_temps = []
                for i, date in enumerate(last_dates):
                    if date.month == month_num:
                        last_month_temps.append(last_temps[i])
                
                if len(first_month_temps) > 5 and len(last_month_temps) > 5:
                    # Create density curves
                    first_kde = gaussian_kde(first_month_temps)
                    last_kde = gaussian_kde(last_month_temps)
                    
                    # Calculate density values
                    first_density = first_kde(temp_range)
                    last_density = last_kde(temp_range)
                    
                    # Normalize densities to fit within month height
                    max_density = max(np.max(first_density), np.max(last_density))
                    first_density_scaled = (first_density / max_density) * month_height
                    last_density_scaled = (last_density / max_density) * month_height
                    
                    # Plot density curves as filled areas above the baseline
                    axes[1, 0].fill_between(temp_range, y_base, y_base + first_density_scaled,
                                           alpha=0.6, color='blue', 
                                           label='First 10 years' if month_idx == 0 else "")
                    
                    axes[1, 0].fill_between(temp_range, y_base, y_base + last_density_scaled,
                                           alpha=0.6, color='red',
                                           label='Last 10 years' if month_idx == 0 else "")
                    
                    # Add baseline for each month
                    axes[1, 0].axhline(y=y_base, color='gray', alpha=0.3, linewidth=0.5)
            
            # Set up axes
            axes[1, 0].set_xlabel(f'Temperature {temp_scale}')
            axes[1, 0].set_ylabel('Month')
            axes[1, 0].set_yticks(y_positions)
            axes[1, 0].set_yticklabels(month_names[::-1])
            axes[1, 0].set_title('Temperature Distribution Change by Month\n(First vs Last 10 Years)')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            axes[1, 0].legend(loc='upper right')
            
            # Set y-axis limits
            axes[1, 0].set_ylim(-0.5, 12)
            
        else:
            # If less than 10 years, show a simpler comparison
            axes[1, 0].text(0.5, 0.5, f'Need ≥10 years for comparison\n(Current: {n_years} years)', 
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.6))
            axes[1, 0].set_title('Temperature Distribution Change')
            axes[1, 0].set_xlabel(f'Temperature {temp_scale}')
            axes[1, 0].set_ylabel('Month')
        
        # Plot 4 (Bottom Right): Temperature Distribution (unchanged)
        axes[1, 1].hist(temps, bins=50, density=True, alpha=0.6, color='blue', 
                       edgecolor='black', linewidth=0.5)
        axes[1, 1].set_title('Temperature Distribution')
        axes[1, 1].set_xlabel(f'Temperature {temp_scale}')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Simulation Parameters:
Baseline: {metadata['alpha0']:.1f}{temp_scale}
Seasonal Amplitude: {metadata['alpha1']:.1f}{temp_scale}
Noise Std: {metadata['sigma']:.1f}{temp_scale}
Daily Trend: {metadata['alpha2']:.2e}{temp_scale}/day
Total Warming: {metadata['total_warming']:.3f}{temp_scale}

Data Statistics:
Mean: {np.mean(temps):.2f}{temp_scale}
Std: {np.std(temps):.2f}{temp_scale}
Min: {np.min(temps):.2f}{temp_scale}
Max: {np.max(temps):.2f}{temp_scale}"""
        
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8), fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            if filename is None:
                filename = f'temperature_simulation_{metadata["n_years"]}years.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        #plt.show()
        
        return fig, axes

def main():
    """
    Example usage and parameter tuning
    """
    print("Temperature Simulation Module - FIXED VERSION")
    print("=" * 50)
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"temperature_data_{timestamp}"
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # Create simulator with realistic parameters
    # These parameters simulate temperate climate (like continental US)
    temp_sim = TemperatureSimulator(
        alpha0=5.0,      # average
        alpha1=20.0,  # seasonal variation
        tau1=196,          # Warmest around Jul 15
        sigma=4.0,           # daily variability
        alpha2=0.00006    # ~0.022°C warming per year
    )
    
    # Simulate 100 years of temperature data
    print("Simulating 100 years of temperature data...")
    temperatures, dates, metadata = temp_sim.simulate_temperatures(100, start_year=1920, random_seed=123)
    
    print(f"Generated {len(temperatures)} daily temperature observations")
    print(f"Temperature range: {np.min(temperatures):.1f}°C to {np.max(temperatures):.1f}°C")
    print(f"Mean temperature: {np.mean(temperatures):.2f}°C")
    
    # Create visualization and save to output directory
    plot_filename = os.path.join(output_dir, 'temperature_simulation_100years.png')
    temp_sim.visualize_temperatures(temperatures, dates, metadata, 
                                   save_plot=True, temps_f=True, filename=plot_filename)
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperatures,
        'temperature_f': np.array(temperatures) * 1.8 + 32,
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day_of_year': [d.timetuple().tm_yday for d in dates]
    })
    
    print("\nSample of generated data:")
    print(df.head(10))
    
    # Generate filename with date range
    start_year = dates[0].year
    end_year = dates[-1].year
    csv_filename = os.path.join(output_dir, f'temperature_data_{start_year}_{end_year}.csv')
        
    # Save to CSV in the output directory
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to directory: {output_dir}")
    print(f"  - Temperature data: {csv_filename}")
    print(f"  - Visualization: {plot_filename}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return temperatures, dates, metadata, df

if __name__ == "__main__":
    main()