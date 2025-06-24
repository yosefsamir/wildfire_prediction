"""Visualization functions for wildfire prediction.

This module contains functions for creating visualizations of wildfire data.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def visualize_frp_distribution(df, output_dir):
    """Create and save visualizations of FRP distribution.
    
    Args:
        df: DataFrame with FRP data
        output_dir: Directory to save the visualizations
        
    Returns:
        None
    """
    if 'frp' not in df.columns:
        print("FRP column not found in dataframe")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate skewness
    frp_skewness = stats.skew(df['frp'].dropna())
    
    # Create histogram for original FRP
    plt.figure(figsize=(10, 6))
    sns.histplot(df['frp'], kde=True, color='orangered')
    plt.axvline(df['frp'].mean(), color='red', linestyle='--', label=f'Mean: {df["frp"].mean():.2f}')
    plt.axvline(df['frp'].median(), color='green', linestyle='-.', label=f'Median: {df["frp"].median():.2f}')
    plt.title(f'Distribution of Fire Radiative Power (FRP)\nSkewness: {frp_skewness:.4f}', fontsize=14)
    plt.xlabel('Fire Radiative Power (FRP)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'frp_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create histogram for log-transformed FRP
    if 'frp_log' in df.columns:
        log_frp_skewness = stats.skew(df['frp_log'].dropna())
        plt.figure(figsize=(10, 6))
        sns.histplot(df['frp_log'], kde=True, color='blue')
        plt.title(f'Log-Transformed Distribution of FRP\nSkewness: {log_frp_skewness:.4f}', fontsize=14)
        plt.xlabel('Log(FRP + 1)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'log_frp_distribution.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"FRP distribution visualizations saved to: {output_dir}")


def visualize_california_fires(california_data, california, output_dir):
    """Create and save visualization of fire points in California.
    
    Args:
        california_data: GeoDataFrame with fire points in California
        california: GeoDataFrame with California boundary
        output_dir: Directory to save the visualization
        
    Returns:
        None
    """
    if california_data.empty or california.empty:
        print("Cannot create California visualization: empty data")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data if it's too large
    plot_data = california_data
    if len(california_data) > 10000:
        plot_data = california_data.sample(10000, random_state=42)
        print(f"Sampled to {len(plot_data)} points for visualization")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    california.plot(ax=ax, color='lightgrey', edgecolor='black')
    plot_data.plot(ax=ax, column='brightness', cmap='hot', 
                markersize=2, legend=True, 
                legend_kwds={'label': 'Brightness', 'orientation': 'horizontal'})
    plt.title('Wildfire Data Points in California', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'california_wildfires.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"California wildfire visualization saved to: {output_dir}")