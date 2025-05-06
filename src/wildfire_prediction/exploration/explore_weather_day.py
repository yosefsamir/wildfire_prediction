import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from matplotlib.colors import Normalize
from datetime import datetime
import sys
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster

# Add the project root to sys.path to allow imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def load_weather_data(date_str, data_dir=None):
    """
    Load weather data for a specific date (format: YYYYMMDD).
    
    Parameters:
    -----------
    date_str : str
        Date string in YYYYMMDD format
    data_dir : str, optional
        Path to the directory containing the weather data
        
    Returns:
    --------
    pd.DataFrame
        Weather data for the specified date
    """
    if data_dir is None:
        # Default path relative to project root
        data_dir = project_root / "data" / "raw" / "weather" / date_str[:4]
    
    file_path = os.path.join(data_dir, f"{date_str}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weather data file not found: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"Loaded weather data for {date_str}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    return df

def load_california_map(geojson_path=None):
    """
    Load California map data.
    
    Parameters:
    -----------
    geojson_path : str, optional
        Path to California GeoJSON file
        
    Returns:
    --------
    GeoDataFrame
        GeoDataFrame containing California boundary
    """
    if geojson_path is None:
        # Default path
        geojson_path = project_root / "notebooks" / "california.geojson"
    
    california = gpd.read_file(geojson_path)
    return california

def explore_weather_data_summary(df):
    """
    Provide a summary of the weather data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        "total_points": len(df),
        "longitude_range": (df['longitude'].min(), df['longitude'].max()),
        "latitude_range": (df['latitude'].min(), df['latitude'].max()),
        "precipitation_stats": df['ppt'].describe(),
        "max_temp_stats": df['tmax'].describe(),
        "max_vpd_stats": df['vbdmax'].describe()
    }
    
    # Print summary
    print("\n=== Weather Data Summary ===")
    print(f"Total data points: {summary['total_points']}")
    print(f"Longitude range: {summary['longitude_range'][0]} to {summary['longitude_range'][1]}")
    print(f"Latitude range: {summary['latitude_range'][0]} to {summary['latitude_range'][1]}")
    print("\nPrecipitation (ppt) statistics:")
    print(summary['precipitation_stats'])
    print("\nMax Temperature (tmax) statistics:")
    print(summary['max_temp_stats'])
    print("\nMax Vapor Pressure Deficit (vbdmax) statistics:")
    print(summary['max_vpd_stats'])
    
    return summary

def plot_point_distribution(df, california=None, figsize=(12, 10)):
    """
    Plot the distribution of weather data points on a map of California.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    california : GeoDataFrame, optional
        California boundary GeoDataFrame
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig, ax
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If California map is provided, plot it as background
    if california is not None:
        california.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Create a scatter plot of points
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'], 
        c=df['tmax'], 
        cmap='coolwarm',
        alpha=0.6,
        s=10,
        edgecolor='none'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Maximum Temperature (°C)')
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    date_str = df['date'].iloc[0] if 'date' in df.columns else "Unknown Date"
    ax.set_title(f'Weather Data Points in California - {date_str}')
    
    plt.tight_layout()
    return fig, ax

def plot_temperature_heatmap(df, california=None, figsize=(12, 10), resolution=0.05):
    """
    Create a temperature heatmap over California.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    california : GeoDataFrame, optional
        California boundary GeoDataFrame
    figsize : tuple, optional
        Figure size
    resolution : float, optional
        Grid resolution for interpolation
        
    Returns:
    --------
    fig, ax
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If California map is provided, plot it as background
    if california is not None:
        california.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Create a heatmap-style plot
    # For simplicity, we'll use a scatter plot with larger point sizes to create a pseudo-heatmap
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'], 
        c=df['tmax'], 
        cmap='coolwarm',
        alpha=0.7,
        s=30,
        edgecolor='none'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Maximum Temperature (°C)')
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    date_str = df['date'].iloc[0] if 'date' in df.columns else "Unknown Date"
    ax.set_title(f'Temperature Heatmap - {date_str}')
    
    plt.tight_layout()
    return fig, ax

def plot_variable_distribution(df, variable='tmax', bins=30, figsize=(10, 6)):
    """
    Plot the distribution of a specific variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    variable : str, optional
        Variable to plot ('tmax', 'ppt', or 'vbdmax')
    bins : int, optional
        Number of bins for histogram
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig, ax
        Figure and axis objects
    """
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame columns")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the histogram
    sns.histplot(df[variable], bins=bins, kde=True, ax=ax)
    
    # Set labels and title
    variable_labels = {
        'tmax': 'Maximum Temperature (°C)',
        'ppt': 'Precipitation (mm)',
        'vbdmax': 'Maximum Vapor Pressure Deficit (kPa)'
    }
    
    ax.set_xlabel(variable_labels.get(variable, variable))
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {variable_labels.get(variable, variable)}')
    
    plt.tight_layout()
    return fig, ax

def explore_precipitation_patterns(df, california=None, figsize=(12, 10)):
    """
    Explore precipitation patterns across California.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    california : GeoDataFrame, optional
        California boundary GeoDataFrame
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig, ax
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If California map is provided, plot it as background
    if california is not None:
        california.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Create a scatter plot with point size proportional to precipitation
    # Add a small epsilon to ensure zero precipitation values are visible
    epsilon = 0.01
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'], 
        c=df['ppt'],
        s=np.sqrt(df['ppt'] + epsilon) * 20,  # Scale the size by sqrt for better visualization
        cmap='Blues',
        alpha=0.7,
        edgecolor='none'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Precipitation (mm)')
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    date_str = df['date'].iloc[0] if 'date' in df.columns else "Unknown Date"
    ax.set_title(f'Precipitation Patterns - {date_str}')
    
    plt.tight_layout()
    return fig, ax

def create_interactive_map(df, california=None, zoom_start=6, variable='tmax', 
                        map_type='markers', html_path=None):
    """
    Create an interactive map visualization of weather data points using Folium.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    california : GeoDataFrame, optional
        California boundary GeoDataFrame
    zoom_start : int, optional
        Initial zoom level for the map
    variable : str, optional
        Variable to use for coloring points ('tmax', 'ppt', or 'vbdmax')
    map_type : str, optional
        Type of map visualization ('markers', 'cluster', or 'heatmap')
    html_path : str, optional
        Path to save the HTML file of the map
        
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame columns")
    
    # Calculate the center of the data points
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create a base map
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=zoom_start,
                   tiles='CartoDB positron')
    
    # Add California boundary if provided
    if california is not None:
        folium.GeoJson(
            california,
            name='California Boundary',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 2
            }
        ).add_to(m)
    
    # Define color scales and variable labels
    variable_labels = {
        'tmax': 'Maximum Temperature (°C)',
        'ppt': 'Precipitation (mm)',
        'vbdmax': 'Maximum Vapor Pressure Deficit (kPa)'
    }
    
    # Get date from dataframe if available
    date_str = df['date'].iloc[0] if 'date' in df.columns else "Unknown Date"
    
    # Create different types of visualizations based on map_type
    if map_type == 'heatmap':
        # Create a heatmap layer
        heat_data = [[row['latitude'], row['longitude'], row[variable]] 
                     for _, row in df.iterrows() if not np.isnan(row[variable])]
        
        HeatMap(
            heat_data,
            radius=15,
            gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'red'},
            name=f'{variable_labels.get(variable, variable)} Heatmap',
            show=True
        ).add_to(m)
        
    elif map_type == 'cluster':
        # Create a marker cluster layer
        marker_cluster = MarkerCluster(
            name=f'{variable_labels.get(variable, variable)} Clusters'
        ).add_to(m)
        
        # Add markers to the cluster
        for _, row in df.iterrows():
            if not np.isnan(row[variable]):
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color='gray',
                    fill_color='blue',
                    fill_opacity=0.7,
                    popup=f"{variable}: {row[variable]:.2f}",
                ).add_to(marker_cluster)
    
    else:  # 'markers' (default)
        # Normalize the variable for color mapping
        vmin = df[variable].min()
        vmax = df[variable].max()
        
        # Create a feature group for the markers
        feature_group = folium.FeatureGroup(
            name=f'{variable_labels.get(variable, variable)} Markers'
        ).add_to(m)
        
        # Add individual markers
        for _, row in df.iterrows():
            if not np.isnan(row[variable]):
                # Calculate color based on variable value
                normalized_value = (row[variable] - vmin) / (vmax - vmin)
                
                # Use different color schemes based on variable
                if variable == 'tmax':
                    # Blue to red for temperature
                    rgb = f'rgb({int(255 * normalized_value)}, {int(50 * (1 - normalized_value))}, {int(255 * (1 - normalized_value))})'
                elif variable == 'ppt':
                    # Light blue to dark blue for precipitation
                    blue_val = int(100 + 155 * normalized_value)
                    rgb = f'rgb(0, {int(100 + 50 * normalized_value)}, {blue_val})'
                else:
                    # Default gradient
                    rgb = f'rgb({int(255 * normalized_value)}, {int(255 * (1 - normalized_value))}, 0)'
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    color='gray',
                    fill=True,
                    fill_color=rgb,
                    fill_opacity=0.7,
                    popup=f"<b>{variable_labels.get(variable, variable)}:</b> {row[variable]:.2f}<br>"
                          f"<b>Lat:</b> {row['latitude']:.4f}<br>"
                          f"<b>Lon:</b> {row['longitude']:.4f}",
                    tooltip=f"{variable}: {row[variable]:.2f}"
                ).add_to(feature_group)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a title
    title_html = f'''
        <h3 align="center" style="font-size:16px">
            <b>Weather Data for {date_str}</b>
        </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map if a path is provided
    if html_path:
        m.save(html_path)
        print(f"Interactive map saved to: {html_path}")
    
    return m

def explore_one_day(date_str='20130101', data_dir=None, geojson_path=None, save_dir=None):
    """
    Explore weather data for a specific day and generate visualizations.
    
    Parameters:
    -----------
    date_str : str, optional
        Date string in YYYYMMDD format
    data_dir : str, optional
        Path to the directory containing the weather data
    geojson_path : str, optional
        Path to California GeoJSON file
    save_dir : str, optional
        Directory to save the generated figures
        
    Returns:
    --------
    dict
        Dictionary containing the generated plots and data summaries
    """
    # Load the data
    df = load_weather_data(date_str, data_dir)
    
    # Load California map
    try:
        california = load_california_map(geojson_path)
    except Exception as e:
        print(f"Warning: Could not load California map: {e}")
        california = None
    
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Generate data summary
    summary = explore_weather_data_summary(df)
    
    # Generate plots
    results = {'summary': summary, 'data': df}
    
    # Plot point distribution
    fig_points, ax_points = plot_point_distribution(df, california)
    results['point_distribution'] = (fig_points, ax_points)
    if save_dir:
        fig_points.savefig(os.path.join(save_dir, f"{date_str}_point_distribution.png"))
    
    # Plot temperature heatmap
    fig_temp, ax_temp = plot_temperature_heatmap(df, california)
    results['temperature_heatmap'] = (fig_temp, ax_temp)
    if save_dir:
        fig_temp.savefig(os.path.join(save_dir, f"{date_str}_temperature_heatmap.png"))
    
    # Plot variable distributions
    for var in ['tmax', 'ppt', 'vbdmax']:
        fig_var, ax_var = plot_variable_distribution(df, var)
        results[f'{var}_distribution'] = (fig_var, ax_var)
        if save_dir:
            fig_var.savefig(os.path.join(save_dir, f"{date_str}_{var}_distribution.png"))
    
    # Plot precipitation patterns
    fig_precip, ax_precip = explore_precipitation_patterns(df, california)
    results['precipitation_patterns'] = (fig_precip, ax_precip)
    if save_dir:
        fig_precip.savefig(os.path.join(save_dir, f"{date_str}_precipitation_patterns.png"))
    
    # Create interactive maps
    # Temperature map
    temp_map = create_interactive_map(
        df, california, variable='tmax', map_type='markers',
        html_path=os.path.join(save_dir, f"{date_str}_temp_interactive_map.html") if save_dir else None
    )
    results['temp_interactive_map'] = temp_map
    
    # Precipitation map
    precip_map = create_interactive_map(
        df, california, variable='ppt', map_type='heatmap',
        html_path=os.path.join(save_dir, f"{date_str}_precip_heatmap.html") if save_dir else None
    )
    results['precip_interactive_map'] = precip_map
    
    # VPD map with clustering
    vpd_map = create_interactive_map(
        df, california, variable='vbdmax', map_type='cluster',
        html_path=os.path.join(save_dir, f"{date_str}_vpd_cluster_map.html") if save_dir else None
    )
    results['vpd_interactive_map'] = vpd_map
    
    print(f"\nExploration completed for date: {date_str}")
    if save_dir:
        print(f"Figures and interactive maps saved to: {save_dir}")
    
    return results

if __name__ == "__main__":
    # Example usage
    # You can run this file directly to explore weather data for January 1, 2013
    date_to_explore = "20130101"
    save_dir = os.path.join(project_root, "reports", "figures", "weather_exploration")
    
    results = explore_one_day(
        date_str=date_to_explore,
        save_dir=save_dir
    )
    
    # Show the plots
    plt.show()