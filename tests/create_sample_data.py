#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for merging fire and weather data on sample datasets.

This script creates sample data for California wildfires and weather stations,
then demonstrates the merging process to create a merged dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

# Configuration
TEMPORAL_TOLERANCE = pd.Timedelta('1D')  # 1-day tolerance for temporal matching
BUFFER_DEGREES = 1.0  # ~111km buffer for spatial filtering

def create_sample_fire_data():
    """Create a sample California wildfire dataset."""
    print("Creating sample California wildfire data...")
    return pd.DataFrame({
        'grid_id': ['A1', 'B2', 'C3', 'D4', 'E5'],
        'week': ['2022-W01', '2022-W02', '2022-W03', '2022-W04', '2022-W05'],
        'latitude': [37.7749, 36.7783, 38.5816, 37.8044, 38.2721],  # California coordinates
        'longitude': [-122.4194, -119.4179, -121.4944, -122.2711, -122.0137],
        'acq_date': pd.to_datetime(['2022-01-05', '2022-01-12', '2022-01-18', '2022-01-25', '2022-02-01']),
        'frp_log': [1.2, 2.3, 0.8, 1.5, 2.1],
        'brightness_normalized': [0.4, 0.7, 0.2, 0.5, 0.6],
        'fire': [1, 1, 1, 1, 1]  # All are fire records
    })

def create_sample_weather_data():
    """Create a sample weather master dataset."""
    print("Creating sample weather master data...")
    # Create data for two years to test filtering
    data = []
    # 2022 data - matches with fire data
    for day in pd.date_range(start='2022-01-01', end='2022-02-15'):
        # Add multiple weather stations for each day
        for i, (lat, lon) in enumerate([
            (37.77, -122.42),  # Close to fire point A1
            (36.78, -119.42),  # Close to fire point B2
            (38.58, -121.49),  # Close to fire point C3
            (37.80, -122.27),  # Close to fire point D4
            (38.27, -122.01),  # Close to fire point E5
            (39.10, -120.95),  # Extra station not near any fire
        ]):
            grid_id = f"W{i+1}"
            data.append({
                'date': day,
                'year': day.year,
                'month': day.month,
                'day_of_year': day.dayofyear,
                'grid_id': grid_id,
                'latitude': lat,
                'longitude': lon,
                'tmax': np.random.uniform(70, 90),  # Random temperature
                'ppt': np.random.uniform(0, 10),    # Random precipitation
                'vbdmax': np.random.uniform(1, 5),  # Random vapor pressure deficit
                'week': day.to_period('W').strftime('%Y-W%W')
            })
    
    return pd.DataFrame(data)

def prepare_fire_data(fire_df):
    """Prepare fire data for merging."""
    print("Preparing fire data...")
    fire_df = fire_df.copy()
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
    fire_df['week'] = fire_df['week'].astype(str)
    fire_df['grid_id'] = fire_df['grid_id'].astype(str)
    
    # Spatial coordinates in radians for indexing
    fire_coords = np.radians(fire_df[['latitude', 'longitude']].values)
    return fire_df, fire_coords

def build_spatial_index(weather_df, fire_df):
    """Build spatial index for nearby weather stations."""
    print("Building spatial index...")
    
    # Filter weather stations within the buffered area around fire data
    weather_stations = weather_df[
        (weather_df['latitude'] >= fire_df['latitude'].min() - BUFFER_DEGREES) &
        (weather_df['latitude'] <= fire_df['latitude'].max() + BUFFER_DEGREES) &
        (weather_df['longitude'] >= fire_df['longitude'].min() - BUFFER_DEGREES) &
        (weather_df['longitude'] <= fire_df['longitude'].max() + BUFFER_DEGREES)
    ][['grid_id', 'latitude', 'longitude']].drop_duplicates('grid_id')
    
    # Convert to radians and build BallTree
    weather_coords = np.radians(weather_stations[['latitude', 'longitude']].values)
    tree = BallTree(weather_coords, metric='haversine')
    
    print(f"Found {len(weather_stations)} unique weather stations in the area")
    return weather_stations, tree

def process_and_merge(fire_df, weather_df, weather_stations, tree):
    """Process the data and perform spatial-temporal merging."""
    print("Performing spatial-temporal merging...")
    
    # Filter fire data for 2022
    year = 2022
    year_fire = fire_df[pd.DatetimeIndex(fire_df['acq_date']).year == year].copy()
    
    # Filter weather data for 2022
    weather_chunk = weather_df[weather_df['year'] == year]
    
    # Spatial matching
    weather_coords = np.radians(weather_chunk[['latitude', 'longitude']].values)
    _, indices = tree.query(weather_coords, k=1)
    weather_chunk['nearest_grid_id'] = weather_stations.iloc[indices.flatten()]['grid_id'].values
    
    # Temporal matching - exact first
    weather_chunk['date'] = pd.to_datetime(weather_chunk['date'])
    merged = pd.merge(
        year_fire,
        weather_chunk,
        left_on=['grid_id', 'acq_date'],
        right_on=['nearest_grid_id', 'date'],
        how='left'
    )
    
    # Fallback to nearest date for unmatched
    unmatched = merged[merged['tmax'].isna()].copy()
    if not unmatched.empty:
        print(f"Falling back to nearest date matching for {len(unmatched)} records")
        fallback_merged = pd.merge_asof(
            unmatched.sort_values('acq_date'),
            weather_chunk.sort_values('date'),
            left_on='acq_date',
            right_on='date',
            by='nearest_grid_id',
            direction='nearest',
            tolerance=TEMPORAL_TOLERANCE
        )
        merged.update(fallback_merged)
    
    print(f"Merged dataset: {merged.shape}")
    return merged

def visualize_data(fire_df, weather_df, merged_df):
    """Visualize the sample data and merging results."""
    print("Creating visualizations...")
    
    # Create the output directory if it doesn't exist
    output_dir = 'data/processed/merged'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot map of fire points and weather stations
    plt.figure(figsize=(12, 10))
    
    # Plot weather stations
    weather_stations = weather_df[['grid_id', 'latitude', 'longitude']].drop_duplicates('grid_id')
    plt.scatter(weather_stations['longitude'], weather_stations['latitude'], 
              c='blue', alpha=0.5, label='Weather Stations')
    
    # Plot fire points
    plt.scatter(fire_df['longitude'], fire_df['latitude'], 
              c='red', marker='*', s=200, label='Fire Events')
    
    # Add labels to fire points
    for idx, row in fire_df.iterrows():
        plt.annotate(row['grid_id'], 
                   (row['longitude'], row['latitude']),
                   xytext=(10, 10),
                   textcoords='offset points')
    
    plt.title('Sample California Wildfires and Weather Stations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'sample_data_map.png'))
    
    # Create a merged weather-fire data visualization
    if not merged_df.empty:
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with temperature as color
        sc = plt.scatter(merged_df['longitude_x'], merged_df['latitude_x'], 
                      c=merged_df['tmax'], cmap='hot', s=200, alpha=0.7)
        
        # Add a colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Maximum Temperature (°F)')
        
        # Add labels to points
        for idx, row in merged_df.iterrows():
            plt.annotate(f"{row['grid_id']} - {row['tmax']:.1f}°F", 
                       (row['longitude_x'], row['latitude_x']),
                       xytext=(10, 10),
                       textcoords='offset points')
        
        plt.title('California Wildfires with Matched Weather Data (Temperature)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'merged_data_temperature.png'))
    
    print(f"Visualizations saved to {output_dir}")

def save_sample_data(fire_df, weather_df, merged_df):
    """Save the sample datasets to CSV files."""
    print("Saving sample datasets...")
    
    # Create the output directory if it doesn't exist
    output_dir = 'data/processed/merged'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fire data
    fire_path = os.path.join(output_dir, 'sample_california_wildfires.csv')
    fire_df.to_csv(fire_path, index=False)
    
    # Save weather data
    weather_path = os.path.join(output_dir, 'sample_weather_master.csv')
    # Only save a subset of weather data to keep file size manageable
    weather_sample = weather_df.drop_duplicates(['date', 'grid_id']).head(100)
    weather_sample.to_csv(weather_path, index=False)
    
    # Save merged data
    if not merged_df.empty:
        merged_path = os.path.join(output_dir, 'sample_merged_fire_weather.csv')
        merged_df.to_csv(merged_path, index=False)
    
    print(f"Sample datasets saved to {output_dir}")
    print(f"  - Fire data: {fire_path}")
    print(f"  - Weather data: {weather_path}")
    print(f"  - Merged data: {os.path.join(output_dir, 'sample_merged_fire_weather.csv')}")

def main():
    """Main function to demonstrate the merging process."""
    print("\n=== Creating Sample Datasets for California Wildfires and Weather ===\n")
    
    # Create sample datasets
    fire_df = create_sample_fire_data()
    weather_df = create_sample_weather_data()
    
    # Prepare fire data for merging
    fire_df, fire_coords = prepare_fire_data(fire_df)
    
    # Build spatial index
    weather_stations, tree = build_spatial_index(weather_df, fire_df)
    
    # Process and merge data
    merged_df = process_and_merge(fire_df, weather_df, weather_stations, tree)
    
    # Save sample datasets
    save_sample_data(fire_df, weather_df, merged_df)
    
    # Create visualizations
    visualize_data(fire_df, weather_df, merged_df)
    
    print("\n=== Sample Data Creation and Merging Complete ===\n")
    print("You can now use these sample datasets to test your code.")

if __name__ == "__main__":
    main()