#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pipeline script for applying weather features to cleaned weather data.

This script loads cleaned weather data from the interim folder,
applies California-specific weather feature engineering techniques,
and saves the enhanced weather datasets to the processed folder.
"""

import os
import sys
import pandas as pd
import glob
from tqdm import tqdm
import yaml

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from wildfire_prediction.data import (
    get_project_paths,
    save_processed_data
)

from wildfire_prediction.features.feature_weather import (
    calculate_hot_dry_index,
    calculate_spi_ca_daily,
    calculate_vpd_extreme_ca,
    add_ca_temporal_features,
    calculate_ca_clusters,
    engineer_ca_features
)


def main():
    """Main function to apply California-specific weather features to cleaned weather data."""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'configs', 'config.yml')
        
        # Load config directly to get the years to process
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        # Get years to process from the configuration
        years_to_process = config.get('weather', {}).get('years_to_process', [])
        if not years_to_process:
            print("Warning: No years specified in config.yml. Will process all available years.")
        else:
            # Convert years to strings for directory matching
            years_to_process = [str(year) for year in years_to_process]
            print(f"Using years from config file: {years_to_process}")
        
        # Get project paths
        paths = get_project_paths(config_path)
        
        # Define the input and output directories
        weather_clean_dir = paths['interim_data']
        weather_processed_dir = paths['processed_data']
        
        # Ensure the output directory exists
        os.makedirs(weather_processed_dir, exist_ok=True)
        
        # Check if the clean weather directory exists
        if not os.path.exists(weather_clean_dir):
            raise FileNotFoundError(f"Cleaned weather data directory not found at: {weather_clean_dir}")
        
        # Get all years in the weather_clean directory 
        available_years = [d for d in os.listdir(weather_clean_dir) 
                if os.path.isdir(os.path.join(weather_clean_dir, d))]
        
        if not available_years:
            raise FileNotFoundError(f"No year folders found in {weather_clean_dir}")
        
        # Filter years based on configuration if specified
        if years_to_process:
            years = [year for year in available_years if year in years_to_process]
            if not years:
                print(f"Warning: None of the specified years {years_to_process} found in {weather_clean_dir}")
                print(f"Available years: {available_years}")
                print("Processing all available years instead.")
                years = available_years
        else:
            years = available_years
        
        print(f"Processing {len(years)} years of California weather data: {years}")
        
        # Get CA-specific feature engineering config if available
        ca_feature_config = config.get('features', {}).get('california', {})
        
        # Process each year's data
        for year in years:
            year_dir = os.path.join(weather_clean_dir, year)
            output_year_dir = os.path.join(weather_processed_dir, year)
            os.makedirs(output_year_dir, exist_ok=True)
            
            # Get all weather data files for this year
            weather_files = glob.glob(os.path.join(year_dir, "*.csv"))
            
            if not weather_files:
                print(f"Warning: No weather data files found for year {year}")
                continue
                
            print(f"Processing {len(weather_files)} California weather files for year {year}...")
            
            # Process each weather file
            for file_path in tqdm(weather_files):
                file_name = os.path.basename(file_path)
                output_file = os.path.join(output_year_dir, f"features_{file_name}")
                
                # Skip if output file already exists (optional)
                if os.path.exists(output_file):
                    continue
                
                # Load the weather data
                df = pd.read_csv(file_path)
                
                # Ensure date column exists and is in datetime format
                if 'date' not in df.columns:
                    # Try to extract date from filename if not in the data
                    date_str = file_name.split('.')[0]  # Assuming filename format like "20130101.csv"
                    try:
                        date = pd.to_datetime(date_str, format='%Y%m%d')
                        df['date'] = date
                    except:
                        print(f"Warning: Could not extract date from filename {file_name}")
                        df['date'] = pd.to_datetime(f"{year}0101")  # Default to January 1st
                else:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Check for required columns
                required_columns = {
                    'hot_dry_index': ['tmax', 'ppt'],
                    'spi': ['ppt'],
                    'vpd_extreme': ['vbdmax'],
                    'temporal': ['date'],
                    'spatial_clusters': ['longitude', 'latitude']
                }
                
                missing_columns = []
                for feature, columns in required_columns.items():
                    missing = [col for col in columns if col not in df.columns]
                    if missing:
                        missing_columns.extend(missing)
                
                if missing_columns:
                    missing_set = set(missing_columns)
                    if len(missing_set) > 3:  # If too many missing columns, skip this file
                        print(f"Skipping {file_name}: Too many required columns missing: {missing_set}")
                        continue
                    else:
                        print(f"Warning: Some columns missing in {file_name}: {missing_set}")
                
                # Apply California-specific weather feature engineering
                enhanced_df = engineer_ca_features(df, config=ca_feature_config)
                
                # Show what features were added
                new_features = [col for col in enhanced_df.columns if col not in df.columns]
                if new_features:
                    print(f"Added California weather features to {file_name}: {new_features}")
                
                # Save the enhanced dataset
                enhanced_df.to_csv(output_file, index=False)
            
            print(f"Completed processing California weather features for year {year}")
        
        print("Weather feature engineering completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())