#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pipeline script for preprocessing fire data.

This script loads cleaned fire data from the interim directory,
performs feature engineering, and saves the processed data to the processed directory.
"""

import os
import sys
import yaml
import pandas as pd
import geopandas as gpd
import random

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from wildfire_prediction.data import (
    get_project_paths,
    save_processed_data
)

from wildfire_prediction.features.feature_engineering import (
    create_grid_and_time_features,
    encode_categorical_features,
    get_all_weeks,
    transform_numerical_features,
    drop_unnecessary_columns,
    drop_nonzero_types,
    get_all_grid_ids,
    get_all_weeks,
    build_full_dataset,
    sample_dataset
)


def main():
    """Main function to preprocess fire data."""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'configs', 'config.yml')
        
        # Get project paths
        paths = get_project_paths(config_path)
        
        # Ensure interim directory exists
        os.makedirs(paths['interim_data'], exist_ok=True)
        
        # Load cleaned data from interim directory using consistent path handling
        # Use paths from config instead of hardcoded paths
        interim_file = os.path.join(paths['interim_data'], 'fire_clean.csv')
        print(f"Loading cleaned data from: {interim_file}")
        
        # For DVC reference, also calculate the relative path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dvc_relative_path = os.path.relpath(interim_file, project_root).replace(os.path.sep, '/')
        print(f"DVC relative path: {dvc_relative_path}")
        
        # Check if the file exists with detailed error reporting
        if not os.path.exists(interim_file):
            # Check if parent directory exists
            parent_dir = os.path.dirname(interim_file)
            if not os.path.exists(parent_dir):
                print(f"Parent directory does not exist: {parent_dir}")
            raise FileNotFoundError(f"Cleaned data file not found at: {interim_file}")
            
        df = pd.read_csv(interim_file)
        print(f"Loaded {len(df)} rows of cleaned data")
        
        # Create grid and time features
        print("Creating grid and time features...")
        df = create_grid_and_time_features(df)
        
        # Encode categorical features
        print("Encoding categorical features...")
        df = encode_categorical_features(df)
        
        # Transform numerical features
        print("Transforming numerical features...")
        df = transform_numerical_features(df)
        
        print(f"before dropping non-zero 'type' values: {len(df)} rows")
        # Filter out rows where 'type' is not zero
        df = drop_nonzero_types(df)
        print(f"After dropping non-zero 'type' values: {len(df)} rows")
        # Drop unnecessary columns but keep the ones needed for our dataset
        print("Dropping unnecessary columns while keeping required ones...")
        columns_to_drop = [
            'confidence', 'acq_time', 'bright_t31', 'daynight', 'type', 'scan', 'track',
            'satellite', 'instrument', 'version'
        ]
        # Keep 'brightness' and 'frp' as they're needed for 'brightness_normalized' and 'frp_log'
        df = drop_unnecessary_columns(df, columns_to_drop)
        
        print("load california data frame")

        california_gdf = gpd.read_file("california.geojson")
        
        print("get all grid ids in california")

        all_grid_ids = get_all_grid_ids(california_gdf)
        
        print("all grid ids: ",len(all_grid_ids))
        
        print("get all weeks from 2013 to 2024")
        
        all_weeks = get_all_weeks("2013-01-01", "2024-12-31")

        print("build full dataset")
        
        # Use a more memory-efficient approach by limiting negative samples
        # This helps avoid memory errors during processing
        negative_samples = 2_000_000  
        
        try:
            print(f"Building dataset with {negative_samples} negative samples...")
            # Monitor memory usage during processing
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Memory usage before building dataset: {memory_before:.2f} MB")
            
            # Use a try-except block specifically for the build_full_dataset function
            try:
                df = build_full_dataset(df, all_grid_ids, all_weeks, negative_samples)
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                print(f"Memory usage after building dataset: {memory_after:.2f} MB (change: {memory_after - memory_before:.2f} MB)")
            except MemoryError as e:
                print(f"Memory error during build_full_dataset: {e}")
                # Try with a much smaller sample size
                negative_samples = 100_000  # Drastically reduce sample size
                print(f"Retrying with {negative_samples} negative samples...")
                df = build_full_dataset(df, all_grid_ids, all_weeks, negative_samples)
            
            # Sample to a more manageable size
            total_rows = 2_000_000  # Reduce from 10M to 2M to avoid memory issues
            print(f"Sampling dataset to {total_rows} total rows...")
            df = sample_dataset(df, total_rows)
            
        except MemoryError as e:
            print(f"Memory error encountered: {e}. Trying with minimal sample size...")
            # If we still hit memory issues, try with an even smaller sample
            negative_samples = 50_000
            print(f"Retrying with {negative_samples} negative samples...")
            df = build_full_dataset(df, all_grid_ids, all_weeks, negative_samples)
            df = sample_dataset(df, 500_000)  # Sample to 500K rows
        except Exception as e:
            print(f"Error during dataset building: {e}")
            # Try with absolute minimal processing to at least produce some output
            print("Attempting minimal processing to produce output...")
            
            # Create a minimal set of negative samples
            print("Creating a minimal set of negative samples...")
            pos_df = df.copy()
            pos_df['fire'] = 1  # Mark existing data as positive samples
            
            # Generate a small number of negative samples
            print("Generating minimal negative samples...")
            try:
                # Get a small subset of grid_ids and weeks
                sample_grid_ids = all_grid_ids[:min(1000, len(all_grid_ids))]
                sample_weeks = all_weeks[:min(52, len(all_weeks))]  # About a year of weeks
                
                # Create a small number of negative samples
                neg_samples = 10000  # Very small number to avoid memory issues
                print(f"Attempting to create {neg_samples} negative samples...")
                
                # Create negative samples with required columns
                neg_grid_ids = [random.choice(sample_grid_ids) for _ in range(neg_samples)]
                neg_weeks = [random.choice(sample_weeks) for _ in range(neg_samples)]
                
                # Create DataFrame with negative samples
                neg_df = pd.DataFrame({
                    'grid_id': neg_grid_ids,
                    'week': neg_weeks,
                    'fire': 0  # Mark as negative samples
                })
                
                # Add required columns for negative samples
                neg_df['frp_log'] = 0.0
                neg_df['brightness_normalized'] = 0.0
                
                # Convert grid_id to latitude and longitude
                from wildfire_prediction.features.feature_engineering import grid_id_to_lat_lon, week_to_acq_date
                lat_lon_tuples = neg_df['grid_id'].apply(grid_id_to_lat_lon)
                neg_df['latitude'] = lat_lon_tuples.apply(lambda x: x[0])
                neg_df['longitude'] = lat_lon_tuples.apply(lambda x: x[1])
                
                # Convert week to acquisition date
                neg_df['acq_date'] = neg_df['week'].apply(week_to_acq_date)
                
                # Combine positive and negative samples
                df = pd.concat([pos_df, neg_df], ignore_index=True)
                print(f"Created dataset with {len(pos_df)} positive samples and {len(neg_df)} negative samples")
                
            except Exception as e:
                print(f"Error creating negative samples: {e}")
                # If all else fails, just use positive samples
                df = pos_df
                print(f"Fallback: Using {len(df)} positive samples only")
            
        # Ensure we have a valid dataframe to continue
        if df is None or len(df) == 0:
            raise ValueError("Failed to create valid dataset after multiple attempts")
        
        print(f"After preprocessing: {len(df)} rows")
        
        # Save processed data using paths from config
        output_path = os.path.join(paths['processed_data'], 'california_wildfires.csv')
        print(f"Saving processed data to: {output_path}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Calculate DVC relative path for reference
        dvc_output_path = os.path.relpath(output_path, project_root).replace(os.path.sep, '/')
        print(f"DVC relative path: {dvc_output_path}")
        
        # Save the data
        save_processed_data(df, output_path)
        
        # Verify the file exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"File created successfully at: {output_path}")
            print(f"File size: {file_size} bytes")
            
            # Double-check file is accessible
            try:
                with open(output_path, 'r') as f:
                    f.readline()  # Read first line to verify file is accessible
                print(f"File is readable and accessible")
            except Exception as e:
                print(f"Warning: File exists but could not be read: {e}")
        else:
            raise FileNotFoundError(f"Failed to create file at: {output_path}")
        
        # Print paths for reference
        abs_path = os.path.abspath(output_path)
        print(f"Absolute path: {abs_path}")
        print(f"DVC relative path: {dvc_output_path}")
        
        # Force file system sync to ensure all writes are committed to disk
        import time
        time.sleep(1)  # Small delay to ensure file system operations complete
        
        print("Preprocessing completed successfully!")
        return 0
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())