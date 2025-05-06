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
    transform_numerical_features,
    drop_unnecessary_columns,
    drop_nonzero_types
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
        # Drop unnecessary columns
        print("Dropping unnecessary columns...")
        columns_to_drop = [
            'confidence','acq_time', 'bright_t31', 'frp', 'daynight', 'type','brightness','scan','track',
            'satellite','instrument','version'
        ]
        df = drop_unnecessary_columns(df,columns_to_drop)
        
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