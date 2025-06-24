#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pipeline script for cleaning fire data.

This script loads raw fire data and California boundary data,
performs initial cleaning, and saves the cleaned data to the interim directory.
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
    load_and_clean_data,
    validate_coordinates,
    convert_to_geodataframe,
    load_california_boundary,
    filter_california_data,
    save_processed_data
)


def main():
    """Main function to clean fire data."""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'configs', 'config.yml')
        
        # Get project paths
        paths = get_project_paths(config_path)
        
        # Ensure interim directory exists with absolute path
        interim_dir = os.path.abspath(os.path.normpath(paths['interim_data']))
        os.makedirs(interim_dir, exist_ok=True)
        print(f"Interim directory: {interim_dir} (exists: {os.path.exists(interim_dir)})")
        
        # Process in chunks with memory optimization
        chunksize = 1000000
        cleaned_chunks = []
        
        for chunk in pd.read_csv(paths['fire_data'], chunksize=chunksize, 
                            dtype={'latitude': 'float32', 'longitude': 'float32'}):
            print(f"Processing chunk of {len(chunk)} rows...")
            
            # Clean and validate
            df = load_and_clean_data(chunk)
            df = validate_coordinates(df)
            
            # Convert to GeoDataFrame and filter
            gdf = convert_to_geodataframe(df)
            california = load_california_boundary(paths['ca_boundary'])
            california_data = filter_california_data(gdf, california)
            
            # Clear memory
            del df, gdf
            cleaned_chunks.append(california_data)
        
        # Combine and save final data
        final_df = pd.concat(cleaned_chunks)
        
        # Ensure the interim directory exists and is accessible
        if not os.path.exists(interim_dir):
            print(f"Creating interim directory: {interim_dir}")
            os.makedirs(interim_dir, exist_ok=True)
        
        # Double-check directory exists before proceeding
        if not os.path.exists(interim_dir):
            raise FileNotFoundError(f"Failed to create or access interim directory: {interim_dir}")
            
        # Create output path using config paths
        output_file = 'fire_clean.csv'
        output_path = os.path.join(paths['interim_data'], output_file)
        print(f"Saving cleaned data to: {output_path}")
        
        # Calculate DVC relative path for reference
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dvc_relative_path = os.path.relpath(output_path, project_root).replace(os.path.sep, '/')
        print(f"DVC will track: {dvc_relative_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir} (exists: {os.path.exists(output_dir)})")
        
        # Use the improved save_processed_data function with built-in retry mechanism
        # This will raise an exception if it fails, which DVC will properly detect
        save_processed_data(final_df, output_path)
        
        # Wait a moment to ensure file system operations complete
        import time
        time.sleep(1)
        
        # Verify file was created one final time with multiple attempts
        max_verify_attempts = 3
        for verify_attempt in range(max_verify_attempts):
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"File successfully created: {output_path} (size: {os.path.getsize(output_path)} bytes)")
                # Print the absolute and relative paths for DVC reference
                abs_path = os.path.abspath(output_path)
                rel_path = os.path.relpath(output_path, project_root)
                print(f"Absolute path: {abs_path}")
                print(f"Relative path from project root: {rel_path}")
                print(f"DVC expected path: {dvc_relative_path}")
                break
            else:
                print(f"Verification attempt {verify_attempt+1}/{max_verify_attempts}: File not found or empty")
                # Check directory existence and permissions
                dir_path = os.path.dirname(output_path)
                print(f"Directory exists: {os.path.exists(dir_path)}, Is writable: {os.access(dir_path, os.W_OK)}")
                if verify_attempt < max_verify_attempts - 1:
                    time.sleep(1)  # Wait before retrying
                else:
                    raise FileNotFoundError(f"File does not exist or is empty after save operation: {output_path}")
        
        # Force file handle closure
        import gc
        gc.collect()
        
        return 0
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
