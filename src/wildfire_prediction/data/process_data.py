"""Data processing module for wildfire prediction.

This module contains functions for loading, cleaning, and processing wildfire data.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import yaml


def get_project_paths(config_path=None):
    """Set up project paths and verify file existence.
    
    Args:
        config_path: Path to the config file. If None, uses default config.
        
    Returns:
        dict: Dictionary of project paths
    """
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to project root (src/wildfire_prediction/data -> src/wildfire_prediction -> src -> project_root)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # If config file is provided, load paths from it
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            paths = {
                'project_root': project_root,
                'raw_data': os.path.join(project_root, config['paths']['raw_data']),
                'external_data': os.path.join(project_root, config['paths']['external_data']),
                'interim_data': os.path.join(project_root, config['paths']['interim_data']),
                'processed_data': os.path.join(project_root, config['paths']['processed_data']),
                'figures': os.path.join(project_root, config['paths']['figures'])
            }
            
            # Set file paths
            paths['fire_data'] = os.path.join(paths['raw_data'], config['files']['fire_data'])
            paths['ca_boundary'] = os.path.join(paths['raw_data'], config['files']['ca_boundary'])
            paths['processed_fire_data'] = os.path.join(paths['processed_data'], config['files']['processed_fire_data'])
    else:
        # Default paths if no config is provided
        paths = {
            'project_root': project_root,
            'raw_data': os.path.join(project_root, 'data', 'raw'),
            'external_data': os.path.join(project_root, 'data', 'external'),
            'interim_data': os.path.join(project_root, 'data', 'interim'),
            'processed_data': os.path.join(project_root, 'data', 'processed'),
            'figures': os.path.join(project_root, 'artifacts', 'figures')
        }
        
        # Set file paths
        paths['fire_data'] = os.path.join(paths['raw_data'], 'fire_archive_SV-C2_607788.csv')
        paths['ca_boundary'] = os.path.join(paths['raw_data'], 'cb_2023_us_state_20m.zip')
        paths['processed_fire_data'] = os.path.join(paths['processed_data'], 'california_wildfires.csv')
    
    # Ensure directories exist
    for path in paths.values():
        if isinstance(path, str) and not path.endswith('.csv') and not path.endswith('.zip'):
            os.makedirs(path, exist_ok=True)
    
    # Verify file existence
    if not os.path.exists(paths['fire_data']):
        raise FileNotFoundError(f"Fire data file not found at: {paths['fire_data']}")
    if not os.path.exists(paths['ca_boundary']):
        raise FileNotFoundError(f"California boundary file not found at: {paths['ca_boundary']}")
    
    print(f"Project root directory: {paths['project_root']}")
    print(f"Fire data file exists: {os.path.exists(paths['fire_data'])}")
    
    return paths


def load_and_clean_data(data_input):
    """Load and clean fire data from either path or DataFrame."""
    try:
        if isinstance(data_input, str):
            # Load from file path
            dtype = {
                'latitude': 'float32',
                'longitude': 'float32',
                'brightness': 'float32',
                'frp': 'float32'
            }
            df = pd.read_csv(data_input, dtype=dtype)
        else:
            # Use directly if input is DataFrame
            df = data_input.copy()
        
        initial_rows = len(df)
        print(f"Processing {initial_rows} rows")
        
        # Handle missing values
        print(f"Before dropping null values: {len(df)} rows")
        df = df.dropna()
        print(f"After dropping null values: {len(df)} rows")
        
        # Remove duplicates
        print(f"Before removing duplicates: {len(df)} rows")
        df = df.drop_duplicates()
        print(f"After removing duplicates: {len(df)} rows")
        
        return df
    except Exception as e:
        raise Exception(f"Error processing data: {e}")


def validate_coordinates(df):
    """Validate coordinate columns exist and contain valid data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        pd.DataFrame: Validated DataFrame
    """
    required_cols = ['longitude', 'latitude']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    print(f"Before coordinate validation: {len(df)} rows")
    if df[required_cols].isnull().any().any():
        print("Warning: Found null values in coordinates")
        df = df.dropna(subset=required_cols)
        print(f"After coordinate validation: {len(df)} rows")
    else:
        print("No null values found in coordinates")
    
    return df


def convert_to_geodataframe(df):
    """Convert DataFrame to GeoDataFrame with Point geometries.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Point geometries
    """
    try:
        print(f"Before converting to GeoDataFrame: {len(df)} rows")
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84 coordinate system
        print(f"After converting to GeoDataFrame: {len(gdf)} rows")
        return gdf
    except Exception as e:
        raise Exception(f"Error creating GeoDataFrame: {e}")


def load_california_boundary(boundary_path):
    """Load California boundary from shapefile.
    
    Args:
        boundary_path: Path to the boundary shapefile
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with California boundary
    """
    try:
        states = gpd.read_file(boundary_path)
        california = states[states.STATEFP == "06"]  # '06' is California's FIPS code
        
        if california.empty:
            raise ValueError("California boundary is empty")
            
        # Save California boundary for potential reuse
        california.to_file("california.geojson", driver="GeoJSON")
        return california
    except Exception as e:
        raise Exception(f"Error loading California boundary: {e}")


def filter_california_data(gdf, california):
    """Filter points to only include those within California.
    
    Args:
        gdf: GeoDataFrame with fire points
        california: GeoDataFrame with California boundary
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with only points in California
    """
    try:
        print(f"Before filtering to California: {len(gdf)} rows")
        # Replace union_all() with unary_union which is available in older versions
        california_data = gdf[gdf.within(california.geometry.unary_union)]
        print(f"After filtering to California: {len(california_data)} rows")
        return california_data
    except Exception as e:
        raise Exception(f"Error filtering points: {e}")


def save_processed_data(df, output_path):
    """Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the DataFrame
        
    Returns:
        bool: True if successful, raises exception otherwise
    """
    try:
        # Convert to absolute path and normalize path separators for Windows
        output_path = os.path.abspath(os.path.normpath(output_path))
        output_dir = os.path.dirname(output_path)
        
        print(f"Saving to directory: {output_dir} (exists: {os.path.exists(output_dir)})")
        print(f"Full output path: {output_path}")
        
        # Ensure directory exists with multiple attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Create all parent directories if they don't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Verify directory was created successfully
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    print(f"Directory created/verified: {output_dir} (exists: True)")
                    # Check if directory is writable
                    if os.access(output_dir, os.W_OK):
                        print(f"Directory is writable: {output_dir}")
                    else:
                        print(f"WARNING: Directory is not writable: {output_dir}")
                    break
                else:
                    raise FileNotFoundError(f"Failed to create directory: {output_dir}")
            except Exception as dir_err:
                if attempt < max_attempts - 1:
                    print(f"Attempt {attempt+1}/{max_attempts} to create directory failed: {str(dir_err)}")
                    import time
                    time.sleep(1)  # Wait before retrying
                else:
                    raise
        
        # Save data with retry mechanism
        print(f"DataFrame shape before saving: {df.shape}")
        max_save_attempts = 3
        for save_attempt in range(max_save_attempts):
            try:
                # Flush any pending file operations before saving
                import gc
                gc.collect()
                
                # Create a temporary filename in the same directory
                temp_path = f"{output_path}.temp"
                
                # Save to temporary file first
                print(f"Saving to temporary file: {temp_path}")
                df.to_csv(temp_path, index=False)
                
                # Force sync to disk
                import gc
                gc.collect()
                
                # Verify temp file was created
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    # Rename temp file to final filename (atomic operation on most file systems)
                    print(f"Renaming {temp_path} to {output_path}")
                    if os.path.exists(output_path):
                        os.remove(output_path)  # Remove existing file if it exists
                    os.rename(temp_path, output_path)
                else:
                    raise FileNotFoundError(f"Temporary file was not created or is empty: {temp_path}")
                
                # Wait a moment to ensure file is fully written
                import time
                time.sleep(0.5)
                
                # Verify final file was created
                file_exists = os.path.exists(output_path)
                file_size = os.path.getsize(output_path) if file_exists else 0
                print(f"Processed data saved to: {output_path} (exists: {file_exists}, size: {file_size} bytes)")
                
                if not file_exists or file_size == 0:
                    raise FileNotFoundError(f"File was not created or is empty: {output_path}")
                
                # Ensure file is properly closed and synced to disk
                gc.collect()
                
                return True
            except Exception as save_err:
                if save_attempt < max_save_attempts - 1:
                    print(f"Attempt {save_attempt+1}/{max_save_attempts} to save file failed: {str(save_err)}")
                    import time
                    time.sleep(1)  # Wait before retrying
                else:
                    raise
    except Exception as e:
        print(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise Exception(f"Error saving processed data: {e}")