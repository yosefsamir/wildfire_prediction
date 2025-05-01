"""Feature engineering module for wildfire prediction.

This module contains functions for creating and transforming features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from scipy import stats
import utm


def lat_lon_to_utm_grid(lat, lon, grid_size_km=1):
    """Convert latitude and longitude to UTM grid ID.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        grid_size_km: Size of the grid in kilometers
        
    Returns:
        str: Grid ID in the format "zone_easting_northing"
    """
    easting, northing, zone_num, zone_letter = utm.from_latlon(lat, lon)
    grid_easting = int(easting // (grid_size_km * 1000))  # Convert to km grid
    grid_northing = int(northing // (grid_size_km * 1000))
    return f"{zone_num}{zone_letter}_{grid_easting}_{grid_northing}"


def create_grid_and_time_features(gdf, grid_size_km=1):
    """Create grid ID and time-based features.
    
    Args:
        gdf: GeoDataFrame with latitude and longitude columns
        grid_size_km: Size of the grid in kilometers
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with added grid and time features
    """
    print(f"Before creating grid and time features: {len(gdf)} rows")
    
    # Create grid ID for each point
    gdf['grid_id'] = gdf.apply(
        lambda row: lat_lon_to_utm_grid(row['latitude'], row['longitude'], grid_size_km), axis=1
    )
    
    # Convert acquisition date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(gdf['acq_date']):
        gdf['acq_date'] = pd.to_datetime(gdf['acq_date'])
    
    # Create time-based features
    gdf['week'] = gdf['acq_date'].dt.to_period('W')
    gdf['month'] = gdf['acq_date'].dt.month
    gdf['year'] = gdf['acq_date'].dt.year
    gdf['day_of_year'] = gdf['acq_date'].dt.dayofyear
    
    print(f"After creating grid and time features: {len(gdf)} rows")
    return gdf


def encode_categorical_features(df, drop_low_confidence=True):
    """Encode categorical features for modeling.
    
    Args:
        df: DataFrame with categorical features
        drop_low_confidence: Whether to drop low confidence values
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical features
    """
    print(f"Before encoding categorical features: {len(df)} rows")
    
    # Encode confidence levels (ordinal encoding)
    if 'confidence' in df.columns:
        # Filter out low confidence values if requested
        if drop_low_confidence:
            print(f"Before filtering low confidence: {len(df)} rows")
            df = df[df['confidence'] != 'l']
            print(f"After filtering low confidence: {len(df)} rows")
        
        # Filter out any remaining invalid confidence values
        valid_confidence = df['confidence'].isin(['n', 'h']) if drop_low_confidence else df['confidence'].isin(['l', 'n', 'h'])
        if not valid_confidence.all():
            print(f"Before removing invalid confidence values: {len(df)} rows")
            print(f"Warning: Found {(~valid_confidence).sum()} invalid confidence values")
            df = df[valid_confidence]
            print(f"After removing invalid confidence values: {len(df)} rows")
        
        # Define confidence order based on whether we're keeping low confidence values
        confidence_order = ['n', 'h'] if drop_low_confidence else ['l', 'n', 'h']
        encoder = OrdinalEncoder(categories=[confidence_order])
        df['confidence_encoded'] = encoder.fit_transform(df[['confidence']])
    
    # One-hot encode day/night
    if 'daynight' in df.columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(df[['daynight']])
        df['is_day'] = encoded_data[:, 0]  # Only need one column since we dropped the first
    
    print(f"After encoding categorical features: {len(df)} rows")
    return df


def transform_numerical_features(df, log_transform_frp=True, normalize_brightness=True):
    """Apply transformations to numerical features.
    
    Args:
        df: DataFrame with numerical features
        log_transform_frp: Whether to apply log transformation to FRP
        normalize_brightness: Whether to normalize brightness values
        
    Returns:
        pd.DataFrame: DataFrame with transformed numerical features
    """
    print(f"Before transforming numerical features: {len(df)} rows")
    
    # Log transform FRP (Fire Radiative Power)
    if 'frp' in df.columns and log_transform_frp:
        df['frp_log'] = np.log1p(df['frp'])  # log(x+1) to handle zeros
        
        # Calculate and print skewness
        original_skew = stats.skew(df['frp'].dropna())
        transformed_skew = stats.skew(df['frp_log'].dropna())
        print(f"FRP Skewness - Original: {original_skew:.4f}, Log-transformed: {transformed_skew:.4f}")
    
    # Normalize brightness
    if 'brightness' in df.columns and normalize_brightness:
        # Use a custom range for MinMaxScaler to avoid zeros
        scaler = MinMaxScaler(feature_range=(0.1, 1.0))
        df['brightness_normalized'] = scaler.fit_transform(df[['brightness']])
        
        # Check if we still have zeros
        zero_count = (df['brightness_normalized'] == 0).sum()
        if zero_count > 0:
            print(f"Warning: Still found {zero_count} zeros in brightness_normalized")
    
    print(f"After transforming numerical features: {len(df)} rows")
    return df


def drop_unnecessary_columns(df, columns_to_drop=None):
    """Remove columns that aren't needed for analysis.
    
    Args:
        df: DataFrame to process
        columns_to_drop: List of columns to drop, or None to use defaults
        
    Returns:
        pd.DataFrame: DataFrame with unnecessary columns removed
    """
    print(f"Before dropping unnecessary columns: {len(df)} rows")
    
    if columns_to_drop is None:
        columns_to_drop = ['scan', 'track', 'version', 'satellite', 'instrument', 'bright_t31']
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        
    print(f"After dropping unnecessary columns: {len(df)} rows")
    return df