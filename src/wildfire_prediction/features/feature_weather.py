"""Weather feature engineering for wildfire prediction.

This module contains functions for creating and transforming weather features 
specifically for wildfire prediction in California.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler , StandardScaler
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_hot_dry_index(df, tmax_col='tmax', ppt_col='ppt', 
                            lon_col='longitude', lat_col='latitude',
                            threshold_temp=25.0, threshold_ppt=1.0):
    """
    Calculate a hot-dry index for wildfire risk assessment per location.
    High temperatures combined with low precipitation increase wildfire risk.

    Args:
        df: DataFrame with temperature, precipitation, longitude, and latitude data.
        tmax_col: Column name for maximum temperature.
        ppt_col: Column name for precipitation.
        lon_col: Column name for longitude.
        lat_col: Column name for latitude.
        threshold_temp: Temperature threshold (C) for high risk.
        threshold_ppt: Precipitation threshold (mm) for low rainfall.

    Returns:
        DataFrame with added hot_dry_index, high_temp_day, low_rain_day, and hot_dry_day columns.
    """
    result_df = df.copy()
    required_cols = {tmax_col, ppt_col, lon_col, lat_col}
    if not required_cols.issubset(df.columns):
        logger.warning("Required columns missing. Cannot calculate hot_dry_index")
        result_df['hot_dry_index'] = 0.0
        return result_df

    # Vectorized normalization across groups
    grouped = result_df.groupby([lon_col, lat_col])
    t_min = grouped[tmax_col].transform('min')
    t_max = grouped[tmax_col].transform('max')
    p_min = grouped[ppt_col].transform('min')
    p_max = grouped[ppt_col].transform('max')

    t_norm = (result_df[tmax_col] - t_min) / (t_max - t_min + 1e-8)
    p_norm = (result_df[ppt_col] - p_min) / (p_max - p_min + 1e-8)
    p_norm_inv = 1 - p_norm

    result_df['hot_dry_index'] = t_norm + p_norm_inv
    result_df['high_temp_day'] = (result_df[tmax_col] > threshold_temp).astype(int)
    result_df['low_rain_day'] = (result_df[ppt_col] < threshold_ppt).astype(int)
    result_df['hot_dry_day'] = (result_df['high_temp_day'] & result_df['low_rain_day']).astype(int)

    return result_df


def calculate_spi_ca_daily(df, ppt_col='ppt', lookback_days=7, min_periods=3,
                           lat_col='latitude', lon_col='longitude'):
    """
    Calculate Standardized Precipitation Index (SPI) for California weather data 
    using a 7-day rolling window per (longitude, latitude) location.

    Args:
        df: DataFrame with precipitation and location data
        ppt_col: Column name for precipitation
        lookback_days: Number of days for rolling window (default: 7)
        min_periods: Minimum number of periods required for calculation
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        
    Returns:
        DataFrame with added SPI and drought category columns
    """
    # Check required columns
    required_cols = {ppt_col, 'date', lat_col, lon_col}
    if not required_cols.issubset(df.columns):
        logger.warning("Required columns missing. Cannot calculate SPI")
        df['spi_7day'] = 0.0
        df['drought_category'] = 0
        return df

    # Convert date to datetime and sort within groups
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', lon_col, lat_col])

    # Group once and perform all rolling operations
    grouped = df.groupby([lon_col, lat_col])[ppt_col]
    
    # Efficiently calculate rolling sum, mean, and std in one pass
    df['rolling_7d_sum'] = grouped.rolling(window=lookback_days, min_periods=min_periods).sum().reset_index(drop=True)
    df['rolling_7d_mean'] = grouped.rolling(window=lookback_days, min_periods=min_periods).mean().reset_index(drop=True)
    df['rolling_7d_std'] = grouped.rolling(window=lookback_days, min_periods=min_periods).std().reset_index(drop=True)

    # Calculate SPI (Standardized Precipitation Index)
    df['spi_7day'] = ((df['rolling_7d_sum'] - df['rolling_7d_mean']) / df['rolling_7d_std'].replace(0, np.nan)).fillna(0)

    # Remove temporary columns
    df.drop(columns=['rolling_7d_sum', 'rolling_7d_mean', 'rolling_7d_std'], inplace=True)

    # Categorize into drought levels
    df['drought_category'] = pd.cut(
        df['spi_7day'],
        bins=[-float('inf'), -2, -1.5, -1, -0.5, 0.5, float('inf')],
        labels=[5, 4, 3, 2, 1, 0],
        include_lowest=True
    ).astype(int)

    return df


def calculate_vpd_extreme_ca(df, vpd_col='vbdmax', percentile_threshold=90, 
                             lat_col='latitude', lon_col='longitude'):
    """
    Calculate extreme vapor pressure deficit (VPD) indicators for California per (longitude, latitude).
    High VPD is associated with increased wildfire risk.

    Args:
        df: DataFrame with VPD and location data
        vpd_col: Column name for VPD
        percentile_threshold: Percentile threshold for extreme values
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        
    Returns:
        DataFrame with added VPD anomaly, extreme indicator, and risk category columns
    """
    if not {vpd_col, lat_col, lon_col}.issubset(df.columns):
        logger.warning("Required columns missing. Cannot calculate VPD extremes")
        df['vpd_anomaly'] = 0.0
        df['vpd_extreme'] = 0
        df['vpd_risk_category'] = 1
        return df

    # Perform a single groupby call
    grouped = df.groupby([lon_col, lat_col])[vpd_col]
    agg_results = grouped.agg(['mean', 'std', lambda x: np.nanpercentile(x, percentile_threshold)])
    agg_results.columns = ['vpd_mean', 'vpd_std', 'vpd_threshold']

    # Merge with the original DataFrame
    df = df.merge(agg_results, on=[lon_col, lat_col], how='left')

    # Calculate the new columns
    df['vpd_extreme'] = (df[vpd_col] > df['vpd_threshold']).astype(int)
    df['vpd_anomaly'] = ((df[vpd_col] - df['vpd_mean']) / df['vpd_std']).fillna(0)

    # Using rank directly without transform
    df['vpd_risk_category'] = grouped.rank(method='first', pct=True)
    df['vpd_risk_category'] = pd.qcut(df['vpd_risk_category'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Drop the extra columns
    df.drop(columns=['vpd_mean', 'vpd_std', 'vpd_threshold'], inplace=True)

    return df

def add_ca_temporal_features(df):
    """
    Add California-specific temporal features for wildfire prediction.
    
    Args:
        df: DataFrame with date column
        
    Returns:
        DataFrame with added temporal features
    """
    result_df = df.copy()
    if 'date' not in df.columns:
        logger.warning("Date column missing. Cannot add temporal features")
        result_df['is_fire_season'] = 0
        return result_df

    result_df['date'] = pd.to_datetime(result_df['date'])
    month = result_df['date'].dt.month
    day_of_year = result_df['date'].dt.dayofyear

    result_df['month'] = month
    result_df['day_of_year'] = day_of_year
    result_df['is_fire_season'] = month.between(6, 10).astype(int)
    result_df['is_santa_ana_season'] = ((month >= 10) | (month <= 4)).astype(int)

    result_df['season'] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [0, 1, 2, 3], default=0
    )

    week_in_year = day_of_year // 7
    result_df['week_sin'] = np.sin(2 * np.pi * week_in_year / 52)
    result_df['week_cos'] = np.cos(2 * np.pi * week_in_year / 52)
    result_df['month_sin'] = np.sin(2 * np.pi * month / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * month / 12)
    result_df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    result_df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    return result_df

def calculate_compound_indices(df, lat_col='latitude', lon_col='longitude'):
    """
    Calculate compound indices from multiple variables for wildfire risk assessment, 
    grouped by (latitude, longitude).

    Args:
        df: DataFrame with processed weather features
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        
    Returns:
        DataFrame with added compound indices
    """
    # Check required columns
    result_df = df.copy()
    required_cols = {lat_col, lon_col}
    if not required_cols.issubset(df.columns):
        logger.warning("Required location columns are missing. Cannot group by region.")
        result_df[['fire_weather_index', 'drought_weather_index', 'fire_risk_index']] = 1
        return result_df

    has_hot_dry = 'hot_dry_index' in df.columns
    has_vpd = 'vpd_risk_category' in df.columns
    has_spi = 'drought_category' in df.columns
    has_wind = 'wind_risk' in df.columns

    if has_hot_dry:
        grouped = result_df.groupby([lon_col, lat_col])
        hot_dry_min = grouped['hot_dry_index'].transform('min')
        hot_dry_max = grouped['hot_dry_index'].transform('max')
        hot_dry_norm = 5 * (result_df['hot_dry_index'] - hot_dry_min) / (hot_dry_max - hot_dry_min + 1e-8)
    else:
        hot_dry_norm = 0

    result_df['fire_weather_index'] = 1
    if has_hot_dry and has_vpd:
        result_df['fire_weather_index'] = 0.5 * hot_dry_norm + 0.5 * result_df['vpd_risk_category']
    elif has_hot_dry:
        result_df['fire_weather_index'] = hot_dry_norm
    elif has_vpd:
        result_df['fire_weather_index'] = result_df['vpd_risk_category']

    result_df['drought_weather_index'] = 1
    if has_spi and has_hot_dry:
        result_df['drought_weather_index'] = 0.7 * result_df['drought_category'] + 0.3 * hot_dry_norm
    elif has_spi:
        result_df['drought_weather_index'] = result_df['drought_category']
    elif has_hot_dry:
        result_df['drought_weather_index'] = hot_dry_norm

    result_df['fire_risk_index'] = result_df['fire_weather_index']
    if has_hot_dry and has_vpd and has_spi:
        if has_wind:
            result_df['fire_risk_index'] = (0.25 * hot_dry_norm + 0.25 * result_df['vpd_risk_category'] +
                                           0.25 * result_df['drought_category'] + 0.25 * result_df['wind_risk'])
        else:
            result_df['fire_risk_index'] = (0.33 * hot_dry_norm + 0.33 * result_df['vpd_risk_category'] +
                                           0.34 * result_df['drought_category'])

    return result_df

def calculate_rolling_means(df, tmax_col='tmax', ppt_col='ppt', vpd_col='vbdmax',
                            lat_col='latitude', lon_col='longitude', 
                            window=7, min_periods=3):
    """
    Calculate rolling means for key weather variables to capture short-term trends,
    grouped by (longitude, latitude).

    Args:
        df: DataFrame with weather data
        tmax_col: Column name for maximum temperature
        ppt_col: Column name for precipitation
        vpd_col: Column name for vapor pressure deficit
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        window: Window size for rolling mean (days)
        min_periods: Minimum number of observations required in window
        
    Returns:
        DataFrame with added rolling mean columns
    """
    # Check required columns
    required_cols = {lat_col, lon_col, 'date'}
    if not required_cols.issubset(df.columns):
        logger.warning("Required columns are missing. Cannot group by region.")
        return df

    # Convert to datetime and sort within groups
    df['date'] = pd.to_datetime(df['date'])

    # Columns to process
    columns_to_check = {
        tmax_col: 'tmax_7day_mean', 
        ppt_col: 'ppt_7day_mean', 
        vpd_col: 'vbd_7day_mean'
    }

    # Perform a single groupby and sort within groups
    df = df.sort_values(by=['latitude', 'longitude', 'date'])

    # Apply rolling mean for each column directly
    for col, new_col_name in columns_to_check.items():
        if col in df.columns:
            df[new_col_name] = (
                df.groupby([lon_col, lat_col])[col]
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
        else:
            logger.warning(f"Column {col} not found, skipping rolling mean calculation")
            df[new_col_name] = np.nan

    return df

def normalize_columns(df, columns, prefix='normalized_'):
    """
    Normalize specified columns using RobustScaler and add them as new columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to normalize.
        prefix (str): Prefix to add to the new normalized columns (default: 'robust_').

    Returns:
        pd.DataFrame: A DataFrame with added normalized columns.
    """
    # Initialize the RobustScaler
    scaler = RobustScaler()
    
    # Fit and transform the specified columns
    scaled_data = scaler.fit_transform(df[columns])
    
    # Create a DataFrame for the scaled data with prefixed column names
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{prefix}{col}" for col in columns])
    
    # Concatenate with the original DataFrame
    result_df = pd.concat([df, scaled_df], axis=1)
    
    return result_df

def engineer_ca_features(df, config=None):
    """
    Apply California-specific weather feature engineering.
    This is the main function that combines all feature engineering steps.
    
    Args:
        df: DataFrame with weather data
        config: Configuration dictionary for feature engineering
    
    Returns:
        DataFrame with engineered features
    """
    if config is None:
        config = {}
    
    # Extract configuration parameters with defaults
    include_hot_dry = config.get('include_hot_dry', True)
    include_spi = config.get('include_spi', True)
    include_vpd = config.get('include_vpd', True)
    include_temporal = config.get('include_temporal', True)
    include_compound = config.get('include_compound', True)
    include_rolling_means = config.get('include_rolling_means', True)  
        

    columns = ['tmax', 'ppt', 'vbdmax' ]

    
    # Validate input dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        logger.warning("Input DataFrame is empty")
        return df.copy()
    
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Log input data characteristics
    logger.info(f"Input data shape: {result_df.shape}")
    logger.info(f"Input data columns: {', '.join(result_df.columns)}")
    
    # Check for spatial columns
    has_coords = 'longitude' in result_df.columns and 'latitude' in result_df.columns
    has_grid = 'grid_id' in result_df.columns
    
    if has_coords:
        logger.info("Using lat/lon coordinates for spatial analysis")
    elif has_grid:
        logger.info("Using grid_id for spatial analysis")
    else:
        logger.warning("No spatial coordinates found. Some features may be limited.")
    
    # Apply feature engineering steps based on configuration
    try:
        # 1. Hot-dry index features
        if include_hot_dry and 'tmax' in result_df.columns and 'ppt' in result_df.columns:
            logger.info("Calculating hot-dry index")
            result_df = calculate_hot_dry_index(result_df)
        
        # 2. SPI features
        if include_spi and 'ppt' in result_df.columns and 'date' in result_df.columns:
            logger.info("Calculating SPI features")
            result_df = calculate_spi_ca_daily(result_df)
        
        # 3. VPD features
        if include_vpd and 'vbdmax' in result_df.columns:
            logger.info("Calculating VPD features")
            result_df = calculate_vpd_extreme_ca(result_df)
        
        # 4. Temporal features
        if include_temporal and 'date' in result_df.columns:
            logger.info("Adding temporal features")
            result_df = add_ca_temporal_features(result_df)
        
        # 5. Rolling mean features (7-day)
        if include_rolling_means and 'date' in result_df.columns:
            logger.info("Calculating 7-day rolling means")
            result_df = calculate_rolling_means(result_df)
        
        # 6. Compound indices
        if include_compound:
            logger.info("Calculating compound risk indices")
            result_df = calculate_compound_indices(result_df)

        # 7. Normalize columns
        if 'tmax' in result_df.columns:
            logger.info("Normalizing temperature columns")
            result_df = normalize_columns(result_df, ['tmax', 'ppt', 'vbdmax'], prefix='normalized_')
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise
    
    # Log output data characteristics
    new_features = set(result_df.columns) - set(df.columns)
    logger.info(f"Added {len(new_features)} new features: {', '.join(new_features)}")
    
    return result_df