"""Weather feature engineering for wildfire prediction.

This module contains functions for creating and transforming weather features 
specifically for wildfire prediction in California.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_hot_dry_index(df, tmax_col='tmax', ppt_col='ppt', threshold_temp=25.0, threshold_ppt=1.0):
    """
    Calculate a hot-dry index for wildfire risk assessment.
    High temperatures combined with low precipitation increase wildfire risk.
    
    Args:
        df: DataFrame with temperature and precipitation data
        tmax_col: Column name for maximum temperature
        ppt_col: Column name for precipitation
        threshold_temp: Temperature threshold (C) for high risk
        threshold_ppt: Precipitation threshold (mm) for low rainfall
        
    Returns:
        DataFrame with added hot_dry_index column
    """
    if tmax_col not in df.columns or ppt_col not in df.columns:
        logger.warning(f"Required columns missing. Cannot calculate hot_dry_index")
        df['hot_dry_index'] = 0.0
        return df
    
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Calculate hot-dry index: normalized temperature - normalized precipitation
    # Higher values indicate hotter and drier conditions (higher fire risk)
    t_norm = (result_df[tmax_col] - result_df[tmax_col].min()) / (result_df[tmax_col].max() - result_df[tmax_col].min() + 1e-8)
    p_norm = (result_df[ppt_col] - result_df[ppt_col].min()) / (result_df[ppt_col].max() - result_df[ppt_col].min() + 1e-8)
    
    # Invert precipitation (lower rainfall = higher risk)
    p_norm_inv = 1 - p_norm
    
    # Calculate index (0-2 range, higher = more risk)
    result_df['hot_dry_index'] = t_norm + p_norm_inv
    
    # Add binary flags for extreme conditions
    result_df['high_temp_day'] = (result_df[tmax_col] > threshold_temp).astype(int)
    result_df['low_rain_day'] = (result_df[ppt_col] < threshold_ppt).astype(int)
    
    # Calculate hot-dry streak - consecutive days with high temps and low rain
    # (As our data may not be continuous by day, we'll use a binary flag)
    result_df['hot_dry_day'] = ((result_df['high_temp_day'] == 1) & 
                              (result_df['low_rain_day'] == 1)).astype(int)
    
    return result_df

def calculate_spi_ca_daily(df, ppt_col='ppt', lookback_days=7, min_periods=3, lat_lon_bins=1.0):
    """
    Calculate Standardized Precipitation Index (SPI) for California weather data using a 7-day rolling window.
    SPI is useful for short-term drought monitoring, affecting wildfire risk.
    
    Args:
        df: DataFrame with precipitation data
        ppt_col: Column name for precipitation
        lookback_days: Number of days for rolling window (default: 7)
        min_periods: Minimum number of periods required for calculation
        lat_lon_bins: Size of latitude/longitude bins (for context, not used in calculation)
        
    Returns:
        DataFrame with added SPI column
    """
    if ppt_col not in df.columns or 'date' not in df.columns:
        logger.warning(f"Required columns missing. Cannot calculate SPI")
        df['spi_7day'] = 0.0
        return df
    
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Ensure date is datetime type and sort by date
    if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
        result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.sort_values('date')
    
    # Initialize SPI column
    result_df['spi_7day'] = 0.0
    
    # Calculate rolling precipitation for 7-day window
    rolling_7d = result_df[ppt_col].rolling(window=lookback_days, min_periods=min_periods).sum()
    
    # Calculate SPI: (rolling_sum - mean) / std
    if rolling_7d.std() > 0:
        result_df['spi_7day'] = (rolling_7d - rolling_7d.mean()) / rolling_7d.std()
    
    # Fill NaN values (e.g., early rows with insufficient data)
    result_df['spi_7day'] = result_df['spi_7day'].fillna(0)
    
    # Categorize into drought levels
    result_df['drought_category'] = pd.cut(
        result_df['spi_7day'],
        bins=[-float('inf'), -2, -1.5, -1, -0.5, 0.5, float('inf')],
        labels=[5, 4, 3, 2, 1, 0]  # Higher values indicate more severe drought
    ).astype('int')
    
    return result_df    

def calculate_vpd_extreme_ca(df, vpd_col='vbdmax', percentile_threshold=90):
    """
    Calculate extreme vapor pressure deficit (VPD) indicators for California.
    High VPD is associated with increased wildfire risk.
    
    Args:
        df: DataFrame with VPD data
        vpd_col: Column name for VPD
        percentile_threshold: Percentile threshold for extreme values
        
    Returns:
        DataFrame with added VPD extreme columns
    """
    if vpd_col not in df.columns:
        logger.warning(f"Required column '{vpd_col}' missing. Cannot calculate VPD extremes")
        df['vpd_anomaly'] = 0.0
        df['vpd_extreme'] = 0
        return df
    
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Calculate the percentile threshold for the entire dataset
    vpd_threshold = np.nanpercentile(result_df[vpd_col], percentile_threshold)
    
    # Create binary indicator for extreme VPD
    result_df['vpd_extreme'] = (result_df[vpd_col] > vpd_threshold).astype(int)
    
    # Calculate anomaly (departure from average)
    vpd_mean = result_df[vpd_col].mean()
    vpd_std = result_df[vpd_col].std()
    
    if vpd_std > 0:
        result_df['vpd_anomaly'] = (result_df[vpd_col] - vpd_mean) / vpd_std
    else:
        result_df['vpd_anomaly'] = 0.0
    
    # Group VPD into categories for risk assessment
    result_df['vpd_risk_category'] = pd.qcut(
        result_df[vpd_col].rank(method='first'),  # Use rank to handle ties
        q=5,  # 5 categories
        labels=[1, 2, 3, 4, 5]  # 5 is highest risk
    ).astype('int')
    
    return result_df

def add_ca_temporal_features(df):
    """
    Add California-specific temporal features for wildfire prediction.
    
    Args:
        df: DataFrame with date column
        
    Returns:
        DataFrame with added temporal features
    """
    if 'date' not in df.columns:
        logger.warning("Date column missing. Cannot add temporal features")
        df['is_fire_season'] = 0
        return df
    
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Ensure date is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
        result_df['date'] = pd.to_datetime(result_df['date'])
    
    # Extract date components if not already present
    if 'month' not in result_df.columns:
        result_df['month'] = result_df['date'].dt.month
    
    if 'day_of_year' not in result_df.columns:
        result_df['day_of_year'] = result_df['date'].dt.dayofyear
    
    # Define CA-specific fire seasons and high-risk periods
    
    # Fire season indicator (mainly summer and fall in California)
    # Main fire season: June through October (months 6-10)
    result_df['is_fire_season'] = result_df['month'].between(6, 10).astype(int)
    
    # Santa Ana wind season (typically Oct-Apr in Southern California)
    result_df['is_santa_ana_season'] = ((result_df['month'] >= 10) | 
                                         (result_df['month'] <= 4)).astype(int)
    
    # Define seasons
    conditions = [
        (result_df['month'].isin([12, 1, 2])),  # Winter
        (result_df['month'].isin([3, 4, 5])),   # Spring
        (result_df['month'].isin([6, 7, 8])),   # Summer
        (result_df['month'].isin([9, 10, 11]))  # Fall
    ]
    seasons = [0, 1, 2, 3]  # 0=Winter, 1=Spring, 2=Summer, 3=Fall
    result_df['season'] = np.select(conditions, seasons, default=0)
    
    # Weekly trend (cyclic transformation to maintain continuity)
    week_in_year = result_df['day_of_year'] // 7
    max_week = 52
    # Cyclic encoding using sine and cosine transform
    result_df['week_sin'] = np.sin(2 * np.pi * week_in_year / max_week)
    result_df['week_cos'] = np.cos(2 * np.pi * week_in_year / max_week)
    
    # Monthly trend (cyclic transformation)
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
    
    return result_df


def calculate_compound_indices(df):
    """
    Calculate compound indices from multiple variables for wildfire risk assessment.
    
    Args:
        df: DataFrame with processed weather features
        
    Returns:
        DataFrame with added compound indices
    """
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Check which features are available
    has_hot_dry = 'hot_dry_index' in df.columns
    has_vpd = 'vpd_risk_category' in df.columns
    has_spi = 'drought_category' in df.columns
    has_wind = 'wind_risk' in df.columns
    
    # Composite Fire Weather Index (FWI) - combined risk from temperature, precipitation, and VPD
    if has_hot_dry and has_vpd:
        # Normalize hot_dry_index to 0-5 scale for consistency with other indicators
        hot_dry_norm = 5 * (result_df['hot_dry_index'] - result_df['hot_dry_index'].min()) / (
            result_df['hot_dry_index'].max() - result_df['hot_dry_index'].min() + 1e-8)
        
        # Calculate FWI as weighted average of hot-dry index and VPD risk
        result_df['fire_weather_index'] = (0.5 * hot_dry_norm + 
                                          0.5 * result_df['vpd_risk_category'])
    else:
        # Use whatever is available
        if has_hot_dry:
            hot_dry_norm = 5 * (result_df['hot_dry_index'] - result_df['hot_dry_index'].min()) / (
                result_df['hot_dry_index'].max() - result_df['hot_dry_index'].min() + 1e-8)
            result_df['fire_weather_index'] = hot_dry_norm
        elif has_vpd:
            result_df['fire_weather_index'] = result_df['vpd_risk_category']
        else:
            result_df['fire_weather_index'] = 1  # Default to low risk
    
    # Long-term Drought and Weather Index - combines drought status with current weather
    if has_spi and has_hot_dry:
        # Weight drought more heavily for long-term index
        result_df['drought_weather_index'] = (0.7 * result_df['drought_category'] + 
                                             0.3 * hot_dry_norm)
    else:
        if has_spi:
            result_df['drought_weather_index'] = result_df['drought_category']
        elif has_hot_dry:
            result_df['drought_weather_index'] = hot_dry_norm
        else:
            result_df['drought_weather_index'] = 1  # Default to low risk
    
    # Comprehensive Fire Risk Index - includes all factors
    if has_hot_dry and has_vpd and has_spi:
        # Equal weights to each factor, adjusted for missing wind
        if has_wind:
            result_df['fire_risk_index'] = (0.25 * hot_dry_norm + 
                                          0.25 * result_df['vpd_risk_category'] +
                                          0.25 * result_df['drought_category'] +
                                          0.25 * result_df['wind_risk'])
        else:
            result_df['fire_risk_index'] = (0.33 * hot_dry_norm + 
                                          0.33 * result_df['vpd_risk_category'] +
                                          0.34 * result_df['drought_category'])
    else:
        # Use fire_weather_index as fallback
        result_df['fire_risk_index'] = result_df['fire_weather_index']
    
    return result_df

def calculate_rolling_means(df, tmax_col='tmax', ppt_col='ppt', vpd_col='vbdmax', window=7, min_periods=3):
    """
    Calculate rolling means for key weather variables to capture short-term trends.
    
    Args:
        df: DataFrame with weather data
        tmax_col: Column name for maximum temperature
        ppt_col: Column name for precipitation
        vpd_col: Column name for vapor pressure deficit
        window: Window size for rolling mean (days)
        min_periods: Minimum number of observations required in window
        
    Returns:
        DataFrame with added rolling mean columns
    """
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Check which features are available
    columns_to_check = {tmax_col: 'tmax_7day_mean', 
                        ppt_col: 'ppt_7day_mean', 
                        vpd_col: 'vbd_7day_mean'}
    
    # Ensure date is sorted for proper rolling calculations
    if 'date' in result_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
            result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.sort_values('date')
    
    # Calculate rolling means for each available column
    for col, new_col_name in columns_to_check.items():
        if col in result_df.columns:
            result_df[new_col_name] = result_df[col].rolling(window=window, min_periods=min_periods).mean()
            logger.info(f"Calculated {window}-day rolling mean for {col}")
        else:
            logger.warning(f"Column {col} not found, skipping rolling mean calculation")
            result_df[new_col_name] = np.nan
    
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
    include_rolling_means = config.get('include_rolling_means', True)  # New parameter
        
    # SPI calculation parameters
    lat_lon_bins = config.get('lat_lon_bins', 0.5)
    
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
            result_df = calculate_spi_ca_daily(result_df, lat_lon_bins=lat_lon_bins)
        
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
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise
    
    # Log output data characteristics
    new_features = set(result_df.columns) - set(df.columns)
    logger.info(f"Added {len(new_features)} new features: {', '.join(new_features)}")
    
    return result_df