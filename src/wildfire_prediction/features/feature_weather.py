"""
California Wildfire Feature Engineering (Daily Data)

Optimized for:
- Daily temporal resolution
- California's geographic diversity
- Large datasets (23,913 points/day)
- Memory efficiency
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import warnings
from tqdm import tqdm  # For progress tracking

# Global feature statistics tracker
FEATURE_STATS = {
    'execution_time': {},
    'memory_usage': {}
}

def calculate_hot_dry_index(df, window=7, normalize=True):
    """
    California-optimized Hot-Dry Index with rolling window.
    
    Parameters:
    -----------
    df : pd.DataFrame with ['tmax', 'ppt']
    window : int (days for rolling average)
    normalize : bool (scale to 0-1)
    
    Returns:
    --------
    pd.Series
    """
    try:
        # 7-day rolling averages (optimal for daily data)
        tmax_avg = df['tmax'].rolling(window, min_periods=3).mean()
        ppt_sum = df['ppt'].rolling(window, min_periods=3).sum()
        
        hdi = tmax_avg * (1 / (ppt_sum + 0.1))  # Avoid division by zero
        
        # California-specific normalization
        stats = {
            'min': 0,  # Theoretical minimum
            'max': 150, # California observed max
            'mean': hdi.mean()
        }
        
        if normalize:
            hdi = np.clip((hdi - stats['min']) / (stats['max'] - stats['min']), 0, 1)
            
        FEATURE_STATS['hot_dry_index'] = stats
        return hdi
        
    except Exception as e:
        warnings.warn(f"Hot-Dry Index failed: {str(e)}")
        return pd.Series(np.nan, index=df.index)

def calculate_spi_ca_daily(precip_series, window=6, min_years=10):
    """
    California-optimized SPI for daily data.
    
    Parameters:
    -----------
    precip_series : pd.Series (daily precipitation)
    window : int (months for SPI calculation)
    min_years : int (minimum years for reliable SPI)
    
    Returns:
    --------
    pd.Series (daily SPI values)
    """
    try:
        # Resample to monthly first
        monthly = precip_series.resample('M').sum()
        
        if len(monthly) < 12 * min_years:
            warnings.warn(f"Insufficient data for {window}-month SPI")
            return pd.Series(np.nan, index=precip_series.index)
            
        # Calculate rolling sums
        rolling_sum = monthly.rolling(window).sum().dropna()
        
        # Fit gamma distribution using L-moments (better for precipitation)
        params = gamma.fit(rolling_sum[rolling_sum > 0], method='lmoments')
        
        # Calculate SPI
        spi_monthly = pd.Series(index=monthly.index, dtype=float)
        valid_mask = rolling_sum > 0
        spi_monthly[valid_mask] = norm.ppf(gamma.cdf(rolling_sum[valid_mask], *params))
        
        # Handle zeros
        zero_mask = (rolling_sum == 0) & ~rolling_sum.isna()
        if zero_mask.any():
            p_zero = zero_mask.mean()
            spi_monthly[zero_mask] = norm.ppf(p_zero / 2)
        
        # Map back to daily
        spi_daily = spi_monthly.reindex(precip_series.index, method='ffill')
        
        FEATURE_STATS['spi'] = {
            'window': f"{window}-month",
            'zeros_pct': zero_mask.mean(),
            'mean': spi_daily.mean()
        }
        
        return spi_daily
        
    except Exception as e:
        warnings.warn(f"SPI calculation failed: {str(e)}")
        return pd.Series(np.nan, index=precip_series.index)

def calculate_vpd_extreme_ca(df, thresholds=(3.0, 4.0)):
    """
    California-specific VPD extremes with elevation adjustment.
    
    Parameters:
    -----------
    df : pd.DataFrame with ['vbdmax', (optional 'elevation')]
    thresholds : tuple (coastal, inland thresholds in kPa)
    
    Returns:
    --------
    tuple (instant_extreme, rolling_7day_extreme)
    """
    try:
        coastal_thresh, inland_thresh = thresholds
        
        # Adjust threshold by elevation if available
        if 'elevation' in df.columns:
            thresh = np.where(df['elevation'] < 500, coastal_thresh, inland_thresh)
        else:
            thresh = coastal_thresh  # Default to coastal threshold
            
        # Calculate extremes
        instant = (df['vbdmax'] > thresh).astype(int)
        rolling = (df['vbdmax'].rolling(7).mean() > thresh).astype(int)
        
        FEATURE_STATS['vpd_extreme'] = {
            'thresholds': thresholds,
            'pct_extreme': instant.mean()
        }
        
        return instant, rolling
        
    except Exception as e:
        warnings.warn(f"VPD Extreme failed: {str(e)}")
        return (pd.Series(0, index=df.index)), (pd.Series(0, index=df.index))

def add_ca_temporal_features(df, date_col='date'):
    """
    California-optimized temporal features.
    
    Parameters:
    -----------
    df : pd.DataFrame
    date_col : str (date column name)
    
    Returns:
    --------
    pd.DataFrame with added features
    """
    try:
        df = df.copy()
        dates = pd.to_datetime(df[date_col])
        
        # Basic features
        df['year'] = dates.dt.year
        df['month'] = dates.dt.month
        df['day_of_year'] = dates.dt.dayofyear
        
        # California fire season (May-Oct)
        df['fire_season'] = df['month'].between(5, 10).astype(int)
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        
        return df
        
    except Exception as e:
        warnings.warn(f"Temporal features failed: {str(e)}")
        return df

def calculate_ca_clusters(df, method='dbscan', eps_km=30, min_samples=10):
    """
    California-optimized spatial clustering.
    
    Parameters:
    -----------
    df : pd.DataFrame with ['latitude', 'longitude']
    method : str ('dbscan' or 'kmeans')
    eps_km : float (DBSCAN radius in km)
    min_samples : int (minimum cluster size)
    
    Returns:
    --------
    pd.Series (cluster labels)
    """
    try:
        coords = df[['latitude', 'longitude']].dropna()
        
        if len(coords) < 100:  # Fallback for small datasets
            method = 'kmeans'
            n_clusters = min(5, len(coords)//20)
        
        # Convert to radians for Haversine
        coords_rad = np.radians(coords.values)
        distances = haversine_distances(coords_rad) * 6371  # km
        
        if method == 'dbscan':
            labels = DBSCAN(eps=eps_km, min_samples=min_samples, 
                           metric='precomputed').fit_predict(distances)
        else:
            labels = KMeans(n_clusters=n_clusters).fit_predict(distances)
        
        # Map back to original index
        result = pd.Series(np.nan, index=df.index)
        result.loc[coords.index] = labels
        
        FEATURE_STATS['spatial_clusters'] = {
            'method': method,
            'n_clusters': len(np.unique(labels[labels != -1])),
            'noise_pct': (labels == -1).mean() if method == 'dbscan' else 0
        }
        
        return result
        
    except Exception as e:
        warnings.warn(f"Clustering failed: {str(e)}")
        return pd.Series(np.nan, index=df.index)

def engineer_ca_features(df, config=None):
    """
    Complete California wildfire feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame (daily California data)
    config : dict (override default parameters)
    
    Returns:
    --------
    pd.DataFrame with engineered features
    """
    # Default configuration (optimized for California)
    default_config = {
        'hot_dry_window': 7,
        'spi_window': 12,  # 12-month SPI for drought monitoring
        'vpd_thresholds': (3.0, 4.0),  # (coastal, inland)
        'cluster_method': 'dbscan',
        'cluster_eps_km': 30,
        'min_cluster_samples': 10
    }
    cfg = {**default_config, **(config or {})}
    
    # Track memory usage
    start_mem = df.memory_usage().sum() / 1024**2
    
    # 1. Temporal features (always calculated if date exists)
    if 'date' in df.columns:
        df = add_ca_temporal_features(df)
    
    # 2. Hot-Dry Index (requires tmax and ppt)
    if {'tmax', 'ppt'}.issubset(df.columns):
        df['hot_dry_index'] = calculate_hot_dry_index(df, window=cfg['hot_dry_window'])
    
    # 3. SPI (requires ppt)
    if 'ppt' in df.columns:
        df['spi'] = calculate_spi_ca_daily(df.set_index('date')['ppt'], 
                                         window=cfg['spi_window'])
    
    # 4. VPD Extremes (requires vbdmax)
    if 'vbdmax' in df.columns:
        df['vpd_instant'], df['vpd_rolling'] = calculate_vpd_extreme_ca(
            df, thresholds=cfg['vpd_thresholds'])
    
    # 5. Spatial Clusters (requires lat/lon)
    if {'latitude', 'longitude'}.issubset(df.columns):
        df['cluster_id'] = calculate_ca_clusters(
            df,
            method=cfg['cluster_method'],
            eps_km=cfg['cluster_eps_km'],
            min_samples=cfg['min_cluster_samples'])
    
    # Track memory usage
    end_mem = df.memory_usage().sum() / 1024**2
    FEATURE_STATS['memory_usage'] = {
        'start_mb': start_mem,
        'end_mb': end_mem,
        'increase_mb': end_mem - start_mem
    }

    return df