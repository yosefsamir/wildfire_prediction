"""Feature engineering module for wildfire prediction.

This module contains functions for creating and transforming features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from scipy import stats
import utm
import geopandas as gpd
from datetime import datetime
from itertools import product
import random
import time
import gc
from concurrent.futures import ProcessPoolExecutor

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


def grid_id_to_lat_lon(grid_id, grid_size_km=1):
    """Convert a UTM grid ID back to approximate latitude and longitude.
    
    Args:
        grid_id: Grid ID in the format "zone_easting_northing"
        grid_size_km: Size of the grid in kilometers
        
    Returns:
        tuple: (latitude, longitude) coordinates representing the center of the grid cell
    """
    # Parse the grid ID
    parts = grid_id.split('_')
    zone_part = parts[0]
    grid_easting = int(parts[1])
    grid_northing = int(parts[2])
    
    # Extract zone number and letter
    zone_num = int(zone_part[:-1])
    zone_letter = zone_part[-1]
    
    # Calculate the center of the grid cell
    easting = (grid_easting * grid_size_km * 1000) + (grid_size_km * 500)  # Center of cell
    northing = (grid_northing * grid_size_km * 1000) + (grid_size_km * 500)  # Center of cell
    
    # Convert back to lat/lon
    lat, lon = utm.to_latlon(easting, northing, zone_num, zone_letter)
    
    return lat, lon


def week_to_acq_date(week):
    """Convert a week period to an acquisition date (middle of the week).
    
    Args:
        week: pandas Period object representing a week
        
    Returns:
        datetime: Date representing the middle of the week (Wednesday)
    """
    # Convert period to timestamp (start of the week)
    start_date = week.start_time.date()
    
    # Add 3 days to get to the middle of the week (Wednesday)
    from datetime import timedelta
    middle_date = start_date + timedelta(days=3)
    
    return pd.Timestamp(middle_date)


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
    # gdf['month'] = gdf['acq_date'].dt.month
    # gdf['year'] = gdf['acq_date'].dt.year
    # gdf['day_of_year'] = gdf['acq_date'].dt.dayofyear
    
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
        # confidence_order = ['n', 'h'] if drop_low_confidence else ['l', 'n', 'h']
        # encoder = OrdinalEncoder(categories=[confidence_order])
        # df['confidence_encoded'] = encoder.fit_transform(df[['confidence']])
    
    # # One-hot encode day/night
    # if 'daynight' in df.columns:
    #     encoder = OneHotEncoder(sparse_output=False, drop='first')
    #     encoded_data = encoder.fit_transform(df[['daynight']])
    #     df['is_day'] = encoded_data[:, 0]  # Only need one column since we dropped the first
    
    print(f"After encoding categorical features: {len(df)} rows")
    return df

def drop_nonzero_types(df):
    """Drop rows with non-zero 'type' values.

    Args:
        df: DataFrame with 'type' column

    Returns:
        pd.DataFrame: DataFrame with non-zero 'type' values removed
    """
    print(f"Before dropping non-zero 'type' values: {len(df)} rows")

    # Filter out rows where 'type' is not zero
    df = df[df['type'] == 0]

    print(f"After dropping non-zero 'type' values: {len(df)} rows")
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



def get_all_grid_ids(ca_gdf: gpd.GeoDataFrame, grid_size_km=1):
    """
    Generate all grid_ids covering the given California boundary GeoDataFrame.
    Args:
        ca_gdf (gpd.GeoDataFrame): California boundary polygon(s)
        grid_size_km (int): size of grid cell in km
    Returns:
        List[str]: list of grid_id strings
    """
    # Merge all geometries
    ca_poly = ca_gdf.unary_union
    minx, miny, maxx, maxy = ca_poly.bounds
    step = grid_size_km / 111.0  # approx degrees per km
    grid_ids = []
    y = miny
    while y <= maxy:
        x = minx
        while x <= maxx:
            if ca_poly.contains(gpd.points_from_xy([x], [y])[0]):
                grid_ids.append(lat_lon_to_utm_grid(y, x, grid_size_km))
            x += step
        y += step
    return list(set(grid_ids))




def get_all_weeks(start_date: str, end_date: str):
    """
    Return all weekly periods between start and end inclusive.
    Args:
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
    Returns:
        List[pd.Period]
    """
    weeks = pd.period_range(start=start_date, end=end_date, freq='W')
    return list(weeks)



def generate_negative_batch(args):
    """
    Helper function to generate a batch of negative samples in parallel.
    
    Args:
        args: Tuple containing (batch_size, all_grid_ids, all_weeks, positive_set, random_state)
        
    Returns:
        pd.DataFrame: DataFrame with negative samples
    """
    import random
    import pandas as pd
    from datetime import timedelta
    
    batch_size, all_grid_ids, all_weeks, positive_set, random_state, grid_id_to_lat_lon_func, week_to_acq_date_func = args
    
    # Set random seed for this process
    random.seed(random_state)
    
    # Pre-generate more combinations than we need to account for duplicates
    # This significantly reduces the number of iterations needed
    multiplier = 3  # Generate 3x more than needed to reduce iterations
    grid_ids = [random.choice(all_grid_ids) for _ in range(batch_size * multiplier)]
    weeks = [random.choice(all_weeks) for _ in range(batch_size * multiplier)]
    
    # Create combinations and filter out positives
    batch_combos = []
    batch_combos_set = set()  # For faster duplicate checking
    
    for i in range(len(grid_ids)):
        if len(batch_combos) >= batch_size:
            break
            
        grid_id = grid_ids[i]
        week = weeks[i]
        combo = (grid_id, str(week))
        
        # Check if this combo is not a positive sample and not already in our batch
        if combo not in positive_set and combo not in batch_combos_set:
            batch_combos.append((grid_id, week))
            batch_combos_set.add(combo)
    
    # If we couldn't generate enough unique combinations, return what we have
    if not batch_combos:
        return pd.DataFrame()
        
    # Convert to DataFrame
    batch_df = pd.DataFrame(batch_combos, columns=['grid_id', 'week'])
    
    # Add required columns for negative samples
    batch_df['fire'] = 0  # Mark as negative samples
    # Verify fire column is set to 0 for all rows
    assert (batch_df['fire'] == 0).all(), "Error: Not all negative samples have fire=0"
    batch_df['frp_log'] = 0.0
    batch_df['brightness_normalized'] = 0.0
    
    # Convert grid_id to latitude and longitude - vectorized approach
    # This is much faster than applying row by row
    lat_lons = [grid_id_to_lat_lon_func(gid) for gid in batch_df['grid_id']]
    batch_df['latitude'] = [ll[0] for ll in lat_lons]
    batch_df['longitude'] = [ll[1] for ll in lat_lons]
    
    # Convert week to acquisition date - vectorized approach
    batch_df['acq_date'] = [week_to_acq_date_func(wk) for wk in batch_df['week']]
    
    return batch_df


def build_full_dataset(wildfire_df: pd.DataFrame,
                        all_grid_ids: list,
                        all_weeks: list,
                        n_negatives: int = None,
                        random_state: int = 42,
                        batch_size: int = 250000,
                        n_jobs: int = 4):
    """
    Build a DataFrame with positive and negative samples of (grid_id, week).
    Optimized implementation with parallel processing for faster generation.
    
    Args:
        wildfire_df (pd.DataFrame): must contain 'grid_id' and 'week'
        all_grid_ids (list): all possible grid_id values
        all_weeks (list): all possible week Periods
        n_negatives (int): number of negative samples to include; if None, include all
        random_state (int): seed for sampling
        batch_size (int): size of batches for processing negative samples
        n_jobs (int): number of parallel jobs to run (default: 4)
    Returns:
        pd.DataFrame: with columns ['grid_id', 'week', 'fire', 'longitude', 'latitude', 
                                   'acq_date', 'frp_log', 'brightness_normalized']
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    import gc
    
    start_time = time.time()
    
    # Prepare positive samples - include all required columns
    required_columns = ['grid_id', 'week', 'latitude', 'longitude', 'acq_date', 'frp_log', 'brightness_normalized']
    
    # Check for typo in column name 'longitude'
    if 'longtiude' in wildfire_df.columns and 'longitude' not in wildfire_df.columns:
        wildfire_df = wildfire_df.rename(columns={'longtiude': 'longitude'})
    
    # Ensure all required columns exist
    missing_columns = [col for col in required_columns if col not in wildfire_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in input dataframe: {missing_columns}")
        # Create missing columns with placeholder values
        for col in missing_columns:
            if col == 'frp_log':
                wildfire_df[col] = 0.0
            elif col == 'brightness_normalized':
                wildfire_df[col] = 0.1  # Minimum value from MinMaxScaler
    
    # Create a copy of the positive samples with all required columns
    pos = wildfire_df[required_columns].copy()
    
    # Ensure all positive samples are labeled correctly
    pos['fire'] = 1
    n_pos = len(pos)
    print(f"Positive samples: {n_pos}")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # If n_negatives is None or too large, set a reasonable default
    if n_negatives is None:
        n_negatives = 1_000_000
        print(f"Setting negative samples to {n_negatives} for memory efficiency")
    
    # Optimize: Convert all_grid_ids and all_weeks to numpy arrays for faster random selection
    all_grid_ids = np.array(all_grid_ids, dtype=object)
    all_weeks = np.array(all_weeks, dtype=object)
    
    print(f"Using optimized parallel processing approach for {n_negatives} samples with {n_jobs} workers")
    
    # Create a hash set of positive samples for fast lookup
    positive_set = set((gid, str(wk)) for gid, wk in zip(pos['grid_id'], pos['week']))
    
    # Calculate optimal batch size based on n_negatives and n_jobs
    # Ensure each worker gets a reasonable amount of work
    optimal_batch_size = max(min(batch_size, n_negatives // n_jobs), 10000)
    
    # Calculate number of batches needed
    n_batches = (n_negatives + optimal_batch_size - 1) // optimal_batch_size  # Ceiling division
    
    print(f"Generating {n_negatives} negative samples in {n_batches} batches of ~{optimal_batch_size} each")
    
    # Prepare batch arguments for parallel processing
    batch_args = []
    for i in range(n_batches):
        # For the last batch, adjust size if needed
        current_batch_size = min(optimal_batch_size, n_negatives - i * optimal_batch_size)
        # Use different random seeds for each batch to avoid duplicates
        batch_seed = random_state + i
        batch_args.append((current_batch_size, all_grid_ids, all_weeks, positive_set, 
                        batch_seed, grid_id_to_lat_lon, week_to_acq_date))
    
    # Generate negative samples in parallel
    neg_dfs = []
    samples_generated = 0
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Process batches in parallel
        for i, batch_df in enumerate(executor.map(generate_negative_batch, batch_args)):
            if not batch_df.empty:
                neg_dfs.append(batch_df)
                samples_generated += len(batch_df)
                
                # Report progress
                if (i+1) % max(1, n_batches//10) == 0 or i+1 == n_batches:
                    elapsed = time.time() - start_time
                    print(f"Progress: {samples_generated}/{n_negatives} samples in {elapsed:.1f}s "
                        f"({100*samples_generated/n_negatives:.1f}%)")
    
    # Combine all negative samples efficiently
    if neg_dfs:
        try:
            # Try to concatenate all batches at once
            print("Combining negative samples...")
            neg = pd.concat(neg_dfs, ignore_index=True)
            
            # Free memory
            del neg_dfs
            gc.collect()
            
        except MemoryError:
            # If memory error, try to concatenate in smaller groups
            print("Memory error during concatenation. Trying with smaller groups...")
            final_dfs = []
            group_size = max(1, len(neg_dfs) // 10)  # Divide into ~10 groups
            
            for i in range(0, len(neg_dfs), group_size):
                group = pd.concat(neg_dfs[i:i+group_size], ignore_index=True)
                final_dfs.append(group)
                # Clear original dataframes to free memory
                for j in range(i, min(i+group_size, len(neg_dfs))):
                    neg_dfs[j] = None
            
            # Free memory
            del neg_dfs
            gc.collect()
            
            neg = pd.concat(final_dfs, ignore_index=True)
            del final_dfs
            gc.collect()
        
        print(f"Final negative samples: {len(neg)}")
        
        # Combine with positive samples
        try:
            print("Creating final dataset...")
            # Verify negative samples have fire=0 before concatenation
            assert (neg['fire'] == 0).all(), "Error: Not all negative samples have fire=0 before concatenation"
            
            full_df = pd.concat([pos, neg], ignore_index=True)
            
            # Verify that after concatenation, negative samples still have fire=0
            neg_count = len(neg)
            zeros_after_concat = (full_df['fire'] == 0).sum()
            print(f"Negative samples before concat: {neg_count}, Zero values after concat: {zeros_after_concat}")
            assert zeros_after_concat == neg_count, "Error: Number of samples with fire=0 doesn't match number of negative samples"
            
            # Free memory
            del neg, pos
            gc.collect()
            
        except MemoryError:
            # If we can't combine all at once, sample the negatives first
            print("Memory error during final concatenation. Sampling negatives first...")
            if len(neg) > 500000:
                neg = neg.sample(n=500000, random_state=random_state)
            
            # Verify negative samples have fire=0 before concatenation
            assert (neg['fire'] == 0).all(), "Error: Not all negative samples have fire=0 before concatenation"
            
            full_df = pd.concat([pos, neg], ignore_index=True)
            
            # Verify that after concatenation, negative samples still have fire=0
            neg_count = len(neg)
            zeros_after_concat = (full_df['fire'] == 0).sum()
            print(f"Negative samples before concat: {neg_count}, Zero values after concat: {zeros_after_concat}")
            assert zeros_after_concat == neg_count, "Error: Number of samples with fire=0 doesn't match number of negative samples"
            
            # Free memory
            del neg, pos
            gc.collect()
        
        total_time = time.time() - start_time
        print(f"Total dataset size: {len(full_df)} rows (created in {total_time:.1f}s)")
        return full_df
    else:
        # If we couldn't generate any negative samples, just return positives
        print("Warning: Could not generate any negative samples")
        return pos
    
    
    
    
def sample_dataset(full_df: pd.DataFrame, total_rows: int, random_state: int = 42):
    """
    Sample a dataset to a specified total number of rows, keeping all positives.
    Args:
        full_df (pd.DataFrame): must contain 'fire' column
        total_rows (int): desired total row count
        random_state (int)
    Returns:
        pd.DataFrame: sampled DataFrame
    """
    # Verify input data has correct fire labels
    pos_count = (full_df['fire'] == 1).sum()
    neg_count = (full_df['fire'] == 0).sum()
    print(f"Before sampling - Positive samples: {pos_count}, Negative samples: {neg_count}")
    
    pos = full_df[full_df['fire'] == 1].copy()
    neg = full_df[full_df['fire'] == 0].copy()
    
    # Double-check that our filtering worked correctly
    assert (pos['fire'] == 1).all(), "Error: Not all positive samples have fire=1"
    assert (neg['fire'] == 0).all(), "Error: Not all negative samples have fire=0"
    
    n_pos = len(pos)
    n_neg = max(total_rows - n_pos, 0)
    
    # Sample negatives
    if n_neg < len(neg):
        # Use more efficient sampling for very large datasets
        if len(neg) > 1000000:  # If more than 1 million rows
            # Calculate fraction instead of absolute number to reduce memory usage
            fraction = n_neg / len(neg)
            neg_sampled = neg.sample(frac=fraction, random_state=random_state)
        else:
            neg_sampled = neg.sample(n=n_neg, random_state=random_state)
        # Verify sampled negatives still have fire=0
        assert (neg_sampled['fire'] == 0).all(), "Error: Not all sampled negative samples have fire=0"
    else:
        neg_sampled = neg
    
    # Combine datasets
    sampled_df = pd.concat([pos, neg_sampled], ignore_index=True)
    
    # Verify final dataset has correct fire labels
    final_pos_count = (sampled_df['fire'] == 1).sum()
    final_neg_count = (sampled_df['fire'] == 0).sum()
    print(f"After sampling - Positive samples: {final_pos_count}, Negative samples: {final_neg_count}")
    assert final_pos_count == n_pos, "Error: Number of positive samples changed after concatenation"
    assert final_neg_count == len(neg_sampled), "Error: Number of negative samples changed after concatenation"
    
    return sampled_df