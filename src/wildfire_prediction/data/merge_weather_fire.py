import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import pyarrow.parquet as pq
import logging
from pathlib import Path
import pyarrow.compute as pc

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
WEATHER_PATH = DATA_DIR / "interim" / "weather_master.parquet"
FIRE_PATH = DATA_DIR / "processed" / "california_wildfires.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "merged" / "merged_fire_weather.csv"
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)

# Constants
BUFFER_DEGREES = 1.0  # ~111km buffer for spatial filtering
TEMPORAL_TOLERANCE = pd.Timedelta('1D')  # 3-day tolerance for temporal matching

def load_and_prepare_fire_data():
    """Load and prepare wildfire data with spatial indexing"""
    logger.info("Loading fire data...")
    fire_df = pd.read_csv(FIRE_PATH, usecols=[
        'grid_id', 'week', 'latitude', 'longitude', 
        'acq_date', 'frp_log', 'brightness_normalized', 'fire'
    ])
    
    # Convert data types
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
    fire_df['week'] = fire_df['week'].astype(str)
    fire_df['grid_id'] = fire_df['grid_id'].astype(str)
    
    # Spatial coordinates in radians
    fire_coords = np.radians(fire_df[['latitude', 'longitude']].values)
    logger.info(f"Loaded {len(fire_df)} fire records")
    return fire_df, fire_coords


def process_year(year, fire_df):
    """Process a single year of data without monthly chunking using KNN for spatial matching"""
    logger.info(f"=== Processing year {year} ===")
    
    year_fire = fire_df[fire_df['acq_date'].dt.year == year].copy()

    logger.info(f"Year {year} fire data shape: {year_fire.shape}")
    print(f"Year {year} fire data shape: {year_fire.shape}")
    if year_fire.empty:
        logger.warning(f"No fire data found for year {year}")
        return pd.DataFrame()

    # Get spatial bounds for this year's fires
    min_lat = year_fire['latitude'].min() - BUFFER_DEGREES
    max_lat = year_fire['latitude'].max() + BUFFER_DEGREES
    min_lon = year_fire['longitude'].min() - BUFFER_DEGREES
    max_lon = year_fire['longitude'].max() + BUFFER_DEGREES
    
    logger.info(f"Spatial bounds for {year}: Lat [{min_lat:.4f}, {max_lat:.4f}], Lon [{min_lon:.4f}, {max_lon:.4f}]")

    try:
        logger.info(f"Loading weather data for year {year}...")
        # Load entire year's weather data with spatial filtering
        weather_data = pq.read_table(
            str(WEATHER_PATH),
            filters=[
                ('year', '=', year),
                ('latitude', '>=', min_lat),
                ('latitude', '<=', max_lat),
                ('longitude', '>=', min_lon),
                ('longitude', '<=', max_lon)
            ]
        ).to_pandas()

        logger.info(f"Loaded weather data for {year}: {len(weather_data)} records")

        if weather_data.empty:
            logger.warning(f"No weather data found for year {year}")
            return pd.DataFrame()
        
        # Log weather data statistics
        logger.info(f"Weather data summary for {year}:")
        logger.info(f"  Temperature (max): min={weather_data['tmax'].min():.2f}, max={weather_data['tmax'].max():.2f}, mean={weather_data['tmax'].mean():.2f}")
        logger.info(f"  Precipitation: min={weather_data['ppt'].min():.2f}, max={weather_data['ppt'].max():.2f}, mean={weather_data['ppt'].mean():.2f}")
        logger.info(f"  VBD (max): min={weather_data['vbdmax'].min():.2f}, max={weather_data['vbdmax'].max():.2f}, mean={weather_data['vbdmax'].mean():.2f}")
        
        # Prepare weather data for merging
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        # Group weather data by date for faster access
        logger.info(f"Grouping weather data by date...")
        weather_by_date = {date: group for date, group in weather_data.groupby('date')}
        logger.info(f"Weather data grouped into {len(weather_by_date)} unique dates")
        
        # Prepare our results list
        merged_data = []
        
        # Process all fire data at once
        logger.info(f"Processing {len(year_fire)} fire records using K-nearest neighbors...")
        
        
        # Create a lookup for weather data by date
        fire_dates = year_fire['acq_date'].dt.normalize().unique()
        logger.info(f"Fire data spans {len(fire_dates)} unique dates")
        
        # Process each fire record
        matches_same_day = 0
        matches_different_day = 0
        no_matches = 0
        
        # Use enumerate to correctly track the index
        for idx, fire_record in enumerate(year_fire.iterrows()):
            # Get the actual record data (fire_record is a tuple of (index, Series))
            _, fire_record_data = fire_record
            
            fire_date = fire_record_data['acq_date'].normalize()  # Normalize to remove time component
            fire_coords_single = np.radians([[fire_record_data['latitude'], fire_record_data['longitude']]])
            
            # First try to find weather data on the same day
            if fire_date in weather_by_date:
                day_weather = weather_by_date[fire_date]
                
                if not day_weather.empty:
                    # Get coordinates for weather stations on this day
                    day_weather_coords = np.radians(day_weather[['latitude', 'longitude']].values)
                    
                    # Build a BallTree for just this day's weather data
                    day_tree = BallTree(day_weather_coords, metric='haversine')
                    
                    # Find nearest neighbor
                    distances, indices = day_tree.query(fire_coords_single, k=1)
                    
                    # Get the nearest weather record
                    nearest_idx = indices[0][0]
                    nearest_weather = day_weather.iloc[nearest_idx].copy()
                    
                    # Combine fire and weather data
                    combined = {**fire_record_data.to_dict(), **nearest_weather.to_dict()}
                    combined['distance_km'] = distances[0][0] * 6371  # Convert to km (Earth radius ≈ 6371 km)
                    combined['temporal_offset_days'] = 0.0  # Same day
                    
                    merged_data.append(combined)
                    matches_same_day += 1
                else:
                    # Try to find nearest temporal match
                    temporal_match = find_nearest_temporal_match(fire_record_data, weather_data, TEMPORAL_TOLERANCE)
                    if temporal_match is not None:
                        merged_data.append(temporal_match)
                        matches_different_day += 1
                    else:
                        no_matches += 1
            else:
                # Try to find nearest temporal match
                temporal_match = find_nearest_temporal_match(fire_record_data, weather_data, TEMPORAL_TOLERANCE)
                if temporal_match is not None:
                    merged_data.append(temporal_match)
                    matches_different_day += 1
                else:
                    no_matches += 1
            
            # Log progress every 10,000 records
            if (idx + 1) % 10000 == 0 or idx == len(year_fire) - 1:
                logger.info(f"Processed {idx + 1}/{len(year_fire)} fire records")
        
        logger.info(f"Matching summary: Same day: {matches_same_day}, Different day: {matches_different_day}, No match: {no_matches}")
        
        # Convert results to DataFrame
        if merged_data:
            merged = pd.DataFrame(merged_data)
            
            # Log statistics about the matches
            logger.info(f"Total matched records: {len(merged)} of {len(year_fire)} ({len(merged)/len(year_fire)*100:.2f}%)")
            
            # Log spatial distance statistics
            logger.info(f"Distance statistics (km):")
            logger.info(f"  Min: {merged['distance_km'].min():.2f}, Max: {merged['distance_km'].max():.2f}, Mean: {merged['distance_km'].mean():.2f}")
            
            # Log temporal offset statistics if applicable
            if 'temporal_offset_days' in merged.columns and (merged['temporal_offset_days'] > 0).any():
                offset_data = merged[merged['temporal_offset_days'] > 0]
                logger.info(f"Temporal offset statistics (days) for {len(offset_data)} records:")
                logger.info(f"  Min: {offset_data['temporal_offset_days'].min():.2f}, Max: {offset_data['temporal_offset_days'].max():.2f}, Mean: {offset_data['temporal_offset_days'].mean():.2f}")
            
            # Log final merge statistics
            missing_weather = merged['tmax'].isna().sum()
            logger.info(f"Final merge stats for {year}:")
            logger.info(f"  Total records: {len(merged)}")
            logger.info(f"  Records with complete weather data: {len(merged) - missing_weather} ({(len(merged) - missing_weather)/len(merged)*100:.2f}%)")
            logger.info(f"  Records with missing weather data: {missing_weather} ({missing_weather/len(merged)*100:.2f}%)")
            
            # Sample data preview
            if not merged.empty:
                sample = merged.head(1)
                logger.info(f"Sample record preview for {year}: Fire location: ({sample['latitude'].iloc[0]:.4f}, {sample['longitude'].iloc[0]:.4f}), "
                            f"Date: {sample['acq_date'].iloc[0]}, Temp: {sample['tmax'].iloc[0]:.2f}, "
                            f"Precip: {sample['ppt'].iloc[0]:.2f}, VBD: {sample['vbdmax'].iloc[0]:.2f}, "
                            f"Distance: {sample['distance_km'].iloc[0]:.2f} km")
            
            logger.info(f"=== Year {year} processing complete ===")
            return merged
        else:
            logger.warning(f"No matches found for any records in year {year}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error processing year {year}: {str(e)}", exc_info=True)
        return pd.DataFrame()

def find_nearest_temporal_match(fire_record, weather_data, tolerance):
    """Find the nearest temporal match for a fire record within the given tolerance"""
    fire_date = fire_record['acq_date']
    fire_coords = np.radians([[fire_record['latitude'], fire_record['longitude']]])
    
    # Filter weather data within temporal tolerance
    date_lower = fire_date - tolerance
    date_upper = fire_date + tolerance
    temporal_window = weather_data[(weather_data['date'] >= date_lower) & (weather_data['date'] <= date_upper)]
    
    if temporal_window.empty:
        return None
    
    # Get coordinates for weather stations in temporal window
    temporal_coords = np.radians(temporal_window[['latitude', 'longitude']].values)
    
    # Build a BallTree for this temporal window
    temp_tree = BallTree(temporal_coords, metric='haversine')
    
    # Find nearest neighbor
    distances, indices = temp_tree.query(fire_coords, k=1)
    
    # Get the nearest weather record
    nearest_idx = indices[0][0]
    nearest_weather = temporal_window.iloc[nearest_idx].copy()
    
    # Combine fire and weather data
    combined = {**fire_record.to_dict(), **nearest_weather.to_dict()}
    combined['distance_km'] = distances[0][0] * 6371  # Convert to km (Earth radius ≈ 6371 km)
    combined['temporal_offset_days'] = abs((nearest_weather['date'] - fire_date).total_seconds()) / 86400  # Convert to days
    
    return combined
