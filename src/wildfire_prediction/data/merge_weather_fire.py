import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import pyarrow.parquet as pq
import logging
from pathlib import Path
import pyarrow.compute as pc
from calendar import monthrange

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
WEATHER_PATH = DATA_DIR / "interim" / "weather_features.parquet"
FIRE_PATH = DATA_DIR / "processed" / "california_wildfires.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "merged" / "merged_fire_weather.csv"
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)

# Constants
BUFFER_DEGREES = 1.0  # ~111km buffer for spatial filtering
TEMPORAL_TOLERANCE = pd.Timedelta('1D')  # 3-day tolerance for temporal matching

def load_and_prepare_fire_data():
    """Load and prepare wildfire data with spatial indexing, optimized for speed"""
    logger.info("Loading fire data...")
    dtype_spec = {
        'grid_id': 'string',
        'week': 'string',
        'latitude': 'float64',
        'longitude': 'float64',
        'frp_log': 'float32',
        'brightness_normalized': 'float32',
        'fire': 'int8'
    }
    
    fire_df = pd.read_csv(
        FIRE_PATH,
        usecols=['grid_id', 'week', 'latitude', 'longitude', 'acq_date', 'frp_log', 'brightness_normalized', 'fire'],
        dtype=dtype_spec,
        parse_dates=['acq_date'],
        engine='c'
    )
    
    fire_coords = np.radians(fire_df[['latitude', 'longitude']].values)
    logger.info(f"Loaded {len(fire_df)} fire records")
    return fire_df, fire_coords



def process_month(year: int, month: int, fire_df: pd.DataFrame) -> pd.DataFrame:
    """Process a single month of data using a single monthly BallTree for spatial matching."""
    logger.info(f"=== Processing year {year}, month {month} ===")

    # 1. Filter fire data for the month
    mask = (
        (fire_df['acq_date'].dt.year == year) &
        (fire_df['acq_date'].dt.month == month)
    )
    month_fire = fire_df.loc[mask].reset_index(drop=True)
    logger.info(f"Year {year}, Month {month} fire data shape: {month_fire.shape}")
    if month_fire.empty:
        logger.warning(f"No fire data for {year}-{month:02d}")
        return pd.DataFrame()

    # Precompute fire radians array
    fire_rad = np.radians(month_fire[['latitude', 'longitude']].values)

    # 2. Determine spatial bounds
    min_lat = month_fire['latitude'].min() - BUFFER_DEGREES
    max_lat = month_fire['latitude'].max() + BUFFER_DEGREES
    min_lon = month_fire['longitude'].min() - BUFFER_DEGREES
    max_lon = month_fire['longitude'].max() + BUFFER_DEGREES
    logger.info(
        f"Spatial bounds {year}-{month:02d}: "
        f"Lat[{min_lat:.4f},{max_lat:.4f}], "
        f"Lon[{min_lon:.4f},{max_lon:.4f}]"
    )

    try:
        # 3. Load & filter weather data once per month
        logger.info(f"Loading weather data for {year}-{month:02d}...")
        table = pq.read_table(
            WEATHER_PATH,
            filters=[
                ('year', '=', year), ('month', '=', month),
                ('latitude', '>=', min_lat), ('latitude', '<=', max_lat),
                ('longitude', '>=', min_lon), ('longitude', '<=', max_lon)
            ]
        )
        weather = table.to_pandas()
        if weather.empty:
            logger.warning(f"No weather data for {year}-{month:02d}")
            return pd.DataFrame()

        # Normalize and group by date
        weather['date'] = pd.to_datetime(weather['date']).dt.normalize()
        weather_by_date = {
            date: group.reset_index(drop=True)
            for date, group in weather.groupby('date')
        }
        logger.info(f"Weather grouped into {len(weather_by_date)} unique dates")

        # 4. Build one spatial index per month on unique station coords
        unique_stations = weather[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        station_coords_rad = np.radians(unique_stations.values)
        monthly_tree = BallTree(station_coords_rad, metric='haversine')

        merged_records = []
        same_day = diff_day = no_match = 0

        # 5. Match each fire record via the monthly_tree
        for idx, row in month_fire.iterrows():
            fire_date = row['acq_date'].normalize()
            coord_rad = fire_rad[idx].reshape(1, -1)

            if fire_date in weather_by_date:
                day_group = weather_by_date[fire_date]
                # Query the monthly tree
                dists, inds = monthly_tree.query(coord_rad, k=1)
                station_idx = inds[0][0]
                station_lat, station_lon = unique_stations.iloc[station_idx]

                # Find the day's weather record for that station
                match = day_group[
                    (day_group['latitude'] == station_lat) &
                    (day_group['longitude'] == station_lon)
                ]
                if not match.empty:
                    weather_rec = match.iloc[0]
                    record = {**row.to_dict(), **weather_rec.to_dict()}
                    record['distance_km'] = dists[0][0] * 6371
                    record['temporal_offset_days'] = 0
                    merged_records.append(record)
                    same_day += 1
                else:
                    # No station record on this day—fallback temporal
                    temp = find_nearest_temporal_match(row, weather, TEMPORAL_TOLERANCE)
                    if temp is not None:
                        merged_records.append(temp)
                        diff_day += 1
                    else:
                        no_match += 1
            else:
                # No same-day group—use temporal fallback
                temp = find_nearest_temporal_match(row, weather, TEMPORAL_TOLERANCE)
                if temp is not None:
                    merged_records.append(temp)
                    diff_day += 1
                else:
                    no_match += 1

            if (idx + 1) % 10000 == 0 or idx == len(month_fire) - 1:
                logger.info(f"Processed {idx+1}/{len(month_fire)} records")

        logger.info(
            f"Matches: same_day={same_day}, diff_day={diff_day}, no_match={no_match}"
        )

        # 6. Compile results
        if merged_records:
            merged_df = pd.DataFrame(merged_records)
            logger.info(
                f"Final: matched {len(merged_df)} of {len(month_fire)} "
                f"({len(merged_df)/len(month_fire)*100:.2f}%)"
            )
            return merged_df
        else:
            logger.warning("No records matched for this month.")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error processing {year}-{month:02d}: {e}", exc_info=True)
        return pd.DataFrame()


def process_year(year, fire_df):
    """Process a single year of data by iterating through months"""
    logger.info(f"=== Processing year {year} by months ===")
    
    year_fire = fire_df[fire_df['acq_date'].dt.year == year].copy()
    
    logger.info(f"Year {year} fire data shape: {year_fire.shape}")
    print(f"Year {year} fire data shape: {year_fire.shape}")
    if year_fire.empty:
        logger.warning(f"No fire data found for year {year}")
        return pd.DataFrame()
    
    # Process each month separately
    month_results = []
    months_in_year = year_fire['acq_date'].dt.month.unique()
    
    for month in sorted(months_in_year):
        month_result = process_month(year, month, fire_df)
        if not month_result.empty:
            month_results.append(month_result)
    
    # Combine monthly results
    if month_results:
        combined_result = pd.concat(month_results)
        logger.info(f"Combined results for year {year}: {len(combined_result)} records from {len(month_results)} months")
        return combined_result
    else:
        logger.warning(f"No results found for any month in year {year}")
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
