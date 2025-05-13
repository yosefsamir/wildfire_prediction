"""Weather data processing module for wildfire prediction.

This module contains functions for loading, cleaning, and processing weather data.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

# Import grid feature functionality
from ..features.feature_engineering import lat_lon_to_utm_grid, create_grid_and_time_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_weather_files_for_year(raw_weather_dir, year, file_pattern="%Y%m%d.csv"):
    """Get all weather data files for a specific year.
    
    Args:
        raw_weather_dir: Directory containing raw weather data organized by year
        year: The year to process (as string or int)
        file_pattern: Pattern for the filename with date format codes
        
    Returns:
        list: List of file paths for the specified year
    """
    year_str = str(year)
    year_dir = os.path.join(raw_weather_dir, year_str)
    
    if not os.path.exists(year_dir):
        raise FileNotFoundError(f"Weather data directory for year {year} not found at: {year_dir}")
    
    # Find all CSV files for the year
    pattern = os.path.join(year_dir, "*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No weather data files found for year {year} in {year_dir}")
    
    logger.info(f"Found {len(files)} weather files for year {year}")
    return sorted(files)

def extract_date_from_filename(filename, pattern="%Y%m%d.csv"):
    """Extract date from weather filename with given pattern.
    
    Args:
        filename: Weather file name
        pattern: Date pattern in filename
        
    Returns:
        datetime: Date extracted from filename
    """
    base_name = os.path.basename(filename)
    # Convert pattern to regex pattern that extracts the date part
    date_format = pattern.replace('.', '\\.').replace('%Y', '([0-9]{4})').replace('%m', '([0-9]{2})').replace('%d', '([0-9]{2})')
    
    import re
    match = re.match(date_format, base_name)
    
    if match:
        # Reconstruct date string based on the pattern
        date_str = ''
        if '%Y' in pattern:
            date_str += match.group(1)
        if '%m' in pattern:
            date_str += match.group(2)
        if '%d' in pattern:
            date_str += match.group(3)
            
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            logger.error(f"Could not parse date from filename: {filename}")
            return None
    else:
        # Fallback to the original method
        date_str = os.path.splitext(base_name)[0]  # Remove extension
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            logger.error(f"Could not parse date from filename: {filename}")
            return None

def clean_weather_file(file_path, config, output_dir=None):
    """Clean an individual weather data file using configuration settings.
    
    Args:
        file_path: Path to the weather data file
        config: Configuration dictionary with cleaning parameters
        output_dir: Directory to save the cleaned file (if None, will not save)
        
    Returns:
        pd.DataFrame: Cleaned weather data
    """
    try:
        # Get configuration values
        file_pattern = config['weather'].get('file_pattern', "%Y%m%d.csv")
        missing_indicators = config['weather'].get('missing_value_indicators', [-9999])
        drop_threshold = config['weather']['clean_columns'].get('drop_threshold', 50.0)
        
        # Get date from filename
        file_date = extract_date_from_filename(file_path, file_pattern)
        if file_date is None:
            return None
            
        # Read weather CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Processing {file_path}: {len(df)} rows")
        
        # Add date column based on filename
        df['date'] = file_date
        
        # Clean data - common operations:
        # 1. Handle missing values
        initial_rows = len(df)
        for indicator in missing_indicators:
            df = df.replace(indicator, np.nan)  # Replace missing value indicators
        
        # 2. Calculate missing value percentage
        missing_percentage = df.isna().mean() * 100
        logger.info(f"Missing value percentage: \n{missing_percentage[missing_percentage > 0].to_string()}")
        
        # 3. Drop columns with too many missing values (e.g., >50%)
        cols_to_drop = missing_percentage[missing_percentage > drop_threshold].index.tolist()
        if cols_to_drop:
            logger.info(f"Dropping columns with >{drop_threshold}% missing values: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # 4. Handle remaining missing values by filling
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # 5. Check for and remove duplicate rows
        df = df.drop_duplicates()
        logger.info(f"After removing duplicates: {len(df)} rows (removed {initial_rows - len(df)})")
        
        # 6. Validate coordinates if present
        if all(col in df.columns for col in ['latitude', 'longitude']):
            # Drop rows with invalid coordinates
            df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90) & 
                    (df['longitude'] >= -180) & (df['longitude'] <= 180)]
            logger.info(f"After coordinate validation: {len(df)} rows")
        
        
        # 7. Save the cleaned file if output_dir is provided
        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get original filename
            original_filename = os.path.basename(file_path)
            
            # Save to output directory with same filename
            output_path = os.path.join(output_dir, original_filename)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned file to: {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def clean_weather_data_for_year(raw_weather_dir, output_dir, year, config_path=None):
    """Clean all weather data for a specific year and save the result.
    
    Args:
        raw_weather_dir: Directory containing raw weather data
        output_dir: Directory to save cleaned data
        year: Year to process
        config_path: Path to configuration file
        
    Returns:
        str: Path to the cleaned weather data file for the year
    """
    try:
        # Load configuration if provided
        if config_path:
            config = load_config(config_path)
        else:
            # Default configuration
            config = {
                'weather': {
                    'file_pattern': "%Y%m%d.csv",
                    'output_file_pattern': "weather_clean_%Y.csv",
                    'missing_value_indicators': [-9999],
                    'clean_columns': {
                        'drop_threshold': 50.0,
                    }
                }
            }
        
        # Get file pattern from config
        file_pattern = config['weather'].get('file_pattern', "%Y%m%d.csv")
        output_pattern = config['weather'].get('output_file_pattern', "weather_clean_%Y.csv")
        
        # Get all weather files for the year
        files = get_weather_files_for_year(raw_weather_dir, year, file_pattern)
        
        # Format output filename for the yearly summary
        output_filename = datetime.strptime(str(year), "%Y").strftime(output_pattern)
        output_path = os.path.join(output_dir, output_filename)
        
        # Process each file and append directly to yearly file without saving individual files
        first_file = True
        total_rows = 0
        
        for file_path in files:
            # Process the file but don't save it individually (pass None as output_dir)
            df = clean_weather_file(file_path, config, None)
            if df is not None:
                # Append to the yearly file
                header = first_file  # Only write header for the first file
                df.to_csv(output_path, mode='a' if not header else 'w', 
                         index=False, header=header)
                
                current_rows = len(df)
                total_rows += current_rows
                logger.info(f"Added {current_rows} rows from {os.path.basename(file_path)} to yearly file (total: {total_rows})")
                
                # After the first file, we're in append mode
                first_file = False
                
        if total_rows == 0:
            raise ValueError(f"No valid weather data files processed for year {year}")
        
        logger.info(f"Processed {len(files)} weather files for year {year}")
        logger.info(f"Merged all files for year {year}: {total_rows} total rows")
        logger.info(f"Saved merged yearly file to: {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error cleaning weather data for year {year}: {str(e)}")
        raise


def merge_yearly_weather_files(output_dir, interim_dir, years, config_path=None, grid_size_km=1):
    """
    Merge yearly weather files into a single master Parquet file without partitioning.
    
    Args:
        output_dir: Directory containing yearly weather files
        interim_dir: Directory to save master file
        years: List of years to process
        config_path: Path to configuration file
        grid_size_km: Size of the grid in kilometers
    
    Returns:
        str: Path to the master weather file
    """
    try:
        # Load configuration
        config = load_config(config_path) if config_path else {
            'weather': {
                'output_file_pattern': "weather_clean_%Y.csv",
                'master_file_name': "weather_master.parquet",
                'grid_precision': 0.1
            }
        }
        
        output_pattern = config['weather'].get('output_file_pattern', "weather_clean_%Y.csv")
        master_file_name = config['weather'].get('master_file_name', "weather_master.parquet")
        master_path = os.path.join(interim_dir, master_file_name)
        os.makedirs(interim_dir, exist_ok=True)
        
        processed_files = []
        parquet_writer = None
        
        for year in years:
            year_filename = str(year).join(output_pattern.split("%Y"))
            year_path = os.path.join(output_dir, year_filename)
            
            if not os.path.exists(year_path):
                logger.warning(f"Yearly file not found: {year_path}")
                continue
            
            logger.info(f"Processing {year_path}")
            processed_files.append(year_path)
            
            # Process in chunks
            for chunk_idx, chunk in enumerate(pd.read_csv(year_path, chunksize=500000)):
                # Convert and enhance data
                chunk['date'] = pd.to_datetime(chunk['date'])
                chunk['year'] = chunk['date'].dt.year
                chunk['month'] = chunk['date'].dt.month
                chunk['day_of_year'] = chunk['date'].dt.dayofyear
                
                # Add UTM grid IDs
                chunk = add_grid_to_weather_data(chunk, grid_size_km)
                
                # Convert to PyArrow Table
                table = pa.Table.from_pandas(chunk)
                
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(master_path, table.schema, compression='snappy')

                parquet_writer.write_table(table)
                
                logger.info(f"Processed chunk {chunk_idx+1} from {year_path}")

        parquet_writer.close()
        if not processed_files:
            raise ValueError("No valid yearly files found for merging!")
        
        logger.info(f"Merged {len(processed_files)} yearly files into {master_path}")
        return master_path
        
    except Exception as e:
        logger.error(f"Error in merge_yearly_weather_files: {e}")
        raise

def add_grid_to_weather_data(df, grid_size_km=1):
    """Add UTM grid IDs to weather data for spatial analysis.
    
    Args:
        df: DataFrame with latitude and longitude columns
        grid_size_km: Size of the grid in kilometers
        
    Returns:
        pd.DataFrame: DataFrame with added grid ID feature
    """
    logger.info(f"Before adding grid IDs to weather data: {len(df)} rows")
    
    # Create grid ID for each point in the weather data
    df['grid_id'] = df.apply(
        lambda row: lat_lon_to_utm_grid(row['latitude'], row['longitude'], grid_size_km), axis=1
    )

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Create time-based features
    df['week'] = df['date'].dt.to_period('W')
    # Log grid ID counts for debugging
    grid_count = df['grid_id'].nunique()
    logger.info(f"Created {grid_count} unique grid cells")
    
    logger.info(f"After adding grid IDs to weather data: {len(df)} rows")
    return df


def load_fire_unique_grids_and_weeks(fire_data_path):
    """
    Load california wildfire data to extract unique grid_id values and unique week values separately.
    
    Args:
        fire_data_path: Path to the california_wildfires.csv file
        
    Returns:
        tuple: (set_of_unique_grid_ids, set_of_unique_weeks)
    """
    logger.info(f"Loading fire data from {fire_data_path} to extract unique grid_ids and weeks...")
    
    # Create sets to store unique values
    unique_grids = set()
    unique_weeks = set()
    
    try:
        for chunk in pd.read_csv(fire_data_path, chunksize=500000):
            # Validate required columns
            if 'grid_id' not in chunk.columns:
                logger.error("Fire data missing required column: grid_id")
                raise ValueError("Fire data missing required column: grid_id")
            
            # Handle week column
            if 'week' not in chunk.columns:
                if 'date' not in chunk.columns:
                    logger.error("Fire data missing both 'week' and 'date' columns")
                    raise ValueError("Fire data missing both 'week' and 'date' columns")
                chunk['date'] = pd.to_datetime(chunk['date'])
                chunk['week'] = chunk['date'].dt.to_period('W').astype(str)
            
            # Standardize week format to string
            chunk['week'] = chunk['week'].astype(str)
            
            # Update unique sets
            unique_grids.update(chunk['grid_id'].unique())
            unique_weeks.update(chunk['week'].unique())
            
            logger.info(f"Processed chunk: {len(unique_grids)} unique grids, {len(unique_weeks)} unique weeks so far")
        
        logger.info(f"Completed loading. Found {len(unique_grids)} unique grids and {len(unique_weeks)} unique weeks")
        return unique_grids, unique_weeks
        
    except Exception as e:
        logger.error(f"Error loading fire data: {e}")
        raise



def filter_chunk_by_fire_data(chunk, unique_grids, unique_weeks):
    """
    Filter a chunk of weather data to keep records where either:
    - grid_id is in unique_grids, OR
    - week is in unique_weeks
    
    Args:
        chunk: DataFrame chunk containing weather data
        unique_grids: Set of unique grid_id values to match
        unique_weeks: Set of unique week values to match
        
    Returns:
        pd.DataFrame: Filtered chunk containing matching rows
    """
    # --- Step 1: Ensure week column exists in consistent string format ---
    if 'week' not in chunk.columns:
        if 'date' not in chunk.columns:
            raise ValueError("Data must contain either 'week' or 'date' column")
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['week'] = chunk['date'].dt.to_period('W').astype(str)
    else:
        chunk['week'] = chunk['week'].astype(str)  # Ensure string format
    
    # --- Step 2: Vectorized filtering (OR condition) ---
    grid_mask = chunk['grid_id'].isin(unique_grids)
    week_mask = chunk['week'].isin(unique_weeks)
    combined_mask = grid_mask | week_mask  
    
    filtered_chunk = chunk[combined_mask]
    
    logger.info(
        f"Filtered from {len(chunk)} to {len(filtered_chunk)} rows "
        f"(grid matches: {grid_mask.sum()}, week matches: {week_mask.sum()})"
    )
    return filtered_chunk