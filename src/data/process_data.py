from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utm  # Make sure this is installed: pip install utm
from scipy import stats


def get_project_paths():
    """Set up project paths and verify file existence."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Create paths dictionary
    paths = {
        'project_root': project_root,
        'raw_data': os.path.join(project_root, 'data', 'raw'),
        'processed_data': os.path.join(project_root, 'data', 'processed'),
        'figures': os.path.join(project_root, 'reports', 'figures')
    }
    
    # Ensure directories exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    # Set file paths
    paths['fire_data'] = os.path.join(paths['raw_data'], 'fire_archive_SV-C2_607788.csv')
    paths['ca_boundary'] = os.path.join(paths['raw_data'], 'cb_2023_us_state_20m.zip')
    paths['processed_fire_data'] = os.path.join(paths['processed_data'], 'california_wildfires.csv')
    
    # Verify file existence
    if not os.path.exists(paths['fire_data']):
        raise FileNotFoundError(f"Fire data file not found at: {paths['fire_data']}")
    if not os.path.exists(paths['ca_boundary']):
        raise FileNotFoundError(f"California boundary file not found at: {paths['ca_boundary']}")
    
    print(f"Project root directory: {paths['project_root']}")
    print(f"Fire data file exists: {os.path.exists(paths['fire_data'])}")
    
    return paths


def load_and_clean_data(file_path):
    """Load fire data and perform initial cleaning."""
    try:
        df = pd.read_csv(file_path)
        initial_rows = len(df)
        print(f"Initially loaded {initial_rows} rows")
        
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
        raise Exception(f"Error loading or cleaning CSV file: {e}")


def validate_coordinates(df):
    """Validate coordinate columns exist and contain valid data."""
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
    """Convert DataFrame to GeoDataFrame with Point geometries."""
    try:
        print(f"Before converting to GeoDataFrame: {len(df)} rows")
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84 coordinate system
        print(f"After converting to GeoDataFrame: {len(gdf)} rows")
        return gdf
    except Exception as e:
        raise Exception(f"Error creating GeoDataFrame: {e}")


def load_california_boundary(boundary_path):
    """Load California boundary from shapefile."""
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
    """Filter points to only include those within California."""
    try:
        print(f"Before filtering to California: {len(gdf)} rows")
        california_data = gdf[gdf.within(california.geometry.union_all())]
        print(f"After filtering to California: {len(california_data)} rows")
        return california_data
    except Exception as e:
        raise Exception(f"Error filtering points: {e}")


def lat_lon_to_utm_grid(lat, lon, grid_size_km=1):
    """Convert latitude and longitude to UTM grid ID."""
    easting, northing, zone_num, zone_letter = utm.from_latlon(lat, lon)
    grid_easting = int(easting // (grid_size_km * 1000))  # Convert to km grid
    grid_northing = int(northing // (grid_size_km * 1000))
    return f"{zone_num}{zone_letter}_{grid_easting}_{grid_northing}"


def create_grid_and_time_features(gdf):
    """Create grid ID and time-based features."""
    print(f"Before creating grid and time features: {len(gdf)} rows")
    
    # Create grid ID for each point
    gdf['grid_id'] = gdf.apply(
        lambda row: lat_lon_to_utm_grid(row['latitude'], row['longitude']), axis=1
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


def encode_categorical_features(df):
    """Encode categorical features for modeling."""
    print(f"Before encoding categorical features: {len(df)} rows")
    
    # Encode confidence levels (ordinal encoding)
    if 'confidence' in df.columns:
        # Filter out low confidence values
        print(f"Before filtering low confidence: {len(df)} rows")
        df = df[df['confidence'] != 'l']
        print(f"After filtering low confidence: {len(df)} rows")
        
        # Filter out any remaining invalid confidence values
        valid_confidence = df['confidence'].isin(['m', 'h'])
        if not valid_confidence.all():
            print(f"Before removing invalid confidence values: {len(df)} rows")
            print(f"Warning: Found {(~valid_confidence).sum()} invalid confidence values")
            df = df[valid_confidence]
            print(f"After removing invalid confidence values: {len(df)} rows")
        
        confidence_order = ['m', 'h']  # From Medium to High (removed Low)
        encoder = OrdinalEncoder(categories=[confidence_order])
        df['confidence_encoded'] = encoder.fit_transform(df[['confidence']])
    
    # One-hot encode day/night
    if 'daynight' in df.columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(df[['daynight']])
        df['is_day'] = encoded_data[:, 0]  # Only need one column since we dropped the first
    
    print(f"After encoding categorical features: {len(df)} rows")
    return df


def transform_numerical_features(df):
    """Apply transformations to numerical features."""
    print(f"Before transforming numerical features: {len(df)} rows")
    
    # Log transform FRP (Fire Radiative Power)
    if 'frp' in df.columns:
        df['frp_log'] = np.log1p(df['frp'])  # log(x+1) to handle zeros
        
        # Calculate and print skewness
        original_skew = stats.skew(df['frp'].dropna())
        transformed_skew = stats.skew(df['frp_log'].dropna())
        print(f"FRP Skewness - Original: {original_skew:.4f}, Log-transformed: {transformed_skew:.4f}")
    
    # Normalize brightness
    if 'brightness' in df.columns:
        scaler = MinMaxScaler()
        df['brightness_normalized'] = scaler.fit_transform(df[['brightness']])
    
    print(f"After transforming numerical features: {len(df)} rows")
    return df


def drop_unnecessary_columns(df):
    """Remove columns that aren't needed for analysis."""
    print(f"Before dropping unnecessary columns: {len(df)} rows")
    
    columns_to_drop = ['scan', 'track', 'version', 'satellite', 'instrument', 'bright_t31']
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        
    print(f"After dropping unnecessary columns: {len(df)} rows")
    return df


def save_processed_data(df, output_path):
    """Save processed data to CSV file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False


def visualize_frp_distribution(df, output_dir):
    """Create and save visualizations of FRP distribution."""
    if 'frp' not in df.columns:
        print("FRP column not found in dataframe")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate skewness
    frp_skewness = stats.skew(df['frp'].dropna())
    
    # Create histogram for original FRP
    plt.figure(figsize=(10, 6))
    sns.histplot(df['frp'], kde=True, color='orangered')
    plt.axvline(df['frp'].mean(), color='red', linestyle='--', label=f'Mean: {df["frp"].mean():.2f}')
    plt.axvline(df['frp'].median(), color='green', linestyle='-.', label=f'Median: {df["frp"].median():.2f}')
    plt.title(f'Distribution of Fire Radiative Power (FRP)\nSkewness: {frp_skewness:.4f}', fontsize=14)
    plt.xlabel('Fire Radiative Power (FRP)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'frp_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create histogram for log-transformed FRP
    if 'frp_log' in df.columns:
        log_frp_skewness = stats.skew(df['frp_log'].dropna())
        plt.figure(figsize=(10, 6))
        sns.histplot(df['frp_log'], kde=True, color='blue')
        plt.title(f'Log-Transformed Distribution of FRP\nSkewness: {log_frp_skewness:.4f}', fontsize=14)
        plt.xlabel('Log(FRP + 1)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'log_frp_distribution.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"FRP distribution visualizations saved to: {output_dir}")


def visualize_california_fires(california_data, california, output_dir):
    """Create and save visualization of fire points in California."""
    if california_data.empty or california.empty:
        print("Cannot create California visualization: empty data")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data if it's too large
    plot_data = california_data
    if len(california_data) > 10000:
        plot_data = california_data.sample(10000, random_state=42)
        print(f"Sampled to {len(plot_data)} points for visualization")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    california.plot(ax=ax, color='lightgrey', edgecolor='black')
    plot_data.plot(ax=ax, column='brightness', cmap='hot', 
                markersize=2, legend=True, 
                legend_kwds={'label': 'Brightness', 'orientation': 'horizontal'})
    plt.title('Wildfire Data Points in California', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'california_wildfires.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"California wildfire visualization saved to: {output_dir}")


def main():
    """Main function to orchestrate the data processing pipeline."""
    try:
        # Get project paths
        paths = get_project_paths()
        
        # Load and clean data
        df = load_and_clean_data(paths['fire_data'])
        
        # Validate coordinates
        df = validate_coordinates(df)
        
        # Convert to GeoDataFrame
        gdf = convert_to_geodataframe(df)
        
        # Load California boundary
        california = load_california_boundary(paths['ca_boundary'])
        
        # Filter to California data
        california_data = filter_california_data(gdf, california)
        
        # Create grid and time features
        california_data = create_grid_and_time_features(california_data)
        
        # Encode categorical features
        california_data = encode_categorical_features(california_data)
        
        # Transform numerical features
        california_data = transform_numerical_features(california_data)
        
        # Drop unnecessary columns
        california_data = drop_unnecessary_columns(california_data)
        
        # Save processed data
        save_processed_data(california_data, paths['processed_fire_data'])
        
        # Create visualizations
        # visualize_frp_distribution(california_data, paths['figures'])
        # visualize_california_fires(california_data, california, paths['figures'])
        
        print("Data processing completed successfully!")
        return california_data
        
    except Exception as e:
        print(f"Error in data processing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()