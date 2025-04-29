
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path to the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/data directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # Go up two levels to project root
print(f"Project root directory: {project_root}")

# Construct the path to the data file
file_path = os.path.join(project_root, 'data', 'raw', 'fire_archive_SV-C2_607788.csv')
abs_file_path = os.path.abspath(file_path)
print(f"Absolute file path: {abs_file_path}")
print(f"File exists: {os.path.exists(abs_file_path)}")

# Try to list files in the target directory
target_dir = os.path.dirname(abs_file_path)
if os.path.exists(target_dir):
    print(f"Files in {target_dir}:")
    for file in os.listdir(target_dir):
        print(f"  - {file}")
else:
    print(f"Directory {target_dir} does not exist")



# Read the CSV file and check for any issues
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} rows")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    raise

# Validate longitude and latitude columns exist and contain valid data
required_cols = ['longitude', 'latitude']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Missing required columns: {required_cols}")

if df[required_cols].isnull().any().any():
    print("Warning: Found null values in coordinates")
    df = df.dropna(subset=required_cols)

# Convert to GeoDataFrame with error handling
try:
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84 coordinate system
except Exception as e:
    print(f"Error creating GeoDataFrame: {e}")
    raise

# Load California boundary with proper error handling
try:
    # Load all U.S. states - Fix the path with proper raw string formatting
    states = gpd.read_file(r"D:\college\computer science level 3\second term\data science\cb_2023_us_state_20m.zip")
    
    # Filter California
    california = states[states.STATEFP == "06"]  # '06' is California's FIPS code
    california.to_file("california.geojson", driver="GeoJSON")
    if california.empty:
        raise ValueError("California boundary file is empty")
except Exception as e:
    print(f"Error loading California boundary: {e}")
    raise

# Filter points within California
try:
    california_data = gdf[gdf.within(california.unary_union)]
    print(f"Filtered to {len(california_data)} points within California")
    
    # Sample the data if it's too large (over 917,000 points is a lot!)
    if len(california_data) > 10000:
        print(f"Sampling data for better plot performance...")
        california_data = california_data.sample(10000, random_state=42)
        print(f"Sampled to {len(california_data)} points")
except Exception as e:
    print(f"Error filtering points: {e}")
    raise

# Set the matplotlib backend explicitly for better compatibility
import matplotlib
matplotlib.use('TkAgg')  # Try 'Agg', 'Qt5Agg', or 'TkAgg'

# Plot California with data points - reduce DPI for better performance
plt.figure(figsize=(10, 8), dpi=100)  # Lower DPI and slightly smaller figure

# Plot California boundary
california.plot(color='lightgrey', edgecolor='black')

# Plot data points with smaller markers
california_data.plot(ax=plt.gca(), column='brightness', cmap='hot', 
                    markersize=2, legend=True,  # Smaller markers
                    legend_kwds={'label': 'Brightness', 'orientation': 'horizontal'})

# Add title and labels
plt.title('Wildfire Data Points in California', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Save the figure with lower resolution
os.makedirs(os.path.join(project_root, 'reports', 'figures'), exist_ok=True)
plt.savefig(os.path.join(project_root, 'reports', 'figures', 'california_wildfires.png'), 
           bbox_inches='tight', dpi=100)
print(f"Plot saved to {os.path.join(project_root, 'reports', 'figures', 'california_wildfires.png')}")

# Show the plot
plt.tight_layout()  # Adjust layout to fit everything
plt.show()

# data = np.array(df['brightness']).reshape(-1, 1)
# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(data)
# print(normalized_data)
# # Preview data
# print("Initial data preview:")
# print(df.head())
# print("\nMissing values per column:")
# print(df.isnull().sum())

# # Handle missing values (drop rows with any missing values)
# df = df.dropna()

# # Remove duplicates
# df = df.drop_duplicates()


# # convert longitude and latitude to Point objects
# df['geometry'] = df.apply(
#     lambda row: Point(row['longitude'], row['latitude']), axis=1)
# firms_gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

# # convert acq_date to weekly periods
# firms_gdf['week'] = firms_gdf['acq_date'].dt.to_period('W')
# weekly_fires = firms_gdf.groupby(['grid_id', 'week']).size().reset_index(name='fire_count')


