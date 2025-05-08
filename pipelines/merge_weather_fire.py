import pandas as pd
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #Read the datasets
fire_path = os.path.join(root_dir, 'data', 'raw', 'california_wildfires.csv')
weather_path = os.path.join(root_dir, 'data', 'interim', 'weather.parquet')

wildfire_df = pd.read_csv(fire_path)
weather_df = pd.read_parquet(weather_path)

# #Ensure the merge keys have compatible types
# #Example: If 'week' is a datetime, convert both; if it's a string/week number, ensure both are string/int
# #wildfire_df['week'] = pd.to_datetime(wildfire_df['week'])
# #weather_df['week'] = pd.to_datetime(weather_df['week'])
# #Perform the merge on 'week' and 'grid_id'
merged_df = pd.merge(wildfire_df, weather_df, on=['week', 'grid_id'], how='inner')
merged_df.to_parquet('merged_data.parquet', index=False)

# #Display the first few rows of the merged DataFrame
# merged_df.head()