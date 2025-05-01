import pytest
import pandas as pd
import os
import sys

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

@pytest.fixture
def sample_weather_data():
    """Create a sample weather dataframe"""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'temperature': [75.4, 78.2, 72.1],
        'humidity': [45.2, 42.8, 50.3],
        'wind_speed': [8.5, 10.2, 7.8]
    })

@pytest.fixture
def sample_satellite_data():
    """Create a sample satellite dataframe"""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'ndvi': [0.65, 0.58, 0.72],
        'lst': [310.2, 315.4, 308.7]
    })

@pytest.fixture
def sample_fire_data():
    """Create a sample fire records dataframe"""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'fire_count': [3, 0, 5],
        'total_frp': [45.2, 0.0, 78.5]
    })

def test_merge_shapes(sample_weather_data, sample_satellite_data, sample_fire_data):
    """Test that merged dataframe has correct shape"""
    # Merge the dataframes on date
    merged_df = sample_weather_data.merge(sample_satellite_data, on='date').merge(sample_fire_data, on='date')
    
    # Check the shape of the merged dataframe
    assert merged_df.shape[0] == 3  # Should have 3 rows
    assert merged_df.shape[1] == 8  # Should have 8 columns (date + 3 weather + 2 satellite + 2 fire)

def test_merge_columns(sample_weather_data, sample_satellite_data, sample_fire_data):
    """Test that merged dataframe has all expected columns"""
    # Merge the dataframes on date
    merged_df = sample_weather_data.merge(sample_satellite_data, on='date').merge(sample_fire_data, on='date')
    
    # Check that all columns from the original dataframes are present
    expected_columns = ['date', 'temperature', 'humidity', 'wind_speed', 'ndvi', 'lst', 'fire_count', 'total_frp']
    for col in expected_columns:
        assert col in merged_df.columns

def test_merge_values(sample_weather_data, sample_satellite_data, sample_fire_data):
    """Test that values are preserved correctly in the merge"""
    # Merge the dataframes on date
    merged_df = sample_weather_data.merge(sample_satellite_data, on='date').merge(sample_fire_data, on='date')
    
    # Check a specific row to ensure values are preserved
    row = merged_df[merged_df['date'] == pd.Timestamp('2023-01-01')].iloc[0]
    assert row['temperature'] == 75.4
    assert row['ndvi'] == 0.65
    assert row['fire_count'] == 3