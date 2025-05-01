import pytest
import pandas as pd
import os
import sys

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from wildfire_prediction.data.process_data import (
    drop_bad_rows,
    validate_coordinates
)

@pytest.fixture
def sample_fire_data():
    """Create a sample dataframe for testing"""
    return pd.DataFrame({
        'latitude': [34.05, 36.12, None, 45.67, -91.23],
        'longitude': [-118.24, -115.67, -117.30, None, -120.45],
        'brightness': [305.4, 298.1, 310.2, 315.6, 290.8],
        'acq_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    })

def test_drop_bad_rows(sample_fire_data):
    """Test that rows with missing coordinates are dropped"""
    cleaned_df = drop_bad_rows(sample_fire_data)
    
    # Should drop rows with None in latitude or longitude
    assert len(cleaned_df) == 3
    assert cleaned_df['latitude'].isna().sum() == 0
    assert cleaned_df['longitude'].isna().sum() == 0

def test_validate_coordinates(sample_fire_data):
    """Test that invalid coordinates are identified"""
    valid_coords = validate_coordinates(sample_fire_data)
    
    # Should mark the row with latitude -91.23 as invalid (out of range)
    assert len(valid_coords) == 2
    assert all(lat >= -90 and lat <= 90 for lat in valid_coords['latitude'])
    assert all(lon >= -180 and lon <= 180 for lon in valid_coords['longitude'])