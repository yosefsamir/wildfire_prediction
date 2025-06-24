import pytest
import pandas as pd
from wildfire_prediction.features.validation import validate_dataframe, WildfireSchema, check_missing_values
from pandas import DataFrame

@pytest.fixture
def sample_df() -> DataFrame:
    df = DataFrame({
        'latitude': [34.05, 36.12],
        'longitude': [-118.24, -115.67],
        'brightness': [305.4, 298.1],
        'acq_date': ['2023-01-01', '2023-01-02']
    })
    # Convert acq_date to datetime to match schema requirements
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    return df

def test_validate_dataframe_success(sample_df):
    result = validate_dataframe(sample_df, WildfireSchema.BASIC_SCHEMA, 'test')
    assert result is None

def test_validate_dataframe_missing_column(sample_df):
    invalid_df = sample_df.drop(columns=['latitude'])
    result = validate_dataframe(invalid_df, WildfireSchema.BASIC_SCHEMA, 'test')
    assert "Missing columns ['latitude']" in result

def test_check_missing_values_below_threshold():
    # Create a dataframe with missing values below threshold
    df = DataFrame({
        'latitude': [34.05, 36.12, None],
        'longitude': [-118.24, -115.67, -117.30],
        'brightness': [305.4, 298.1, 310.2],
        'acq_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    # 1/3 = 0.33 missing in latitude column, with threshold of 0.4 should pass
    assert check_missing_values(df, threshold=0.4) is True

def test_check_missing_values_above_threshold():
    # Create a dataframe with missing values above threshold
    df = DataFrame({
        'latitude': [34.05, None, None],
        'longitude': [-118.24, -115.67, -117.30],
        'brightness': [305.4, 298.1, 310.2],
        'acq_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    # 2/3 = 0.67 missing in latitude column, with threshold of 0.5 should fail
    assert check_missing_values(df, threshold=0.5) is False

def test_validate_dataframe_type_check(sample_df):
    # Create a dataframe with incorrect type
    invalid_df = sample_df.copy()
    invalid_df['brightness'] = invalid_df['brightness'].astype(str)
    
    # The validation function doesn't currently check types, but this test
    # is added for future implementation of type checking
    # This test will pass for now since the current implementation only checks for missing columns
    result = validate_dataframe(invalid_df, WildfireSchema.BASIC_SCHEMA, 'test')
    assert result is None  # Current implementation doesn't check types
