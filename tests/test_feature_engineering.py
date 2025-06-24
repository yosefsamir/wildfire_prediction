import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from wildfire_prediction.features.feature_engineering import (
    log_transform,
    normalize_features,
    encode_categorical
)

@pytest.fixture
def sample_feature_data():
    """Create a sample dataframe for feature engineering testing"""
    return pd.DataFrame({
        'frp': [10.5, 25.3, 50.2, 100.8, 5.1],
        'brightness': [310.4, 350.2, 298.7, 325.6, 290.3],
        'confidence': ['nominal', 'high', 'nominal', 'high', 'low'],
        'daynight': ['D', 'D', 'N', 'N', 'D'],
        'acq_date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-20', '2023-04-10', '2023-05-05'])
    })

def test_log_transform(sample_feature_data):
    """Test logarithmic transformation of features"""
    df = sample_feature_data.copy()
    
    # Apply log transform to frp column
    df_transformed = log_transform(df, 'frp')
    
    # Check that a new column was created
    assert 'frp_log' in df_transformed.columns
    
    # Check that the transformation was applied correctly
    for i, val in enumerate(df['frp']):
        assert np.isclose(df_transformed['frp_log'][i], np.log1p(val))

def test_normalize_features(sample_feature_data):
    """Test feature normalization"""
    df = sample_feature_data.copy()
    
    # Normalize the brightness column
    df_normalized = normalize_features(df, ['brightness'])
    
    # Check that a new column was created
    assert 'brightness_normalized' in df_normalized.columns
    
    # Check that values are between 0 and 1
    assert df_normalized['brightness_normalized'].min() >= 0
    assert df_normalized['brightness_normalized'].max() <= 1

def test_encode_categorical(sample_feature_data):
    """Test categorical feature encoding"""
    df = sample_feature_data.copy()
    
    # Encode the confidence column
    df_encoded = encode_categorical(df, 'confidence')
    
    # Check that a new column was created
    assert 'confidence_encoded' in df_encoded.columns
    
    # Check that the encoding is consistent
    confidence_mapping = {}
    for i, val in enumerate(df['confidence']):
        if val not in confidence_mapping:
            confidence_mapping[val] = df_encoded['confidence_encoded'][i]
        else:
            assert confidence_mapping[val] == df_encoded['confidence_encoded'][i]

def test_date_features(sample_feature_data):
    """Test extraction of date features"""
    df = sample_feature_data.copy()
    
    # Extract month and day of year from acq_date
    df['month'] = df['acq_date'].dt.month
    df['day_of_year'] = df['acq_date'].dt.dayofyear
    
    # Check that the features were extracted correctly
    assert df['month'][0] == 1  # January
    assert df['month'][1] == 2  # February
    
    assert df['day_of_year'][0] == 1  # January 1st
    assert df['day_of_year'][1] == 46  # February 15th