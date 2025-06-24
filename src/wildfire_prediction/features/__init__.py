"""Feature engineering module.

This module contains functions for creating and transforming features for wildfire prediction.
"""

from .feature_engineering import (
    lat_lon_to_utm_grid,
    create_grid_and_time_features,
    encode_categorical_features,
    transform_numerical_features,
    drop_unnecessary_columns,
    drop_nonzero_types,
    get_all_grid_ids,
    get_all_weeks,
    build_full_dataset,
    sample_dataset
)


from .feature_weather import(
    engineer_ca_features
)