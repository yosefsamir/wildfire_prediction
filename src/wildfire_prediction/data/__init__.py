"""Data loading and processing module.

This module contains functions for loading, cleaning, and processing wildfire data.
"""

from .process_data import (
    get_project_paths,
    load_and_clean_data,
    validate_coordinates,
    convert_to_geodataframe,
    load_california_boundary,
    filter_california_data,
    save_processed_data
)