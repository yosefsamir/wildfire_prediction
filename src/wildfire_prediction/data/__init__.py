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

from .weather_processing import (
    clean_weather_data_for_year,
    load_config,
    get_weather_files_for_year,
    extract_date_from_filename,
    clean_weather_file,
    merge_yearly_weather_files,
    add_grid_to_weather_data
)