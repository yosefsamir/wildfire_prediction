#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pipeline script for cleaning weather data by year.

This script processes weather data files for a specific year,
performs cleaning operations, and saves the cleaned data.
It now also merges daily files into yearly files and all yearly files into a master file.
"""

import os
import sys
import yaml
import pandas as pd
import argparse
from datetime import datetime
import logging

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from wildfire_prediction.data import (
    get_project_paths,
    clean_weather_data_for_year,
    merge_yearly_weather_files
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean weather data for a specific year')
    parser.add_argument('--year', type=int, required=False, 
                        help='Year to process (e.g., 2013)')
    parser.add_argument('--all-years', action='store_true',
                        help='Process all years defined in config')
    parser.add_argument('--merge-only', action='store_true',
                        help='Skip cleaning and only merge existing yearly files')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main function to clean weather data for a specific year and merge files."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        if args.config:
            config_path = args.config
        else:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs', 'config.yml')
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup logging based on config
        log_level = getattr(logging, config.get('general', {}).get('log_level', 'INFO'))
        logging.basicConfig(level=log_level, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # Get project paths
        paths = get_project_paths(config_path)
        
        # Create weather paths
        raw_weather_dir = os.path.join(paths['raw_data'], config['weather']['raw_dir'])
        interim_weather_dir = os.path.join(paths['interim_data'], config['weather']['interim_dir'])
        
        # Ensure output directory exists
        os.makedirs(interim_weather_dir, exist_ok=True)
        
        # Determine which years to process
        if args.all_years:
            years_to_process = config['weather'].get('years_to_process', [])
            if not years_to_process:
                logger.error("No years defined in config for processing.")
                return 1
        elif args.year:
            years_to_process = [args.year]
        else:
            logger.error("No year specified. Use --year or --all-years.")
            return 1
        
        logger.info(f"Processing years: {years_to_process}")
        
        # Process each year if not merge-only
        if not args.merge_only:
            for year in years_to_process:
                logger.info(f"Processing weather data for year {year}")
                logger.info(f"Raw weather directory: {raw_weather_dir}")
                logger.info(f"Output directory: {interim_weather_dir}")


                # Get output filename for the year to check if it already exists
                output_pattern = config['weather'].get('output_file_pattern', "weather_clean_%Y.csv")
                output_filename = datetime.strptime(str(year), "%Y").strftime(output_pattern)
                output_path = os.path.join(interim_weather_dir, output_filename)
                
                # Skip processing if the file already exists
                if os.path.exists(output_path):
                    logger.info(f"Yearly file for {year} already exists at {output_path}. Skipping cleaning.")
                    continue


                
                try:
                    # Process weather data for the year
                    year_output_path = clean_weather_data_for_year(
                        raw_weather_dir, 
                        interim_weather_dir, 
                        year, 
                        config_path
                    )
                    
                    # Calculate relative path for DVC
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    dvc_relative_path = os.path.relpath(os.path.dirname(year_output_path), project_root).replace(os.path.sep, '/')
                    logger.info(f"Cleaned and merged weather data for year {year} saved to: {year_output_path}")
                    logger.info(f"DVC relative path: {dvc_relative_path}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning weather data for year {year}: {str(e)}")
                    continue
        
        # Merge all yearly files into a master file
        logger.info("Merging all yearly weather files into a master file...")
        
        try:
            # Path to the interim directory for the merged master file
            master_interim_dir = paths['interim_data']
            
            # Merge all yearly files
            master_path = merge_yearly_weather_files(
                interim_weather_dir,
                master_interim_dir,
                years_to_process,
                config_path
            )
            
            logger.info(f"Master weather file created successfully at: {master_path}")
            
            # Calculate relative path for DVC
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            master_dvc_path = os.path.relpath(master_path, project_root).replace(os.path.sep, '/')
            logger.info(f"Master file DVC relative path: {master_dvc_path}")
            
        except Exception as e:
            logger.error(f"Error merging yearly weather files: {str(e)}")
            return 1
        
        logger.info("Weather data processing and merging completed successfully.")
        return 0
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())