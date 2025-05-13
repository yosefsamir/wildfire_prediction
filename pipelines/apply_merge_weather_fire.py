import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import os
import sys
import concurrent.futures
from functools import partial

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

# Import functions from the main module
from wildfire_prediction.data import (
    load_and_prepare_fire_data,
    process_year,
    WEATHER_PATH,
    FIRE_PATH, 
    OUTPUT_PATH,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fire_weather_merge')


def process_year_wrapper(year, fire_df):
    """Wrapper function for parallel processing of years"""
    try:
        logger.info(f"Starting processing for year {year}")
        result = process_year(year, fire_df)
        logger.info(f"Completed processing for year {year}")
        return result
    except Exception as e:
        logger.error(f"Error processing year {year}: {str(e)}", exc_info=True)
        return pd.DataFrame()


def main():
    # Load and prepare data
    fire_df, fire_coords = load_and_prepare_fire_data()
    
    # Years to process
    years_to_process = range(2013, 2025)  
    
   
    merged_chunks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        # Create a partial function with the fire_df parameter
        process_func = partial(process_year_wrapper, fire_df=fire_df)
        
        # Submit all years to the executor and track with tqdm
        future_to_year = {executor.submit(process_func, year): year for year in years_to_process}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_year), total=len(future_to_year),  desc="Processing years"):
            year = future_to_year[future]
            try:
                year_result = future.result()
                if not year_result.empty:
                    logger.info(f"Adding results from year {year} ({len(year_result)} records)")
                    merged_chunks.append(year_result)
                else:
                    logger.warning(f"Empty result for year {year}")
            except Exception as e:
                logger.error(f"Year {year} generated an exception: {str(e)}")
    
    # Combine results
    if merged_chunks:
        logger.info(f"Combining {len(merged_chunks)} year chunks")
        merged_df = pd.concat(merged_chunks)
        
        # Save output
        logger.info(f"Saving merged data ({len(merged_df)} records)")
        merged_df.to_csv(OUTPUT_PATH, index=False)
        
        # Validation
        logger.info("Merge completed successfully!")
        logger.info(f"Final dataset shape: {merged_df.shape}")
        logger.info("Missing values summary:")
        for col in ['tmax', 'ppt', 'vbdmax']:
            missing = merged_df[col].isna().sum()
            logger.info(f"{col}: {missing} missing ({missing/len(merged_df):.2%})")
    else:
        logger.error("No data was processed successfully!")

if __name__ == "__main__":
    main()