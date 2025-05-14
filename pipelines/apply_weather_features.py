#!/usr/bin/env python
"""Weather Feature Application Script.

This script applies weather feature engineering to the master weather dataset.
It processes the data in chunks based on time periods to efficiently
handle large datasets, then combines all processed data into a single parquet file.
"""
import os
import logging
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import math
import json
import sys
import tempfile


src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)
# Import the weather feature engineering functions
from wildfire_prediction.features import engineer_ca_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_chunk(chunk_data, start_date, end_date):
    """
    Process a single chunk of the weather data.
    
    Args:
        chunk_data: DataFrame with weather data for a specific date range
        start_date: Start date of the chunk
        end_date: End date of the chunk
        
    Returns:
        Processed DataFrame or None if error
    """
    # Apply weather feature engineering
    logger.info(f"Processing chunk: Date range [{start_date} to {end_date}]")
    
    # Skip if the chunk is empty
    if chunk_data.empty:
        logger.warning(f"Empty chunk found: Date range [{start_date} to {end_date}]")
        return None
    
    # Apply feature engineering
    try:
        processed_data = engineer_ca_features(chunk_data)
        
        # Add date range identifier (useful for debugging and analysis)
        processed_data['date_bin'] = f"{start_date}_{end_date}"
        
        logger.info(f"Processed chunk with {len(processed_data)} rows")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return None


def process_and_write_chunk(args):
    """
    Process a chunk and write it to a temporary parquet file.
    
    Args:
        args: Tuple of (start_date, end_date, input_file, temp_dir)
        
    Returns:
        Tuple of (temp_file_path, number of rows processed) or (None, 0) if error
    """
    start_date, end_date, input_file, temp_dir = args
    
    # Create a temporary file path for this chunk's output
    temp_file = Path(temp_dir) / f"temp_chunk_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    
    try:
        # Load data directly using pandas with date filters
        filters = [('date', '>=', pd.Timestamp(start_date)), 
                   ('date', '<=', pd.Timestamp(end_date))]
        
        chunk_data = pd.read_parquet(input_file, filters=filters)
        
        if chunk_data.empty:
            logger.warning(f"Empty chunk for date range [{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}]")
            return None, 0
        
        # Process the chunk
        processed_data = process_chunk(
            chunk_data, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # If processing was successful, write to temporary parquet file
        if processed_data is not None:
            processed_data.to_parquet(
                temp_file,
                index=False,
                compression='snappy',
                engine='pyarrow'
            )
            
            return str(temp_file), len(processed_data)
            
    except Exception as e:
        logger.error(f"Error processing chunk [{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}]: {str(e)}")
    
    return None, 0


def main():
    """Main function to apply weather features to the entire dataset."""
    start_time = time.time()
    
    # Configure paths
    root_dir = Path(__file__).resolve().parent.parent
    input_file = root_dir / "data" / "interim" / "weather_master.parquet"
    output_directory = root_dir / "data" / "interim"
    output_file = output_directory / "weather_features.parquet"
    
    # Create output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for processed chunks
    temp_dir = output_directory / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing weather data from {input_file}")
    logger.info(f"Output will be saved to {output_file}")
    
    # Verify file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Read parquet metadata to get schema and statistics without loading the whole file
    parquet_file = pq.ParquetFile(input_file)
    metadata = parquet_file.metadata
    
    logger.info(f"Total rows in weather master file: {metadata.num_rows:,}")
    
    try:
        # Read date range from the metadata if available, otherwise read from file
        # This is faster than loading the entire dataset just to determine date range
        table = pq.read_table(input_file, columns=['date'])
        df_date = table.to_pandas()
        
        # Ensure date column is datetime type
        if 'date' not in df_date.columns:
            logger.error("No 'date' column found in the input file. Cannot chunk by date.")
            return
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_date['date']):
            df_date['date'] = pd.to_datetime(df_date['date'])
        
        # Get min/max date
        min_date = df_date['date'].min()
        max_date = df_date['date'].max()
        
        logger.info(f"Data date range: [{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}]")
        
        # Define time chunks - 7 day intervals
        days_delta = (max_date - min_date).days
        num_chunks = (days_delta // 7) + 1  # 7-day chunks
        
        logger.info(f"Creating {num_chunks} chunks of 7-day intervals")
        
        # Remove any existing output file
        if output_file.exists():
            logger.info(f"Removing existing output file: {output_file}")
            import shutil
            if os.path.isdir(output_file):
                shutil.rmtree(output_file)
            else:
                os.remove(output_file)
                
        # Track statistics for summary
        processed_chunk_count = 0
        total_rows_processed = 0
        
        # Prepare all chunk processing tasks - 7-day intervals
        chunks_to_process = []
        # Use range(num_chunks) for all chunks or range(10) for testing with 10 chunks
        for i in tqdm(range(num_chunks), desc="Preparing chunks", unit="chunk"):
            start_date = min_date + pd.Timedelta(days=i*7)
            # Make sure end_date doesn't exceed max_date
            end_date = min(min_date + pd.Timedelta(days=(i+1)*7 - 1), max_date)
            
            # Add to list of chunks to process
            chunks_to_process.append((start_date, end_date, input_file, temp_dir))
        
        logger.info(f"Found {len(chunks_to_process)} 7-day chunks to process")
        
        # Number of workers should be adjusted based on available CPU cores
        max_workers = min(8, os.cpu_count() or 4)
        logger.info(f"Using {max_workers} workers for parallel processing")
        
        # Process chunks in parallel
        temp_files = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_and_write_chunk, args) for args in chunks_to_process]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                temp_file, rows_processed = future.result()
                if temp_file is not None and rows_processed > 0:
                    temp_files.append(temp_file)
                    processed_chunk_count += 1
                    total_rows_processed += rows_processed
        
        # After all chunks are processed, combine the temporary files
        if temp_files:
            logger.info(f"Combining {len(temp_files)} processed chunks into final output")
            
            try:
                # Get schema from first file by reading the table
                first_table = pq.read_table(temp_files[0])
                schema = first_table.schema
                
                # Create writer with the schema
                with pq.ParquetWriter(
                    output_file,
                    schema=schema,
                    compression='snappy',
                    version='2.6'
                ) as writer:
                    
                    # Write the first table
                    writer.write_table(first_table)
                    
                    # Append remaining files
                    for temp_file in tqdm(temp_files[1:], desc="Combining files"):
                        try:
                            table = pq.read_table(temp_file)
                            writer.write_table(table)
                        except Exception as e:
                            logger.error(f"Error reading temporary file {temp_file}: {str(e)}")
                
                logger.info(f"Successfully saved all processed data to {output_file}")
                
            except Exception as e:
                logger.error(f"Error during final file combination: {str(e)}")
                raise
            
            # Clean up temporary files
            logger.info("Cleaning up temporary files")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
            
            # Remove temp directory if empty
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            # Create a summary file with metadata about the processing
            summary = {
                'input_file': str(input_file),
                'output_file': str(output_file),
                'total_chunks_processed': processed_chunk_count,
                'total_rows_processed': total_rows_processed,
                'chunk_interval_days': 7,  # 7-day chunks
                'date_range': [min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d')],
                'total_date_range_days': days_delta,
                'processing_time_seconds': time.time() - start_time
            }
            
            # Save summary
            summary_file = output_directory / "weather_features_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Processed {processed_chunk_count} chunks with {total_rows_processed} total rows")
            logger.info(f"Summary saved to {summary_file}")
            
        else:
            logger.warning("No data was processed. Check for errors.")
            summary = {
                'input_file': str(input_file),
                'error': "No data was processed",
            }
            
            # Save summary even if no data was processed
            summary_file = output_directory / "weather_features_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
    
    # Calculate and log total processing time
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
