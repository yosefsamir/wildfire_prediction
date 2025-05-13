import logging
import os
from pathlib import Path
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WildfireFeaturePipeline:
    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        self.config = config or {
            'input_path': None,
            'output_dir': './processed_features',
            'feature_config': {
                'include_hot_dry': True,
                'include_spi': True,
                'include_vpd': True,
                'include_temporal': True,
                'include_compound': True
            },
            'dask_config': {
                'n_workers': 8,               # Increased for 100M records
                'threads_per_worker': 1,
                'memory_limit': '16GB',       # Increased memory
                'shuffle_method': 'disk'      # Prevent OOM
            }
        }
        
        # Validate paths
        self.input_path = Path(self.config['input_path'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Dask
        self.cluster = LocalCluster(
            n_workers=self.config['dask_config']['n_workers'],
            threads_per_worker=self.config['dask_config']['threads_per_worker'],
            memory_limit=self.config['dask_config']['memory_limit']
        )
        self.client = Client(self.cluster)
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")

    def _add_location_ids(self, ddf):
        """Create unique IDs for each exact lat/lon combination"""
        logger.info("Adding location IDs...")
        with tqdm(total=1, desc="Creating location IDs") as pbar:
            ddf = ddf.assign(
                location_id=lambda x: (
                    x['latitude'].round(6).astype(str) + 
                    '_' + 
                    x['longitude'].round(6).astype(str)
                ).astype('category')  # Better for memory with many locations
            )
            pbar.update(1)
        return ddf

    def _process_single_location(self, location_df):
        """Process data for a single location (guaranteed complete temporal data)"""
        try:
            # Convert to Pandas (now safe - all data is here)
            pdf = location_df.compute().sort_values('date')
            
            # Your feature engineering here
            # Example:
            if self.config['feature_config']['include_temporal']:
                pdf['7d_avg_temp'] = pdf['tmax'].rolling(7).mean()
            
            return pdf
            
        except Exception as e:
            logger.error(f"Failed processing location {location_df.name}: {e}")
            return pd.DataFrame()

    def _verify_partitions(self, ddf):
        """Verify no location is split across partitions"""
        logger.info("Verifying partition integrity...")
        with tqdm(total=3, desc="Verifying partitions") as pbar:
            location_counts = ddf['location_id'].value_counts().compute()
            pbar.update(1)
            
            partition_stats = ddf.map_partitions(
                lambda df: pd.Series({
                    'n_locations': df['location_id'].nunique(),
                    'min_date': df['date'].min(),
                    'max_date': df['date'].max()
                })
            ).compute()
            pbar.update(1)
            
            logger.info(f"Partition stats:\n{partition_stats}")
            if any(partition_stats['n_locations'] < len(location_counts)):
                logger.warning("Some locations are split across partitions!")
            else:
                logger.info("All locations have complete data in their partitions")
            pbar.update(1)

    def run(self):
        """Execute the full processing pipeline"""
        try:
            logger.info("Starting wildfire feature pipeline")
            
            # 1. Read input data
            logger.info(f"Reading input from {self.input_path}")
            with tqdm(total=1, desc="Reading input data") as pbar:
                ddf = dd.read_parquet(
                    self.input_path,
                    engine='pyarrow',
                    filters=[('date', '>=', '1980-01-01')]  # Optional filtering
                )
                pbar.update(1)
            
            # 2. Add location IDs
            ddf = self._add_location_ids(ddf)
            
            # 3. Ensure complete location data per partition
            logger.info("Shuffling data by location_id")
            with tqdm(total=1, desc="Shuffling data") as pbar:
                ddf = ddf.shuffle(
                    'location_id',
                    shuffle=self.config['dask_config']['shuffle_method'],
                    npartitions='auto'
                )
                pbar.update(1)
            
            # 4. Verification (optional)
            self._verify_partitions(ddf)
            
            # 5. Process each location
            logger.info("Processing individual locations")
            with tqdm(total=1, desc="Processing locations") as pbar:
                result = ddf.groupby('location_id').apply(
                    self._process_single_location,
                    meta=ddf._meta
                )
                # Track progress of computation
                progress(result)
                pbar.update(1)
            
            # 6. Write output
            logger.info(f"Writing results to {self.output_dir}")
            with tqdm(total=1, desc="Writing output") as pbar:
                result.to_parquet(
                    self.output_dir,
                    engine='pyarrow',
                    compression='snappy',
                    write_index=False,
                    partition_on=['location_id'],
                    schema={
                        'latitude': pa.float32(),
                        'longitude': pa.float32(),
                        'date': pa.date32(),
                        'tmax': pa.float32(),
                        # Add other columns as needed
                    }
                )
                pbar.update(1)
            
            logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.client.close()
            self.cluster.close()

if __name__ == "__main__":
    # Example configuration for 100M records
    config = {
        'input_path': '/data/interim/weather_master1.parquet',
        'output_dir': '/data/interim/weather_features.parquet',
        'dask_config': {
            'n_workers': 12,              # Adjust based on your cores
            'threads_per_worker': 1,      # Better for memory-bound tasks
            'memory_limit': '24GB',       # For large locations
            'shuffle_method': 'disk'      # Essential for big data
        },
        'feature_config': {
            'include_temporal': True,     # Example features
            'include_vpd': True
        }
    }
    
    pipeline = WildfireFeaturePipeline(config)
    pipeline.run()