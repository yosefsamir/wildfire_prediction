"""Prediction pipeline for XGBoost wildfire prediction model.

This script uses a trained XGBoost model to make predictions on new data.
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.wildfire_prediction.features.feature_engineering import (
    create_grid_and_time_features,
    transform_numerical_features
)
from src.wildfire_prediction.models.xgboost_model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_prediction_data(data_path, config):
    """Prepare data for prediction.
    
    Args:
        data_path: Path to data for prediction
        config: Configuration dictionary
        
    Returns:
        tuple: (X, df) feature matrix and original dataframe
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Data loaded with shape: {df.shape}")
    
    # # Apply feature engineering if needed
    # if 'grid_id' not in df.columns and all(col in df.columns for col in ['latitude', 'longitude']):
    #     logger.info("Creating grid and time features...")
    #     grid_size_km = config['feature_engineering']['grid_size_km']
    #     df = create_grid_and_time_features(df, grid_size_km=grid_size_km)
    
    # # Apply transformations to numerical features
    # if 'frp_log' not in df.columns and 'frp' in df.columns:
    #     logger.info("Transforming numerical features...")
    #     log_transform_frp = config['feature_engineering']['log_transform_frp']
    #     normalize_brightness = config['feature_engineering']['normalize_brightness']
    #     df = transform_numerical_features(df, log_transform_frp=log_transform_frp, normalize_brightness=normalize_brightness)
    
    # Get feature columns from config
    feature_columns = config['feature_engineering'].get('feature_columns', [])
    
    # If no specific features are defined, use all available numerical features
    if not feature_columns:
        # Exclude non-feature columns
        exclude_cols = ['fire', 'acq_date', 'latitude', 'longitude']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Using features: {feature_columns}")
    
    # Check if all feature columns exist in the dataframe
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        # Remove missing columns from feature list
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Create feature matrix
    X = df[feature_columns]
    
    logger.info(f"Prepared features with shape: {X.shape}")
    
    return X, df


def main():
    """Main function to make predictions using the trained XGBoost model."""
    parser = argparse.ArgumentParser(description='Make predictions with XGBoost wildfire model')
    parser.add_argument('--config', type=str, default='configs/params.yml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, 
                        default='artifacts/models/xgb_wildfire_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data for prediction')
    parser.add_argument('--output',defualt= 'artifacts/predictions', type=str, required=True,
                        help='Path to save prediction results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model)
    
    # Prepare data
    X, df = prepare_prediction_data(args.data, config)
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    # Add predictions to dataframe
    df['fire_probability'] = y_pred_proba
    df['fire_predicted'] = y_pred
    
    # Save results
    logger.info(f"Saving prediction results to {args.output}")
    df.to_csv(args.output, index=False)
    
    # Print summary
    logger.info(f"Prediction summary:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Predicted fires: {sum(y_pred)}")
    logger.info(f"Prediction pipeline completed successfully!")


if __name__ == '__main__':
    main()