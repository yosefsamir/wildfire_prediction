"""Training pipeline for XGBoost wildfire prediction model.

This script trains an XGBoost model for wildfire prediction using processed data.
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.wildfire_prediction.features.feature_engineering import (
    drop_unnecessary_columns,
    sample_dataset
)
from src.wildfire_prediction.models.xgboost_model import (
    create_xgboost_model,
    train_xgboost_model,
    evaluate_xgboost_model,
    save_model
)

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


def prepare_features(data_path, config):
    """Prepare features for model training.
    
    Args:
        data_path: Path to processed data
        config: Configuration dictionary
        
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Sample data if needed (for development/testing)
    if config.get('sample_data', False):
        sample_size = config.get('sample_size', 100000)
        df = sample_dataset(df, sample_size=sample_size, random_state=config['train']['random_state'])
    
    logger.info(f"Data loaded with shape: {df.shape}")
    
    # Define target variable - we'll use 'fire' column if it exists, otherwise create it
    if 'fire' not in df.columns:
        # Assuming fire is indicated by non-zero FRP values
        df['fire'] = (df['frp'] > 0).astype(int)
    
    # Get feature columns from config
    feature_columns = config['feature_engineering'].get('feature_columns', [])
    
    # If no specific features are defined, use all available numerical features
    if not feature_columns:
        # Exclude target and non-feature columns
        exclude_cols = ['fire', 'grid_id', 'week', 'acq_date', 'latitude', 'longitude']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Using features: {feature_columns}")
    
    # Check if all feature columns exist in the dataframe
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        # Remove missing columns from feature list
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Create feature matrix and target vector
    X = df[feature_columns]
    y = df['fire']
    
    logger.info(f"Prepared features with shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def main():
    """Main function to train and evaluate the XGBoost model."""
    parser = argparse.ArgumentParser(description='Train XGBoost model for wildfire prediction')
    parser.add_argument('--config', type=str, default='configs/params.yml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, 
                        default='data/processed/california_wildfires.csv',
                        help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='artifacts/models',
                        help='Directory to save model and metrics')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare features
    X, y = prepare_features(args.data, config)
    
    # Train model
    logger.info("Training XGBoost model...")
    model, X_train, X_test, y_train, y_test, cv_scores = train_xgboost_model(
        X, y,
        params=config['model'].get('params', None),
        test_size=config['train']['test_size'],
        random_state=config['train']['random_state'],
        cv_folds=config['train']['cv_folds']
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics_dir = os.path.join(args.output_dir, '..', 'metrics')
    metrics = evaluate_xgboost_model(model, X_test, y_test, 
                                    output_dir=os.path.join(args.output_dir, '..', 'figures'))
    
    # Save model
    model_path = os.path.join(args.output_dir, 'wildfire_xgboost_model.pkl')
    save_model(model, model_path)
    
    # Save metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'xgboost_model_metrics.json')
    
    # Convert numpy values to Python native types for JSON serialization
    for k, v in metrics.items():
        if isinstance(v, np.floating):
            metrics[k] = float(v)
        elif isinstance(v, np.integer):
            metrics[k] = int(v)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Training pipeline completed successfully!")


if __name__ == '__main__':
    main()