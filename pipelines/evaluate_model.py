"""Evaluation pipeline for wildfire prediction models.

This script evaluates trained models on test data and generates performance metrics and visualizations.
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
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


def evaluate_model(model, X_test, y_test, output_dir=None, model_name='model'):
    """Evaluate model performance and generate visualizations.
    
    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test target vector
        output_dir: Directory to save evaluation results
        model_name: Name of the model for file naming
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Log metrics
    logger.info(f"Model evaluation metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"{metric}: {value:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    # Generate and save visualizations if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, lw=2, 
                label=f'Precision-Recall curve (AP = {metrics["average_precision"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_precision_recall_curve.png'))
        
        # Plot confusion matrix
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0, 1], ['No Fire', 'Fire'])
        plt.yticks([0, 1], ['No Fire', 'Fire'])
        
        # Add text annotations
        thresh = np.max(cm) / 2
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i][j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        
        # Save metrics to JSON file
        metrics_path = os.path.join(output_dir, f'{model_name}_detailed_metrics.json')
        
        # Convert numpy values to Python native types for JSON serialization
        for k, v in metrics.items():
            if isinstance(v, np.floating):
                metrics[k] = float(v)
            elif isinstance(v, np.integer):
                metrics[k] = int(v)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics


def main():
    """Main function to evaluate trained models."""
    parser = argparse.ArgumentParser(description='Evaluate wildfire prediction models')
    parser.add_argument('--config', type=str, default='configs/params_xgboost.yml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, 
                        default='artifacts/models/wildfire_xgboost_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='artifacts/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--model-name', type=str, default='xgboost',
                        help='Name of the model for file naming')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model)
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    test_data = pd.read_csv(args.data)
    
    # Extract features and target
    feature_columns = config['feature_engineering'].get('feature_columns', [])
    
    # If no specific features are defined, use all available numerical features
    if not feature_columns:
        # Exclude target and non-feature columns
        exclude_cols = ['fire', 'grid_id', 'week', 'acq_date', 'latitude', 'longitude']
        feature_columns = [col for col in test_data.columns if col not in exclude_cols]
    
    # Check if all feature columns exist in the dataframe
    missing_cols = [col for col in feature_columns if col not in test_data.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        # Remove missing columns from feature list
        feature_columns = [col for col in feature_columns if col in test_data.columns]
    
    # Create feature matrix and target vector
    X_test = test_data[feature_columns]
    
    # Define target variable - we'll use 'fire' column if it exists, otherwise create it
    if 'fire' not in test_data.columns:
        # Assuming fire is indicated by non-zero FRP values
        test_data['fire'] = (test_data['frp'] > 0).astype(int)
    
    y_test = test_data['fire']
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluate_model(model, X_test, y_test, output_dir=args.output_dir, model_name=args.model_name)
    
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()