"""XGBoost model implementation for wildfire prediction.

This module contains functions for creating, training, evaluating, and saving XGBoost models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_xgboost_model(params=None):
    """Create an XGBoost model with specified parameters.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        xgb.XGBClassifier: Configured XGBoost model
    """
    # Default parameters optimized for imbalanced wildfire data
    default_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'scale_pos_weight': 10,  # Helps with class imbalance
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'use_label_encoder': False,  # Avoid warning
        'verbosity': 0  # Reduce output noise
    }
    
    # Update default parameters with provided parameters
    if params:
        default_params.update(params)
    
    # Create and return the model
    model = xgb.XGBClassifier(**default_params)
    logger.info(f"Created XGBoost model with parameters: {default_params}")
    return model


def train_xgboost_model(X, y, params=None, test_size=0.2, random_state=42, cv_folds=5):
    """Train an XGBoost model with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        params: Dictionary of model parameters
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        
    Returns:
        tuple: (trained model, X_train, X_test, y_train, y_test, cv_scores)
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    logger.info(f"Class distribution in training set: {np.bincount(y_train)}")
    
    # Create model
    model = create_xgboost_model(params)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train the model on the full training set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    logger.info(f"Model trained with best iteration: {model.best_iteration}")
    
    return model, X_train, X_test, y_train, y_test, cv_scores


def evaluate_xgboost_model(model, X_test, y_test, output_dir=None):
    """Evaluate the XGBoost model and generate performance metrics.
    
    Args:
        model: Trained XGBoost model
        X_test: Test feature matrix
        y_test: Test target vector
        output_dir: Directory to save evaluation plots
        
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
    
    # Plot feature importance if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot and save feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=20, height=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        
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
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        logger.info(f"Evaluation plots saved to {output_dir}")
    
    return metrics


def save_model(model, filepath):
    """Save the trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path to save the model
        
    Returns:
        str: Path where model was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filepath):
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        object: Loaded model
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    # Load the model
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model