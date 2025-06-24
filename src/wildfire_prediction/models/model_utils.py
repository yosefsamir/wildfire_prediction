"""Utility functions for wildfire prediction models.

This module contains utility functions for model interpretation and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance
import shap
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_feature_importance(model, X, feature_names=None, top_n=20, output_dir=None, filename='feature_importance.png'):
    """Plot feature importance for the model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        X: Feature matrix used for training
        feature_names: List of feature names
        top_n: Number of top features to display
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        pd.DataFrame: DataFrame with feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Create DataFrame with feature importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance
    importances = importances.sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importances.head(top_n))
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        logger.info(f"Feature importance plot saved to {os.path.join(output_dir, filename)}")
    
    plt.close()
    return importances


def plot_permutation_importance(model, X, y, feature_names=None, n_repeats=10, 
                              top_n=20, output_dir=None, filename='permutation_importance.png'):
    """Plot permutation feature importance for the model.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        n_repeats: Number of times to permute each feature
        top_n: Number of top features to display
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        pd.DataFrame: DataFrame with permutation importances
    """
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Calculate permutation importance
    logger.info("Calculating permutation importance...")
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    
    # Create DataFrame with permutation importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    })
    
    # Sort by importance
    importances = importances.sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importances.head(top_n))
    plt.title(f'Top {top_n} Permutation Feature Importances')
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        logger.info(f"Permutation importance plot saved to {os.path.join(output_dir, filename)}")
    
    plt.close()
    return importances


def plot_shap_values(model, X, feature_names=None, max_display=20, 
                   output_dir=None, filename_prefix='shap'):
    """Plot SHAP values for model interpretation.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        max_display: Maximum number of features to display
        output_dir: Directory to save the plots
        filename_prefix: Prefix for saved plot filenames
        
    Returns:
        tuple: (explainer, shap_values)
    """
    try:
        # Get feature names if not provided
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # Create explainer
        logger.info("Creating SHAP explainer...")
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = explainer(X)
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{filename_prefix}_summary.png'))
            logger.info(f"SHAP summary plot saved to {os.path.join(output_dir, f'{filename_prefix}_summary.png')}")
        
        plt.close()
        
        # Plot bar summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', max_display=max_display, show=False)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{filename_prefix}_bar.png'))
            logger.info(f"SHAP bar plot saved to {os.path.join(output_dir, f'{filename_prefix}_bar.png')}")
        
        plt.close()
        
        return explainer, shap_values
    
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {str(e)}")
        return None, None


def plot_model_comparison(models, X_test, y_test, model_names=None, 
                        output_dir=None, filename='model_comparison.png'):
    """Plot ROC and Precision-Recall curves for multiple models.
    
    Args:
        models: List of trained models
        X_test: Test feature matrix
        y_test: Test target vector
        model_names: List of model names
        output_dir: Directory to save the plots
        filename: Filename for the saved plot
        
    Returns:
        dict: Dictionary with AUC and AP scores for each model
    """
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(models))]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Dictionary to store scores
    scores = {}
    
    # Plot ROC curves
    for model, name in zip(models, model_names):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap_score = auc(recall, precision)
        
        ax2.plot(recall, precision, lw=2, label=f'{name} (AP = {ap_score:.4f})')
        
        # Store scores
        scores[name] = {'roc_auc': roc_auc, 'average_precision': ap_score}
    
    # ROC curve plot settings
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall curve plot settings
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        logger.info(f"Model comparison plot saved to {os.path.join(output_dir, filename)}")
    
    plt.close()
    return scores


def analyze_spatial_temporal_patterns(predictions_df, output_dir=None):
    """Analyze spatial and temporal patterns in wildfire predictions.
    
    Args:
        predictions_df: DataFrame with predictions, must contain grid_id, week, and fire_probability columns
        output_dir: Directory to save the plots
        
    Returns:
        tuple: (spatial_patterns, temporal_patterns) DataFrames
    """
    if not all(col in predictions_df.columns for col in ['grid_id', 'week', 'fire_probability']):
        logger.error("DataFrame must contain grid_id, week, and fire_probability columns")
        return None, None
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Analyze spatial patterns (by grid_id)
    spatial_patterns = predictions_df.groupby('grid_id').agg({
        'fire_probability': ['mean', 'std', 'count'],
        'fire_predicted': ['sum', 'mean']
    }).reset_index()
    
    spatial_patterns.columns = ['grid_id', 'mean_probability', 'std_probability', 
                              'sample_count', 'predicted_fires', 'fire_rate']
    
    # Sort by fire rate
    spatial_patterns = spatial_patterns.sort_values('fire_rate', ascending=False)
    
    # Plot top grid cells by fire rate
    plt.figure(figsize=(12, 8))
    top_grids = spatial_patterns.head(20)
    sns.barplot(x='fire_rate', y='grid_id', data=top_grids)
    plt.title('Top 20 Grid Cells by Fire Rate')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'top_fire_rate_grids.png'))
    
    plt.close()
    
    # Analyze temporal patterns (by week)
    temporal_patterns = predictions_df.groupby('week').agg({
        'fire_probability': ['mean', 'std', 'count'],
        'fire_predicted': ['sum', 'mean']
    }).reset_index()
    
    temporal_patterns.columns = ['week', 'mean_probability', 'std_probability', 
                               'sample_count', 'predicted_fires', 'fire_rate']
    
    # Convert week to datetime for better plotting
    if isinstance(temporal_patterns['week'].iloc[0], str):
        temporal_patterns['week'] = pd.to_datetime(temporal_patterns['week'])
    
    # Sort by week
    temporal_patterns = temporal_patterns.sort_values('week')
    
    # Plot temporal patterns
    plt.figure(figsize=(15, 8))
    plt.plot(temporal_patterns['week'], temporal_patterns['fire_rate'], 'o-')
    plt.title('Temporal Pattern of Fire Rate')
    plt.xlabel('Week')
    plt.ylabel('Fire Rate')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'temporal_fire_patterns.png'))
    
    plt.close()
    
    return spatial_patterns, temporal_patterns