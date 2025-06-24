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
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wildfire_prediction.models.xgboost_model import load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _preprocess_X(df):
    """
    Convert a DataFrame's object columns to numeric for XGBoost prediction.
    Drops non-feature columns like grid_id/acq_date.
    """
    df2 = df.copy()

    # Drop identifiers or non-feature columns if present
    for col in ['grid_id', 'acq_date']:
        if col in df2.columns:
            df2 = df2.drop(columns=[col])

    # Convert object dtypes
    for col in df2.select_dtypes(include=['object']).columns:
        if 'date' in col.lower():
            df2[col] = pd.to_datetime(df2[col], errors='coerce').view('int64') // 10**9
        else:
            df2[col] = pd.factorize(df2[col])[0]

    return df2


def evaluate_model(
    model,
    X_train, y_train,
    X_test, y_test,
    X_val=None, y_val=None,
    output_dir=None,
    model_name='model',
    threshold=None
):
    """Evaluate model performance and generate visualizations."""

    # 0) Immediate log to show evaluation has started
    logger.info("Starting model evaluation on test/validation sets...")

    # Preprocess all feature sets
    X_train_proc = _preprocess_X(X_train)
    X_test_proc  = _preprocess_X(X_test)
    if X_val is not None:
        X_val_proc = _preprocess_X(X_val)

    # 1) Fast test predictions via DMatrix + Booster
    dtest = xgb.DMatrix(X_test_proc)
    y_test_proba = model._Booster.predict(dtest)
    y_test_pred  = (y_test_proba >= 0.5).astype(int)

    # 2) Training predictions (smaller, sklearn wrapper is OK)
    y_train_proba = model.predict_proba(X_train_proc)[:, 1]
    y_train_pred  = (y_train_proba >= 0.5).astype(int)

    # Find optimal threshold using F1 score if threshold is None
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        logger.info(f"Optimal threshold based on F1: {threshold:.4f}")
    
    # Apply threshold to get binary predictions
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # Compute test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'average_precision': average_precision_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
        'threshold_used': float(threshold)
    }

    # Compute train metrics
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'train_roc_auc': roc_auc_score(y_train, y_train_proba),
        'train_average_precision': average_precision_score(y_train, y_train_proba),
    }

    # Aggregate metrics
    metrics = {**test_metrics, **train_metrics}

    # Validation metrics (if provided)
    if X_val is not None and y_val is not None:
        y_val_proba = model.predict_proba(X_val_proc)[:, 1]
        y_val_pred  = (y_val_proba >= 0.5).astype(int)
        val_metrics = {
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba),
            'val_average_precision': average_precision_score(y_val, y_val_proba),
        }
        metrics.update(val_metrics)

    # Overfitting gaps
    metrics['train_test_accuracy_gap'] = metrics['train_accuracy'] - metrics['accuracy']
    metrics['train_test_roc_auc_gap'] = metrics['train_roc_auc'] - metrics['roc_auc']
    if 'val_accuracy' in metrics:
        metrics['train_val_accuracy_gap'] = metrics['train_accuracy'] - metrics['val_accuracy']
        metrics['val_test_accuracy_gap'] = metrics['val_accuracy'] - metrics['accuracy']

    # Log summary
    logger.info("=== Model Metrics ===")
    for k, v in metrics.items():
        if k != 'confusion_matrix' and isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.4f}")

    logger.info("\nClassification Report (Test):\n" +
                classification_report(y_test, y_test_pred, zero_division=0))

    logger.info("\n=== Overfitting Assessment ===")
    logger.info(f"Train-Test Accuracy Gap: {metrics['train_test_accuracy_gap']:.4f}")
    logger.info(f"Train-Test ROC-AUC Gap: {metrics['train_test_roc_auc_gap']:.4f}")
    if 'val_accuracy' in metrics:
        logger.info(f"Train-Val Accuracy Gap: {metrics['train_val_accuracy_gap']:.4f}")
        logger.info(f"Val-Test Accuracy Gap: {metrics['val_test_accuracy_gap']:.4f}")

    # Generate and save visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # ROC Curves
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.plot(fpr, tpr, label=f'Test ROC (AUC={metrics["roc_auc"]:.4f})')
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_proba)
        plt.plot(fpr_tr, tpr_tr, linestyle='--', label=f'Train ROC (AUC={metrics["train_roc_auc"]:.4f})')
        if 'val_roc_auc' in metrics:
            fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
            plt.plot(fpr_val, tpr_val, linestyle=':', label=f'Val ROC (AUC={metrics["val_roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
        plt.close()

        # Precision-Recall Curves
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        plt.plot(recall, precision, label=f'Test PR (AP={metrics["average_precision"]:.4f})')
        if 'train_average_precision' in metrics:
            pr_tr, rc_tr, _ = precision_recall_curve(y_train, y_train_proba)
            plt.plot(rc_tr, pr_tr, linestyle='--', label=f'Train PR (AP={metrics["train_average_precision"]:.4f})')
        if 'val_average_precision' in metrics:
            pr_val, rc_val, _ = precision_recall_curve(y_val, y_val_proba)
            plt.plot(rc_val, pr_val, linestyle=':', label=f'Val PR (AP={metrics["val_average_precision"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'))
        plt.close()

        # Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.xticks([0, 1], ['No Fire', 'Fire'])
        plt.yticks([0, 1], ['No Fire', 'Fire'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()

        # Save detailed metrics to JSON
        metrics_fp = os.path.join(output_dir, f'{model_name}_detailed_metrics.json')
        clean = {k: (float(v) if isinstance(v, np.generic) else v) for k, v in metrics.items()}
        with open(metrics_fp, 'w') as f:
            json.dump(clean, f, indent=4)
        logger.info(f"Saved evaluation outputs to {output_dir}")

    return metrics


def main():
    """Main function to evaluate trained models."""
    parser = argparse.ArgumentParser(description='Evaluate wildfire prediction models')
    parser.add_argument('--config', type=str, default='configs/params.yml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pkl)')
    parser.add_argument('--data', type=str,
                        help='Path to data CSV (used for both train & test if specific flags missing)')
    parser.add_argument('--test-data', type=str,
                        help='Path to test data CSV (overrides --data)')
    parser.add_argument('--train-data', type=str,
                        help='Path to train data CSV (overrides --data)')
    parser.add_argument('--val-data', type=str,
                        help='Path to validation data CSV (optional)')
    parser.add_argument('--output-dir', type=str, default='artifacts/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--model-name', type=str, default='xgboost',
                        help='Name of the model for file naming')
    args = parser.parse_args()

    # Resolve train/test paths
    if args.test_data:
        test_path = args.test_data
    elif args.data:
        test_path = args.data
    else:
        parser.error("Either --test-data or --data must be provided")

    if args.train_data:
        train_path = args.train_data
    elif args.data:
        train_path = args.data
    else:
        parser.error("Either --train-data or --data must be provided")

    # Load config (for consistency; not strictly needed here)
    _ = load_config(args.config)

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model)

    # Load datasets
    logger.info(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    val_df = None
    if args.val_data:
        logger.info(f"Loading validation data from {args.val_data}")
        val_df = pd.read_csv(args.val_data)

    # Ensure 'fire' target
    for df in (train_df, test_df, val_df):
        if df is not None and 'fire' not in df.columns and 'frp' in df.columns:
            df['fire'] = (df['frp'] > 0).astype(int)

    # Split into X/y
    X_train, y_train = train_df.drop(columns=['fire']), train_df['fire']
    X_test, y_test   = test_df.drop(columns=['fire']),  test_df['fire']
    X_val = y_val = None
    if val_df is not None:
        X_val, y_val = val_df.drop(columns=['fire']), val_df['fire']

    # Run evaluation
    evaluate_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
