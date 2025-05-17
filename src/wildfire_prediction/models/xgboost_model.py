"""XGBoost model implementation for wildfire prediction.

This module contains functions for creating, training, evaluating, and saving XGBoost models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import logging
import json
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_xgboost_model(params=None):
    """Create an XGBoost classifier with optimized parameters."""
    # Default parameters - improved based on hyperparameter recommendations
    default_params = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'min_child_weight': 10,
        'gamma': 1.0,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss', 'error'],
        'tree_method': 'hist',  # Faster histogram-based algorithm
        'grow_policy': 'lossguide',  # Focus on nodes with higher loss
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0,
        # `enable_categorical` only applies when passing pandas categoricals
    }

    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    logger.info(f"Created XGBoost model with parameters: {default_params}")
    return model


def train_xgboost_model(
    X, y,
    params=None,
    test_size=0.2,
    random_state=42,
    cv_folds=5,
    early_stopping_rounds=20,
    validation_data=None,
    memory_efficient=True
):
    """Train an XGBoost model with manual stratified cross-validation."""
    # 1) Preprocess object dtypes (including dates)
    X_proc = X.copy()
    for col in X_proc.select_dtypes(include=['object']).columns:
        if 'date' in col.lower():
            # convert to seconds since epoch
            X_proc[col] = pd.to_datetime(X_proc[col], errors='coerce').view('int64') // 10**9
            logger.info(f"Converted date column '{col}' to timestamp")
        else:
            X_proc[col], _ = pd.factorize(X_proc[col])
            logger.info(f"Encoded categorical column '{col}' via factorize")

    # 2) Split train/test
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    else:
        X_train, y_train = X_proc, y
        X_test = y_test = None
        logger.info(f"Using full data for training: {X_train.shape}")

    # 3) Adjust scale_pos_weight for imbalance, if not provided
    if params is None:
        params = {}
    if 'scale_pos_weight' not in params:
        neg, pos = np.bincount(y_train)
        params['scale_pos_weight'] = neg / pos
        logger.info(f"Auto scale_pos_weight set to: {params['scale_pos_weight']:.3f}")

    # 4) Create initial model
    model = create_xgboost_model(params)

    # 5) Cross‑validation with memory optimization
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    
    # Extract only booster parameters
    booster_params = {
        k: v for k, v in model.get_params().items()
        if k not in ('n_estimators', 'use_label_encoder', 'verbosity')
    }
    
    # Add memory efficiency parameters if enabled
    if memory_efficient:
        if 'tree_method' not in booster_params:
            booster_params['tree_method'] = 'hist'  # Memory-efficient histogram method
        if 'max_bin' not in booster_params:
            booster_params['max_bin'] = 256  # Reduce number of bins for histogram
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # Use indices instead of loading all data into memory at once
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        # Free memory
        import gc
        gc.collect()
        
        # Create DMatrix with memory efficiency
        dtrain = xgb.DMatrix(X_tr, label=y_tr, nthread=-1)
        dvalid = xgb.DMatrix(X_val, label=y_val, nthread=-1)
        
        # Free memory from original dataframes
        del X_tr, X_val
        gc.collect()

        bst = xgb.train(
            params=booster_params,
            dtrain=dtrain,
            num_boost_round=model.get_params()['n_estimators'],
            evals=[(dvalid, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        y_val_pred = bst.predict(dvalid)
        score = roc_auc_score(y_val, y_val_pred)
        cv_scores.append(score)
        logger.info(f"Fold {fold} ROC‑AUC: {score:.4f}")
        
        # Free memory
        del dtrain, dvalid, bst, y_val_pred
        gc.collect()

    cv_scores = np.array(cv_scores)
    logger.info(f"CV ROC‑AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 6) Final training on full train set with memory optimization
    import gc
    gc.collect()
    
    # Create DMatrix with memory efficiency
    logger.info("Creating DMatrix for final training...")
    dtrain_full = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    evals = [(dtrain_full, 'train')]
    
    # Add validation data if provided
    eval_matrices = []
    if validation_data is not None:
        X_val_ext, y_val_ext = validation_data
        logger.info("Creating validation DMatrix...")
        ev = xgb.DMatrix(X_val_ext, label=y_val_ext, nthread=-1)
        evals.append((ev, 'validation'))
        eval_matrices.append(ev)
    
    # Add test data if available
    if X_test is not None:
        logger.info("Creating test DMatrix...")
        ev_test = xgb.DMatrix(X_test, label=y_test, nthread=-1)
        evals.append((ev_test, 'test'))
        eval_matrices.append(ev_test)
    
    # Free memory before training
    gc.collect()
    
    logger.info("Starting final training...")
    final_bst = xgb.train(
        params=booster_params,
        dtrain=dtrain_full,
        num_boost_round=model.get_params()['n_estimators'],
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    # Free memory
    del dtrain_full
    for matrix in eval_matrices:
        del matrix
    gc.collect()
    model._Booster = final_bst  # attach low‑level booster
    setattr(model, '_best_iteration', final_bst.best_iteration)

    logger.info(f"Final model best_iteration: {final_bst.best_iteration}")
    return model, X_train, X_test, y_train, y_test, cv_scores


def _preprocess_X(df):
    """Helper to preprocess object columns in evaluation."""
    df2 = df.copy()
    for col in df2.select_dtypes(include=['object']).columns:
        if 'date' in col.lower():
            df2[col] = pd.to_datetime(df2[col], errors='coerce').view('int64') // 10**9
        else:
            df2[col], _ = pd.factorize(df2[col])
    return df2

def evaluate_xgboost_model(
    model,
    X_train, y_train,
    X_test,  y_test,
    X_val=None, y_val=None,
    output_dir=None,
    model_name='model',
    threshold=None
):
    """
    Evaluate a trained XGBoost model (booster) on train/test/(optional) val splits.
    Automatically picks an F1‐optimal threshold if none provided.
    Saves ROC, PR, confusion matrix plots and a detailed JSON.
    """
    logger.info("Starting model evaluation...")

    # 1) Preprocess
    X_tr = _preprocess_X(X_train)
    X_te = _preprocess_X(X_test)
    X_va = _preprocess_X(X_val) if X_val is not None else None

    # 2) Predict with the low-level Booster
    bst = model._Booster
    y_tr_p = bst.predict(xgb.DMatrix(X_tr))
    y_te_p = bst.predict(xgb.DMatrix(X_te))
    y_va_p = bst.predict(xgb.DMatrix(X_va)) if X_va is not None else None

    # 3) Determine threshold (F1-optimal on test)
    if threshold is None:
        prec, rec, ths = precision_recall_curve(y_test, y_te_p)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        idx = np.nanargmax(f1s)
        threshold = float(ths[idx]) if idx < len(ths) else 0.5
        logger.info(f"Optimal threshold by F1: {threshold:.4f}")

    # 4) Apply threshold
    y_tr_pred = (y_tr_p >= threshold).astype(int)
    y_te_pred = (y_te_p >= threshold).astype(int)
    y_va_pred = (y_va_p >= threshold).astype(int) if y_va_p is not None else None

    # 5) Compute metrics
    metrics = {}
    # Test
    metrics.update({
        'accuracy':          accuracy_score(y_test,  y_te_pred),
        'precision':         precision_score(y_test, y_te_pred, zero_division=0),
        'recall':            recall_score(y_test,    y_te_pred, zero_division=0),
        'f1':                f1_score(y_test,       y_te_pred, zero_division=0),
        'roc_auc':           roc_auc_score(y_test,  y_te_p),
        'average_precision': average_precision_score(y_test, y_te_p),
        'confusion_matrix':  confusion_matrix(y_test, y_te_pred).tolist(),
        'threshold_used':    threshold
    })
    # Train
    metrics.update({
        'train_accuracy':          accuracy_score(y_train,  y_tr_pred),
        'train_precision':         precision_score(y_train, y_tr_pred, zero_division=0),
        'train_recall':            recall_score(y_train,    y_tr_pred, zero_division=0),
        'train_f1':                f1_score(y_train,       y_tr_pred, zero_division=0),
        'train_roc_auc':           roc_auc_score(y_train,  y_tr_p),
        'train_average_precision': average_precision_score(y_train, y_tr_p),
    })
    # Validation (if given)
    if X_va is not None:
        metrics.update({
            'val_accuracy':          accuracy_score(y_val,  y_va_pred),
            'val_precision':         precision_score(y_val, y_va_pred, zero_division=0),
            'val_recall':            recall_score(y_val,    y_va_pred, zero_division=0),
            'val_f1':                f1_score(y_val,       y_va_pred, zero_division=0),
            'val_roc_auc':           roc_auc_score(y_val,  y_va_p),
            'val_average_precision': average_precision_score(y_val, y_va_p),
        })

    # 6) Overfitting gaps
    metrics['train_test_accuracy_gap'] = metrics['train_accuracy'] - metrics['accuracy']
    metrics['train_test_roc_auc_gap'] = metrics['train_roc_auc'] - metrics['roc_auc']
    if X_va is not None:
        metrics['train_val_accuracy_gap'] = metrics['train_accuracy'] - metrics['val_accuracy']
        metrics['val_test_accuracy_gap']  = metrics['val_accuracy'] - metrics['accuracy']

    # 7) Log key metrics
    logger.info("=== Model Metrics ===")
    for k, v in metrics.items():
        if k != 'confusion_matrix' and isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.4f}")
    logger.info("\nClassification Report (Test):\n" +
                classification_report(y_test, y_te_pred, zero_division=0))

    # 8) Visualizations + JSON
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # A) ROC Curves
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_te_p)
        plt.plot(fpr, tpr, label=f'Test ROC (AUC={metrics["roc_auc"]:.4f})')
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_tr_p)
        plt.plot(fpr_tr, tpr_tr, linestyle='--', label=f'Train ROC (AUC={metrics["train_roc_auc"]:.4f})')
        if 'val_roc_auc' in metrics:
            fpr_val, tpr_val, _ = roc_curve(y_val, y_va_p)
            plt.plot(fpr_val, tpr_val, linestyle=':', label=f'Val ROC (AUC={metrics["val_roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
        plt.close()

        # B) Precision-Recall Curves
        plt.figure(figsize=(8, 6))
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_te_p)
        plt.plot(recall, precision, label=f'Test PR (AP={metrics["average_precision"]:.4f})')
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        plt.scatter(
            recall[best_idx],
            precision[best_idx],
            color='red',
            s=100,
            label=f'Best F1={f1_scores[best_idx]:.3f}'
        )
        if 'train_average_precision' in metrics:
            pr_tr, rc_tr, _ = precision_recall_curve(y_train, y_tr_p)
            plt.plot(rc_tr, pr_tr, linestyle='--',
                    label=f'Train PR (AP={metrics["train_average_precision"]:.4f})')
        if 'val_average_precision' in metrics:
            pr_val, rc_val, _ = precision_recall_curve(y_val, y_va_p)
            plt.plot(rc_val, pr_val, linestyle=':',
                    label=f'Val PR (AP={metrics["val_average_precision"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curves')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'))
        plt.close()


        # c) Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(); plt.xlabel('Pred'); plt.ylabel('True')
        thresh = cm.max()/2.
        for i in range(2):
            for j in range(2):
                plt.text(j,i,cm[i,j],ha='center',
                        color='white' if cm[i,j]>thresh else 'black')
        plt.xticks([0,1],['No Fire','Fire'])
        plt.yticks([0,1],['No Fire','Fire'])
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()

        
        #D) Feature importance
        plt.figure()
        xgb.plot_importance(model._Booster, max_num_features=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
        plt.close()
        fp = os.path.join('artifacts/metrics', f'{model_name}_detailed_metrics.json')
        with open(fp, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved all evaluation outputs to {output_dir}")

    return metrics


def save_model(model, filepath):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filepath):
    """Load a trained model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {filepath}")
    return model
