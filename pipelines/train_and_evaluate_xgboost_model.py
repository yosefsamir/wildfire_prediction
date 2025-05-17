"""Training pipeline for XGBoost wildfire prediction model."""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

import xgboost as xgb
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wildfire_prediction.features.feature_engineering import drop_unnecessary_columns
from src.wildfire_prediction.models.xgboost_model import (
    train_xgboost_model,
    evaluate_xgboost_model,
    save_model,
    _preprocess_X
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def prepare_features(data_path, cfg, sample_size=None):
    logger.info(f"Loading dataset from {data_path}")
    
    # Use chunking to reduce memory usage
    if sample_size:
        # Load only a sample for testing
        logger.info(f"Loading sample of {sample_size} rows")
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        # Load in chunks to reduce memory usage
        logger.info("Loading data in chunks to reduce memory usage")
        chunk_size = 500000  # Adjust based on available memory
        chunks = []
        
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            # Process each chunk
            if 'fire' not in chunk.columns:
                chunk['fire'] = (chunk['frp'] > 0).astype(int)
            
            # Drop unwanted columns early to save memory
            chunk = drop_unnecessary_columns(chunk)
            
            # Keep only necessary columns
            feats = cfg['feature_engineering']['feature_columns']
            valid_feats = [c for c in feats if c in chunk.columns]
            chunks.append(chunk[valid_feats + ['fire']])
            
            # Log progress
            logger.info(f"Processed chunk with {len(chunk)} rows")
            
            # Force garbage collection
            import gc
            del chunk
            gc.collect()
        
        # Combine chunks
        logger.info("Combining processed chunks...")
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    
    # Ensure binary target if not already processed in chunks
    if sample_size and 'fire' not in df.columns:
        df['fire'] = (df['frp'] > 0).astype(int)
    
    # Drop unwanted columns if not already done in chunks
    if sample_size:
        df = drop_unnecessary_columns(df)
    
    logger.info(f"Data shape: {df.shape}, class counts: {df['fire'].value_counts().to_dict()}")
    
    # Select features
    feats = cfg['feature_engineering']['feature_columns']
    missing = [c for c in feats if c not in df.columns]
    if missing:
        logger.warning(f"Missing features dropped: {missing}")
        feats = [c for c in feats if c in df.columns]
    
    X = df[feats]
    y = df['fire']
    
    # Free memory
    del df
    gc.collect()
    
    return X, y


def main():
    p = argparse.ArgumentParser("Wildfire XGBoost Train")
    p.add_argument('--config', default='configs/params.yml')
    p.add_argument('--data', default='data/processed/merged/merged_fire_weather.csv')
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--sample-size', type=int, help='Optional: Use only a sample of data for testing')
    p.add_argument('--memory-efficient', action='store_true', help='Use memory-efficient training')
    args = p.parse_args()

    # Configure memory settings
    if args.memory_efficient:
        logger.info("Running in memory-efficient mode")
        # Limit XGBoost threads
        os.environ['OMP_NUM_THREADS'] = '4'  # Adjust based on your system
        # Force garbage collection
        import gc
        gc.collect()
    
    cfg = load_config(args.config)
    X, y = prepare_features(args.data, cfg, sample_size=args.sample_size)

    if y.nunique() < 2:
        logger.error("Need both classes in data; exiting.")
        sys.exit(1)

    # 1) Split: Train+Val / Test
    rs = cfg['train']['random_state']
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y,
        test_size=cfg['train']['test_size'],
        random_state=rs,
        stratify=y
    )

    # 2) Split: Train / Validation
    val_sz = cfg['train'].get('val_size', 0.25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval,
        test_size=val_sz,
        random_state=rs,
        stratify=y_trval
    )
    logger.info(f"Shapes → train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # 3) Auto‐compute class weight if null
    params = cfg['model']['params'].copy()
    if params.get('scale_pos_weight') is None:
        neg, pos = np.bincount(y_train)
        params['scale_pos_weight'] = neg / pos
        logger.info(f"Auto scale_pos_weight set to {params['scale_pos_weight']:.3f}")

    # 4) Train with CV + early stopping (memory-efficient mode)
    model, _, _, _, _, cv_scores = train_xgboost_model(
        X_train, y_train,
        params=params,
        test_size=0.0,
        random_state=rs,
        cv_folds=cfg['train']['cv_folds'],
        early_stopping_rounds=cfg['train']['early_stopping_rounds'],
        validation_data=(X_val, y_val),
        memory_efficient=args.memory_efficient if hasattr(args, 'memory_efficient') else True
    )
    
    # Force garbage collection after training
    import gc
    gc.collect()

    # 5) Make output dirs
    md = os.path.join(args.output_dir, 'models')
    mg = os.path.join(args.output_dir, 'metrics')
    fg = os.path.join(args.output_dir, 'figures')
    for d in (md, mg, fg):
        os.makedirs(d, exist_ok=True)

    # 6) Evaluate & plot with optimal threshold
    metrics = evaluate_xgboost_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        output_dir=fg,
        model_name='xgboost'
    )

    # 7) Get raw probabilities and extend metrics if you like
    dtest = xgb.DMatrix(_preprocess_X(X_test))
    probs = model._Booster.predict(dtest)
    precision, recall, thr = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    metrics['average_precision'] = float(ap)

    # 8) Find and log best threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else metrics.get('threshold_used', 0.5)
    metrics.update({
        'best_pr_threshold': best_thr,
        'f1_at_best_threshold': float(f1_scores[best_idx])
    })
    logger.info(f"Best PR threshold: {best_thr:.3f}, F1: {f1_scores[best_idx]:.3f}")

    # 9) Save the combined metrics to disk
    metrics_fp = os.path.join(mg, 'metrics.json')
    with open(metrics_fp, 'w') as f:
        json.dump(metrics, f, default=lambda o: o.item() if isinstance(o, np.generic) else o, indent=4)
    logger.info(f"Saved metrics → {metrics_fp}")

    # 10) Save model
    model_fp = os.path.join(md, 'xgb_wildfire_model.pkl')
    save_model(model, model_fp)
    logger.info(f"Saved model → {model_fp}")

    logger.info("Training pipeline completed successfully!")





if __name__ == '__main__':
    main()
