import os
import json
import logging
import random
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from wildfire_prediction.features.feature_engineering import (
    create_grid_and_time_features,
    transform_numerical_features,

)
from wildfire_prediction.models.xgboost_model import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
def load_config(path='configs/params.yml'):
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _drop_period_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any PeriodArray columns to numeric timestamps."""
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.PeriodDtype):
            logger.warning(f"Converting PeriodArray column '{col}' to Unix timestamp")
            df[col] = df[col].dt.to_timestamp().astype('int64') // 10**9
    return df


def create_sample_data():
    """Create sample data for testing predictions."""
    logger.info("Creating sample data for testing predictions...")
    output_dir = 'data/processed/sample'
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_config('configs/params.yml')
    feature_columns = list(dict.fromkeys(cfg['feature_engineering']['feature_columns']))

    n = 200
    lat = np.random.uniform(32.5, 42.0, n)
    lon = np.random.uniform(-124.5, -114.0, n)

    # 70% summer, 30% other
    n_s = int(n * 0.7)
    n_o = n - n_s
    summer_start = datetime(2023, 6, 1)
    other_start  = datetime(2023,10,1)

    summer_days = np.random.randint(0, (datetime(2023,9,30)-summer_start).days, n_s)
    other_days  = np.random.randint(0, (datetime(2024,5,31)-other_start).days, n_o)

    dates = [summer_start + timedelta(days=int(d)) for d in summer_days] + \
            [other_start  + timedelta(days=int(d)) for d in other_days]
    random.shuffle(dates)

    df = pd.DataFrame({
        'latitude':  lat,
        'longitude': lon,
        'acq_date':  pd.to_datetime(dates)
    })

    # grid + time features
    try:
        df = create_grid_and_time_features(df, grid_size_km=cfg['feature_engineering']['grid_size_km'])
    except Exception as e:
        logger.warning(f"create_grid_and_time_features failed: {e}")
        df['grid_id'] = [f"g{i}" for i in range(n)]
        df['week']    = df['acq_date'].dt.isocalendar().week.astype(int)

    # Drop PeriodArray columns
    df = _drop_period_arrays(df)

    # Month & week as ints
    df['month'] = df['acq_date'].dt.month.astype(int)
    df['week']  = df['acq_date'].dt.isocalendar().week.astype(int)

    # Seasonal flags & cyclicals
    df['is_fire_season']      = df['month'].between(6,9).astype(int)
    df['is_santa_ana_season'] = ((df['month']>=10)|(df['month']<=2)).astype(int)
    df['month_sin'] = np.sin(2*np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2*np.pi * df['month'] / 12)
    df['week_sin']  = np.sin(2*np.pi * df['week']  / 52)
    df['week_cos']  = np.cos(2*np.pi * df['week']  / 52)

    # Season category
    conds = [
        df['month'].between(3,5),
        df['month'].between(6,8),
        df['month'].between(9,11),
        (df['month']==12)|(df['month']<=2)
    ]
    df['season'] = np.select(conds, [0,1,2,3])

    # Weather-like features
    df['tmax']   = np.clip(np.random.normal(80,10,n), 30,110)
    precip_scale = np.select(
        [df['season']==0, df['season']==1, df['season']==2, df['season']==3],
        [1.0,0.2,0.8,1.5], default=1.0
    ).astype(float)
    df['ppt']    = np.random.exponential(precip_scale, size=n)

    vpd_base     = np.select(
        [df['season']==0, df['season']==1, df['season']==2, df['season']==3],
        [15,30,20,10], default=20
    ).astype(float)
    df['vpdmax'] = np.clip(np.random.normal(vpd_base,5,n), 5,45)

    # Derived indices
    df['hot_dry_index']     = df['tmax'] * df['vpdmax'] / 100
    df['high_temp_day']     = (df['tmax']>90).astype(int)
    df['low_rain_day']      = (df['ppt']<0.1).astype(int)
    df['hot_dry_day']       = ((df['tmax']>85)&(df['ppt']<0.1)).astype(int)
    df['drought_category']  = np.clip(df['season'] + np.random.randint(-1,2,n), 0,4)
    df['vpd_extreme']       = (df['vpdmax']>35).astype(int)
    df['vpd_anomaly']       = df['vpdmax'] - vpd_base
    df['vpd_risk_category'] = np.floor(df['vpdmax']/10).clip(0,4).astype(int)

    df['tmax_7day_mean']    = df['tmax'] + np.random.normal(0,3,n)
    df['ppt_7day_mean']     = df['ppt'] * 7 * np.random.uniform(0.7,1.3,n)
    df['vpd_7day_mean']     = df['vpdmax'] + np.random.normal(0,2,n)

    df['fire_weather_index']    = (df['tmax']/10)*(df['vpdmax']/10)*(1-np.minimum(df['ppt'],1)*0.5)
    df['drought_weather_index'] = df['drought_category'] * df['hot_dry_index'] / 10
    df['fire_risk_index']       = df['fire_weather_index'] * (df['drought_category']+1) / 3

    # Synthetic fire label
    base_prob = (
        (df['tmax']-50)/60*0.3 +
        df['vpdmax']/45*0.2 +
        df['drought_category']/4*0.2 +
        df['fire_weather_index']/df['fire_weather_index'].max()*0.2 +
        df['is_fire_season']*0.1
    )
    fire_prob = np.clip(base_prob + np.random.normal(0,0.1,n), 0.01, 0.99)
    df['fire']        = (np.random.rand(n) < fire_prob).astype(int)
    df['frp']         = df['fire'] * np.random.exponential(50,n)
    df['brightness']  = 300 + df['fire'] * np.random.uniform(20,100,n)

    df = transform_numerical_features(df, log_transform_frp=True, normalize_brightness=True)

    # Drop duplicate columns if any
    cols = pd.Index(df.columns)
    dupes = cols[cols.duplicated()].unique()
    if len(dupes):
        logger.warning(f"Dropping duplicate columns: {list(dupes)}")
        df = df.loc[:, ~df.columns.duplicated()]

    # Ensure config features exist
    for c in feature_columns:
        if c not in df:
            logger.warning(f"Missing '{c}', filling with N(0,1)")
            df[c] = np.random.normal(0,1,n)

    sample_fp = os.path.join(output_dir, 'sample_data.csv')
    df.to_csv(sample_fp, index=False)
    logger.info(f"Saved sample_data to {sample_fp}")

    return df

def test_prediction(sample_data=None):
    """Test the prediction functionality with comprehensive evaluation."""
    logger.info("Testing prediction functionality...")
    preds_dir = 'artifacts/predictions'
    figs_dir  = 'artifacts/figures/predictions'
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # 1) Load or generate sample data
    if sample_data is None:
        sample_fp = 'data/processed/sample/sample_data.csv'
        sample_data = create_sample_data() if not os.path.exists(sample_fp) else pd.read_csv(sample_fp)

    # 2) Load the trained model
    model = load_model('artifacts/models/xgb_wildfire_model.pkl')

    # 3) Prepare feature matrix
    cfg = load_config('configs/params.yml')
    feat = list(dict.fromkeys(cfg['feature_engineering']['feature_columns']))
    for c in feat:
        if c not in sample_data:
            sample_data[c] = np.random.normal(0,1,len(sample_data))
    X = sample_data[feat].copy()
    for col in X.select_dtypes(include=['object']).columns:
        if 'date' in col:
            X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64')//10**9
        else:
            X[col], _ = pd.factorize(X[col])

    # 4) Predict probabilities
    dmat    = xgb.DMatrix(X)
    y_proba = model._Booster.predict(dmat)

    # 5) Determine best threshold by F1 on sample (if true labels exist)
    thr = 0.5
    prec, rec, ths = None, None, None
    f1s = None
    idx = None
    if 'fire' in sample_data:
        y_true = sample_data['fire']
        prec, rec, ths = precision_recall_curve(y_true, y_proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        idx = np.nanargmax(f1s)
        thr = float(ths[idx]) if idx < len(ths) else 0.5
        logger.info(f"Chosen threshold={thr:.3f}, F1={f1s[idx]:.3f}")

    y_pred = (y_proba >= thr).astype(int)
    sample_data['fire_probability'] = y_proba
    sample_data['fire_predicted']   = y_pred
    sample_data['threshold_used']   = thr

    # 6) Compute metrics & save JSON
    metrics = {}
    if 'fire' in sample_data:
        y_true = sample_data['fire']
        cm     = confusion_matrix(y_true, y_pred)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'confusion_matrix': cm.tolist(),
            'threshold': thr
        }
        with open(os.path.join(preds_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics → {preds_dir}/metrics.json")

        # 7a) ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label=f'Test ROC (AUC={metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(figs_dir, 'roc_curve.png'))
        plt.close()

        # 7b) Precision–Recall curve with best‑F1 dot
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, label=f'PR (AP={metrics["average_precision"]:.3f})')
        if idx is not None:
            plt.scatter(rec[idx], prec[idx],
                        color='red', s=80,
                        label=f'Best F1={f1s[idx]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(figs_dir, 'pr_curve.png'))
        plt.close()

        # 7c) Confusion matrix heatmap
        plt.figure(figsize=(6, 5))
        cm_arr = np.array(cm)
        plt.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                plt.text(j, i, cm_arr[i, j],
                         ha='center', va='center',
                         color='white' if cm_arr[i, j] > cm_arr.max()/2 else 'black')
        plt.xticks([0, 1], ['No Fire', 'Fire'])
        plt.yticks([0, 1], ['No Fire', 'Fire'])
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, 'confusion_matrix.png'))
        plt.close()

        # 7d) Probability distribution by class
        plt.figure(figsize=(8,6))
        plt.hist(y_proba[y_true==1], bins=20, alpha=0.6, label='Actual Fire')
        plt.hist(y_proba[y_true==0], bins=20, alpha=0.6, label='No Fire')
        plt.axvline(x=thr, color='red', linestyle='--', label=f'Threshold={thr:.2f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(figs_dir, 'probability_distribution.png'))
        plt.close()

    # 8) Save predictions CSV
    pred_fp = os.path.join(preds_dir, 'sample_predictions.csv')
    sample_data.to_csv(pred_fp, index=False)
    logger.info(f"Saved predictions → {pred_fp}")

    return metrics


def main():
    logger.info("=== Step 3: Creating Sample Data ===")
    try:
        create_sample_data()
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")

    logger.info("=== Step 4: Testing Prediction ===")
    try:
        test_prediction()
    except Exception as e:
        logger.error(f"Error during prediction testing: {e}")

    logger.info("=== All steps completed! ===")


if __name__ == '__main__':
    main()
