import os
import json
import logging
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
def load_config(path='configs/params.yml'):
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
from wildfire_prediction.models.xgboost_model import load_model


def predict(data: pd.DataFrame) -> float:
    """
    :param data: dataframe
    :return: prediction
    """
    logger.info("predict")    

    model = load_model('artifacts/models/xgb_wildfire_model.pkl')
    
    cfg = load_config('configs/params.yml')
    
    features = list(dict.fromkeys(cfg['feature_engineering']['feature_columns']))

    X = data[features]
    
    
    dmat    = xgb.DMatrix(X)
    
    y_pred = model._Booster.predict(dmat)
    logger.info(y_pred)
    return float(y_pred[0])