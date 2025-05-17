# Wildfire Prediction Models

This directory contains model implementations for wildfire prediction.

## XGBoost Model

The XGBoost model is implemented in `xgboost_model.py` and provides the following functionality:

- `create_xgboost_model()`: Creates an XGBoost classifier with optimized parameters for wildfire prediction
- `train_xgboost_model()`: Trains the model with cross-validation
- `evaluate_xgboost_model()`: Evaluates model performance and generates metrics
- `save_model()`: Saves the trained model to disk
- `load_model()`: Loads a trained model from disk

## Usage

### Training

To train the XGBoost model, use the training pipeline:

```bash
python pipelines/train_xgboost_model.py --config configs/params_xgboost.yml --data data/processed/california_wildfires.csv
```

This will:

1. Load and prepare the data
2. Train the XGBoost model with cross-validation
3. Evaluate the model on a test set
4. Save the model and metrics

### Prediction

To make predictions with a trained model:

```bash
python pipelines/predict_wildfires.py --model artifacts/models/wildfire_xgboost_model.pkl --data new_data.csv --output predictions.csv
```

### Evaluation

To evaluate a trained model with comprehensive overfitting analysis:

```bash
python pipelines/evaluate_model.py \
    --model artifacts/models/wildfire_xgboost_model.pkl \
    --train-data data/processed/train_data.csv \
    --test-data data/processed/test_data.csv \
    --val-data data/processed/val_data.csv \
    --output-dir artifacts/evaluation \
    --model-name xgboost
```

The validation data is optional but recommended for better overfitting analysis.

## Model Features

The XGBoost model uses the following features:

- `frp_log`: Log-transformed Fire Radiative Power
- `brightness_normalized`: Normalized brightness values
- `vpdmax`: Maximum vapor pressure deficit (weather variable)
- `tmax`: Maximum temperature (weather variable)
- `ppt`: Precipitation (weather variable)
- Spatial-temporal features from `grid_id` and `week`

## Performance

The XGBoost model is well-suited for this wildfire prediction task because:

1. It handles imbalanced datasets effectively with `scale_pos_weight` parameter
2. It works well with the spatial-temporal features from the grid system
3. It can capture non-linear relationships between weather variables and fire occurrence
4. It provides feature importance rankings to understand key predictors