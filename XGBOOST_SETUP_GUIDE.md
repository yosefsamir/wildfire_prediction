r# XGBoost Model Training Setup Guide

## Overview

This guide provides instructions for setting up a compatible environment for training the XGBoost wildfire prediction model. The setup addresses potential dependency issues and ensures all required packages are properly installed.

## Problem Description

The XGBoost model training pipeline may encounter compilation errors or dependency conflicts due to incompatible versions of Python, XGBoost, scikit-learn, and other dependencies. This guide provides a solution by creating a dedicated conda environment with compatible package versions.

## Solution

### 1. New Environment File

A new environment file (`environment_xgboost.yml`) has been created with carefully selected package versions to ensure compatibility:

- Python 3.9
- XGBoost 1.7.5 (compatible with scikit-learn 1.2.2)
- SHAP 0.41.0 for model interpretation
- All other required dependencies with compatible versions

### 2. Automated Setup Script

A PowerShell script (`setup_and_train_xgboost.ps1`) has been created to automate the environment setup and model training process.

## Setup Instructions

### Option 1: Using the Automated Script (Recommended)

1. Open PowerShell in the project directory
2. Run the setup script:
   ```
   .\setup_and_train_xgboost.ps1
   ```
   This will:
   - Create the conda environment
   - Activate the environment
   - Verify package versions
   - Run the XGBoost training pipeline

### Option 2: Manual Setup

1. Create the conda environment:

   ```
   conda env create -f environment_xgboost.yml
   ```

2. Activate the environment:

   ```
   conda activate wildfire_xgboost
   ```

3. Verify the installation:

   ```
   python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"
   python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
   python -c "import shap; print(f'SHAP version: {shap.__version__}')"
   ```

4. Run the training pipeline:
   ```
   python pipelines/train_xgboost_model.py --config configs/params.yml
   ```

## Troubleshooting

If you encounter issues:

1. **Conda environment creation fails**:

   - Try updating conda: `conda update -n base -c defaults conda`
   - Check for conflicts in existing environments

2. **Import errors after environment activation**:

   - Ensure you've activated the correct environment
   - Try reinstalling the specific package: `pip install xgboost==1.7.5`

3. **Training script errors**:
   - Check that all data files exist in the expected locations
   - Verify that the configs/params.yml file contains valid parameters

## Additional Notes

- The environment includes SHAP for model interpretation, which is used in the evaluation phase
- The specified versions have been tested for compatibility
- This setup maintains the existing DVC pipeline structure

## Next Steps

After successful training, you can:

1. Evaluate the model using: `python pipelines/evaluate_model.py`
2. Make predictions using: `python pipelines/predict_wildfires.py`
3. Track model performance metrics in the artifacts directory