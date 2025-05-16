@echo off
REM Batch script to set up XGBoost environment and run the training pipeline

echo Current directory: %cd%

REM Step 1: Create the conda environment
echo Creating conda environment from environment_xgboost.yml...
call conda env create -f environment_xgboost.yml
if %ERRORLEVEL% neq 0 (
    echo Error creating conda environment
    exit /b %ERRORLEVEL%
)

REM Step 2: Activate the environment
echo Activating wildfire_xgboost environment...
call conda activate wildfire_xgboost
if %ERRORLEVEL% neq 0 (
    echo Error activating conda environment
    exit /b %ERRORLEVEL%
)

REM Step 3: Verify the environment
echo Verifying Python and XGBoost versions:
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
python -c "import shap; print(f'SHAP version: {shap.__version__}')"

REM Step 4: Run the training pipeline
echo Running XGBoost training pipeline...
python pipelines\train_xgboost_model.py --config configs\params.yml
if %ERRORLEVEL% neq 0 (
    echo Error running training pipeline
    exit /b %ERRORLEVEL%
)

echo Training process completed!
pause