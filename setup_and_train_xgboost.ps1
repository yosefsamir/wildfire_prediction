# PowerShell script to set up XGBoost environment and run the training pipeline

# Stop on first error
$ErrorActionPreference = "Stop"

# Display current directory
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green

# Step 1: Create the conda environment
Write-Host "Creating conda environment from environment_xgboost.yml..." -ForegroundColor Green
conda env create -f environment_xgboost.yml

# Step 2: Activate the environment
Write-Host "Activating wildfire_xgboost environment..." -ForegroundColor Green
# Note: In PowerShell scripts, conda activate doesn't work directly
# We need to use the following approach instead
$CondaPath = (& conda info --base)
$ActivatePath = "$CondaPath\Scripts\activate.ps1"
& $ActivatePath wildfire_xgboost

# Step 3: Verify the environment
Write-Host "Verifying Python and XGBoost versions:" -ForegroundColor Green
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
python -c "import shap; print(f'SHAP version: {shap.__version__}')"

# Step 4: Run the training pipeline
Write-Host "Running XGBoost training pipeline..." -ForegroundColor Green
python pipelines/train_xgboost_model.py --config configs/params.yml

Write-Host "Training process completed!" -ForegroundColor Green