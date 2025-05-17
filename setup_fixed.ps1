# PowerShell script to set up XGBoost environment and run the training pipeline
# Fixed version to address conda environment issues

# Stop on first error
$ErrorActionPreference = "Stop"

# Display current directory
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green

# Step 1: Check if conda is available
$condaExists = $null
try {
    $condaExists = Get-Command conda -ErrorAction SilentlyContinue
} catch {
    Write-Host "Conda not found. Please install Miniconda or Anaconda." -ForegroundColor Red
    exit 1
}

# Step 2: Get conda base path
$condaInfo = conda info --json | ConvertFrom-Json
$condaPath = $condaInfo.root_prefix
Write-Host "Conda base path: $condaPath" -ForegroundColor Green

# Step 3: Check if environment already exists and remove if needed
$envExists = conda env list | Select-String "wildfire_xgboost"
if ($envExists) {
    Write-Host "Removing existing wildfire_xgboost environment..." -ForegroundColor Yellow
    conda env remove -n wildfire_xgboost
}

# Step 4: Create environment manually with compatible packages
Write-Host "Creating conda environment with compatible packages..." -ForegroundColor Green
conda create -n wildfire_xgboost python=3.9 -y

# Step 5: Install packages one by one to avoid conflicts
Write-Host "Installing required packages..." -ForegroundColor Green
conda install -n wildfire_xgboost -c conda-forge pandas=1.5.3 numpy=1.24.3 matplotlib=3.7.1 scikit-learn=1.2.2 -y
conda install -n wildfire_xgboost -c conda-forge xgboost=1.5.0 -y
conda install -n wildfire_xgboost -c conda-forge scikit-learn=1.0.2 -y
conda install -n wildfire_xgboost -c conda-forge pip=23.0 pyyaml=6.0 -y

# Step 6: Activate environment using conda run
Write-Host "Running commands in wildfire_xgboost environment..." -ForegroundColor Green

# Step 7: Verify the environment
Write-Host "Verifying Python and XGBoost versions:" -ForegroundColor Green
conda run -n wildfire_xgboost python -c "import sys; print(f'Python version: {sys.version}')"
conda run -n wildfire_xgboost python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"
conda run -n wildfire_xgboost python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
conda run -n wildfire_xgboost python -c "import yaml; print(f'PyYAML is installed')"

# Step 8: Install SHAP using pip
Write-Host "Installing SHAP package..." -ForegroundColor Green
conda run -n wildfire_xgboost pip install shap==0.41.0

# Step 9: Verify SHAP installation
conda run -n wildfire_xgboost python -c "import shap; print(f'SHAP version: {shap.__version__}')"

# Step 10: Run the training pipeline
Write-Host "Running XGBoost training pipeline..." -ForegroundColor Green
conda run -n wildfire_xgboost python pipelines/train_xgboost_model.py --config configs/params.yml

Write-Host "Training process completed!" -ForegroundColor Green