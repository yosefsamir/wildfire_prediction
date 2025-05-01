# Wildfire Prediction Project Restructuring Plan

## Current Structure Analysis

The current project has a basic structure with:

- `.dvc` and `.dvcignore` files (DVC is already initialized)
- `data.dvc` and `models.dvc` files for data versioning
- `data/` directory with `raw/` and `processed/` subdirectories
- `src/data/` with data processing scripts
- `src/models/` with model-related scripts
- `reports/figures/` with visualization outputs

## Restructuring Plan

### 1. Create New Directory Structure

Create the following directories to match the proposed DVC-powered structure:

```
wildfire_prediction/
├── configs/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── notebooks/
├── pipelines/
├── src/wildfire_prediction/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── utils/
├── tests/
├── artifacts/
│   ├── models/
│   ├── figures/
│   └── metrics/
└── docs/
```

### 2. Refactor Code Organization

Refactor the existing code into the new structure:

1. **src/wildfire_prediction/**: Create a proper Python package

   - Move core functionality from `src/data/process_data.py` into modular components
   - Create proper `__init__.py` files
   - Organize into submodules (data, features, models, visualization, utils)

2. **pipelines/**: Create pipeline scripts that call the core library

   - `download_fire_data.py`
   - `clean_fire_data.py`
   - `preprocess_fire_data.py`
   - `visualize_fire_data.py`

3. **configs/**: Create configuration files
   - `config.yml`: Data paths and general settings
   - `params.yml`: Model hyperparameters
   - `logging.yml`: Logging configuration

### 3. Create DVC Pipeline

Create a `dvc.yaml` file defining the data processing and modeling pipeline:

```yaml
stages:
  clean_fire_data:
    cmd: python pipelines/clean_fire_data.py
    deps:
      - data/raw/fire_archive_SV-C2_607788.csv
      - data/raw/cb_2023_us_state_20m.zip
    outs:
      - data/interim/fire_clean.csv

  preprocess_fire_data:
    cmd: python pipelines/preprocess_fire_data.py
    deps:
      - data/interim/fire_clean.csv
    outs:
      - data/processed/california_wildfires.csv

  visualize_fire_data:
    cmd: python pipelines/visualize_fire_data.py
    deps:
      - data/processed/california_wildfires.csv
    outs:
      - artifacts/figures/california_wildfires.png
      - artifacts/figures/frp_distribution.png
```

### 4. Create Environment Files

Create environment files for reproducibility:

- `environment.yml`: Conda environment specification
- `Makefile`: Common commands for project management

### 5. Documentation

Create documentation files:

- Update `README.md` with project overview and setup instructions
- Create `docs/` directory with additional documentation

## Implementation Steps

1. Create the new directory structure
2. Refactor the code into the new structure
3. Create configuration files
4. Set up the DVC pipeline
5. Create environment files
6. Update documentation

## Benefits of Restructuring

- **Modularity**: Clear separation of concerns with modular code
- **Reproducibility**: DVC pipeline ensures reproducible workflows
- **Maintainability**: Organized structure makes the project easier to maintain
- **Collaboration**: Standard structure makes it easier for others to understand and contribute
- **Scalability**: Structure supports adding new data sources and models