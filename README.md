# Wildfire Prediction

A data science project for predicting wildfires in California using satellite data, weather information, and historical fire records.

## Project Structure

This project follows a DVC-powered structure for reproducible data science:

```
wildfire_prediction/
├── .gitignore           # Git ignore file
├── .dvcignore           # DVC ignore file
├── dvc.yaml             # DVC pipeline definition
├── dvc.lock             # DVC pipeline state
├── .dvc/                # DVC internal files
├── README.md            # Project documentation
├── environment.yml      # Conda environment specification
├── Makefile             # Common commands
├── configs/             # Configuration files
├── data/                # Data directories
│   ├── external/        # Original data from external sources
│   ├── raw/             # Initial cleaned data
│   ├── interim/         # Intermediate processed data
│   └── processed/       # Final processed data ready for modeling
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code package
│   └── wildfire_prediction/
│       ├── data/        # Data loading and processing
│       ├── features/    # Feature engineering
│       ├── models/      # Model training and prediction
│       ├── visualization/ # Visualization utilities
│       └── utils/       # Utility functions
├── pipelines/           # Pipeline scripts
├── tests/               # Unit and integration tests
├── artifacts/           # Model outputs and visualizations
│   ├── models/          # Trained models
│   ├── figures/         # Generated figures
│   └── metrics/         # Model metrics
└── docs/                # Additional documentation
```

## Setup

### Prerequisites

- Python 3.8+
- Conda or Miniconda
- DVC

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd wildfire_prediction

# Create and activate conda environment
conda env create -f environment.yml
conda activate wildfire-env

# Pull data using DVC
dvc pull
```

## Data Pipeline

The data processing pipeline consists of the following stages:

1. **Clean Fire Data**: Initial cleaning of raw fire data
2. **Preprocess Fire Data**: Feature engineering and preprocessing
3. **Visualize Fire Data**: Generate visualizations of the processed data
4. **Train Model**: Train the wildfire prediction model

To run the entire pipeline:

```bash
dvc repro
```

To run a specific stage:

```bash
dvc repro <stage-name>
```

## Data Sources

- Fire data: MODIS and VIIRS satellite fire detections
- California boundary: U.S. Census Bureau state boundaries

## Features

- Spatial grid-based features
- Temporal features (day of year, month, etc.)
- Fire radiative power (FRP) and brightness
- Day/night classification
- Confidence levels

## Models

The project uses machine learning models to predict wildfire occurrence and intensity based on the processed features.

## License

[MIT License](LICENSE)