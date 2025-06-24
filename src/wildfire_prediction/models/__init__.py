"""Models module for wildfire prediction.

This module contains model implementations for wildfire prediction.
"""

from .xgboost_model import (
    create_xgboost_model,
    train_xgboost_model,
    evaluate_xgboost_model,
    save_model,
    load_model
)

from .model_utils import (
    plot_feature_importance,
    plot_permutation_importance,
    plot_shap_values,
    plot_model_comparison,
    analyze_spatial_temporal_patterns
)