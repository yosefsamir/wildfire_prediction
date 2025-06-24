import pytest
import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier

# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

@pytest.fixture
def sample_model_data():
    """Create a sample dataset for model testing"""
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic features
    X = pd.DataFrame({
        'frp_log': np.random.normal(3, 1, n_samples),
        'brightness_normalized': np.random.uniform(0, 1, n_samples),
        'confidence_encoded': np.random.choice([0, 1, 2], n_samples),
        'is_day': np.random.choice([0, 1], n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_year': np.random.randint(1, 366, n_samples)
    })
    
    # Create synthetic target (0 = no fire, 1 = fire)
    y = np.random.choice([0, 1], n_samples)
    
    return X, y

def test_model_training(sample_model_data):
    """Test that a model can be trained on the data"""
    X, y = sample_model_data
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    # Check that the model was trained successfully
    assert hasattr(model, 'estimators_')
    assert len(model.estimators_) == 10
    
    # Check that the model can make predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)

def test_model_serialization(sample_model_data, tmp_path):
    """Test that a model can be serialized and deserialized"""
    X, y = sample_model_data
    
    # Create and train a model
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Save the model
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Load the model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Check that the loaded model is the same as the original
    assert type(loaded_model) == type(model)
    assert loaded_model.n_estimators == model.n_estimators
    
    # Check that the loaded model makes the same predictions
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    assert np.array_equal(original_preds, loaded_preds)