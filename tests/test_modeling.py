import pandas as pd
import numpy as np
from src.modeling import train_linear_model, train_adaboost_model

def test_train_linear_model():
    df = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
        'y': np.random.rand(100)
    })
    metrics, importance = train_linear_model(df, 'y', ['x1', 'x2'], 0.2)
    assert 'Test RMSE' in metrics
    assert 'Test MAE' in metrics
    assert 'Test R2' in metrics
    assert 'Train RMSE' in metrics
    assert len(importance) == 2

def test_train_linear_model_with_categorical():
    df = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.choice(['A', 'B'], 100),
        'y': np.random.rand(100)
    })
    metrics, importance = train_linear_model(df, 'y', ['x1', 'x2'], 0.2)
    assert 'Test RMSE' in metrics
    assert len(importance) >= 2 # x2 might be one-hot encoded to 1 column (drop_first=True) if 2 values

def test_train_adaboost_model():
    df = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
        'y': np.random.choice(['A', 'B'], 100)
    })
    metrics, importance = train_adaboost_model(df, 'y', ['x1', 'x2'], 0.2)
    assert 'Test Accuracy' in metrics
    assert 'Test Precision' in metrics
    assert 'Test F1-Score' in metrics
    assert 'Train Accuracy' in metrics
    assert len(importance) == 2
