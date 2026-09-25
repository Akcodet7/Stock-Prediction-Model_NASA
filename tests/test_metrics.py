import pytest
import numpy as np
from app.services.metrics_service import calculate_metrics

def test_rmse_and_mae_identical_arrays():
    y_true = np.array([100.0, 105.0, 110.0, 115.0])
    y_pred = np.array([100.0, 105.0, 110.0, 115.0])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["directional_accuracy"] == 100.0

def test_directional_accuracy_perfect_match():
    # Both going up on all days
    y_true = np.array([10.0, 12.0, 14.0, 16.0, 20.0])
    y_pred = np.array([11.0, 13.0, 15.0, 18.0, 22.0])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["directional_accuracy"] == 100.0

def test_directional_accuracy_opposite():
    # True goes up, Pred goes down
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([16.0, 14.0, 12.0, 10.0])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["directional_accuracy"] == 0.0

def test_judging_score_returns_float():
    y_true = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 107.0])
    y_pred = np.array([100.0, 101.5, 101.2, 103.8, 103.5, 106.8])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert isinstance(metrics["judging_score"], float)
    assert -5.0 <= metrics["judging_score"] <= 5.0
