import numpy as np
import math
from typing import Dict, Any

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prev_actual: np.ndarray = None) -> Dict[str, float]:
    """
    Computes rigorous machine learning and quantitative finance evaluation metrics.
    
    1. RMSE (Root Mean Squared Error): standard magnitude error.
    2. MAE (Mean Absolute Error): average deviation in currency.
    3. Directional Accuracy (%): Percentage of times the model correctly predicts
       whether the stock will go UP or DOWN compared to the previous trading day.
    4. Judging Score: Annualized Sharpe Ratio / Risk-adjusted return of following
       the model's directional signals compared to a 6% risk-free rate hurdle.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 1. RMSE & MAE
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # 2. Directional Accuracy
    # If prev_actual is provided, compare day-over-day changes.
    # Otherwise compare consecutive actual vs consecutive predicted movements.
    if prev_actual is not None and len(prev_actual) == len(y_true):
        actual_direction = np.sign(y_true - prev_actual)
        predicted_direction = np.sign(y_pred - prev_actual)
    else:
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))

    matching_directions = np.sum(actual_direction == predicted_direction)
    total_directions = len(actual_direction)
    directional_accuracy = float((matching_directions / total_directions) * 100.0) if total_directions > 0 else 50.0

    # 3. Mathematically Sound Judging Score (Annualized Sharpe Ratio of Signal Strategy)
    # Daily returns if taking long/short positions based on model's predicted direction:
    actual_returns = np.diff(y_true) / y_true[:-1]
    predicted_signals = np.sign(np.diff(y_pred))
    strategy_returns = predicted_signals * actual_returns

    mean_daily_return = np.mean(strategy_returns)
    std_daily_return = np.std(strategy_returns)

    # Assume 6% annual risk-free rate (6% / 252 trading days = ~0.000238 daily)
    daily_rf = 0.06 / 252.0

    if std_daily_return > 1e-8:
        # Annualized Sharpe Ratio = (Daily Excess Return / Daily Volatility) * sqrt(252)
        annualized_sharpe = float(((mean_daily_return - daily_rf) / std_daily_return) * math.sqrt(252))
        judging_score = round(max(-5.0, min(5.0, annualized_sharpe)), 2)
    else:
        judging_score = 0.0

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "directional_accuracy": round(directional_accuracy, 2),
        "judging_score": float(judging_score)
    }
