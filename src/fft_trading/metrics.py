"""Evaluation metrics module for FFT-Trading predictions."""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .prediction import PredictionResult


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    # Forecast accuracy metrics
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float

    # Trading metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    cumulative_return: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None

    # Summary
    overall_score: Optional[float] = None


def compute_rmse(actual: List[float], predicted: List[float]) -> float:
    """
    Compute Root Mean Square Error.

    RMSE = sqrt(mean((actual - predicted)^2))

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE value (same units as input)
    """
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    if len(actual_arr) != len(predicted_arr):
        raise ValueError("Actual and predicted must have same length")

    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def compute_mae(actual: List[float], predicted: List[float]) -> float:
    """
    Compute Mean Absolute Error.

    MAE = mean(|actual - predicted|)

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAE value (same units as input)
    """
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    if len(actual_arr) != len(predicted_arr):
        raise ValueError("Actual and predicted must have same length")

    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def compute_mape(actual: List[float], predicted: List[float]) -> float:
    """
    Compute Mean Absolute Percentage Error.

    MAPE = 100 * mean(|(actual - predicted) / actual|)

    Args:
        actual: Actual values (must be non-zero)
        predicted: Predicted values

    Returns:
        MAPE as percentage
    """
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    if len(actual_arr) != len(predicted_arr):
        raise ValueError("Actual and predicted must have same length")

    # Avoid division by zero
    mask = actual_arr != 0
    if not np.any(mask):
        return float('inf')

    return float(100 * np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])))


def compute_directional_accuracy(actual: List[float], predicted: List[float]) -> float:
    """
    Compute percentage of correct direction predictions.

    Measures how often the prediction correctly identifies whether
    the price will go up or down.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Directional accuracy as percentage (0-100)
    """
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    if len(actual_arr) < 2:
        return 0.0

    # Compute actual direction changes
    actual_direction = np.diff(actual_arr) > 0

    # Compute predicted direction changes
    if len(predicted_arr) >= len(actual_arr):
        predicted_direction = np.diff(predicted_arr[:len(actual_arr)]) > 0
    else:
        # If predicted is shorter, compare what we can
        predicted_direction = np.diff(predicted_arr) > 0
        actual_direction = actual_direction[:len(predicted_direction)]

    # Calculate accuracy
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)

    return float(100 * correct / total) if total > 0 else 0.0


def compute_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> float:
    """
    Compute Sharpe ratio.

    Sharpe = (mean_return - risk_free_rate) / std_return

    Args:
        returns: Daily returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default: 2%)
        annualization_factor: Trading days per year (default: 252)

    Returns:
        Sharpe ratio (annualized)
    """
    returns_arr = np.array(returns)

    if len(returns_arr) < 2:
        return 0.0

    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize
    daily_rf = risk_free_rate / annualization_factor
    annualized_return = mean_return * annualization_factor
    annualized_std = std_return * np.sqrt(annualization_factor)

    sharpe = (annualized_return - risk_free_rate) / annualized_std

    return float(sharpe)


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """
    Compute maximum drawdown from equity curve.

    Drawdown is the peak-to-trough decline, expressed as a percentage.

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Maximum drawdown as positive percentage (0-100)
    """
    equity_arr = np.array(equity_curve)

    if len(equity_arr) < 2:
        return 0.0

    # Compute running maximum
    running_max = np.maximum.accumulate(equity_arr)

    # Compute drawdowns
    drawdowns = (running_max - equity_arr) / running_max * 100

    return float(np.max(drawdowns))


def compute_cumulative_return(equity_curve: List[float]) -> float:
    """
    Compute cumulative return from equity curve.

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Cumulative return as percentage
    """
    if len(equity_curve) < 2:
        return 0.0

    initial = equity_curve[0]
    final = equity_curve[-1]

    if initial == 0:
        return 0.0

    return float((final - initial) / initial * 100)


def compute_volatility(returns: List[float], annualization_factor: int = 252) -> float:
    """
    Compute annualized volatility (standard deviation of returns).

    Args:
        returns: Daily returns (as decimals)
        annualization_factor: Trading days per year (default: 252)

    Returns:
        Annualized volatility as percentage
    """
    returns_arr = np.array(returns)

    if len(returns_arr) < 2:
        return 0.0

    daily_volatility = np.std(returns_arr, ddof=1)
    annualized_volatility = daily_volatility * np.sqrt(annualization_factor) * 100

    return float(annualized_volatility)


def compute_returns_from_prices(prices: List[float]) -> List[float]:
    """
    Compute daily returns from price series.

    Args:
        prices: List of prices

    Returns:
        List of daily returns (as decimals)
    """
    prices_arr = np.array(prices)

    if len(prices_arr) < 2:
        return []

    returns = np.diff(prices_arr) / prices_arr[:-1]
    return returns.tolist()


def compute_win_rate(trades: List[Dict]) -> float:
    """
    Compute win rate from list of trades.

    Args:
        trades: List of dicts with 'profit' or 'pnl' key

    Returns:
        Win rate as percentage (0-100)
    """
    if not trades:
        return 0.0

    wins = 0
    for trade in trades:
        profit = trade.get('profit', trade.get('pnl', 0))
        if profit > 0:
            wins += 1

    return float(100 * wins / len(trades))


def evaluate_prediction(result: PredictionResult) -> Dict[str, float]:
    """
    Comprehensive evaluation of prediction quality.

    Computes all relevant metrics for a PredictionResult.

    Args:
        result: PredictionResult with test_prices and predicted_prices

    Returns:
        Dict with all metrics:
        - rmse, mae, mape: Accuracy metrics
        - directional_accuracy: Direction prediction accuracy
        - sharpe_ratio, max_drawdown, cumulative_return: Trading metrics
    """
    if result.predicted_prices is None:
        raise ValueError("PredictionResult must have predicted_prices")

    actual = result.test_prices
    predicted = result.predicted_prices[:len(actual)]  # Ensure same length

    # Accuracy metrics
    rmse = compute_rmse(actual, predicted)
    mae = compute_mae(actual, predicted)
    mape = compute_mape(actual, predicted)
    dir_acc = compute_directional_accuracy(actual, predicted)

    # Trading metrics
    # Compute returns from actual and predicted
    actual_returns = compute_returns_from_prices(actual)
    predicted_returns = compute_returns_from_prices(predicted)

    # Compute equity curves (starting from 100)
    actual_equity = [100]
    predicted_equity = [100]

    for r in actual_returns:
        actual_equity.append(actual_equity[-1] * (1 + r))

    for r in predicted_returns:
        predicted_equity.append(predicted_equity[-1] * (1 + r))

    sharpe = compute_sharpe_ratio(predicted_returns)
    max_dd = compute_max_drawdown(predicted_equity)
    cum_ret = compute_cumulative_return(predicted_equity)
    vol = compute_volatility(predicted_returns)

    return {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'mape': round(mape, 4),
        'directional_accuracy': round(dir_acc, 2),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_dd, 4),
        'cumulative_return': round(cum_ret, 4),
        'volatility': round(vol, 4)
    }


def compute_prediction_confidence(
    actual: List[float],
    predicted: List[float],
    lookback_window: int = 10
) -> Dict[str, float]:
    """
    Compute confidence metrics for a prediction.

    Args:
        actual: Actual values
        predicted: Predicted values
        lookback_window: Window for rolling metrics

    Returns:
        Dict with confidence metrics:
        - confidence_score: Overall confidence (0-1)
        - recent_mape: MAPE over recent window
        - stability: Consistency of errors
    """
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    errors = np.abs(actual_arr - predicted_arr)
    pct_errors = errors / np.abs(actual_arr) * 100

    # Recent performance
    recent_errors = pct_errors[-lookback_window:]
    recent_mape = float(np.mean(recent_errors))

    # Stability (inverse of error variance)
    error_variance = float(np.var(pct_errors))
    stability = 1.0 / (1.0 + error_variance)

    # Overall confidence (based on recent MAPE)
    # Lower MAPE = higher confidence
    confidence_score = 1.0 / (1.0 + recent_mape / 100)

    return {
        'confidence_score': round(confidence_score, 4),
        'recent_mape': round(recent_mape, 4),
        'stability': round(stability, 4),
        'average_error_pct': round(float(np.mean(pct_errors)), 4)
    }
