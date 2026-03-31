"""Backtesting module for FFT-Trading strategy validation."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .data_fetcher import fetch_stock_data, StockData
from .fft_analysis import analyze_fft, FFTResult
from .prediction import predict_future
from .metrics import (
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_cumulative_return,
    compute_volatility,
    compute_returns_from_prices
)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    prediction_window: int = 30,
    hold_period: int = 5,
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> BacktestResult:
    """
    Run FFT-based trading strategy backtest.

    Strategy:
    - Use rolling window of prediction_window days
    - Predict next hold_period days
    - Go long if forecast slope is positive, short if negative
    - Hold for hold_period days, then exit

    Args:
        ticker: Stock symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        prediction_window: Days of historical data for FFT
        hold_period: Days to hold each position
        initial_capital: Starting capital
        transaction_cost: Cost per trade (0.001 = 0.1%)
        slippage: Price slippage (0.0005 = 0.05%)

    Returns:
        BacktestResult with full performance metrics
    """
    # Fetch data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    prices = stock_data.prices
    dates = stock_data.dates

    if len(prices) < prediction_window + hold_period:
        raise ValueError(
            f"Not enough data: need {prediction_window + hold_period} days, "
            f"got {len(prices)}"
        )

    # Track positions and equity
    capital = initial_capital
    equity_curve = [initial_capital]
    daily_returns = []
    trades = []

    # Rolling window backtest
    for window_start in range(0, len(prices) - prediction_window - hold_period, hold_period):
        window_end = window_start + prediction_window

        # Get training data
        train_prices = prices[window_start:window_end]

        # Predict next hold_period days
        predicted = predict_future(train_prices, hold_period)

        # Calculate forecast slope
        if len(predicted) < 2:
            continue

        slope = predicted[-1] - predicted[0]

        # Current price (at window end)
        current_price = prices[window_end]

        # Apply slippage to entry
        if slope > 0:
            position = 'long'
            entry_price = current_price * (1 + slippage)
        elif slope < 0:
            position = 'short'
            entry_price = current_price * (1 - slippage)
        else:
            continue  # No trade

        # Hold for hold_period days
        exit_idx = min(window_end + hold_period, len(prices) - 1)
        exit_price = prices[exit_idx]

        # Apply slippage to exit
        if position == 'long':
            exit_price *= (1 - slippage)
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            exit_price *= (1 + slippage)
            pnl_pct = (entry_price - exit_price) / entry_price

        # Transaction costs (round trip)
        pnl_pct -= 2 * transaction_cost

        # Calculate PnL
        pnl = capital * pnl_pct
        capital += pnl

        # Record trade
        trade = Trade(
            entry_date=dates[window_end],
            exit_date=dates[exit_idx],
            entry_price=entry_price,
            exit_price=exit_price,
            position=position,
            pnl=pnl,
            pnl_pct=pnl_pct * 100,
            holding_days=exit_idx - window_end
        )
        trades.append(trade)

        # Update equity curve (simplified: add PnL over hold period)
        for day in range(hold_period):
            if len(equity_curve) < len(prices):
                daily_return = pnl / hold_period
                equity_curve.append(equity_curve[-1] + daily_return)
                daily_returns.append(daily_return / equity_curve[-2])

    # Compute metrics
    total_return = (capital - initial_capital) / initial_capital * 100

    # Annualized return
    n_days = len(dates)
    annualized_return = ((capital / initial_capital) ** (365 / n_days) - 1) * 100

    # Sharpe ratio
    if daily_returns:
        sharpe = compute_sharpe_ratio(daily_returns)
    else:
        sharpe = 0.0

    # Max drawdown
    max_dd = compute_max_drawdown(equity_curve)

    # Win rate
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0.0

    return BacktestResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=capital,
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        trades=trades,
        equity_curve=equity_curve,
        daily_returns=daily_returns
    )


def walk_forward_analysis(
    ticker: str,
    start_date: str,
    end_date: str,
    window_size: int = 250,
    step_size: int = 20,
    forecast_horizon: int = 10
) -> Dict[str, List[float]]:
    """
    Perform walk-forward analysis to evaluate FFT prediction stability.

    Unlike backtesting, this focuses on prediction accuracy rather than
    trading performance.

    Args:
        ticker: Stock symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        window_size: Training window size (default: 250 trading days)
        step_size: Steps between windows (default: 20 days)
        forecast_horizon: Days to predict in each window

    Returns:
        Dict with lists:
        - window_start_dates: Window start indices
        - prediction_errors: MAPE for each window
        - directional_accuracies: Direction accuracy for each window
        - predicted_ranges: Range of predictions
        - actual_ranges: Range of actual prices
    """
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    prices = stock_data.prices
    dates = stock_data.dates

    results = {
        'window_start_dates': [],
        'prediction_errors': [],
        'directional_accuracies': [],
        'predicted_ranges': [],
        'actual_ranges': []
    }

    window_start = 0
    while window_start + window_size + forecast_horizon <= len(prices):
        window_end = window_start + window_size

        # Training data
        train_prices = prices[window_start:window_end]

        # Predict
        predicted = predict_future(train_prices, forecast_horizon)

        # Actual prices for forecast period
        actual_end = min(window_end + forecast_horizon, len(prices))
        actual = prices[window_end:actual_end]

        # Ensure same length
        min_len = min(len(predicted), len(actual))
        predicted = predicted[:min_len]
        actual = actual[:min_len]

        if not actual or not predicted:
            window_start += step_size
            continue

        # Compute MAPE
        mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100

        # Directional accuracy
        actual_directions = np.diff(actual) > 0
        predicted_directions = np.diff(predicted) > 0
        dir_acc = np.mean(actual_directions == predicted_directions) * 100

        results['window_start_dates'].append(dates[window_start])
        results['prediction_errors'].append(mape)
        results['directional_accuracies'].append(dir_acc)
        results['predicted_ranges'].append(max(predicted) - min(predicted))
        results['actual_ranges'].append(max(actual) - min(actual))

        window_start += step_size

    return results


def analyze_rolling_fft_stability(
    prices: List[float],
    window_size: int = 250,
    step_size: int = 20
) -> Dict:
    """
    Analyze stability of dominant FFT cycles across rolling windows.

    Helps determine if FFT-based predictions are reliable by checking
    if dominant frequencies remain consistent across time windows.

    Args:
        prices: Price series
        window_size: Rolling window size
        step_size: Step between windows

    Returns:
        Dict with stability metrics:
        - dominant_cycles_by_window: List of dominant periods per window
        - cycle_stability_score: 0-1 score (1 = very stable)
        - avg_amplitude_by_cycle: Average amplitude for each cycle
        - recommendation: Trading recommendation based on stability
    """
    n = len(prices)
    windows = []
    all_dominant_cycles = []

    # Collect dominant cycles for each window
    for start in range(0, n - window_size, step_size):
        window = prices[start:start + window_size]
        windows.append(window)

        fft_result = analyze_fft(window)

        # Get dominant periods (excluding DC and very long periods)
        frequencies = np.fft.fftfreq(len(window))
        amplitudes = np.abs(fft_result.fft_coefficients)

        # Sort by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]

        dominant_periods = []
        for idx in sorted_indices[1:10]:  # Skip DC
            if frequencies[idx] > 0:
                period = 1.0 / frequencies[idx]
                if 5 <= period <= 250:  # Reasonable range
                    dominant_periods.append(period)

        all_dominant_cycles.append(dominant_periods[:5])  # Top 5

    if not all_dominant_cycles:
        return {
            'dominant_cycles_by_window': [],
            'cycle_stability_score': 0.0,
            'avg_amplitude_by_cycle': {},
            'recommendation': 'Insufficient data for stability analysis'
        }

    # Analyze stability
    all_cycles = [c for cycles in all_dominant_cycles for c in cycles]

    if not all_cycles:
        return {
            'dominant_cycles_by_window': all_dominant_cycles,
            'cycle_stability_score': 0.0,
            'avg_amplitude_by_cycle': {},
            'recommendation': 'No consistent cycles found'
        }

    # Cluster similar cycles and count occurrences
    cycle_counts = {}
    for cycles in all_dominant_cycles:
        for cycle in cycles:
            # Round to nearest 5 days for clustering
            rounded = round(cycle / 5) * 5
            cycle_counts[rounded] = cycle_counts.get(rounded, 0) + 1

    # Stability score: how often do the same cycles appear?
    max_possible = len(all_dominant_cycles) * 5
    actual_occurrences = sum(cycle_counts.values())
    unique_cycles = len(cycle_counts)

    # Higher score = fewer unique cycles appearing more often
    stability_score = actual_occurrences / (unique_cycles * len(all_dominant_cycles)) if unique_cycles > 0 else 0
    stability_score = min(1.0, stability_score)

    # Recommendation
    if stability_score > 0.7:
        recommendation = 'High stability - FFT predictions likely reliable'
    elif stability_score > 0.4:
        recommendation = 'Moderate stability - Use FFT predictions with caution'
    else:
        recommendation = 'Low stability - FFT predictions may be unreliable'

    return {
        'dominant_cycles_by_window': all_dominant_cycles,
        'cycle_stability_score': round(stability_score, 3),
        'avg_amplitude_by_cycle': {k: v / len(all_dominant_cycles) for k, v in cycle_counts.items()},
        'recommendation': recommendation,
        'n_windows_analyzed': len(all_dominant_cycles)
    }


def create_backtest_report(result: BacktestResult) -> str:
    """
    Create a text report summarizing backtest results.

    Args:
        result: BacktestResult object

    Returns:
        Formatted report string
    """
    report = f"""
{'='*60}
BACKTEST REPORT - {result.ticker}
{'='*60}

Period: {result.start_date} to {result.end_date}
Initial Capital: ${result.initial_capital:,.2f}
Final Capital: ${result.final_capital:,.2f}

RETURNS
-------
Total Return: {result.total_return:.2f}%
Annualized Return: {result.annualized_return:.2f}%

RISK METRICS
------------
Sharpe Ratio: {result.sharpe_ratio:.3f}
Max Drawdown: {result.max_drawdown:.2f}%
Volatility: {compute_volatility(result.daily_returns) if result.daily_returns else 0:.2f}%

TRADING STATISTICS
------------------
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Win Rate: {result.win_rate:.2f}%

{'='*60}
"""
    return report
