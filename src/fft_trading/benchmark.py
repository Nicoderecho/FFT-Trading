"""Benchmark framework for systematic comparison of FFT prediction methods.

Compares baseline vs auto-N vs stability-weighted vs ensemble across
multiple tickers using walk-forward evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
from datetime import datetime

from src.fft_trading.data_fetcher import fetch_stock_data
from src.fft_trading.prediction import (
    predict_future_with_trend,
    find_optimal_n_components,
    compute_stability_weights,
)
from src.fft_trading.ensemble import ensemble_predict


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single method on a single ticker."""
    ticker: str
    method_name: str
    params: Dict
    mean_mape: float
    median_mape: float
    mean_dir_accuracy: float
    n_windows: int
    window_results: List[Dict] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Full benchmark report across methods and tickers."""
    results: List[BenchmarkResult]
    tickers: List[str]
    method_names: List[str]
    timestamp: str = ""


def _walk_forward_evaluate(
    prices: List[float],
    predict_fn: Callable,
    window_size: int = 1260,
    forecast_horizon: int = 20,
    step_size: int = 60,
    max_folds: int = 20
) -> Dict:
    """
    Run walk-forward evaluation on a prediction function.

    Args:
        prices: Full price series
        predict_fn: Function(train_prices, prediction_days) -> List[float]
        window_size: Training window size
        forecast_horizon: Days to predict per fold
        step_size: Stride between folds
        max_folds: Maximum number of evaluation folds

    Returns:
        Dict with 'mape_values', 'dir_acc_values', 'n_windows'
    """
    n = len(prices)
    mape_values = []
    dir_acc_values = []

    fold = 0
    pos = window_size
    while pos + forecast_horizon <= n and fold < max_folds:
        train = prices[pos - window_size:pos]
        actual = prices[pos:pos + forecast_horizon]

        try:
            predicted = predict_fn(train, forecast_horizon)
        except Exception:
            pos += step_size
            fold += 1
            continue

        min_len = min(len(predicted), len(actual))
        if min_len < 2:
            pos += step_size
            fold += 1
            continue

        pred = np.array(predicted[:min_len])
        act = np.array(actual[:min_len])

        mape = float(np.mean(np.abs((act - pred) / act)) * 100)
        mape_values.append(mape)

        act_dir = np.diff(act) > 0
        pred_dir = np.diff(pred) > 0
        dir_acc = float(np.mean(act_dir == pred_dir) * 100)
        dir_acc_values.append(dir_acc)

        pos += step_size
        fold += 1

    return {
        'mape_values': mape_values,
        'dir_acc_values': dir_acc_values,
        'n_windows': len(mape_values)
    }


def _make_baseline_fn(n_components=10, trend_type='log', max_cycle_ratio=0.33):
    """Create a baseline prediction function."""
    def fn(train_prices, prediction_days):
        predicted, _ = predict_future_with_trend(
            train_prices, prediction_days,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
        return predicted
    return fn


def _make_auto_n_fn(trend_type='log', max_cycle_ratio=0.33):
    """Create prediction function with auto-optimized N components."""
    def fn(train_prices, prediction_days):
        best_n, _ = find_optimal_n_components(
            train_prices,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
        predicted, _ = predict_future_with_trend(
            train_prices, prediction_days,
            n_components=best_n,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
        return predicted
    return fn


def _make_stability_fn(n_components=10, trend_type='log', max_cycle_ratio=0.33):
    """Create prediction function with stability-weighted components."""
    def fn(train_prices, prediction_days):
        stab_weights = compute_stability_weights(train_prices)
        predicted, _ = predict_future_with_trend(
            train_prices, prediction_days,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio,
            stability_weights=stab_weights
        )
        return predicted
    return fn


def _make_ensemble_fn(
    full_prices: List[float],
    n_components=10, trend_type='log', max_cycle_ratio=0.33,
    weighting='performance'
):
    """Create ensemble prediction function.

    Note: ensemble_predict needs the full price history to select
    sub-windows, so we pass it in at creation time.
    """
    def fn(train_prices, prediction_days):
        result = ensemble_predict(
            train_prices, prediction_days,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio,
            weighting=weighting
        )
        return result.predicted_prices
    return fn


def run_benchmark(
    tickers: List[str] = None,
    methods: List[str] = None,
    start_date: str = '2015-01-01',
    end_date: str = None,
    forecast_horizon: int = 20,
    window_size: int = 1260,
    step_size: int = 60,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33,
    n_components: int = 10,
    max_folds: int = 20
) -> BenchmarkReport:
    """
    Systematic comparison of prediction methods across tickers.

    Args:
        tickers: Stock symbols to test (default: ['^GSPC', 'AAPL', 'MSFT'])
        methods: Methods to compare. Options:
            'baseline' — fixed n_components, log trend
            'auto_n' — cross-validated n_components
            'stability' — stability-weighted component selection
            'ensemble' — multi-window ensemble
            'full' — auto_n + stability + ensemble combined
        start_date: Data start date
        end_date: Data end date (default: today)
        forecast_horizon: Days to predict per evaluation fold
        window_size: Training window for walk-forward
        step_size: Stride between folds
        trend_type: Trend extraction method
        max_cycle_ratio: Period filter ratio
        n_components: Baseline number of components
        max_folds: Maximum evaluation folds per ticker

    Returns:
        BenchmarkReport with results for all methods and tickers
    """
    if tickers is None:
        tickers = ['^GSPC', 'AAPL', 'MSFT']
    if methods is None:
        methods = ['baseline', 'auto_n', 'stability', 'ensemble']
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    all_results = []

    for ticker in tickers:
        print(f"\n  Benchmarking {ticker}...")
        try:
            stock_data = fetch_stock_data(ticker, start_date, end_date)
        except Exception as e:
            print(f"    Failed to fetch {ticker}: {e}")
            continue

        prices = stock_data.prices
        if len(prices) < window_size + forecast_horizon + 50:
            print(f"    Insufficient data for {ticker} ({len(prices)} points)")
            continue

        # Build prediction functions for each method
        method_fns = {}
        if 'baseline' in methods:
            method_fns['baseline'] = _make_baseline_fn(
                n_components, trend_type, max_cycle_ratio
            )
        if 'auto_n' in methods:
            method_fns['auto_n'] = _make_auto_n_fn(trend_type, max_cycle_ratio)
        if 'stability' in methods:
            method_fns['stability'] = _make_stability_fn(
                n_components, trend_type, max_cycle_ratio
            )
        if 'ensemble' in methods:
            method_fns['ensemble'] = _make_ensemble_fn(
                prices, n_components, trend_type, max_cycle_ratio
            )

        for method_name, predict_fn in method_fns.items():
            print(f"    Method: {method_name}...", end=" ", flush=True)

            eval_result = _walk_forward_evaluate(
                prices, predict_fn,
                window_size=window_size,
                forecast_horizon=forecast_horizon,
                step_size=step_size,
                max_folds=max_folds
            )

            if eval_result['n_windows'] == 0:
                print("no valid windows")
                continue

            mape_arr = eval_result['mape_values']
            dir_arr = eval_result['dir_acc_values']

            result = BenchmarkResult(
                ticker=ticker,
                method_name=method_name,
                params={
                    'n_components': n_components,
                    'trend_type': trend_type,
                    'max_cycle_ratio': max_cycle_ratio,
                    'window_size': window_size,
                    'forecast_horizon': forecast_horizon
                },
                mean_mape=round(float(np.mean(mape_arr)), 3),
                median_mape=round(float(np.median(mape_arr)), 3),
                mean_dir_accuracy=round(float(np.mean(dir_arr)), 2) if dir_arr else 50.0,
                n_windows=eval_result['n_windows'],
                window_results=[
                    {'mape': m, 'dir_acc': d}
                    for m, d in zip(mape_arr, dir_arr)
                ]
            )
            all_results.append(result)
            print(f"MAPE={result.mean_mape:.1f}%, DirAcc={result.mean_dir_accuracy:.1f}%")

    return BenchmarkReport(
        results=all_results,
        tickers=tickers,
        method_names=methods,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )


def create_benchmark_report(report: BenchmarkReport) -> str:
    """Generate a human-readable comparison table."""
    lines = [
        f"\n{'='*70}",
        f"FFT Prediction Benchmark Report",
        f"Generated: {report.timestamp}",
        f"{'='*70}",
    ]

    # Group by ticker
    for ticker in report.tickers:
        ticker_results = [r for r in report.results if r.ticker == ticker]
        if not ticker_results:
            continue

        lines.append(f"\n  {ticker}")
        lines.append(f"  {'─'*60}")
        lines.append(f"  {'Method':<15} {'MAPE%':>8} {'Med.MAPE%':>10} {'DirAcc%':>8} {'Folds':>6}")
        lines.append(f"  {'─'*60}")

        # Sort by mean MAPE (lower is better)
        ticker_results.sort(key=lambda r: r.mean_mape)

        for i, r in enumerate(ticker_results):
            marker = " <-- best" if i == 0 else ""
            lines.append(
                f"  {r.method_name:<15} "
                f"{r.mean_mape:>7.2f} "
                f"{r.median_mape:>9.2f} "
                f"{r.mean_dir_accuracy:>7.1f} "
                f"{r.n_windows:>5}"
                f"{marker}"
            )

        lines.append(f"  {'─'*60}")

    # Summary: best method per ticker
    lines.append(f"\n  Summary: Best method per ticker")
    lines.append(f"  {'─'*40}")
    for ticker in report.tickers:
        ticker_results = [r for r in report.results if r.ticker == ticker]
        if ticker_results:
            best = min(ticker_results, key=lambda r: r.mean_mape)
            lines.append(f"  {ticker:<10} → {best.method_name} (MAPE={best.mean_mape:.2f}%)")
    lines.append(f"  {'─'*40}")
    lines.append("")

    return "\n".join(lines)
