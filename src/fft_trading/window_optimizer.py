"""Adaptive window selection for FFT-based prediction.

Tests multiple historical window lengths and recommends the best one
per ticker using walk-forward MAPE and spectral stability as criteria.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.fft_trading.prediction import predict_future_with_trend
from src.fft_trading.fft_analysis import analyze_fft


# Default candidate windows in trading days
# 504≈2yr, 756≈3yr, 1260≈5yr, 2520≈10yr, 3780≈15yr, 5040≈20yr
DEFAULT_CANDIDATE_WINDOWS = [504, 756, 1260, 2520, 3780, 5040]


@dataclass
class WindowEvaluation:
    """Result of evaluating a single window length."""
    window_days: int
    mean_mape: float
    median_mape: float
    directional_accuracy: float
    spectral_stability: float
    composite_score: float
    n_folds: int = 0


@dataclass
class WindowRecommendation:
    """Recommendation from adaptive window analysis."""
    best_window: int
    evaluations: List[WindowEvaluation]
    reasoning: str
    ticker: str = ""


def _compute_rolling_stability(
    prices: List[float],
    window_size: int = 250,
    step_size: int = 20,
    n_top_cycles: int = 5,
    period_bucket: float = 5.0
) -> float:
    """
    Measure how stable dominant FFT cycles are across rolling windows.

    Returns a score between 0 and 1, where 1 means the same cycles
    appear in every window (highly periodic signal).
    """
    n = len(prices)
    if n < window_size + step_size:
        return 0.0

    all_dominant_periods: List[List[float]] = []

    for start in range(0, n - window_size, step_size):
        window = prices[start:start + window_size]
        fft_result = analyze_fft(window)

        frequencies = np.fft.fftfreq(len(window))
        amplitudes = np.abs(fft_result.fft_coefficients)
        sorted_indices = np.argsort(amplitudes)[::-1]

        periods = []
        for idx in sorted_indices[1:15]:  # skip DC
            if frequencies[idx] > 0:
                period = 1.0 / frequencies[idx]
                if 5 <= period <= 500:
                    periods.append(period)
        all_dominant_periods.append(periods[:n_top_cycles])

    if not all_dominant_periods:
        return 0.0

    # Cluster by period_bucket and count occurrences
    cycle_counts: Dict[float, int] = {}
    for periods in all_dominant_periods:
        for p in periods:
            rounded = round(p / period_bucket) * period_bucket
            cycle_counts[rounded] = cycle_counts.get(rounded, 0) + 1

    if not cycle_counts:
        return 0.0

    n_windows = len(all_dominant_periods)
    total_occurrences = sum(cycle_counts.values())
    unique_cycles = len(cycle_counts)

    stability = total_occurrences / (unique_cycles * n_windows) if unique_cycles > 0 else 0
    return min(1.0, stability)


def evaluate_window(
    prices: List[float],
    window_days: int,
    forecast_horizon: int = 20,
    n_components: int = 10,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33,
    n_folds: int = 5,
    step_size: int = None
) -> Optional[WindowEvaluation]:
    """
    Evaluate a specific window length using walk-forward cross-validation.

    Takes the last `window_days` of prices, runs n_folds of walk-forward
    prediction, and computes MAPE and directional accuracy.

    Args:
        prices: Full price history
        window_days: Number of trading days to use as the analysis window
        forecast_horizon: Days to predict in each fold
        n_components: FFT components
        trend_type: Trend extraction method
        max_cycle_ratio: Period filter ratio
        n_folds: Number of walk-forward folds
        step_size: Days between folds (default: forecast_horizon)

    Returns:
        WindowEvaluation or None if insufficient data
    """
    if len(prices) < window_days + forecast_horizon:
        return None

    if step_size is None:
        step_size = forecast_horizon

    # Use the most recent data up to window_days + enough room for folds
    total_needed = window_days + forecast_horizon + (n_folds - 1) * step_size
    if len(prices) < total_needed:
        # Reduce folds to fit available data
        n_folds = max(1, (len(prices) - window_days - forecast_horizon) // step_size + 1)
        total_needed = window_days + forecast_horizon + (n_folds - 1) * step_size

    # Start index so we use the most recent data
    base_start = len(prices) - total_needed

    mape_values = []
    dir_acc_values = []

    for fold in range(n_folds):
        train_start = base_start + fold * step_size
        train_end = train_start + window_days
        test_end = min(train_end + forecast_horizon, len(prices))

        train = prices[train_start:train_end]
        actual = prices[train_end:test_end]

        if len(actual) < 2 or len(train) < 50:
            continue

        try:
            predicted, _ = predict_future_with_trend(
                train, len(actual),
                n_components=n_components,
                trend_type=trend_type,
                max_cycle_ratio=max_cycle_ratio
            )
        except Exception:
            continue

        min_len = min(len(predicted), len(actual))
        pred = np.array(predicted[:min_len])
        act = np.array(actual[:min_len])

        # MAPE
        mape = np.mean(np.abs((act - pred) / act)) * 100
        mape_values.append(mape)

        # Directional accuracy
        if min_len > 1:
            act_dir = np.diff(act) > 0
            pred_dir = np.diff(pred) > 0
            dir_acc = np.mean(act_dir == pred_dir) * 100
            dir_acc_values.append(dir_acc)

    if not mape_values:
        return None

    # Spectral stability on the window
    window_prices = prices[-window_days:] if len(prices) >= window_days else prices
    stability = _compute_rolling_stability(
        window_prices,
        window_size=min(250, window_days // 3),
        step_size=20
    )

    mean_mape = float(np.mean(mape_values))
    median_mape = float(np.median(mape_values))
    mean_dir_acc = float(np.mean(dir_acc_values)) if dir_acc_values else 50.0

    # Composite score: lower is better
    # Normalize: MAPE (lower=better), dir_acc (higher=better), stability (higher=better)
    # We invert dir_acc and stability so all components are "lower is better"
    composite = 0.5 * mean_mape + 0.3 * (100 - mean_dir_acc) + 0.2 * (1 - stability) * 100

    return WindowEvaluation(
        window_days=window_days,
        mean_mape=round(mean_mape, 3),
        median_mape=round(median_mape, 3),
        directional_accuracy=round(mean_dir_acc, 2),
        spectral_stability=round(stability, 3),
        composite_score=round(composite, 3),
        n_folds=len(mape_values)
    )


def find_optimal_window(
    prices: List[float],
    candidate_windows: List[int] = None,
    forecast_horizon: int = 20,
    n_components: int = 10,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33,
    ticker: str = ""
) -> WindowRecommendation:
    """
    Test multiple window lengths and recommend the best one.

    Args:
        prices: Full price history
        candidate_windows: Window lengths to test (trading days).
            Default: [504, 756, 1260, 2520, 3780, 5040] (2yr-20yr)
        forecast_horizon: Days to predict in each evaluation fold
        n_components: FFT components
        trend_type: Trend extraction method
        max_cycle_ratio: Period filter ratio
        ticker: Ticker symbol for reporting

    Returns:
        WindowRecommendation with best window and full evaluation table
    """
    if candidate_windows is None:
        candidate_windows = DEFAULT_CANDIDATE_WINDOWS

    # Filter candidates that require more data than available
    max_usable = len(prices) - forecast_horizon - 20  # need at least 1 fold + buffer
    candidate_windows = [w for w in candidate_windows if w <= max_usable]

    if not candidate_windows:
        return WindowRecommendation(
            best_window=len(prices) // 2,
            evaluations=[],
            reasoning="Insufficient data for any candidate window. Using half of available data.",
            ticker=ticker
        )

    evaluations = []
    for window in candidate_windows:
        ev = evaluate_window(
            prices, window,
            forecast_horizon=forecast_horizon,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
        if ev is not None:
            evaluations.append(ev)

    if not evaluations:
        return WindowRecommendation(
            best_window=candidate_windows[0],
            evaluations=[],
            reasoning="All window evaluations failed. Using smallest candidate.",
            ticker=ticker
        )

    # Sort by composite score (lower is better)
    evaluations.sort(key=lambda e: e.composite_score)
    best = evaluations[0]

    # Build reasoning
    years = best.window_days / 252
    reasoning = (
        f"Best window: {best.window_days} days (~{years:.1f} years). "
        f"MAPE: {best.mean_mape:.1f}%, Dir.Acc: {best.directional_accuracy:.1f}%, "
        f"Stability: {best.spectral_stability:.2f}. "
    )

    if best.spectral_stability > 0.7:
        reasoning += "High spectral stability — dominant cycles are consistent."
    elif best.spectral_stability > 0.4:
        reasoning += "Moderate spectral stability — use predictions with caution."
    else:
        reasoning += "Low spectral stability — consider ensemble approach for robustness."

    return WindowRecommendation(
        best_window=best.window_days,
        evaluations=evaluations,
        reasoning=reasoning,
        ticker=ticker
    )


def create_window_analysis_report(recommendation: WindowRecommendation) -> str:
    """Generate a human-readable report of window analysis results."""
    lines = [
        f"\n{'='*60}",
        f"Window Optimization Report - {recommendation.ticker}",
        f"{'='*60}",
        f"\nRecommended: {recommendation.best_window} days "
        f"(~{recommendation.best_window/252:.1f} years)",
        f"\n{recommendation.reasoning}",
        f"\n{'─'*60}",
        f"{'Window':>10} {'MAPE%':>8} {'Dir.Acc%':>9} {'Stability':>10} {'Score':>8} {'Folds':>6}",
        f"{'─'*60}",
    ]

    for ev in recommendation.evaluations:
        years = ev.window_days / 252
        marker = " ◀" if ev.window_days == recommendation.best_window else ""
        lines.append(
            f"{ev.window_days:>7}d ({years:.0f}y) "
            f"{ev.mean_mape:>7.2f} "
            f"{ev.directional_accuracy:>8.1f} "
            f"{ev.spectral_stability:>9.3f} "
            f"{ev.composite_score:>7.2f} "
            f"{ev.n_folds:>5}{marker}"
        )

    lines.append(f"{'─'*60}")
    lines.append("Score = 0.5*MAPE + 0.3*(100-DirAcc) + 0.2*(1-Stability)*100  [lower = better]")
    lines.append("")
    return "\n".join(lines)
