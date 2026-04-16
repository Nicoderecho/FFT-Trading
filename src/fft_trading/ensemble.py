"""Multi-window ensemble predictions for FFT-based forecasting.

Runs predictions from multiple historical window lengths and combines
them into a single robust forecast with principled confidence bands
derived from model disagreement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.fft_trading.prediction import (
    predict_future_with_trend,
    compute_stability_weights,
)
from src.fft_trading.window_optimizer import _compute_rolling_stability


# Sensible defaults: 2yr, 3yr, 5yr, 10yr
DEFAULT_ENSEMBLE_WINDOWS = [504, 756, 1260, 2520]


@dataclass
class EnsemblePrediction:
    """Result of a multi-window ensemble prediction."""
    predicted_prices: List[float]
    confidence_band_lower: List[float]
    confidence_band_upper: List[float]
    window_weights: Dict[int, float]
    individual_predictions: Dict[int, List[float]]
    disagreement_score: float  # mean std / mean price — higher = more uncertainty
    n_windows_used: int = 0


def _compute_performance_weights(
    prices: List[float],
    windows: List[int],
    validation_days: int = 20,
    n_components: int = 10,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33
) -> Dict[int, float]:
    """
    Compute weights inversely proportional to recent out-of-sample MAPE.

    Uses the last `validation_days` of prices as a held-out validation set.
    Each window predicts those days, and the weight is inversely proportional
    to its MAPE.
    """
    # Hold out last validation_days
    actual = prices[-validation_days:]
    train_base = prices[:-validation_days]

    raw_weights: Dict[int, float] = {}
    for window in windows:
        if len(train_base) < window:
            continue
        train = train_base[-window:]
        try:
            predicted, _ = predict_future_with_trend(
                train, validation_days,
                n_components=n_components,
                trend_type=trend_type,
                max_cycle_ratio=max_cycle_ratio
            )
            min_len = min(len(predicted), len(actual))
            pred = np.array(predicted[:min_len])
            act = np.array(actual[:min_len])
            mape = float(np.mean(np.abs((act - pred) / act)) * 100)
            # Inverse MAPE as weight (add epsilon to avoid division by zero)
            raw_weights[window] = 1.0 / (mape + 0.1)
        except Exception:
            raw_weights[window] = 0.0

    # Normalize
    total = sum(raw_weights.values())
    if total > 0:
        return {w: v / total for w, v in raw_weights.items()}
    # Fallback to equal weights
    n = len(raw_weights)
    return {w: 1.0 / n for w in raw_weights} if n > 0 else {}


def _compute_stability_weights_for_ensemble(
    prices: List[float],
    windows: List[int]
) -> Dict[int, float]:
    """
    Compute weights proportional to spectral stability of each window.
    """
    raw_weights: Dict[int, float] = {}
    for window in windows:
        if len(prices) < window:
            continue
        window_prices = prices[-window:]
        stability = _compute_rolling_stability(
            window_prices,
            window_size=min(250, window // 3),
            step_size=20
        )
        raw_weights[window] = stability + 0.01  # small floor to avoid zero

    total = sum(raw_weights.values())
    if total > 0:
        return {w: v / total for w, v in raw_weights.items()}
    n = len(raw_weights)
    return {w: 1.0 / n for w in raw_weights} if n > 0 else {}


def ensemble_predict(
    prices: List[float],
    prediction_days: int,
    windows: List[int] = None,
    weighting: str = 'performance',
    n_components: int = 10,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33,
    use_stability_components: bool = False
) -> EnsemblePrediction:
    """
    Run predictions from multiple window lengths and combine them.

    Args:
        prices: Full price history
        prediction_days: Days to predict
        windows: Window lengths to ensemble (default: [504, 756, 1260, 2520])
        weighting: How to weight windows:
            'equal' — simple average
            'performance' — inversely proportional to recent MAPE
            'stability' — proportional to spectral stability
        n_components: FFT components per window
        trend_type: Trend extraction method
        max_cycle_ratio: Period filter ratio
        use_stability_components: If True, compute stability weights for
            component selection within each window prediction

    Returns:
        EnsemblePrediction with combined forecast and confidence bands
    """
    if windows is None:
        windows = DEFAULT_ENSEMBLE_WINDOWS

    # Filter windows that require more data than available
    usable_windows = [w for w in windows if w <= len(prices) - 20]
    if not usable_windows:
        # Fallback: use half of available data
        usable_windows = [len(prices) // 2]

    # Compute weights
    if weighting == 'performance' and len(prices) > 40:
        weights = _compute_performance_weights(
            prices, usable_windows,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
    elif weighting == 'stability':
        weights = _compute_stability_weights_for_ensemble(prices, usable_windows)
    else:
        n = len(usable_windows)
        weights = {w: 1.0 / n for w in usable_windows}

    # Run predictions for each window
    individual_predictions: Dict[int, List[float]] = {}
    for window in usable_windows:
        train = prices[-window:]

        stab_weights = None
        if use_stability_components:
            stab_weights = compute_stability_weights(train)

        try:
            predicted, _ = predict_future_with_trend(
                train, prediction_days,
                n_components=n_components,
                trend_type=trend_type,
                max_cycle_ratio=max_cycle_ratio,
                stability_weights=stab_weights
            )
            individual_predictions[window] = predicted
        except Exception:
            continue

    if not individual_predictions:
        # Complete fallback: use all data
        predicted, _ = predict_future_with_trend(
            prices, prediction_days,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
        return EnsemblePrediction(
            predicted_prices=predicted,
            confidence_band_lower=predicted,
            confidence_band_upper=predicted,
            window_weights={len(prices): 1.0},
            individual_predictions={len(prices): predicted},
            disagreement_score=0.0,
            n_windows_used=1
        )

    # Combine predictions using weights
    pred_matrix = np.array([
        individual_predictions[w] for w in individual_predictions
    ])
    weight_vector = np.array([
        weights.get(w, 1.0 / len(individual_predictions))
        for w in individual_predictions
    ])
    # Normalize weight_vector to sum to 1
    weight_sum = weight_vector.sum()
    if weight_sum > 0:
        weight_vector = weight_vector / weight_sum

    # Weighted average
    combined = np.average(pred_matrix, axis=0, weights=weight_vector)

    # Confidence bands from ensemble spread
    pred_std = np.std(pred_matrix, axis=0)
    band_lower = (combined - 1.5 * pred_std).tolist()
    band_upper = (combined + 1.5 * pred_std).tolist()

    # Disagreement score: mean std / mean price level
    mean_price = np.mean(combined)
    disagreement = float(np.mean(pred_std) / mean_price) if mean_price > 0 else 0.0

    # Normalize weights for output
    final_weights = {
        w: float(weight_vector[i])
        for i, w in enumerate(individual_predictions.keys())
    }

    return EnsemblePrediction(
        predicted_prices=combined.tolist(),
        confidence_band_lower=band_lower,
        confidence_band_upper=band_upper,
        window_weights=final_weights,
        individual_predictions={w: p for w, p in individual_predictions.items()},
        disagreement_score=round(disagreement, 4),
        n_windows_used=len(individual_predictions)
    )
