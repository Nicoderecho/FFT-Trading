"""Prediction module for train/test split and future price prediction with trend component."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from scipy.fft import fft, ifft

from src.fft_trading.data_fetcher import StockData
from src.fft_trading.fft_analysis import FFTResult


@dataclass
class PredictionResult:
    """Result of train/test split and prediction."""
    stock_data: StockData
    train_end_date: str
    train_dates: List[str]
    train_prices: List[float]
    test_dates: List[str]
    test_prices: List[float]
    fft_result: Optional[FFTResult] = None
    predicted_prices: Optional[List[float]] = None
    trend_info: Optional[dict] = None  # Store trend information


@dataclass
class TrendResult:
    """Result of trend extraction."""
    trend: List[float]  # Fitted trend values
    detrended: List[float]  # Original - trend (stationary component)
    trend_type: str  # 'linear', 'polynomial', 'exponential'
    params: dict  # Trend parameters for extrapolation


def extract_linear_trend(prices: List[float]) -> TrendResult:
    """
    Extract linear trend from price series.

    Fits: price[t] = a + b*t
    Returns detrended series for FFT analysis.

    Args:
        prices: Price series

    Returns:
        TrendResult with trend and detrended components
    """
    n = len(prices)
    x = np.arange(n)
    y = np.array(prices)

    # Linear regression: y = a + b*x
    A = np.vstack([x, np.ones(n)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]

    trend = (a + b * x).tolist()
    detrended = (y - (a + b * x)).tolist()

    return TrendResult(
        trend=trend,
        detrended=detrended,
        trend_type='linear',
        params={'slope': float(b), 'intercept': float(a)}
    )


def extract_log_trend(prices: List[float]) -> TrendResult:
    """
    Extract log-linear (exponential) trend from price series.

    Fits: log(price[t]) = a + b*t  →  price[t] ≈ exp(a + b*t)

    This is the correct model for inflationary/compounding markets because:
    - Stock prices grow multiplicatively, not additively
    - Log-residuals are stationary (constant variance) — ideal for FFT
    - Equivalent to modeling log-returns as a constant drift

    Args:
        prices: Price series (must be positive)

    Returns:
        TrendResult where:
        - trend: exp(a + b*t) in price-space (for visualization)
        - detrended: log(price[t]) - (a + b*t)  ← stationary, passed to FFT
        - params: slope b and intercept a in log-space
    """
    n = len(prices)
    x = np.arange(n)
    log_y = np.log(np.array(prices, dtype=float))

    # Linear fit in log-space: log(price) = a + b*t
    A = np.vstack([x, np.ones(n)]).T
    b, a = np.linalg.lstsq(A, log_y, rcond=None)[0]

    log_trend = a + b * x
    trend = np.exp(log_trend).tolist()
    detrended = (log_y - log_trend).tolist()  # log-residuals for FFT

    return TrendResult(
        trend=trend,
        detrended=detrended,
        trend_type='log',
        params={'slope': float(b), 'intercept': float(a), 'space': 'log'}
    )


def extract_polynomial_trend(prices: List[float], degree: int = 2) -> TrendResult:
    """
    Extract polynomial trend from price series.

    Fits: price[t] = a0 + a1*t + a2*t^2 + ... + ad*t^d
    Better for capturing curved trends (acceleration/deceleration).

    Args:
        prices: Price series
        degree: Polynomial degree (default: 2 for quadratic)

    Returns:
        TrendResult with trend and detrended components
    """
    n = len(prices)
    x = np.arange(n)
    y = np.array(prices)

    # Polynomial fit
    coeffs = np.polyfit(x, y, degree)
    trend_poly = np.poly1d(coeffs)
    trend = trend_poly(x).tolist()
    detrended = (y - trend_poly(x)).tolist()

    return TrendResult(
        trend=trend,
        detrended=detrended,
        trend_type=f'polynomial_{degree}',
        params={'coefficients': coeffs.tolist()}
    )


def extrapolate_trend(trend_result: TrendResult, n_future: int) -> List[float]:
    """
    Extrapolate trend into the future.

    Args:
        trend_result: TrendResult from extract_*_trend
        n_future: Number of future points to extrapolate

    Returns:
        Extrapolated trend values
    """
    n_original = len(trend_result.trend)
    future_x = np.arange(n_original, n_original + n_future)

    if trend_result.trend_type == 'log':
        slope = trend_result.params['slope']
        intercept = trend_result.params['intercept']
        return np.exp(intercept + slope * future_x).tolist()

    elif trend_result.trend_type == 'linear':
        slope = trend_result.params['slope']
        intercept = trend_result.params['intercept']
        return (intercept + slope * future_x).tolist()

    elif trend_result.trend_type.startswith('polynomial'):
        coeffs = np.array(trend_result.params['coefficients'])
        trend_poly = np.poly1d(coeffs)
        return trend_poly(future_x).tolist()

    else:
        # Default: constant trend (last value)
        last_value = trend_result.trend[-1] if trend_result.trend else 0
        return [last_value] * n_future


def _select_dominant_components(
    fft_coeffs: np.ndarray,
    frequencies: np.ndarray,
    n_components: int,
    max_cycle_ratio: float,
    n: int,
    stability_weights: Optional[Dict[float, float]] = None
) -> List[int]:
    """
    Select dominant positive-frequency FFT components for prediction.

    Shared logic used by both predict_future_with_trend() and
    reconstruct_training_fit() to ensure identical component selection.

    Args:
        fft_coeffs: Complex FFT coefficients
        frequencies: FFT frequency array (from np.fft.fftfreq)
        n_components: Maximum number of components to keep
        max_cycle_ratio: Exclude components with period > n * ratio
        n: Length of original signal
        stability_weights: Optional {period_days: persistence_score} from
            compute_stability_weights(). When provided, components are ranked
            by amplitude * (0.5 + 0.5 * weight) instead of pure amplitude.

    Returns:
        List of FFT coefficient indices for the dominant components
    """
    amplitudes = np.abs(fft_coeffs)
    n_keep = min(n // 2, n_components)
    max_period = n * max_cycle_ratio

    # Build candidate list with scores
    candidates = []
    for idx in range(len(fft_coeffs)):
        freq = frequencies[idx]
        if freq <= 0:
            continue
        period = 1.0 / freq
        if period > max_period:
            continue

        amp = amplitudes[idx]
        if stability_weights is not None:
            # Find closest period in stability_weights (within 5-day tolerance)
            best_weight = 0.0
            for ref_period, weight in stability_weights.items():
                if abs(period - ref_period) <= 5.0:
                    best_weight = max(best_weight, weight)
            score = amp * (0.5 + 0.5 * best_weight)
        else:
            score = amp

        candidates.append((idx, score))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    dominant_indices = [idx for idx, _ in candidates[:n_keep]]

    # Edge-case fallback: all components filtered — use best available
    if not dominant_indices:
        sorted_indices = np.argsort(amplitudes)[::-1]
        for idx in sorted_indices:
            if frequencies[idx] > 0:
                dominant_indices = [idx]
                break
        if not dominant_indices:
            dominant_indices = [0]  # Last resort: DC

    return dominant_indices


def compute_stability_weights(
    prices: List[float],
    window_size: int = 250,
    step_size: int = 20,
    n_top_cycles: int = 5,
    period_tolerance: float = 5.0
) -> Dict[float, float]:
    """
    Measure persistence of each cycle period across rolling windows.

    Slides a window across the price series, extracts dominant periods in
    each window, and computes how often each period appears. Periods are
    clustered within `period_tolerance` days.

    Args:
        prices: Price series
        window_size: Size of each rolling window
        step_size: Stride between windows
        n_top_cycles: Number of top cycles to extract per window
        period_tolerance: Days within which periods are considered identical

    Returns:
        {period_days: persistence_score} where persistence_score is in [0, 1].
        Score of 1.0 means the period appeared in every window.
    """
    n = len(prices)
    if n < window_size + step_size:
        return {}

    # Collect dominant periods for each window
    all_periods: List[List[float]] = []
    for start in range(0, n - window_size, step_size):
        window = np.array(prices[start:start + window_size])
        fft_coeffs = fft(window)
        frequencies = np.fft.fftfreq(len(window))
        amplitudes = np.abs(fft_coeffs)
        sorted_indices = np.argsort(amplitudes)[::-1]

        periods = []
        for idx in sorted_indices:
            if frequencies[idx] <= 0:
                continue
            period = 1.0 / frequencies[idx]
            if 5 <= period <= len(window) * 0.33:  # same max_cycle_ratio logic
                periods.append(period)
            if len(periods) >= n_top_cycles:
                break
        all_periods.append(periods)

    if not all_periods:
        return {}

    n_windows = len(all_periods)

    # Cluster periods by tolerance and count occurrences
    # First, collect all unique periods with bucketing
    period_counts: Dict[float, int] = {}
    for periods in all_periods:
        for p in periods:
            # Find closest existing bucket
            matched = False
            for bucket in list(period_counts.keys()):
                if abs(p - bucket) <= period_tolerance:
                    period_counts[bucket] += 1
                    matched = True
                    break
            if not matched:
                period_counts[p] = 1

    # Normalize to [0, 1] — fraction of windows where this period appeared
    weights = {period: count / n_windows for period, count in period_counts.items()}
    return weights


def find_optimal_n_components(
    prices: List[float],
    candidate_n: List[int] = None,
    forecast_horizon: int = 20,
    n_folds: int = 5,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33
) -> Tuple[int, Dict]:
    """
    Find the optimal number of FFT components via contiguous-block cross-validation.

    Splits the training prices into folds, holds out the last forecast_horizon
    days of each fold as validation, and picks the N that minimizes mean MAPE.

    Args:
        prices: Training price series
        candidate_n: List of N values to test (default: [3, 5, 7, 10, 15, 20])
        forecast_horizon: Days to hold out per fold
        n_folds: Number of cross-validation folds
        trend_type: Trend extraction method
        max_cycle_ratio: Period filter ratio

    Returns:
        Tuple of (best_n, details_dict) where details_dict contains
        'scores_by_n' mapping each N to its mean MAPE, and 'best_n'.
    """
    if candidate_n is None:
        candidate_n = [3, 5, 7, 10, 15, 20]

    n_total = len(prices)
    # Minimum training size: at least 100 data points
    min_train = max(100, forecast_horizon * 3)

    # Calculate step size so folds don't overlap too much
    available = n_total - min_train - forecast_horizon
    if available <= 0:
        # Not enough data for CV — return middle candidate
        mid = candidate_n[len(candidate_n) // 2]
        return mid, {'scores_by_n': {n: float('nan') for n in candidate_n}, 'best_n': mid}

    step = max(forecast_horizon, available // n_folds)
    actual_folds = min(n_folds, available // step + 1)

    scores_by_n: Dict[int, List[float]] = {n: [] for n in candidate_n}

    for fold in range(actual_folds):
        # Each fold holds out a different validation window
        val_end = n_total - fold * step
        val_start = val_end - forecast_horizon
        if val_start < min_train:
            break

        train = prices[:val_start]
        actual = prices[val_start:val_end]

        for n_comp in candidate_n:
            try:
                predicted, _ = predict_future_with_trend(
                    train, len(actual),
                    n_components=n_comp,
                    trend_type=trend_type,
                    max_cycle_ratio=max_cycle_ratio
                )
                min_len = min(len(predicted), len(actual))
                pred = np.array(predicted[:min_len])
                act = np.array(actual[:min_len])
                mape = float(np.mean(np.abs((act - pred) / act)) * 100)
                scores_by_n[n_comp].append(mape)
            except Exception:
                continue

    # Compute mean MAPE per candidate
    mean_scores = {}
    for n_comp, mapes in scores_by_n.items():
        mean_scores[n_comp] = float(np.mean(mapes)) if mapes else float('inf')

    best_n = min(mean_scores, key=mean_scores.get)

    return best_n, {'scores_by_n': mean_scores, 'best_n': best_n}


def predict_future_with_trend(
    train_prices: List[float],
    prediction_days: int,
    n_components: int = 5,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33,
    apply_window: bool = True,
    boundary_continuity: bool = True,
    stability_weights: Optional[Dict[float, float]] = None
) -> Tuple[List[float], dict]:
    """
    Predict future prices using FFT with trend component.

    Strategy:
    1. Extract trend from training data
    2. Apply FFT to detrended series (cyclical component)
    3. Extrapolate both trend and cycles
    4. Combine: prediction = trend_extrapolation + cycle_forecast

    For 'log' trend (default, recommended for inflationary markets):
    - Detrend in log-space → FFT on log-residuals → exp(cycle + log_trend)

    Args:
        train_prices: Historical prices for training
        prediction_days: Number of days to predict
        n_components: Number of FFT components to use
        trend_type: 'log' (default), 'linear', 'polynomial_2', 'polynomial_3', or 'none'
        max_cycle_ratio: Exclude components whose period > n * ratio (default 0.33 = at
            least 3 full cycles required). Prevents window-length artifacts from dominating.
        apply_window: If True, apply a Hann window before FFT to suppress spectral leakage
            from boundary discontinuities. Must match reconstruct_training_fit setting.
        boundary_continuity: If True, apply an exponential-decay correction that closes
            any gap between the last in-sample cycle value and the first forecast value.

    Returns:
        Tuple of (predicted_prices, metadata_dict)
    """
    n = len(train_prices)
    prices_array = np.array(train_prices)

    # Step 1: Extract trend
    if trend_type == 'none':
        trend_result = TrendResult(
            trend=[0.0] * n,
            detrended=prices_array.tolist(),
            trend_type='none',
            params={}
        )
    elif trend_type == 'log':
        trend_result = extract_log_trend(train_prices)
    elif trend_type.startswith('polynomial'):
        degree = int(trend_type.split('_')[1]) if '_' in trend_type else 2
        trend_result = extract_polynomial_trend(train_prices, degree)
    else:  # linear
        trend_result = extract_linear_trend(train_prices)

    # Step 2: FFT on detrended series
    detrended_array = np.array(trend_result.detrended)
    if apply_window:
        w = np.hanning(n)
        norm = np.sqrt(n / np.sum(w ** 2))  # RMS-preserve amplitude scale
        detrended_array = detrended_array * w * norm
    fft_coeffs = fft(detrended_array)
    frequencies = np.fft.fftfreq(n)

    # Step 3: Find dominant components (shared logic)
    dominant_indices = _select_dominant_components(
        fft_coeffs, frequencies, n_components, max_cycle_ratio, n,
        stability_weights=stability_weights
    )

    # Step 4: Extrapolate cyclical component
    cycle_forecast = []
    for t in range(n, n + prediction_days):
        value = 0.0
        for idx in dominant_indices:
            coeff = fft_coeffs[idx]
            value += coeff * np.exp(2j * np.pi * idx * t / n)
        cycle_forecast.append(np.real(value / n))

    # Boundary continuity: exponential decay closes gap between in-sample end and forecast start
    if boundary_continuity and dominant_indices:
        last_fit = np.real(
            sum(fft_coeffs[i] * np.exp(2j * np.pi * i * (n - 1) / n)
                for i in dominant_indices) / n
        )
        gap = last_fit - cycle_forecast[0]
        tau = max(len(cycle_forecast) // 5, 5)
        for t_idx in range(len(cycle_forecast)):
            cycle_forecast[t_idx] += gap * np.exp(-t_idx / tau)

    # Step 5: Extrapolate trend
    trend_forecast = extrapolate_trend(trend_result, prediction_days)

    # Step 6: Combine
    # For log-trend: cycle_forecast is in log-space → exp(log_cycle + log_trend)
    # For all others: additive combination in price-space
    if trend_result.trend_type == 'log':
        # trend_forecast is already in price-space (exp(...)), so we need log
        log_trend_forecast = [
            trend_result.params['intercept'] + trend_result.params['slope'] * (n + t)
            for t in range(prediction_days)
        ]
        predicted = [np.exp(c + lt) for c, lt in zip(cycle_forecast, log_trend_forecast)]
    else:
        predicted = [c + t for c, t in zip(cycle_forecast, trend_forecast)]

    # Metadata for analysis
    metadata = {
        'trend_type': trend_result.trend_type,
        'trend_params': trend_result.params,
        'dominant_frequencies': [float(frequencies[i]) for i in dominant_indices[:3]],
        'mean_cycle_amplitude': float(np.mean(np.abs(detrended_array))),
        'trend_slope': trend_result.params.get('slope', 0.0)
    }

    return predicted, metadata


def reconstruct_training_fit(
    train_prices: List[float],
    n_components: int = 5,
    trend_type: str = 'log',
    max_cycle_ratio: float = 0.33,
    apply_window: bool = True,
    stability_weights: Optional[Dict[float, float]] = None
) -> List[float]:
    """
    Return the FFT model's in-sample fit over the training period.

    Evaluates the same Fourier components used for forecasting at time
    points t = 0..n-1, giving a "what the model learned" curve that can
    be plotted alongside the real prices and the future prediction to
    form one continuous model line across the full time range.

    Args:
        train_prices: Historical prices used for training
        n_components: Number of FFT components (must match predict call)
        trend_type: Trend type (must match predict call)
        max_cycle_ratio: Same period-cutoff ratio as predict_future_with_trend. Must match.
        apply_window: Same Hann-window flag as predict_future_with_trend. Must match.

    Returns:
        List of reconstructed prices over the training period
    """
    n = len(train_prices)
    prices_array = np.array(train_prices)

    # Extract trend (same logic as predict_future_with_trend)
    if trend_type == 'none':
        trend_result = TrendResult(
            trend=[0.0] * n,
            detrended=prices_array.tolist(),
            trend_type='none',
            params={}
        )
    elif trend_type == 'log':
        trend_result = extract_log_trend(train_prices)
    elif trend_type.startswith('polynomial'):
        degree = int(trend_type.split('_')[1]) if '_' in trend_type else 2
        trend_result = extract_polynomial_trend(train_prices, degree)
    else:
        trend_result = extract_linear_trend(train_prices)

    # FFT on detrended series (must use identical window/filter as predict_future_with_trend)
    detrended_array = np.array(trend_result.detrended)
    if apply_window:
        w = np.hanning(n)
        norm = np.sqrt(n / np.sum(w ** 2))
        detrended_array = detrended_array * w * norm
    fft_coeffs = fft(detrended_array)
    frequencies = np.fft.fftfreq(n)

    # Same dominant component selection as in predict_future_with_trend (shared logic)
    dominant_indices = _select_dominant_components(
        fft_coeffs, frequencies, n_components, max_cycle_ratio, n,
        stability_weights=stability_weights
    )

    # Evaluate Fourier series at training time points t = 0..n-1
    cycle_fit = []
    for t in range(n):
        value = 0.0
        for idx in dominant_indices:
            coeff = fft_coeffs[idx]
            value += coeff * np.exp(2j * np.pi * idx * t / n)
        cycle_fit.append(np.real(value / n))

    # Recombine with trend (same logic as predict_future_with_trend)
    if trend_result.trend_type == 'log':
        log_trend = [
            trend_result.params['intercept'] + trend_result.params['slope'] * t
            for t in range(n)
        ]
        return [np.exp(c + lt) for c, lt in zip(cycle_fit, log_trend)]
    else:
        return [c + tr for c, tr in zip(cycle_fit, trend_result.trend)]


def predict_future(
    train_prices: List[float],
    prediction_days: int,
    n_components: int = 5
) -> List[float]:
    """
    Predict future prices using FFT-based extrapolation (legacy, no trend).

    Uses the dominant frequencies from FFT to extrapolate the signal
    into the future.

    Args:
        train_prices: Historical prices for training
        prediction_days: Number of days to predict
        n_components: Number of FFT components to use

    Returns:
        List of predicted prices
    """
    n = len(train_prices)
    prices_array = np.array(train_prices)

    # Compute FFT
    fft_coeffs = fft(prices_array)

    # Get frequencies
    frequencies = np.fft.fftfreq(n)

    # Find dominant components (excluding DC and negative frequencies)
    amplitudes = np.abs(fft_coeffs)
    sorted_indices = np.argsort(amplitudes)[::-1]

    # Keep top components for reconstruction
    n_keep = min(n // 2, n_components)
    dominant_indices = [0]  # Always include DC component
    for idx in sorted_indices:
        if idx == 0:
            continue
        if frequencies[idx] < 0:
            continue
        dominant_indices.append(idx)
        if len(dominant_indices) >= n_keep + 1:
            break

    # Extrapolate by evaluating Fourier series at future time points
    # Formula: x[t] = (1/n) * sum_k X[k] * exp(2*pi*i*k*t/n)
    reconstructed = []
    for t in range(n, n + prediction_days):
        value = 0.0
        for idx in dominant_indices:
            coeff = fft_coeffs[idx]
            value += coeff * np.exp(2j * np.pi * idx * t / n)
        reconstructed.append(np.real(value / n))  # Critical: divide by n

    return reconstructed


def prepare_train_test_split(
    stock_data: StockData,
    train_end_date: str
) -> PredictionResult:
    """
    Split stock data into train and test sets by date.

    Args:
        stock_data: StockData object with dates and prices
        train_end_date: Date string 'YYYY-MM-DD' marking end of training period

    Returns:
        PredictionResult with separated train and test data
    """
    # Find the split index
    split_idx = 0
    for i, date in enumerate(stock_data.dates):
        if date <= train_end_date:
            split_idx = i + 1
        else:
            break

    train_dates = stock_data.dates[:split_idx]
    train_prices = stock_data.prices[:split_idx]
    test_dates = stock_data.dates[split_idx:]
    test_prices = stock_data.prices[split_idx:]

    return PredictionResult(
        stock_data=stock_data,
        train_end_date=train_end_date,
        train_dates=train_dates,
        train_prices=train_prices,
        test_dates=test_dates,
        test_prices=test_prices
    )
