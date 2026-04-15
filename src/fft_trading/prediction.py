"""Prediction module for train/test split and future price prediction with trend component."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
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

    if trend_result.trend_type == 'linear':
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


def predict_future_with_trend(
    train_prices: List[float],
    prediction_days: int,
    n_components: int = 5,
    trend_type: str = 'linear'
) -> Tuple[List[float], dict]:
    """
    Predict future prices using FFT with trend component.

    Strategy:
    1. Extract trend from training data
    2. Apply FFT to detrended series (cyclical component)
    3. Extrapolate both trend and cycles
    4. Combine: prediction = trend_extrapolation + cycle_forecast

    Args:
        train_prices: Historical prices for training
        prediction_days: Number of days to predict
        n_components: Number of FFT components to use
        trend_type: 'linear', 'polynomial_2', 'polynomial_3', or 'none'

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
    elif trend_type.startswith('polynomial'):
        degree = int(trend_type.split('_')[1]) if '_' in trend_type else 2
        trend_result = extract_polynomial_trend(train_prices, degree)
    else:  # linear default
        trend_result = extract_linear_trend(train_prices)

    # Step 2: FFT on detrended series
    detrended_array = np.array(trend_result.detrended)
    fft_coeffs = fft(detrended_array)
    frequencies = np.fft.fftfreq(n)

    # Step 3: Find dominant components
    amplitudes = np.abs(fft_coeffs)
    sorted_indices = np.argsort(amplitudes)[::-1]

    # Keep top components (excluding DC for detrended data)
    n_keep = min(n // 2, n_components)
    dominant_indices = []
    for idx in sorted_indices:
        if frequencies[idx] < 0:
            continue
        dominant_indices.append(idx)
        if len(dominant_indices) >= n_keep:
            break

    if not dominant_indices:
        dominant_indices = [0]  # Fallback to DC

    # Step 4: Extrapolate cyclical component
    cycle_forecast = []
    for t in range(n, n + prediction_days):
        value = 0.0
        for idx in dominant_indices:
            coeff = fft_coeffs[idx]
            value += coeff * np.exp(2j * np.pi * idx * t / n)
        cycle_forecast.append(np.real(value / n))

    # Step 5: Extrapolate trend
    trend_forecast = extrapolate_trend(trend_result, prediction_days)

    # Step 6: Combine
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
