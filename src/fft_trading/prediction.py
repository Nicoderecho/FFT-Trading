"""Prediction module for train/test split and future price prediction."""

from dataclasses import dataclass
from typing import List, Optional
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


def predict_future(
    train_prices: List[float],
    prediction_days: int
) -> List[float]:
    """
    Predict future prices using FFT-based extrapolation.

    Uses the dominant frequencies from FFT to extrapolate the signal
    into the future.

    Args:
        train_prices: Historical prices for training
        prediction_days: Number of days to predict

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
    n_components = min(n // 2, 5)
    dominant_indices = [0]  # Always include DC component
    for idx in sorted_indices:
        if idx == 0:
            continue
        if frequencies[idx] < 0:
            continue
        dominant_indices.append(idx)
        if len(dominant_indices) >= n_components + 1:
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
