"""Signal reconstruction module for FFT-Trading."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.fft import fft, ifft

from .fft_analysis import FFTResult, analyze_fft


def reconstruct_signal_from_top_frequencies(
    fft_result: FFTResult,
    n_components: int = 10
) -> Dict:
    """
    Reconstruct signal using top N dominant frequencies.

    Creates a smoothed version of the original signal by keeping only
    the most significant frequency components.

    Args:
        fft_result: Result from analyze_fft()
        n_components: Number of dominant frequencies to use (default: 10)

    Returns:
        Dict with keys:
        - original: Original price series
        - reconstructed: Reconstructed signal from top frequencies
        - trend: Low-frequency trend component
        - residuals: Difference between original and reconstructed
        - components_info: List of (frequency, amplitude, phase) for each component
    """
    n = fft_result.n_samples
    fft_coeffs = fft_result.fft_coefficients
    original_prices = fft_result.original_prices

    # Get amplitudes for sorting
    amplitudes = np.abs(fft_coeffs)

    # Find indices of top components (including DC)
    sorted_indices = np.argsort(amplitudes)[::-1]

    # Keep DC component and top n_components
    indices_to_keep = [0]  # DC component (mean)
    for idx in sorted_indices:
        if idx == 0:
            continue
        indices_to_keep.append(idx)
        if len(indices_to_keep) >= n_components + 1:
            break

    # Create filtered coefficients
    filtered_coeffs = np.zeros(n, dtype=complex)
    for idx in indices_to_keep:
        filtered_coeffs[idx] = fft_coeffs[idx]

    # Reconstruct via IFFT
    reconstructed = ifft(filtered_coeffs).real

    # Compute residuals
    residuals = np.array(original_prices) - reconstructed

    # Extract component information
    components_info = []
    frequencies = np.fft.fftfreq(n)
    for idx in indices_to_keep:
        if idx == 0:
            components_info.append({
                'type': 'DC',
                'frequency': 0.0,
                'period': float('inf'),
                'amplitude': float(np.real(fft_coeffs[0]) / n),
                'phase': float(np.angle(fft_coeffs[0]))
            })
        else:
            freq = frequencies[idx]
            if freq > 0:
                components_info.append({
                    'type': 'sinusoid',
                    'frequency': float(freq),
                    'period': float(1.0 / freq) if freq > 0 else float('inf'),
                    'amplitude': float(np.abs(fft_coeffs[idx]) / n) * 2,  # Factor of 2 for real signal
                    'phase': float(np.angle(fft_coeffs[idx]))
                })

    return {
        'original': original_prices,
        'reconstructed': reconstructed.tolist(),
        'trend': _extract_trend(reconstructed, n_components // 3),
        'residuals': residuals.tolist(),
        'components_info': components_info,
        'n_components_used': len(indices_to_keep),
        'reconstruction_error': float(np.mean(np.abs(residuals)))
    }


def _extract_trend(
    reconstructed: np.ndarray,
    n_trend_components: int = 3
) -> List[float]:
    """
    Extract low-frequency trend from reconstructed signal.

    Args:
        reconstructed: Reconstructed signal
        n_trend_components: Number of lowest frequency components to use

    Returns:
        Trend component
    """
    n = len(reconstructed)
    fft_coeffs = fft(reconstructed)

    # Keep only lowest frequencies
    trend_coeffs = np.zeros(n, dtype=complex)
    for i in range(min(n_trend_components, n // 2)):
        trend_coeffs[i] = fft_coeffs[i]
        if i > 0:
            trend_coeffs[n - i] = fft_coeffs[n - i]  # Symmetric component

    trend = ifft(trend_coeffs).real
    return trend.tolist()


def decompose_signal(
    prices: List[float],
    n_components: int = 10
) -> Dict:
    """
    Decompose signal into individual sine wave components.

    Useful for visualization - shows which frequencies contribute
    most to the signal.

    Args:
        prices: Original price series
        n_components: Number of components to extract

    Returns:
        Dict with keys:
        - original: Original prices
        - sum_components: Sum of all extracted components
        - individual_components: List of individual sine waves
        - residual: What's left after removing all components
    """
    fft_result = analyze_fft(prices)
    n = fft_result.n_samples
    fft_coeffs = fft_result.fft_coefficients
    frequencies = np.fft.fftfreq(n)

    # Get top components
    amplitudes = np.abs(fft_coeffs)
    sorted_indices = np.argsort(amplitudes)[::-1]

    indices_to_keep = [0]  # DC
    for idx in sorted_indices:
        if idx == 0:
            continue
        indices_to_keep.append(idx)
        if len(indices_to_keep) >= n_components + 1:
            break

    # Build individual components
    t = np.arange(n)
    individual_components = []

    # DC component
    dc_value = np.real(fft_coeffs[0]) / n
    individual_components.append({
        'type': 'DC',
        'values': [dc_value] * n,
        'frequency': 0.0,
        'amplitude': dc_value
    })

    # Sinusoidal components
    for idx in indices_to_keep[1:]:
        freq = frequencies[idx]
        if freq <= 0:
            continue

        amp = np.abs(fft_coeffs[idx]) / n * 2
        phase = np.angle(fft_coeffs[idx])

        component_values = (amp * np.cos(2 * np.pi * freq * t + phase)).tolist()
        individual_components.append({
            'type': 'sinusoid',
            'values': component_values,
            'frequency': float(freq),
            'period': float(1.0 / freq),
            'amplitude': float(amp),
            'phase': float(phase)
        })

    # Sum of all components
    sum_components = np.zeros(n)
    for comp in individual_components:
        sum_components += np.array(comp['values'])

    # Residual
    residual = np.array(prices) - sum_components

    return {
        'original': prices,
        'sum_components': sum_components.tolist(),
        'individual_components': individual_components,
        'residual': residual.tolist(),
        'explained_variance': float(1 - np.var(residual) / np.var(prices)) if np.var(prices) > 0 else 0.0
    }


def extend_forecast(
    fft_result: FFTResult,
    reconstruction_result: Dict,
    forecast_horizon: int = 30
) -> Dict:
    """
    Extend reconstructed signal into forecast horizon.

    Uses the dominant frequencies from FFT to extrapolate the signal
    forward in time, with confidence bands based on reconstruction error.

    Args:
        fft_result: Result from analyze_fft()
        reconstruction_result: Result from reconstruct_signal_from_top_frequencies()
        forecast_horizon: Number of days to forecast

    Returns:
        Dict with keys:
        - forecast_signal: Extrapolated signal
        - forecast_dates_offset: Offset from end of training data
        - confidence_band: (lower, upper) bounds
        - confidence_score: Overall confidence (0-1)
    """
    n = fft_result.n_samples
    fft_coeffs = fft_result.fft_coefficients
    frequencies = np.fft.fftfreq(n)

    # Get components used in reconstruction
    components_info = reconstruction_result.get('components_info', [])

    # Build forecast from components
    forecast_signal = []
    for t in range(n, n + forecast_horizon):
        value = 0.0
        for comp in components_info:
            if comp['type'] == 'DC':
                value += comp['amplitude']
            else:
                freq = comp['frequency']
                amp = comp['amplitude']
                phase = comp['phase']
                value += amp * np.cos(2 * np.pi * freq * t + phase)
        forecast_signal.append(value)

    # Compute confidence band based on reconstruction error
    reconstruction_error = reconstruction_result.get('reconstruction_error', 0)
    residuals = reconstruction_result.get('residuals', [])

    # Estimate residual standard deviation
    if residuals:
        residual_std = float(np.std(residuals))
    else:
        residual_std = reconstruction_error

    # Confidence widens with forecast horizon
    confidence_widths = []
    for i in range(forecast_horizon):
        # Width increases with sqrt of time
        width = residual_std * (1 + np.sqrt(i / forecast_horizon))
        confidence_widths.append(width)

    # Build confidence bands
    lower_band = [forecast_signal[i] - confidence_widths[i] for i in range(forecast_horizon)]
    upper_band = [forecast_signal[i] + confidence_widths[i] for i in range(forecast_horizon)]

    # Compute confidence score (inverse of relative error)
    original_prices = fft_result.original_prices
    price_range = max(original_prices) - min(original_prices)
    relative_error = residual_std / price_range if price_range > 0 else 1
    confidence_score = max(0, min(1, 1 - relative_error))

    return {
        'forecast_signal': forecast_signal,
        'forecast_dates_offset': list(range(1, forecast_horizon + 1)),
        'confidence_band': {
            'lower': lower_band,
            'upper': upper_band,
            'widths': confidence_widths
        },
        'confidence_score': round(confidence_score, 4),
        'residual_std': round(residual_std, 4),
        'forecast_components': len(components_info)
    }


def soft_projection(
    prices: List[float],
    forecast_horizon: int = 30,
    n_components: int = 10
) -> Dict:
    """
    Create a soft projection (not hard prediction) of future prices.

    This is a "soft" projection because it:
    - Shows the likely continuation of current patterns
    - Includes uncertainty bands
    - Does not claim to predict exact future values

    Args:
        prices: Historical price series
        forecast_horizon: Days to project forward
        n_components: Number of frequency components to use

    Returns:
        Dict with complete projection information
    """
    # Analyze and reconstruct
    fft_result = analyze_fft(prices)
    reconstruction = reconstruct_signal_from_top_frequencies(fft_result, n_components)
    forecast = extend_forecast(fft_result, reconstruction, forecast_horizon)

    return {
        'historical_prices': prices,
        'reconstructed_signal': reconstruction['reconstructed'],
        'forecast': forecast['forecast_signal'],
        'confidence_band': forecast['confidence_band'],
        'confidence_score': forecast['confidence_score'],
        'dominant_periods': [
            comp['period'] for comp in reconstruction['components_info']
            if comp.get('period', float('inf')) != float('inf')
        ][:5],
        'metrics': {
            'reconstruction_error': reconstruction['reconstruction_error'],
            'explained_variance': decompose_signal(prices, n_components).get('explained_variance', 0)
        }
    }
