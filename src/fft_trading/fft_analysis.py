"""FFT analysis module using scipy.fft for fast Fourier transform."""

import numpy as np
from scipy.fft import fft, ifft
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FFTResult:
    """Result of FFT analysis on price data."""
    original_prices: List[float]
    n_samples: int
    fft_coefficients: np.ndarray
    dominant_frequencies: List[float]
    dominant_amplitudes: List[float]
    dominant_phases: List[float]


def analyze_fft(prices: List[float]) -> FFTResult:
    """
    Perform Fast Fourier Transform on price data.

    Args:
        prices: List of stock prices

    Returns:
        FFTResult containing coefficients and dominant frequencies
    """
    n = len(prices)
    prices_array = np.array(prices)

    # Compute FFT
    fft_coeffs = fft(prices_array)

    # Get frequencies (normalized)
    frequencies = np.fft.fftfreq(n)

    # Get amplitudes and phases
    amplitudes = np.abs(fft_coeffs)
    phases = np.angle(fft_coeffs)

    # Find dominant frequencies (exclude DC component at index 0)
    # Sort by amplitude and get top frequencies
    sorted_indices = np.argsort(amplitudes)[::-1]

    # Get dominant frequencies (excluding DC and negative frequencies)
    dominant_indices = []
    for idx in sorted_indices:
        if idx == 0:
            continue  # Skip DC component
        if frequencies[idx] < 0:
            continue  # Skip negative frequencies
        dominant_indices.append(idx)
        if len(dominant_indices) >= min(n // 2, 10):
            break

    dominant_freqs = [float(frequencies[i]) for i in dominant_indices]
    dominant_amps = [float(amplitudes[i]) for i in dominant_indices]
    dominant_phas = [float(phases[i]) for i in dominant_indices]

    return FFTResult(
        original_prices=prices,
        n_samples=n,
        fft_coefficients=fft_coeffs,
        dominant_frequencies=dominant_freqs,
        dominant_amplitudes=dominant_amps,
        dominant_phases=dominant_phas
    )


def reconstruct_signal(result: FFTResult, n_components: Optional[int] = None) -> List[float]:
    """
    Reconstruct signal using inverse FFT.

    Args:
        result: FFTResult from analyze_fft
        n_components: Number of dominant frequencies to use (None = all)

    Returns:
        Reconstructed price array
    """
    n = result.n_samples

    if n_components is None:
        # Use all frequencies - perfect reconstruction
        reconstructed = ifft(result.fft_coefficients)
    else:
        # Use only top n_components frequencies
        # Zero out all but the dominant components
        filtered_coeffs = np.zeros(n, dtype=complex)

        # Get dominant frequency indices
        sorted_indices = np.argsort(np.abs(result.fft_coefficients))[::-1]

        # Keep DC component (index 0) and top n_components
        indices_to_keep = [0]
        for idx in sorted_indices:
            if idx == 0:
                continue
            indices_to_keep.append(idx)
            if len(indices_to_keep) >= n_components + 1:
                break

        for idx in indices_to_keep:
            filtered_coeffs[idx] = result.fft_coefficients[idx]

        reconstructed = ifft(filtered_coeffs)

    return reconstructed.real.tolist()
