"""Spectral analysis module for frequency domain analysis."""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .fft_analysis import FFTResult


@dataclass
class SpectralComponent:
    """A single frequency component in the spectrum."""
    frequency: float      # cycles per day
    period: float         # days
    amplitude: float      # price units
    power: float          # energy (amplitude^2)
    phase: float          # radians


@dataclass
class FrequencySpectrum:
    """Complete frequency spectrum analysis result."""
    frequencies: List[float]    # cycles per day
    periods: List[float]        # days
    amplitudes: List[float]     # price units
    power: List[float]          # energy
    phases: List[float]         # radians
    dominant_components: List[SpectralComponent]


def compute_frequency_spectrum(
    fft_result: FFTResult
) -> FrequencySpectrum:
    """
    Compute complete frequency spectrum from FFT result.

    Converts FFT coefficients to physically meaningful quantities:
    - Frequency in cycles per day
    - Period in days
    - Amplitude in price units
    - Power (energy) as amplitude squared

    Args:
        fft_result: Result from analyze_fft()

    Returns:
        FrequencySpectrum with all spectral components
    """
    n = fft_result.n_samples
    fft_coeffs = fft_result.fft_coefficients

    # Compute normalized frequencies
    freq_indices = np.arange(n)
    frequencies = freq_indices / n  # cycles per sample (day)

    # Compute amplitudes (normalized by n)
    amplitudes = np.abs(fft_coeffs) / n

    # Compute power spectrum (Parseval's theorem: power = |X[k]|^2 / N)
    power = (np.abs(fft_coeffs) ** 2) / n

    # Compute phases
    phases = np.angle(fft_coeffs)

    # Convert frequencies to periods (days per cycle)
    # Handle DC component (infinite period) and very small frequencies
    periods = []
    for f in frequencies:
        if f == 0:
            periods.append(float('inf'))  # DC component
        else:
            periods.append(1.0 / f)

    # Find dominant components (positive frequencies only, exclude DC)
    dominant_components = []
    positive_freq_mask = (frequencies > 0) & (frequencies <= 0.5)

    # Sort by power descending
    sorted_indices = np.argsort(power[positive_freq_mask])[::-1]

    # Get top components
    for idx in sorted_indices[:10]:
        actual_idx = np.where(positive_freq_mask)[0][idx]
        component = SpectralComponent(
            frequency=float(frequencies[actual_idx]),
            period=float(periods[actual_idx]),
            amplitude=float(amplitudes[actual_idx]),
            power=float(power[actual_idx]),
            phase=float(phases[actual_idx])
        )
        dominant_components.append(component)

    return FrequencySpectrum(
        frequencies=frequencies.tolist(),
        periods=periods,
        amplitudes=amplitudes.tolist(),
        power=power.tolist(),
        phases=phases.tolist(),
        dominant_components=dominant_components
    )


def compute_power_spectrum(
    fft_coeffs: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    Compute power spectral density.

    Power is defined as |FFT|^2 / N (Parseval's theorem)

    Args:
        fft_coeffs: FFT coefficients
        n_samples: Number of samples

    Returns:
        Power spectrum array
    """
    return (np.abs(fft_coeffs) ** 2) / n_samples


def detect_dominant_cycles(
    fft_result: FFTResult,
    top_cycles: int = 5,
    min_period: int = 5,
    max_period: int = 250
) -> List[Dict]:
    """
    Detect dominant periodic cycles in the signal.

    Filters frequencies to only include those corresponding to
    periods within the specified range, then returns the top cycles.

    Args:
        fft_result: Result from analyze_fft()
        top_cycles: Number of top cycles to return
        min_period: Minimum period in days (default: 5)
        max_period: Maximum period in days (default: 250)

    Returns:
        List of dicts with keys: period_days, amplitude, power, frequency

    Example output:
        [
            {'period_days': 20, 'amplitude': 5.2, 'power': 27.04, 'frequency': 0.05},
            {'period_days': 45, 'amplitude': 3.1, 'power': 9.61, 'frequency': 0.022},
            {'period_days': 90, 'amplitude': 2.8, 'power': 7.84, 'frequency': 0.011}
        ]
    """
    spectrum = compute_frequency_spectrum(fft_result)

    # Filter by period range
    valid_components = []
    for comp in spectrum.dominant_components:
        if min_period <= comp.period <= max_period:
            valid_components.append(comp)

    # Sort by power and return top N
    valid_components.sort(key=lambda c: c.power, reverse=True)

    return [
        {
            'period_days': round(comp.period, 1),
            'amplitude': round(comp.amplitude, 4),
            'power': round(comp.power, 4),
            'frequency': round(comp.frequency, 6)
        }
        for comp in valid_components[:top_cycles]
    ]


def get_significant_periods(
    fft_result: FFTResult,
    power_threshold: float = 0.1
) -> List[float]:
    """
    Get periods with power above threshold.

    Args:
        fft_result: Result from analyze_fft()
        power_threshold: Relative threshold (0.0 to 1.0) as fraction of max power

    Returns:
        List of significant periods in days
    """
    spectrum = compute_frequency_spectrum(fft_result)

    if not spectrum.power:
        return []

    max_power = max(spectrum.power)
    threshold = max_power * power_threshold

    significant_periods = []
    for i, (period, pwr) in enumerate(zip(spectrum.periods, spectrum.power)):
        if pwr >= threshold and period != float('inf'):
            significant_periods.append(round(period, 1))

    return sorted(set(significant_periods))


def frequency_to_period(frequency: float) -> float:
    """
    Convert frequency (cycles/day) to period (days).

    Args:
        frequency: Frequency in cycles per day

    Returns:
        Period in days
    """
    if frequency <= 0:
        return float('inf')
    return 1.0 / frequency


def period_to_frequency(period: float) -> float:
    """
    Convert period (days) to frequency (cycles/day).

    Args:
        period: Period in days

    Returns:
        Frequency in cycles per day
    """
    if period <= 0 or period == float('inf'):
        return 0.0
    return 1.0 / period


def analyze_spectral_concentration(
    fft_result: FFTResult,
    frequency_range: tuple = (0.01, 0.2)
) -> Dict:
    """
    Analyze how concentrated the signal power is in a frequency range.

    Useful for determining if the signal has clear periodic structure
    vs being noise-like (broadband).

    Args:
        fft_result: Result from analyze_fft()
        frequency_range: (min_freq, max_freq) in cycles/day

    Returns:
        Dict with concentration metrics:
        - total_power: Total signal power
        - band_power: Power in specified frequency range
        - concentration_ratio: band_power / total_power
        - dominant_frequency: Frequency with highest power in range
    """
    spectrum = compute_frequency_spectrum(fft_result)

    # Total power (excluding DC)
    total_power = sum(p for p, f in zip(spectrum.power, spectrum.frequencies) if f > 0)

    # Power in specified band
    min_freq, max_freq = frequency_range
    band_power = 0.0
    dominant_freq = 0.0
    max_band_power = 0.0

    for freq, pwr in zip(spectrum.frequencies, spectrum.power):
        if min_freq <= freq <= max_freq:
            band_power += pwr
            if pwr > max_band_power:
                max_band_power = pwr
                dominant_freq = freq

    concentration_ratio = band_power / total_power if total_power > 0 else 0.0

    return {
        'total_power': total_power,
        'band_power': band_power,
        'concentration_ratio': concentration_ratio,
        'dominant_frequency': dominant_freq,
        'dominant_period': 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
    }
