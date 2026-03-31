"""Tests for FFT analysis module."""

import pytest
import numpy as np
from src.fft_trading.fft_analysis import analyze_fft, reconstruct_signal, FFTResult


class TestAnalyzeFFT:
    """Test FFT analysis on price data."""

    def test_analyze_returns_fft_result_object(self):
        """Should return an FFTResult object."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0]
        result = analyze_fft(prices)
        assert isinstance(result, FFTResult)

    def test_analyze_extracts_dominant_frequencies(self):
        """Should identify dominant frequencies from FFT."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0]
        result = analyze_fft(prices)
        assert len(result.dominant_frequencies) > 0
        assert len(result.dominant_amplitudes) > 0

    def test_analyze_stores_original_length(self):
        """Should store the original data length."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0]
        result = analyze_fft(prices)
        assert result.n_samples == 5

    def test_analyze_computes_fft_coefficients(self):
        """Should compute FFT coefficients."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0]
        result = analyze_fft(prices)
        assert result.fft_coefficients is not None
        assert len(result.fft_coefficients) == len(prices)


class TestReconstructSignal:
    """Test signal reconstruction using inverse FFT."""

    def test_reconstruct_returns_reconstructed_prices(self):
        """Should return reconstructed price array."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0]
        result = analyze_fft(prices)
        reconstructed = reconstruct_signal(result)
        assert len(reconstructed) == len(prices)

    def test_reconstruct_uses_all_frequencies_by_default(self):
        """Should use all frequencies when n_components not specified."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0]
        result = analyze_fft(prices)
        reconstructed = reconstruct_signal(result)
        assert len(reconstructed) == 5

    def test_reconstruct_with_n_components(self):
        """Should reconstruct using only top N frequencies."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0]
        result = analyze_fft(prices)
        reconstructed = reconstruct_signal(result, n_components=3)
        assert len(reconstructed) == 8
