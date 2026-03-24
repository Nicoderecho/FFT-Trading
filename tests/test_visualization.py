"""Tests for visualization module."""

import pytest
import os
from fft_trading.visualization import create_prediction_plot, create_fft_plot
from fft_trading.data_fetcher import StockData


class TestCreatePredictionPlot:
    """Test prediction visualization."""

    def test_plot_creates_html_file(self, tmp_path):
        """Should create an interactive HTML file."""
        stock_data = StockData("AAPL", "2023-01-01", "2023-06-01")
        stock_data.dates = ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"]
        stock_data.prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0]
        predicted = [103.5, 104.2, 105.0, 104.8]
        output_path = str(tmp_path / "test_plot.html")
        create_prediction_plot(stock_data, predicted, output_path)
        assert os.path.exists(output_path)

    def test_plot_contains_real_and_predicted(self, tmp_path):
        """Should include both real and predicted prices."""
        stock_data = StockData("AAPL", "2023-01-01", "2023-06-01")
        stock_data.dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        stock_data.prices = [100.0, 102.0, 101.0]
        predicted = [102.5, 103.0]
        output_path = str(tmp_path / "test_plot.html")
        create_prediction_plot(stock_data, predicted, output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Real Price" in content or "Actual" in content
            assert "Predicted" in content


class TestCreateFFTPlot:
    """Test FFT spectrum visualization."""

    def test_fft_plot_creates_html_file(self, tmp_path):
        """Should create an FFT spectrum HTML file."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0]
        output_path = str(tmp_path / "fft_plot.html")
        create_fft_plot(prices, output_path)
        assert os.path.exists(output_path)

    def test_fft_plot_contains_frequency_info(self, tmp_path):
        """Should include frequency/amplitude information."""
        prices = [100.0, 102.0, 101.0, 103.0, 105.0]
        output_path = str(tmp_path / "fft_plot.html")
        create_fft_plot(prices, output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Frequency" in content or "Amplitude" in content
