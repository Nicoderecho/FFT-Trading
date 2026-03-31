"""Tests for stock data fetching module."""

import pytest
from src.fft_trading.data_fetcher import fetch_stock_data, StockData


class TestFetchStockData:
    """Test data fetching from yfinance."""

    def test_fetch_returns_stock_data_object(self):
        """Should return a StockData object with price data."""
        result = fetch_stock_data("AAPL", "2023-01-01", "2023-06-01")
        assert isinstance(result, StockData)
        assert len(result.prices) > 0

    def test_fetch_stores_ticker_symbol(self):
        """Should store the ticker symbol correctly."""
        result = fetch_stock_data("GOOGL", "2023-01-01", "2023-03-01")
        assert result.ticker == "GOOGL"

    def test_fetch_stores_date_range(self):
        """Should store the start and end dates."""
        result = fetch_stock_data("MSFT", "2023-02-01", "2023-04-01")
        assert result.start_date == "2023-02-01"
        assert result.end_date == "2023-04-01"

    def test_fetch_returns_dates_and_prices(self):
        """Should return aligned dates and prices arrays."""
        result = fetch_stock_data("AAPL", "2023-01-01", "2023-02-01")
        assert len(result.dates) == len(result.prices)
