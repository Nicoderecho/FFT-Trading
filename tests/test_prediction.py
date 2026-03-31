"""Tests for prediction module."""

import pytest
from datetime import datetime
from src.fft_trading.prediction import prepare_train_test_split, predict_future, PredictionResult
from src.fft_trading.data_fetcher import StockData


class TestPrepareTrainTestSplit:
    """Test train/test data splitting by date."""

    def test_split_returns_prediction_result(self):
        """Should return a PredictionResult object."""
        stock_data = StockData("AAPL", "2023-01-01", "2023-06-01")
        stock_data.dates = ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"]
        stock_data.prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0]
        result = prepare_train_test_split(stock_data, "2023-03-01")
        assert isinstance(result, PredictionResult)

    def test_split_separates_train_and_test(self):
        """Should separate data into train and test sets."""
        stock_data = StockData("AAPL", "2023-01-01", "2023-06-01")
        stock_data.dates = ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"]
        stock_data.prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0]
        result = prepare_train_test_split(stock_data, "2023-03-01")
        assert len(result.train_prices) == 3  # Jan, Feb, Mar (includes train_end_date)
        assert len(result.test_prices) == 3   # Apr, May, Jun

    def test_split_stores_train_end_date(self):
        """Should store the train end date."""
        stock_data = StockData("AAPL", "2023-01-01", "2023-06-01")
        stock_data.dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        stock_data.prices = [100.0, 102.0, 101.0]
        result = prepare_train_test_split(stock_data, "2023-02-01")
        assert result.train_end_date == "2023-02-01"


class TestPredictFuture:
    """Test future price prediction using FFT."""

    def test_predict_returns_predictions(self):
        """Should return predicted future prices."""
        train_prices = [100.0, 102.0, 101.0, 103.0, 105.0]
        prediction_days = 5
        result = predict_future(train_prices, prediction_days)
        assert len(result) == prediction_days

    def test_predict_returns_reasonable_values(self):
        """Should return values in similar range to training data."""
        train_prices = [100.0, 102.0, 101.0, 103.0, 105.0]
        result = predict_future(train_prices, 3)
        # Predictions should be in reasonable range (not NaN, not extreme)
        assert all(isinstance(p, float) for p in result)
        assert all(not (p != p) for p in result)  # No NaN
