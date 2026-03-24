"""Stock data fetching module using yfinance."""

import yfinance as yf
from datetime import datetime
from typing import List


class StockData:
    """Container for stock price data."""

    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.dates: List[str] = []
        self.prices: List[float] = []


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> StockData:
    """
    Fetch historical stock data from yfinance.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        StockData object with dates and prices
    """
    stock = yf.download(ticker, start=start_date, end=end_date)

    data = StockData(ticker, start_date, end_date)
    data.dates = stock.index.strftime('%Y-%m-%d').tolist()
    data.prices = stock['Close'].to_numpy().tolist()

    return data
