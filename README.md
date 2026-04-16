# FFT-Trading

## What is This?

FFT-Trading is a market prediction tool that uses **Fast Fourier Transform (FFT)** analysis to forecast stock and index prices over medium-to-long timeframes (weeks to months). Instead of traditional machine learning approaches, this project analyzes the frequency spectrum of historical price data to identify dominant cycles and trends, then extrapolates those patterns into the future.

### Key Features

- 📊 **Spectral Analysis**: Decomposes price signals into their dominant frequency components
- 🔮 **Price Prediction**: Forecasts future prices using Fourier series extrapolation
- 📈 **Backtesting**: Walk-forward analysis to validate prediction accuracy
- 🎯 **Ensemble Methods**: Combines predictions across multiple time windows for robust forecasts
- 📉 **Interactive Dashboards**: Beautiful HTML visualizations of predictions and metrics
- 💾 **Multi-format Export**: Save results to CSV or SQLite databases

### Supported Assets

- Stock indices (S&P 500, Nasdaq, Dow Jones, IBEX35, etc.)
- Individual stocks (AAPL, MSFT, GOOGL, etc.)
- Any ticker available on Yahoo Finance

### Who Should Use This?

- **Traders & Investors**: Identify price trends and potential turning points
- **Researchers**: Explore FFT-based approaches to financial forecasting
- **Data Scientists**: Experiment with spectral analysis for time series prediction

## Quick Start

### Requirements
- Python 3.9+
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Predict S&P 500 prices
python main.py ^GSPC --all-data

# Predict Apple stock with auto-tuned components
python main.py AAPL --all-data --auto-components

# Run ensemble prediction with benchmark comparison
python main.py ^IBEX --all-data --use-ensemble --benchmark
```

## Full Documentation

For a complete guide on using the CLI, **see `cli_reference.html`** in the outputs directory after running the program. It includes:
- Detailed command-line options
- Example use cases
- Interpretation of results
- Performance metrics explanation

## Project Structure

```
src/fft_trading/          # Core prediction modules
├── data_fetcher.py       # Download price data
├── fft_analysis.py       # FFT spectral decomposition
├── prediction.py         # Price forecasting
├── visualization.py      # Interactive dashboards
└── ...

tests/                    # Unit tests
main.py                   # Command-line entry point
```

## How It Works

1. **Data Fetching**: Historical price data is downloaded from Yahoo Finance
2. **FFT Analysis**: Prices are transformed into frequency domain to identify dominant cycles
3. **Prediction**: Dominant frequency components are extrapolated into the future
4. **Validation**: Predictions are backtested against historical data
5. **Visualization**: Results are rendered as interactive HTML dashboards

## Important Notes

- Predictions are based on historical frequency patterns and assume market behavior is cyclical
- Not suitable for short-term trading or sudden market disruptions
- Always validate predictions with domain expertise before making financial decisions
- This is educational/research software—use at your own risk

## Getting Help

Check `cli_reference.html` for tutorials and examples, or review the module docstrings in `src/fft_trading/`.
