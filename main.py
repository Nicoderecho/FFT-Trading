"""FFT-Trading main pipeline - Fetch, Analyze, Predict, Visualize."""

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

from src.fft_trading.data_fetcher import fetch_stock_data, StockData
from src.fft_trading.fft_analysis import analyze_fft, reconstruct_signal, FFTResult
from src.fft_trading.prediction import prepare_train_test_split, predict_future, PredictionResult
from src.fft_trading.visualization import create_prediction_plot, create_fft_plot


def run_pipeline(
    ticker: str,
    start_date: str,
    end_date: str,
    train_end_date: str,
    prediction_days: int,
    output_dir: str = "outputs"
) -> None:
    """
    Run the complete FFT trading pipeline for a single ticker.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', '^GSPC')
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        train_end_date: Train/test split date 'YYYY-MM-DD'
        prediction_days: Number of days to predict
        output_dir: Directory to save outputs
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FFT Trading Pipeline - {ticker}")
    print(f"{'='*60}")

    # Step 1: Fetch data
    print(f"\n[1/5] Fetching data for {ticker}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(f"      Retrieved {len(stock_data.dates)} data points")
    print(f"      Range: {stock_data.dates[0]} to {stock_data.dates[-1]}")
    print(f"      Price range: ${float(min(stock_data.prices)):.2f} - ${float(max(stock_data.prices)):.2f}")

    # Step 2: FFT Analysis
    print(f"\n[2/5] Performing FFT analysis...")
    fft_result = analyze_fft(stock_data.prices)
    print(f"      Sample size: {fft_result.n_samples}")
    print(f"      Top 3 dominant frequencies: {fft_result.dominant_frequencies[:3]}")

    # Step 3: Prepare train/test split
    print(f"\n[3/5] Splitting data (train end: {train_end_date})...")
    prediction_result = prepare_train_test_split(stock_data, train_end_date)
    print(f"      Training samples: {len(prediction_result.train_prices)}")
    print(f"      Test samples: {len(prediction_result.test_prices)}")

    # Step 4: Predict future prices
    print(f"\n[4/5] Predicting {prediction_days} days...")
    predicted_prices = predict_future(prediction_result.train_prices, prediction_days)
    print(f"      Predicted range: ${min(predicted_prices):.2f} - ${max(predicted_prices):.2f}")

    # Attach results to prediction_result for visualization
    prediction_result.fft_result = fft_result
    prediction_result.predicted_prices = predicted_prices

    # Step 5: Generate visualizations
    print(f"\n[5/5] Generating visualizations...")

    # Prediction plot
    pred_plot_path = os.path.join(output_dir, f"{ticker}_prediction.html")
    train_end_idx = stock_data.dates.index(train_end_date) if train_end_date in stock_data.dates else None
    create_prediction_plot(
        stock_data,
        predicted_prices,
        output_path=pred_plot_path,
        train_end_idx=train_end_idx
    )
    print(f"      Prediction plot: {pred_plot_path}")

    # FFT spectrum plot
    fft_plot_path = os.path.join(output_dir, f"{ticker}_fft_spectrum.html")
    create_fft_plot(stock_data.prices, output_path=fft_plot_path, ticker=ticker)
    print(f"      FFT spectrum: {fft_plot_path}")

    # Save predictions to CSV
    csv_path = os.path.join(output_dir, f"{ticker}_predictions.csv")
    save_predictions_csv(
        csv_path,
        prediction_result.test_dates,
        prediction_result.test_prices,
        predicted_prices[:len(prediction_result.test_dates)]
    )
    print(f"      Predictions CSV: {csv_path}")

    print(f"\n{'='*60}")
    print(f"Pipeline completed for {ticker}!")
    print(f"{'='*60}\n")


def save_predictions_csv(
    path: str,
    test_dates: List[str],
    test_prices: List[float],
    predicted_prices: List[float]
) -> None:
    """Save predictions to CSV file."""
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'actual_price', 'predicted_price', 'difference', 'error_pct'])

        for i, date in enumerate(test_dates):
            if i < len(predicted_prices):
                actual = test_prices[i]
                pred = predicted_prices[i]
                diff = pred - actual
                error_pct = (diff / actual) * 100 if actual != 0 else 0
                writer.writerow([date, f"{actual:.2f}", f"{pred:.2f}", f"{diff:.2f}", f"{error_pct:.2f}"])


def parse_date(date_str: str) -> str:
    """Validate date format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def main():
    parser = argparse.ArgumentParser(
        description="FFT-Trading: Predict stock prices using Fast Fourier Transform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL --start 2025-01-01 --end 2025-12-01 --train-end 2025-10-01
  python main.py ^GSPC --prediction-days 30
  python main.py AAPL GOOGL MSFT --output-dir results
        """
    )

    parser.add_argument(
        'tickers',
        nargs='+',
        help='Stock ticker symbols (e.g., AAPL, ^GSPC, ^IXIC)'
    )
    parser.add_argument(
        '--start',
        type=parse_date,
        default='2025-01-01',
        help='Start date YYYY-MM-DD (default: 2025-01-01)'
    )
    parser.add_argument(
        '--end',
        type=parse_date,
        default='2025-12-01',
        help='End date YYYY-MM-DD (default: 2025-12-01)'
    )
    parser.add_argument(
        '--train-end',
        type=parse_date,
        default='2025-10-01',
        help='Train/test split date YYYY-MM-DD (default: 2025-10-01)'
    )
    parser.add_argument(
        '--prediction-days',
        type=int,
        default=30,
        help='Days to predict (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )

    args = parser.parse_args()

    # Validate train_end is between start and end
    if not (args.start <= args.train_end <= args.end):
        print(f"Error: --train-end must be between --start and --end")
        return

    # Run pipeline for each ticker
    for ticker in args.tickers:
        try:
            run_pipeline(
                ticker=ticker,
                start_date=args.start,
                end_date=args.end,
                train_end_date=args.train_end,
                prediction_days=args.prediction_days,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"\nError processing {ticker}: {e}\n")


if __name__ == '__main__':
    main()
