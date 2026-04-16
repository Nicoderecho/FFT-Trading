"""FFT-Trading main pipeline - Fetch, Analyze, Predict, Visualize."""

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

from src.fft_trading.data_fetcher import fetch_stock_data, StockData
from src.fft_trading.fft_analysis import analyze_fft, reconstruct_signal, FFTResult
from src.fft_trading.prediction import prepare_train_test_split, predict_future, predict_future_with_trend, reconstruct_training_fit, PredictionResult
from src.fft_trading.visualization import (
    create_dashboard,
    create_prediction_plot,
    create_fft_plot,
    create_reconstruction_plot,
    create_spectrum_plot,
    create_forecast_plot,
    create_metrics_summary_table
)
from src.fft_trading.spectral_analysis import compute_frequency_spectrum, detect_dominant_cycles
from src.fft_trading.reconstruction import reconstruct_signal_from_top_frequencies, soft_projection
from src.fft_trading.metrics import evaluate_prediction, compute_prediction_confidence
from src.fft_trading.backtest import run_backtest, create_backtest_report
from src.fft_trading.storage import export_predictions_csv, save_predictions_sqlite
from src.fft_trading.logging_config import get_pipeline_logger, log_signal_metrics


def run_pipeline(
    ticker: str,
    start_date: Optional[str],
    end_date: Optional[str],
    train_end_date: Optional[str],
    prediction_days: int,
    output_dir: str = "outputs",
    all_data: bool = False,
    reconstruct: bool = False,
    n_components: int = 10,
    forecast_horizon: int = 30,
    use_soft_projection: bool = False,
    save_to_db: bool = False,
    db_path: Optional[str] = None,
    trend_type: str = "linear",
    max_cycle_ratio: float = 0.33
) -> None:
    """
    Run the complete FFT trading pipeline for a single ticker.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', '^GSPC')
        start_date: Start date 'YYYY-MM-DD' (ignored if all_data=True)
        end_date: End date 'YYYY-MM-DD' (ignored if all_data=True)
        train_end_date: Train/test split date 'YYYY-MM-DD'
        prediction_days: Number of days to predict
        output_dir: Directory to save outputs
        all_data: If True, fetch all available historical data
        reconstruct: If True, generate reconstruction plots
        n_components: Number of frequency components for reconstruction
        forecast_horizon: Days to forecast ahead
        use_soft_projection: If True, use soft projection with confidence bands
        save_to_db: If True, save predictions to SQLite database
        db_path: Path to SQLite database (default: outputs/predictions.db)
        trend_type: Trend extraction type ('none', 'linear', 'polynomial_2', 'polynomial_3')
        max_cycle_ratio: Exclude FFT components with period > n * ratio (default 0.33)
    """
    # Initialize logger
    logger = get_pipeline_logger(os.path.join(output_dir, "pipeline.log"))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FFT Trading Pipeline - {ticker}")
    print(f"{'='*60}")

    # Step 1: Fetch data
    print(f"\n[1/6] Fetching data for {ticker}...")
    logger.info(f"Fetching data for {ticker}")

    if all_data:
        data_start = "2000-01-01"
        data_end = datetime.now().strftime("%Y-%m-%d")
        print(f"      Fetching all available data (max range: {data_start} to {data_end})")
        stock_data = fetch_stock_data(ticker, data_start, data_end)
    else:
        stock_data = fetch_stock_data(ticker, start_date, end_date)

    print(f"      Retrieved {len(stock_data.dates)} data points")
    print(f"      Range: {stock_data.dates[0]} to {stock_data.dates[-1]}")
    print(f"      Price range: ${float(min(stock_data.prices)):.2f} - ${float(max(stock_data.prices)):.2f}")
    logger.info(f"Retrieved {len(stock_data.dates)} data points for {ticker}")

    # Step 2: FFT Analysis
    print(f"\n[2/6] Performing FFT analysis...")
    fft_result = analyze_fft(stock_data.prices)
    print(f"      Sample size: {fft_result.n_samples}")

    # Compute spectral analysis
    spectral_data = compute_frequency_spectrum(fft_result)
    dominant_cycles = detect_dominant_cycles(fft_result, top_cycles=5)

    print(f"      Top 3 dominant frequencies: {fft_result.dominant_frequencies[:3]}")
    print(f"      Dominant cycles (periods): {[c['period_days'] for c in dominant_cycles[:3]]} days")

    # Log signal metrics
    periods = [c['period_days'] for c in dominant_cycles]
    log_signal_metrics(
        logger,
        fft_result.n_samples,
        fft_result.dominant_frequencies[:5],
        periods,
        fft_result.dominant_amplitudes[:5]
    )

    # Step 3: Prepare train/test split
    print(f"\n[3/6] Splitting data (train end: {train_end_date})...")
    prediction_result = prepare_train_test_split(stock_data, train_end_date)
    print(f"      Training samples: {len(prediction_result.train_prices)}")
    print(f"      Test samples: {len(prediction_result.test_prices)}")

    # Step 4: Predict future prices
    test_days = len(prediction_result.test_dates)
    predict_days = test_days if test_days > 0 else prediction_days
    print(f"\n[4/6] Predicting {predict_days} days (test period)...")

    if use_soft_projection:
        # Use soft projection with confidence bands
        projection = soft_projection(
            prediction_result.train_prices,
            forecast_horizon=predict_days,
            n_components=n_components
        )
        predicted_prices = projection['forecast']
        confidence_score = projection['confidence_score']
        print(f"      Predicted range: ${min(predicted_prices):.2f} - ${max(predicted_prices):.2f}")
        print(f"      Confidence score: {confidence_score:.0%}")
        logger.info(f"Soft projection: confidence={confidence_score:.0%}, range=${min(predicted_prices):.2f}-${max(predicted_prices):.2f}")
    elif trend_type != "none":
        # Use FFT with trend component (crucial for inflationary markets)
        predicted_prices, trend_metadata = predict_future_with_trend(
            prediction_result.train_prices,
            predict_days,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )
        confidence_score = None
        print(f"      Predicted range: ${min(predicted_prices):.2f} - ${max(predicted_prices):.2f}")
        print(f"      Trend type: {trend_metadata['trend_type']}")
        print(f"      Trend slope: {trend_metadata['trend_params'].get('slope', 'N/A'):.4f}")
        logger.info(f"FFT with {trend_type} trend: range=${min(predicted_prices):.2f}-${max(predicted_prices):.2f}, slope={trend_metadata['trend_params'].get('slope', 0):.4f}")
        prediction_result.trend_info = trend_metadata
    else:
        predicted_prices = predict_future(prediction_result.train_prices, predict_days, n_components=n_components)
        confidence_score = None
        print(f"      Predicted range: ${min(predicted_prices):.2f} - ${max(predicted_prices):.2f}")
        logger.info(f"FFT prediction (no trend): range=${min(predicted_prices):.2f}-${max(predicted_prices):.2f}")

    # Attach results to prediction_result for visualization
    prediction_result.fft_result = fft_result
    prediction_result.predicted_prices = predicted_prices

    # Step 5: Evaluate + visualize
    print(f"\n[5/6] Evaluating and generating dashboard...")

    train_end_idx = stock_data.dates.index(train_end_date) if train_end_date in stock_data.dates else None
    test_predicted = predicted_prices[:len(prediction_result.test_dates)]

    # In-sample FFT fit for full prediction line
    train_fit = None
    if not use_soft_projection and trend_type != "none" and train_end_idx is not None:
        train_fit = reconstruct_training_fit(
            prediction_result.train_prices,
            n_components=n_components,
            trend_type=trend_type,
            max_cycle_ratio=max_cycle_ratio
        )

    # Evaluate metrics before building dashboard
    metrics = None
    if len(prediction_result.test_prices) > 0 and len(test_predicted) > 0:
        metrics = evaluate_prediction(prediction_result)
        print(f"      Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"        {metric}: {value}")
        logger.info(f"Evaluation metrics: {metrics}")

    # Optional reconstruction
    reconstruction = None
    if reconstruct:
        reconstruction = reconstruct_signal_from_top_frequencies(fft_result, n_components)

    # Convert FrequencySpectrum dataclass to dict
    spectral_dict = {
        'periods': spectral_data.periods,
        'power': spectral_data.power,
        'frequencies': spectral_data.frequencies,
        'amplitudes': spectral_data.amplitudes,
        'phases': spectral_data.phases
    }

    # Single dashboard HTML
    dashboard_path = os.path.join(output_dir, f"{ticker}_dashboard.html")
    create_dashboard(
        stock_data=stock_data,
        predicted_prices=predicted_prices,
        train_end_idx=train_end_idx,
        spectral_data=spectral_dict,
        dominant_cycles=dominant_cycles,
        output_path=dashboard_path,
        train_fit=train_fit,
        metrics=metrics,
        reconstruction=reconstruction,
        trend_info=prediction_result.trend_info,
    )
    print(f"      Dashboard: {dashboard_path}")
    logger.info(f"Dashboard saved: {dashboard_path}")

    # Step 6: Save data outputs
    print(f"\n[6/6] Saving data outputs...")

    # Save predictions to CSV
    csv_path = os.path.join(output_dir, f"{ticker}_predictions.csv")
    export_predictions_csv(
        csv_path,
        prediction_result.test_dates,
        prediction_result.test_prices,
        test_predicted,
        metadata={
            'ticker': ticker,
            'train_end': train_end_date,
            'prediction_days': prediction_days,
            'n_components': n_components
        }
    )
    print(f"      Predictions CSV: {csv_path}")

    # Save to database (optional)
    if save_to_db:
        if db_path is None:
            db_path = os.path.join(output_dir, "predictions.db")

        predictions_list = [
            {
                'date': prediction_result.test_dates[i],
                'actual_price': prediction_result.test_prices[i],
                'predicted_price': test_predicted[i] if i < len(test_predicted) else None
            }
            for i in range(len(prediction_result.test_dates))
        ]
        save_predictions_sqlite(db_path, ticker, predictions_list)
        print(f"      Saved to database: {db_path}")
        logger.info(f"Saved predictions to database: {db_path}")

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


def run_backtest_cli(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    prediction_window: int = 252,
    hold_period: int = 5,
    initial_capital: float = 10000,
    trend_type: str = 'log'
) -> None:
    """
    Run backtest for a single ticker.

    Args:
        ticker: Stock symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        output_dir: Output directory
        prediction_window: Days of historical data for FFT
        hold_period: Days to hold each position
        initial_capital: Starting capital
    """
    print(f"\n{'='*60}")
    print(f"Backtest - {ticker}")
    print(f"{'='*60}")

    result = run_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        prediction_window=prediction_window,
        hold_period=hold_period,
        initial_capital=initial_capital,
        trend_type=trend_type
    )

    # Print report
    print(create_backtest_report(result))

    # Save trades to CSV
    trades_path = os.path.join(output_dir, f"{ticker}_backtest_trades.csv")
    from src.fft_trading.storage import export_backtest_trades_csv
    export_backtest_trades_csv(
        trades_path,
        result.trades,
        metadata={
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio
        }
    )
    print(f"Trades saved to: {trades_path}")

    # Save equity curve
    equity_path = os.path.join(output_dir, f"{ticker}_equity_curve.csv")
    from src.fft_trading.storage import export_equity_curve_csv
    # Generate dates for equity curve
    from datetime import timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    equity_dates = [
        (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(len(result.equity_curve))
    ]
    export_equity_curve_csv(
        equity_path,
        result.equity_curve,
        equity_dates,
        metadata={'ticker': ticker}
    )
    print(f"Equity curve saved to: {equity_path}")


def main():
    parser = argparse.ArgumentParser(
        description="FFT-Trading: Predict stock prices using Fast Fourier Transform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction pipeline
  python main.py AAPL --start 2025-01-01 --end 2025-12-01 --train-end 2025-10-01

  # Use all available data with soft projection
  python main.py AAPL --all-data --soft-projection --forecast-horizon 30

  # Generate reconstruction plots
  python main.py AAPL --reconstruct --n-components 10

  # Run backtest
  python main.py AAPL --backtest --start 2024-01-01 --end 2024-12-31

  # Save to database
  python main.py AAPL --save-to-db --db-path outputs/my_predictions.db
        """
    )

    # Positional arguments
    parser.add_argument(
        'tickers',
        nargs='+',
        help='Stock ticker symbols (e.g., AAPL, ^GSPC, ^IXIC)'
    )

    # Date range options
    parser.add_argument(
        '--start',
        type=parse_date,
        default='2020-01-01',
        help='Start date YYYY-MM-DD (default: 2020-01-01)'
    )
    parser.add_argument(
        '--end',
        type=parse_date,
        default='2026-04-01',
        help='End date YYYY-MM-DD (default: 2026-04-01)'
    )
    parser.add_argument(
        '--train-end',
        type=parse_date,
        default='2024-01-01',
        help='Train/test split date YYYY-MM-DD (default: 2024-01-01)'
    )

    # Prediction options
    parser.add_argument(
        '--prediction-days',
        type=int,
        default=30,
        help='Days to predict (default: 30)'
    )
    parser.add_argument(
        '--forecast-horizon',
        type=int,
        default=30,
        help='Forecast horizon for soft projection (default: 30)'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of frequency components for reconstruction (default: 10)'
    )

    # Mode options
    parser.add_argument(
        '--all-data', '-a',
        action='store_true',
        help='Fetch all available historical data (ignores --start and --end)'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest instead of prediction pipeline'
    )
    parser.add_argument(
        '--reconstruct',
        action='store_true',
        help='Generate signal reconstruction plots'
    )
    parser.add_argument(
        '--soft-projection',
        action='store_true',
        help='Use soft projection with confidence bands'
    )
    parser.add_argument(
        '--trend-type',
        type=str,
        choices=['none', 'log', 'linear', 'polynomial_2', 'polynomial_3'],
        default='log',
        help='Trend extraction type: log (default, for inflationary markets), linear, polynomial_2, polynomial_3, none'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    parser.add_argument(
        '--save-to-db',
        action='store_true',
        help='Save predictions to SQLite database'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to SQLite database (default: outputs/predictions.db)'
    )

    # Backtest options
    parser.add_argument(
        '--prediction-window',
        type=int,
        default=252,
        help='Prediction window for backtest in trading days (default: 252 = 1 year)'
    )
    parser.add_argument(
        '--max-cycle-ratio',
        type=float,
        default=0.33,
        help='Exclude FFT components with period > n * ratio (default: 0.33 = at least 3 full cycles)'
    )
    parser.add_argument(
        '--hold-period',
        type=int,
        default=5,
        help='Hold period for backtest (default: 5)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000,
        help='Initial capital for backtest (default: 10000)'
    )

    args = parser.parse_args()

    # Skip date validation if using all available data
    if not args.all_data and not args.backtest:
        # Validate train_end is between start and end
        if not (args.start <= args.train_end <= args.end):
            print(f"Error: --train-end must be between --start and --end")
            return
    elif args.backtest:
        # Validate dates for backtest
        if args.start >= args.end:
            print(f"Error: --start must be before --end")
            return

    # Run appropriate mode
    if args.backtest:
        # Run backtest mode
        for ticker in args.tickers:
            try:
                run_backtest_cli(
                    ticker=ticker,
                    start_date=args.start,
                    end_date=args.end,
                    output_dir=args.output_dir,
                    prediction_window=args.prediction_window,
                    hold_period=args.hold_period,
                    initial_capital=args.initial_capital,
                    trend_type=args.trend_type
                )
            except Exception as e:
                print(f"\nError backtesting {ticker}: {e}\n")
    else:
        # Run prediction pipeline mode
        for ticker in args.tickers:
            try:
                run_pipeline(
                    ticker=ticker,
                    start_date=args.start,
                    end_date=args.end,
                    train_end_date=args.train_end,
                    prediction_days=args.prediction_days,
                    output_dir=args.output_dir,
                    all_data=args.all_data,
                    reconstruct=args.reconstruct,
                    n_components=args.n_components,
                    forecast_horizon=args.forecast_horizon,
                    use_soft_projection=args.soft_projection,
                    save_to_db=args.save_to_db,
                    db_path=args.db_path,
                    trend_type=args.trend_type,
                    max_cycle_ratio=args.max_cycle_ratio
                )
            except Exception as e:
                print(f"\nError processing {ticker}: {e}\n")


if __name__ == '__main__':
    main()
