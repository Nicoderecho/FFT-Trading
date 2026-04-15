"""
FFT-Trading: Stock price prediction using Fast Fourier Transform.

This package provides tools for analyzing stock price data using FFT
and generating predictions based on dominant frequency components.

Example usage:
    from fft_trading import (
        fetch_stock_data,
        analyze_fft,
        predict_future,
        evaluate_prediction
    )

    # Fetch data
    stock_data = fetch_stock_data('AAPL', '2024-01-01', '2024-12-01')

    # Analyze
    fft_result = analyze_fft(stock_data.prices)

    # Predict
    predictions = predict_future(stock_data.prices, prediction_days=30)
"""

# Data fetching
from .data_fetcher import fetch_stock_data, StockData

# FFT Analysis
from .fft_analysis import analyze_fft, FFTResult, reconstruct_signal

# Spectral Analysis
from .spectral_analysis import (
    compute_frequency_spectrum,
    compute_power_spectrum,
    detect_dominant_cycles,
    get_significant_periods,
    frequency_to_period,
    period_to_frequency,
    analyze_spectral_concentration,
    FrequencySpectrum,
    SpectralComponent
)

# Signal Reconstruction
from .reconstruction import (
    reconstruct_signal_from_top_frequencies,
    decompose_signal,
    extend_forecast,
    soft_projection
)

# Prediction
from .prediction import predict_future, predict_future_with_trend, reconstruct_training_fit, PredictionResult, prepare_train_test_split

# Metrics & Evaluation
from .metrics import (
    evaluate_prediction,
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_directional_accuracy,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_cumulative_return,
    compute_volatility,
    compute_returns_from_prices,
    compute_win_rate,
    compute_prediction_confidence,
    EvaluationResult
)

# Backtesting
from .backtest import (
    run_backtest,
    walk_forward_analysis,
    analyze_rolling_fft_stability,
    create_backtest_report,
    BacktestResult,
    Trade
)

# Visualization
from .visualization import (
    create_dashboard,
    create_prediction_plot,
    create_fft_plot,
    create_reconstruction_plot,
    create_spectrum_plot,
    create_component_decomposition_plot,
    create_forecast_plot,
    create_metrics_summary_table
)

# Storage
from .storage import (
    export_predictions_csv,
    export_backtest_trades_csv,
    export_equity_curve_csv,
    save_predictions_sqlite,
    save_backtest_result_sqlite,
    save_fft_analysis_sqlite,
    load_historical_predictions,
    load_backtest_history,
    get_backtest_trades,
    get_prediction_statistics,
    init_database
)

# Logging
from .logging_config import (
    setup_logger,
    get_pipeline_logger,
    log_signal_metrics,
    log_forecast_metrics,
    log_stability_metrics
)

__version__ = '0.2.0'
__author__ = 'FFT-Trading Team'

__all__ = [
    # Data
    'fetch_stock_data',
    'StockData',

    # FFT Analysis
    'analyze_fft',
    'FFTResult',
    'reconstruct_signal',

    # Spectral Analysis
    'compute_frequency_spectrum',
    'compute_power_spectrum',
    'detect_dominant_cycles',
    'get_significant_periods',
    'frequency_to_period',
    'period_to_frequency',
    'analyze_spectral_concentration',
    'FrequencySpectrum',
    'SpectralComponent',

    # Signal Reconstruction
    'reconstruct_signal_from_top_frequencies',
    'decompose_signal',
    'extend_forecast',
    'soft_projection',

    # Prediction
    'predict_future',
    'PredictionResult',
    'prepare_train_test_split',

    # Metrics
    'evaluate_prediction',
    'compute_rmse',
    'compute_mae',
    'compute_mape',
    'compute_directional_accuracy',
    'compute_sharpe_ratio',
    'compute_max_drawdown',
    'compute_cumulative_return',
    'compute_volatility',
    'compute_returns_from_prices',
    'compute_win_rate',
    'compute_prediction_confidence',
    'EvaluationResult',

    # Backtesting
    'run_backtest',
    'walk_forward_analysis',
    'analyze_rolling_fft_stability',
    'create_backtest_report',
    'BacktestResult',
    'Trade',

    # Visualization
    'create_prediction_plot',
    'create_fft_plot',
    'create_reconstruction_plot',
    'create_spectrum_plot',
    'create_component_decomposition_plot',
    'create_forecast_plot',
    'create_metrics_summary_table',

    # Storage
    'export_predictions_csv',
    'export_backtest_trades_csv',
    'export_equity_curve_csv',
    'save_predictions_sqlite',
    'save_backtest_result_sqlite',
    'save_fft_analysis_sqlite',
    'load_historical_predictions',
    'load_backtest_history',
    'get_backtest_trades',
    'get_prediction_statistics',
    'init_database',

    # Logging
    'setup_logger',
    'get_pipeline_logger',
    'log_signal_metrics',
    'log_forecast_metrics',
    'log_stability_metrics',
]
