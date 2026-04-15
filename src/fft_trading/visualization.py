"""Visualization module using plotly for interactive HTML plots."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
import numpy as np
from scipy.fft import fft


def create_prediction_plot(
    stock_data,
    predicted_prices: List[float],
    output_path: str = "prediction_plot.html",
    train_end_idx: Optional[int] = None,
    train_fit: Optional[List[float]] = None
) -> None:
    """
    Create interactive plot comparing real vs predicted prices.

    Args:
        stock_data: StockData object with dates and prices
        predicted_prices: Predicted prices (for test period)
        output_path: Path to save HTML file
        train_end_idx: Index where train/test split occurs
        train_fit: Optional FFT model fit over the training period.
                   When provided, draws a continuous model line across
                   train + test spanning the full time range.
    """
    dates = stock_data.dates
    real_prices = stock_data.prices
    ticker = stock_data.ticker

    # Create figure
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"{ticker} Price Prediction",)
    )

    # Plot real prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=real_prices,
            mode='lines',
            name='Real Price',
            line=dict(color='blue', width=2)
        )
    )

    # Build the full model line: train fit + test prediction concatenated
    if train_fit is not None and train_end_idx is not None:
        # Training segment: model fit over dates[0..train_end_idx]
        train_dates = dates[:train_end_idx + 1]
        fit_segment = train_fit[:len(train_dates)]

        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=fit_segment,
                mode='lines',
                name='FFT Model (train fit)',
                line=dict(color='green', width=1.5, dash='dot'),
                opacity=0.8
            )
        )

        # Test / prediction segment: dates after split
        pred_dates = dates[train_end_idx + 1: train_end_idx + 1 + len(predicted_prices)]
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=predicted_prices[:len(pred_dates)],
                mode='lines',
                name='FFT Prediction (test)',
                line=dict(color='red', width=2, dash='dash')
            )
        )
    else:
        # Fallback: align prediction with last N dates as before
        pred_dates = dates[-len(predicted_prices):]
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=predicted_prices,
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            )
        )

    # Vertical line and annotation for train/test split
    if train_end_idx is not None:
        fig.add_shape(
            type='line',
            x0=dates[train_end_idx],
            y0=0,
            x1=dates[train_end_idx],
            y1=1,
            xref='x',
            yref='y domain',
            line=dict(color='gray', width=2, dash='dot')
        )
        fig.add_annotation(
            x=dates[train_end_idx],
            y=1,
            text="Train/Test Split",
            xref='x',
            yref='y domain',
            showarrow=False,
            textangle=-90
        )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Real vs FFT Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    # Save as interactive HTML
    fig.write_html(output_path)


def create_fft_plot(
    prices: List[float],
    output_path: str = "fft_spectrum.html",
    ticker: str = "Stock"
) -> None:
    """
    Create interactive FFT spectrum plot showing dominant frequencies.

    Args:
        prices: Stock price data
        output_path: Path to save HTML file
        ticker: Stock ticker symbol for title
    """
    n = len(prices)
    prices_array = np.array(prices)

    # Compute FFT
    fft_coeffs = fft(prices_array)
    frequencies = np.fft.fftfreq(n)
    amplitudes = np.abs(fft_coeffs)

    # Only positive frequencies (excluding DC)
    positive_freq_mask = frequencies > 0
    pos_freqs = frequencies[positive_freq_mask]
    pos_amps = amplitudes[positive_freq_mask]

    # Create figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("FFT Amplitude Spectrum", "Phase Spectrum")
    )

    # Amplitude spectrum
    fig.add_trace(
        go.Scatter(
            x=pos_freqs,
            y=pos_amps,
            mode='markers',
            name='Amplitude',
            marker=dict(size=8, color='blue')
        ),
        row=1, col=1
    )

    # Phase spectrum
    phases = np.angle(fft_coeffs)
    pos_phases = phases[positive_freq_mask]

    fig.add_trace(
        go.Scatter(
            x=pos_freqs,
            y=pos_phases,
            mode='markers',
            name='Phase',
            marker=dict(size=8, color='red')
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - FFT Frequency Analysis",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="Normalized Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Normalized Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)

    # Save as interactive HTML
    fig.write_html(output_path)


def create_reconstruction_plot(
    dates: List[str],
    original_prices: List[float],
    reconstruction: Dict,
    output_path: str = "reconstruction_plot.html",
    ticker: str = "Stock"
) -> None:
    """
    Create plot showing original price vs FFT reconstructed signal.

    Args:
        dates: List of date strings
        original_prices: Original price series
        reconstruction: Result from reconstruct_signal_from_top_frequencies()
        output_path: Path to save HTML file
        ticker: Stock ticker symbol
    """
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"{ticker} - Original vs FFT Reconstructed Signal",)
    )

    # Original prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=original_prices,
            mode='lines',
            name='Original Price',
            line=dict(color='blue', width=2),
            opacity=0.7
        )
    )

    # Reconstructed signal
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=reconstruction['reconstructed'],
            mode='lines',
            name='FFT Reconstructed',
            line=dict(color='green', width=2)
        )
    )

    # Trend component
    if 'trend' in reconstruction:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=reconstruction['trend'],
                mode='lines',
                name='Trend Component',
                line=dict(color='orange', width=2, dash='dot')
            )
        )

    fig.update_layout(
        title=f"{ticker} - Signal Reconstruction",
        xaxis_title="Time (days)",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    fig.write_html(output_path)


def create_spectrum_plot(
    spectral_data: Dict,
    dominant_cycles: List[Dict],
    output_path: str = "spectrum_plot.html",
    ticker: str = "Stock"
) -> None:
    """
    Create frequency spectrum plot with power on Y-axis and period (days) on X-axis.

    Args:
        spectral_data: Result from compute_frequency_spectrum()
        dominant_cycles: List of dominant cycle dicts from detect_dominant_cycles()
        output_path: Path to save HTML file
        ticker: Stock ticker symbol
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Power Spectrum", "Dominant Cycles Highlighted"),
        vertical_spacing=0.12
    )

    # Convert to periods (days), excluding DC and very long periods
    periods = []
    power = []
    for i, (p, pw) in enumerate(zip(spectral_data['periods'], spectral_data['power'])):
        if p != float('inf') and p < 500:  # Reasonable range
            periods.append(p)
            power.append(pw)

    # Power spectrum scatter
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=power,
            mode='markers',
            name='Power',
            marker=dict(size=6, color='steelblue'),
            hovertemplate='Period: %{x:.1f} days<br>Power: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Highlight dominant cycles
    dominant_periods = [c['period_days'] for c in dominant_cycles]
    dominant_powers = [c['power'] for c in dominant_cycles]

    fig.add_trace(
        go.Scatter(
            x=dominant_periods,
            y=dominant_powers,
            mode='markers+text',
            name='Dominant Cycles',
            marker=dict(size=12, color='red', symbol='star'),
            text=[f"{p:.0f}d" for p in dominant_periods],
            textposition="top center",
            hovertemplate='Period: %{x:.1f} days<br>Power: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{ticker} - Frequency Domain Analysis",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="Period (days)", row=1, col=1)
    fig.update_xaxes(title_text="Period (days)", row=2, col=1)
    fig.update_yaxes(title_text="Power", row=1, col=1)
    fig.update_yaxes(title_text="Power", row=2, col=1)

    fig.write_html(output_path)


def create_component_decomposition_plot(
    decomposition: Dict,
    output_path: str = "component_decomposition.html",
    ticker: str = "Stock"
) -> None:
    """
    Create plot showing individual sine wave components.

    Args:
        decomposition: Result from decompose_signal()
        output_path: Path to save HTML file
        ticker: Stock ticker symbol
    """
    n_components = len(decomposition.get('individual_components', []))

    # Create subplots: original + each component
    n_rows = min(n_components + 1, 8)  # Max 8 rows for readability
    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=["Original Signal"] + [
            f"Component {i+1} (T={c.get('period', 'inf'):.1f}d)"
            for i, c in enumerate(decomposition.get('individual_components', [])[:7])
        ],
        vertical_spacing=0.08
    )

    # Original signal
    t = list(range(len(decomposition['original'])))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=decomposition['original'],
            mode='lines',
            name='Original',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Individual components
    for i, comp in enumerate(decomposition.get('individual_components', [])[:7]):
        row = i + 2
        if row > n_rows:
            break

        fig.add_trace(
            go.Scatter(
                x=t,
                y=comp['values'],
                mode='lines',
                name=comp.get('type', f'Component {i+1}'),
                line=dict(width=1.5)
            ),
            row=row, col=1
        )

    fig.update_layout(
        title=f"{ticker} - Signal Decomposition into Frequency Components",
        height=200 * n_rows,
        showlegend=False,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time (samples)", row=n_rows, col=1)
    fig.update_yaxes(title_text="Amplitude")

    fig.write_html(output_path)


def create_forecast_plot(
    historical_dates: List[str],
    historical_prices: List[float],
    forecast: Dict,
    output_path: str = "forecast_plot.html",
    ticker: str = "Stock"
) -> None:
    """
    Create forecast plot with confidence bands.

    Args:
        historical_dates: List of historical date strings
        historical_prices: Historical price series
        forecast: Result from extend_forecast() or soft_projection()
        output_path: Path to save HTML file
        ticker: Stock ticker symbol
    """
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"{ticker} - Price Forecast with Confidence Band",)
    )

    n_hist = len(historical_dates)
    n_forecast = len(forecast['forecast_signal'])

    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )

    # Forecast dates (generate from last historical date)
    from datetime import datetime, timedelta
    last_date = datetime.strptime(historical_dates[-1], '%Y-%m-%d')
    forecast_dates = [
        (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
        for i in range(n_forecast)
    ]

    # Forecast signal
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast['forecast_signal'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        )
    )

    # Confidence band
    confidence_band = forecast.get('confidence_band', {})
    if 'lower' in confidence_band and 'upper' in confidence_band:
        # Create filled area for confidence band
        fig.add_trace(
            go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=confidence_band['upper'] + confidence_band['lower'][::-1],
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,165,0,0)'),
                name='Confidence Band',
                hoverinfo='skip'
            )
        )

    # Add vertical line separating historical and forecast
    fig.add_shape(
        type='line',
        x0=historical_dates[-1],
        y0=0,
        x1=historical_dates[-1],
        y1=1,
        xref='x',
        yref='y domain',
        line=dict(color='gray', width=2, dash='dot')
    )

    # Add confidence score annotation
    confidence_score = forecast.get('confidence_score', 0)
    fig.add_annotation(
        x=forecast_dates[len(forecast_dates)//2],
        y=1.05,
        text=f"Confidence: {confidence_score:.0%}",
        xref='x',
        yref='y domain',
        showarrow=False,
        bgcolor='rgba(255,165,0,0.8)',
        font=dict(color='white', size=12)
    )

    fig.update_layout(
        title=f"{ticker} - FFT-Based Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    fig.update_xaxes(range=[historical_dates[0], forecast_dates[-1]])

    fig.write_html(output_path)


def create_metrics_summary_table(
    metrics: Dict[str, float],
    output_path: str = "metrics_summary.html",
    ticker: str = "Stock"
) -> None:
    """
    Create an HTML table summarizing evaluation metrics.

    Args:
        metrics: Dict of metric name -> value
        output_path: Path to save HTML file
        ticker: Stock ticker symbol
    """
    # Categorize metrics
    accuracy_metrics = ['rmse', 'mae', 'mape', 'directional_accuracy']
    trading_metrics = ['sharpe_ratio', 'max_drawdown', 'cumulative_return', 'volatility', 'win_rate']

    fig = go.Figure()

    # Build table rows
    row_labels = []
    row_values = []

    for metric in accuracy_metrics:
        if metric in metrics:
            row_labels.append(metric.replace('_', ' ').title())
            val = metrics[metric]
            if metric == 'mape' or metric == 'directional_accuracy' or metric == 'win_rate':
                row_values.append(f"{val:.2f}%")
            else:
                row_values.append(f"${val:.2f}")

    for metric in trading_metrics:
        if metric in metrics:
            row_labels.append(metric.replace('_', ' ').title())
            val = metrics[metric]
            if metric == 'sharpe_ratio':
                row_values.append(f"{val:.3f}")
            elif metric in ['max_drawdown', 'cumulative_return', 'volatility']:
                row_values.append(f"{val:.2f}%")
            else:
                row_values.append(f"{val:.2f}")

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='royalblue',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[row_labels, row_values],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )
    )

    fig.update_layout(
        title=f"{ticker} - Prediction Metrics Summary",
        height=400,
        width=500
    )

    fig.write_html(output_path)
