"""Visualization module using plotly for interactive HTML plots."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
import numpy as np
from scipy.fft import fft


def create_prediction_plot(
    stock_data,
    predicted_prices: List[float],
    output_path: str = "prediction_plot.html",
    train_end_idx: Optional[int] = None
) -> None:
    """
    Create interactive plot comparing real vs predicted prices.

    Args:
        stock_data: StockData object with dates and prices
        predicted_prices: Predicted prices (for test period or future)
        output_path: Path to save HTML file
        train_end_idx: Index where train/test split occurs
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
            mode='lines+markers',
            name='Real Price',
            line=dict(color='blue', width=2)
        )
    )

    # Plot predicted prices - align with test period at end of real prices
    pred_dates = dates[-len(predicted_prices):]

    fig.add_trace(
        go.Scatter(
            x=pred_dates,
            y=predicted_prices,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        )
    )

    # Add vertical line for train/test split if provided
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
