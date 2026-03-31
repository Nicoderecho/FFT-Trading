"""Logging configuration module for FFT-Trading."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers to avoid duplicates

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_pipeline_logger(log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger configured for the main pipeline.

    Args:
        log_file: Optional log file path (default: logs/pipeline.log)

    Returns:
        Configured pipeline logger
    """
    if log_file is None:
        log_file = "logs/pipeline.log"

    return setup_logger(
        "fft_trading.pipeline",
        level=logging.INFO,
        log_file=log_file
    )


def log_signal_metrics(
    logger: logging.Logger,
    n_samples: int,
    dominant_frequencies: list,
    dominant_periods: list,
    dominant_amplitudes: list
) -> None:
    """
    Log signal analysis metrics.

    Args:
        logger: Logger instance
        n_samples: Number of samples in signal
        dominant_frequencies: List of dominant frequencies (cycles/day)
        dominant_periods: List of dominant periods (days)
        dominant_amplitudes: List of amplitudes
    """
    logger.info(f"Signal analysis: {n_samples} samples")
    logger.info(f"Dominant frequencies: {len(dominant_frequencies)} components found")

    for i, (freq, period, amp) in enumerate(
        zip(dominant_frequencies[:5], dominant_periods[:5], dominant_amplitudes[:5])
    ):
        logger.debug(
            f"  Component {i+1}: "
            f"freq={freq:.6f} cycles/day, "
            f"period={period:.1f} days, "
            f"amplitude={amp:.2f}"
        )


def log_forecast_metrics(
    logger: logging.Logger,
    forecast_horizon: int,
    forecast_range: tuple,
    confidence_level: float
) -> None:
    """
    Log forecast metrics.

    Args:
        logger: Logger instance
        forecast_horizon: Number of days forecasted
        forecast_range: (min_price, max_price) tuple
        confidence_level: Confidence score (0.0 to 1.0)
    """
    logger.info(f"Forecast horizon: {forecast_horizon} days")
    logger.info(
        f"Predicted range: ${forecast_range[0]:.2f} - ${forecast_range[1]:.2f}"
    )
    logger.info(f"Confidence level: {confidence_level:.1%}")


def log_stability_metrics(
    logger: logging.Logger,
    stability_score: float,
    cycle_variance: float,
    recommendation: str
) -> None:
    """
    Log signal stability metrics.

    Args:
        logger: Logger instance
        stability_score: Stability score (0.0 to 1.0)
        cycle_variance: Variance of dominant cycles across windows
        recommendation: Trading recommendation based on stability
    """
    logger.info(f"Stability score: {stability_score:.2f}")
    logger.debug(f"Cycle variance: {cycle_variance:.6f}")
    logger.info(f"Recommendation: {recommendation}")
