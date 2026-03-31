"""Storage module for persisting FFT-Trading predictions and results."""

import sqlite3
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .data_fetcher import StockData
from .prediction import PredictionResult
from .backtest import BacktestResult, Trade


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CSV Export Functions
# ============================================================================

def export_predictions_csv(
    path: str,
    test_dates: List[str],
    test_prices: List[float],
    predicted_prices: List[float],
    metadata: Optional[Dict] = None
) -> None:
    """
    Export predictions to CSV file.

    Args:
        path: Output file path
        test_dates: List of test period dates
        test_prices: Actual prices
        predicted_prices: Predicted prices
        metadata: Optional metadata dict to save as header comments
    """
    ensure_dir(path)

    with open(path, 'w', newline='') as f:
        # Write metadata as comments if provided
        if metadata:
            f.write(f"# FFT-Trading Predictions Export\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")

        # Write header
        writer = csv.writer(f)
        writer.writerow(['date', 'actual_price', 'predicted_price', 'difference', 'error_pct'])

        # Write data
        for i, date in enumerate(test_dates):
            if i < len(predicted_prices):
                actual = test_prices[i]
                pred = predicted_prices[i]
                diff = pred - actual
                error_pct = (diff / actual) * 100 if actual != 0 else 0
                writer.writerow([date, f"{actual:.2f}", f"{pred:.2f}", f"{diff:.2f}", f"{error_pct:.2f}"])


def export_backtest_trades_csv(
    path: str,
    trades: List[Trade],
    metadata: Optional[Dict] = None
) -> None:
    """
    Export backtest trades to CSV.

    Args:
        path: Output file path
        trades: List of Trade objects
        metadata: Optional metadata dict
    """
    ensure_dir(path)

    with open(path, 'w', newline='') as f:
        if metadata:
            f.write(f"# FFT-Trading Backtest Trades\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")

        writer = csv.writer(f)
        writer.writerow([
            'entry_date', 'exit_date', 'entry_price', 'exit_price',
            'position', 'pnl', 'pnl_pct', 'holding_days'
        ])

        for trade in trades:
            writer.writerow([
                trade.entry_date, trade.exit_date,
                f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}",
                trade.position, f"{trade.pnl:.2f}", f"{trade.pnl_pct:.2f}",
                trade.holding_days
            ])


def export_equity_curve_csv(
    path: str,
    equity_curve: List[float],
    dates: List[str],
    metadata: Optional[Dict] = None
) -> None:
    """
    Export equity curve to CSV.

    Args:
        path: Output file path
        equity_curve: List of portfolio values
        dates: Corresponding dates
        metadata: Optional metadata dict
    """
    ensure_dir(path)

    with open(path, 'w', newline='') as f:
        if metadata:
            f.write(f"# FFT-Trading Equity Curve\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")

        writer = csv.writer(f)
        writer.writerow(['date', 'equity'])

        for i, equity in enumerate(equity_curve):
            date = dates[i] if i < len(dates) else f"day_{i}"
            writer.writerow([date, f"{equity:.2f}"])


# ============================================================================
# SQLite Database Functions
# ============================================================================

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Get database connection with proper initialization.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection
    """
    ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str) -> None:
    """
    Initialize database with required tables.

    Args:
        db_path: Path to SQLite database file
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            actual_price REAL,
            predicted_price REAL,
            difference REAL,
            error_pct REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Backtest results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_capital REAL,
            final_capital REAL,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            parameters TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_id INTEGER,
            entry_date TEXT,
            exit_date TEXT,
            entry_price REAL,
            exit_price REAL,
            position TEXT,
            pnl REAL,
            pnl_pct REAL,
            holding_days INTEGER,
            FOREIGN KEY (backtest_id) REFERENCES backtests(id)
        )
    ''')

    # FFT analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fft_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            n_samples INTEGER,
            dominant_frequencies TEXT,
            dominant_periods TEXT,
            dominant_amplitudes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create indices
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtests_ticker ON backtests(ticker)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades(backtest_id)')

    conn.commit()
    conn.close()


def save_predictions_sqlite(
    db_path: str,
    ticker: str,
    predictions: List[Dict],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save predictions to SQLite database.

    Args:
        db_path: Path to SQLite database
        ticker: Stock ticker symbol
        predictions: List of dicts with keys: date, actual_price, predicted_price
        metadata: Optional metadata (not stored in DB, for logging)
    """
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    for pred in predictions:
        actual = pred.get('actual_price')
        predicted = pred.get('predicted_price')
        difference = predicted - actual if actual else 0
        error_pct = (difference / actual * 100) if actual else 0

        cursor.execute('''
            INSERT INTO predictions (
                ticker, prediction_date, actual_price, predicted_price,
                difference, error_pct
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            pred.get('date'),
            actual,
            predicted,
            difference,
            error_pct
        ))

    conn.commit()
    conn.close()


def save_backtest_result_sqlite(
    db_path: str,
    result: BacktestResult,
    parameters: Optional[Dict] = None
) -> int:
    """
    Save backtest result to SQLite database.

    Args:
        db_path: Path to SQLite database
        result: BacktestResult object
        parameters: Optional parameters dict

    Returns:
        ID of inserted backtest record
    """
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Insert backtest result
    cursor.execute('''
        INSERT INTO backtests (
            ticker, start_date, end_date, initial_capital, final_capital,
            total_return, annualized_return, sharpe_ratio, max_drawdown,
            win_rate, total_trades, winning_trades, losing_trades, parameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result.ticker,
        result.start_date,
        result.end_date,
        result.initial_capital,
        result.final_capital,
        result.total_return,
        result.annualized_return,
        result.sharpe_ratio,
        result.max_drawdown,
        result.win_rate,
        result.total_trades,
        result.winning_trades,
        result.losing_trades,
        json.dumps(parameters) if parameters else None
    ))

    backtest_id = cursor.lastrowid

    # Insert trades
    for trade in result.trades:
        cursor.execute('''
            INSERT INTO trades (
                backtest_id, entry_date, exit_date, entry_price, exit_price,
                position, pnl, pnl_pct, holding_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            backtest_id,
            trade.entry_date,
            trade.exit_date,
            trade.entry_price,
            trade.exit_price,
            trade.position,
            trade.pnl,
            trade.pnl_pct,
            trade.holding_days
        ))

    conn.commit()
    conn.close()

    return backtest_id


def save_fft_analysis_sqlite(
    db_path: str,
    ticker: str,
    analysis_date: str,
    n_samples: int,
    dominant_frequencies: List[float],
    dominant_periods: List[float],
    dominant_amplitudes: List[float]
) -> None:
    """
    Save FFT analysis results to SQLite database.

    Args:
        db_path: Path to SQLite database
        ticker: Stock ticker symbol
        analysis_date: Date of analysis
        n_samples: Number of samples analyzed
        dominant_frequencies: List of dominant frequencies
        dominant_periods: List of dominant periods (days)
        dominant_amplitudes: List of amplitudes
    """
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO fft_analysis (
            ticker, analysis_date, n_samples,
            dominant_frequencies, dominant_periods, dominant_amplitudes
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        ticker,
        analysis_date,
        n_samples,
        json.dumps([round(f, 6) for f in dominant_frequencies]),
        json.dumps([round(p, 2) for p in dominant_periods]),
        json.dumps([round(a, 4) for a in dominant_amplitudes])
    ))

    conn.commit()
    conn.close()


def load_historical_predictions(
    db_path: str,
    ticker: str,
    limit: int = 100
) -> List[Dict]:
    """
    Load historical predictions from database.

    Args:
        db_path: Path to SQLite database
        ticker: Stock ticker symbol
        limit: Maximum number of records to return

    Returns:
        List of prediction dicts
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT prediction_date, actual_price, predicted_price, difference, error_pct
        FROM predictions
        WHERE ticker = ?
        ORDER BY prediction_date DESC
        LIMIT ?
    ''', (ticker, limit))

    results = []
    for row in cursor.fetchall():
        results.append({
            'date': row['prediction_date'],
            'actual_price': row['actual_price'],
            'predicted_price': row['predicted_price'],
            'difference': row['difference'],
            'error_pct': row['error_pct']
        })

    conn.close()
    return results


def load_backtest_history(
    db_path: str,
    ticker: Optional[str] = None,
    limit: int = 50
) -> List[Dict]:
    """
    Load historical backtest results.

    Args:
        db_path: Path to SQLite database
        ticker: Optional ticker filter
        limit: Maximum number of records

    Returns:
        List of backtest result dicts
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    if ticker:
        cursor.execute('''
            SELECT * FROM backtests
            WHERE ticker = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (ticker, limit))
    else:
        cursor.execute('''
            SELECT * FROM backtests
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row['id'],
            'ticker': row['ticker'],
            'start_date': row['start_date'],
            'end_date': row['end_date'],
            'total_return': row['total_return'],
            'sharpe_ratio': row['sharpe_ratio'],
            'max_drawdown': row['max_drawdown'],
            'win_rate': row['win_rate'],
            'total_trades': row['total_trades'],
            'parameters': json.loads(row['parameters']) if row['parameters'] else {}
        })

    conn.close()
    return results


def get_backtest_trades(
    db_path: str,
    backtest_id: int
) -> List[Dict]:
    """
    Load trades for a specific backtest.

    Args:
        db_path: Path to SQLite database
        backtest_id: Backtest record ID

    Returns:
        List of trade dicts
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM trades
        WHERE backtest_id = ?
        ORDER BY entry_date
    ''', (backtest_id,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'entry_date': row['entry_date'],
            'exit_date': row['exit_date'],
            'entry_price': row['entry_price'],
            'exit_price': row['exit_price'],
            'position': row['position'],
            'pnl': row['pnl'],
            'pnl_pct': row['pnl_pct'],
            'holding_days': row['holding_days']
        })

    conn.close()
    return results


def get_prediction_statistics(
    db_path: str,
    ticker: str
) -> Dict:
    """
    Get aggregate statistics for predictions.

    Args:
        db_path: Path to SQLite database
        ticker: Stock ticker symbol

    Returns:
        Dict with aggregate statistics
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            COUNT(*) as total_predictions,
            AVG(error_pct) as avg_error_pct,
            MIN(error_pct) as min_error_pct,
            MAX(error_pct) as max_error_pct,
            AVG(actual_price) as avg_actual_price,
            AVG(predicted_price) as avg_predicted_price
        FROM predictions
        WHERE ticker = ?
    ''', (ticker,))

    row = cursor.fetchone()
    conn.close()

    return {
        'total_predictions': row['total_predictions'],
        'avg_error_pct': row['avg_error_pct'],
        'min_error_pct': row['min_error_pct'],
        'max_error_pct': row['max_error_pct'],
        'avg_actual_price': row['avg_actual_price'],
        'avg_predicted_price': row['avg_predicted_price']
    }
