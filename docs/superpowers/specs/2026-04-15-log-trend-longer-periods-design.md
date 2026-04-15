# Design: Log-Linear Trend + Longer Test Periods

**Date:** 2026-04-15  
**Status:** Approved

## Problem

Two critical gaps in the current FFT trading model:

1. **No exponential trend support** — existing trend options (`linear`, `polynomial`) model growth additively. Stock markets grow multiplicatively (compounding). A linear trend in price-space leaves heteroscedastic residuals that violate FFT's stationarity assumption.

2. **Test periods too short** — CLI defaults cover only 1 year of data. FFT needs multiple complete cycles to identify dominant frequencies; medium-term market cycles (quarterly, annual) require 5+ years of history.

## Design

### 1. Log-linear trend in `prediction.py`

Add `extract_log_trend(prices)`:

- Fits `log(price[t]) = a + b·t` via least squares
- Returns:
  - `trend`: `exp(a + b·t)` in price-space (for visualization)
  - `detrended`: `log(price[t]) - (a + b·t)` — stationary log-residuals for FFT
  - `trend_type`: `'log'`
  - `params`: `{'slope': b, 'intercept': a, 'space': 'log'}`

Update `extrapolate_trend` to handle `trend_type='log'`:
- Future trend: `exp(a + b·(n + t_future))`

Update `predict_future_with_trend`:
- Add `trend_type='log'` as a supported option
- When `trend_type='log'`:
  - Detrend in log-space
  - Apply FFT to log-residuals
  - Reconstruct: `prediction = exp(log_cycle_forecast + log_trend_forecast)`
- Change default from `'linear'` to `'log'`

Add `'log'` to the `trend_type` choices in the CLI.

### 2. Backtest uses trend in `backtest.py`

`run_backtest`:
- Add `trend_type: str = 'log'` parameter
- Replace `predict_future(train_prices, hold_period)` with `predict_future_with_trend(train_prices, hold_period, trend_type=trend_type)`

`walk_forward_analysis`:
- Add `trend_type: str = 'log'` parameter
- Replace `predict_future` with `predict_future_with_trend`
- Increase default `window_size` from 250 → 500 trading days

### 3. Longer default periods in `main.py`

| Parameter | Old default | New default |
|-----------|-------------|-------------|
| `--start` | 2025-01-01 | 2020-01-01 |
| `--end` | 2025-12-01 | 2026-04-01 |
| `--train-end` | 2025-10-01 | 2024-01-01 |
| `--trend-type` | `linear` | `log` |
| `--trend-type` choices | 4 options | adds `log` |

This gives ~4 years of training data and ~1.5 years of test period — a sensible ratio for medium-term FFT prediction.

## Files Changed

| File | Change |
|------|--------|
| `src/fft_trading/prediction.py` | Add `extract_log_trend`, update `extrapolate_trend`, update `predict_future_with_trend` |
| `src/fft_trading/backtest.py` | Wire `predict_future_with_trend` into `run_backtest` and `walk_forward_analysis` |
| `main.py` | Update CLI defaults and add `'log'` to `--trend-type` choices |

## Out of Scope

- Changes to visualization modules
- Changes to metrics or storage modules
- Alternative trend models (EMD, polynomial blend)
- New CLI commands or flags beyond `'log'` in `--trend-type`
