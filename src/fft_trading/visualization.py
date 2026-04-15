"""Visualization module using plotly for interactive HTML plots."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
from scipy.fft import fft

# ── Dashboard palette ──────────────────────────────────────────────────────────
_BG_PAGE  = '#0a0e1a'
_BG_CARD  = '#0f1724'
_BG_CHART = '#0c1322'
_BORDER   = '#1e2d45'
_TEXT     = '#e2e8f0'
_MUTED    = '#64748b'
_BLUE     = '#3b82f6'
_GREEN    = '#10b981'
_RED      = '#ef4444'
_YELLOW   = '#f59e0b'
_INDIGO   = '#6366f1'
_PURPLE   = '#8b5cf6'
# ──────────────────────────────────────────────────────────────────────────────


def _base_layout(title: str = '', height: int = 480, rows: int = 1) -> dict:
    """Shared dark layout settings for all dashboard Plotly figures."""
    axis = dict(
        gridcolor='#1a2740',
        linecolor='#243350',
        zerolinecolor='#243350',
        tickfont=dict(color=_MUTED, size=11),
    )
    layout = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=_BG_CHART,
        font=dict(family='Inter, system-ui, sans-serif', color=_TEXT, size=12),
        title=dict(text=title, font=dict(size=14, color=_TEXT, family='Inter'), x=0.01, y=0.98),
        margin=dict(l=16, r=16, t=44, b=16),
        height=height,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(10,14,26,0.85)',
            bordercolor=_BORDER,
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    # Apply axis styles to all subplot axes
    for i in range(1, rows + 1):
        sfx = '' if i == 1 else str(i)
        layout[f'xaxis{sfx}'] = axis.copy()
        layout[f'yaxis{sfx}'] = axis.copy()
    return layout


def create_dashboard(
    stock_data,
    predicted_prices: List[float],
    train_end_idx: Optional[int],
    spectral_data: Dict,
    dominant_cycles: List[Dict],
    output_path: str = "dashboard.html",
    train_fit: Optional[List[float]] = None,
    metrics: Optional[Dict] = None,
    reconstruction: Optional[Dict] = None,
    trend_info: Optional[Dict] = None,
) -> None:
    """
    Generate a single modern dark-themed dashboard HTML combining all analysis.

    Tabs: Prediction | Frequency Spectrum | [Reconstruction] | Metrics
    Metric cards are rendered as styled HTML (not a Plotly table).

    Args:
        stock_data: StockData with dates, prices, ticker
        predicted_prices: Model predictions for the test/forecast period
        train_end_idx: Index of the train/test split in stock_data.dates
        spectral_data: Dict from compute_frequency_spectrum (periods, power, …)
        dominant_cycles: List of dominant cycle dicts from detect_dominant_cycles
        output_path: Path to write the single HTML file
        train_fit: Optional in-sample FFT reconstruction over training period
        metrics: Optional dict of evaluation metrics (rmse, mape, …)
        reconstruction: Optional dict from reconstruct_signal_from_top_frequencies
        trend_info: Optional metadata dict from predict_future_with_trend
    """
    dates = stock_data.dates
    prices = stock_data.prices
    ticker = stock_data.ticker

    # ── 1. Prediction figure ───────────────────────────────────────────────────
    pred_fig = go.Figure()

    pred_fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode='lines', name='Real Price',
        line=dict(color=_BLUE, width=2),
        hovertemplate='%{y:$.2f}<extra>Real</extra>',
    ))

    if train_fit is not None and train_end_idx is not None:
        train_dates = dates[:train_end_idx + 1]
        pred_fig.add_trace(go.Scatter(
            x=train_dates, y=train_fit[:len(train_dates)],
            mode='lines', name='FFT Fit (train)',
            line=dict(color=_GREEN, width=1.5, dash='dot'),
            opacity=0.85,
            hovertemplate='%{y:$.2f}<extra>FFT Fit</extra>',
        ))
        pred_dates = dates[train_end_idx + 1: train_end_idx + 1 + len(predicted_prices)]
        pred_fig.add_trace(go.Scatter(
            x=pred_dates, y=predicted_prices[:len(pred_dates)],
            mode='lines', name='FFT Prediction',
            line=dict(color=_RED, width=2, dash='dash'),
            hovertemplate='%{y:$.2f}<extra>Prediction</extra>',
        ))
    else:
        pred_dates = dates[-len(predicted_prices):]
        pred_fig.add_trace(go.Scatter(
            x=pred_dates, y=predicted_prices,
            mode='lines', name='FFT Prediction',
            line=dict(color=_RED, width=2, dash='dash'),
            hovertemplate='%{y:$.2f}<extra>Prediction</extra>',
        ))

    if train_end_idx is not None:
        pred_fig.add_shape(
            type='line',
            x0=dates[train_end_idx], x1=dates[train_end_idx],
            y0=0, y1=1, xref='x', yref='y domain',
            line=dict(color=_MUTED, width=1, dash='dot'),
        )
        pred_fig.add_annotation(
            x=dates[train_end_idx], y=0.98,
            text='Train / Test', xref='x', yref='y domain',
            showarrow=False, textangle=-90,
            font=dict(color=_MUTED, size=11),
        )

    pred_fig.update_layout(**_base_layout('Price vs FFT Prediction', height=500))
    pred_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis_tickprefix='$',
    )

    # ── 2. Spectrum figure (2 rows) ────────────────────────────────────────────
    spec_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Power Spectrum', 'Dominant Cycles'),
        vertical_spacing=0.14,
    )

    filt_periods, filt_power = [], []
    for p, pw in zip(spectral_data['periods'], spectral_data['power']):
        if p != float('inf') and p < 500:
            filt_periods.append(p)
            filt_power.append(pw)

    spec_fig.add_trace(go.Scatter(
        x=filt_periods, y=filt_power, mode='markers',
        name='Power',
        marker=dict(size=5, color=_INDIGO, opacity=0.7),
        hovertemplate='Period: %{x:.1f}d<br>Power: %{y:.2f}<extra></extra>',
    ), row=1, col=1)

    dom_periods = [c['period_days'] for c in dominant_cycles]
    dom_powers  = [c['power'] for c in dominant_cycles]
    spec_fig.add_trace(go.Scatter(
        x=dom_periods, y=dom_powers, mode='markers+text',
        name='Dominant Cycles',
        marker=dict(size=14, color=_YELLOW, symbol='star'),
        text=[f"{p:.0f}d" for p in dom_periods],
        textposition='top center',
        textfont=dict(color=_YELLOW, size=11),
        hovertemplate='Period: %{x:.1f}d<br>Power: %{y:.2f}<extra></extra>',
    ), row=2, col=1)

    spec_layout = _base_layout('Frequency Domain Analysis', height=560, rows=2)
    spec_layout['xaxis_title'] = 'Period (days)'
    spec_layout['xaxis2_title'] = 'Period (days)'
    spec_layout['yaxis_title'] = 'Power'
    spec_layout['yaxis2_title'] = 'Power'
    spec_fig.update_layout(**spec_layout)

    # ── 3. Reconstruction figure (optional) ───────────────────────────────────
    recon_html_block = ''
    recon_tab_btn = ''
    if reconstruction is not None:
        recon_fig = go.Figure()
        recon_fig.add_trace(go.Scatter(
            x=dates, y=prices, mode='lines',
            name='Original', line=dict(color=_BLUE, width=2), opacity=0.6,
            hovertemplate='%{y:$.2f}<extra>Original</extra>',
        ))
        recon_fig.add_trace(go.Scatter(
            x=dates, y=reconstruction['reconstructed'], mode='lines',
            name='FFT Reconstructed', line=dict(color=_GREEN, width=2),
            hovertemplate='%{y:$.2f}<extra>Reconstructed</extra>',
        ))
        if 'trend' in reconstruction:
            recon_fig.add_trace(go.Scatter(
                x=dates, y=reconstruction['trend'], mode='lines',
                name='Low-freq Trend', line=dict(color=_YELLOW, width=1.5, dash='dot'),
                hovertemplate='%{y:$.2f}<extra>Trend</extra>',
            ))
        recon_fig.update_layout(**_base_layout('Signal Reconstruction', height=480))
        recon_fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)', yaxis_tickprefix='$')

        recon_div = recon_fig.to_html(full_html=False, include_plotlyjs=False,
                                       config={'displaylogo': False})
        recon_html_block = f'''
  <div id="tab-reconstruction" class="tab-content">
    <div class="chart-card">{recon_div}</div>
  </div>'''
        recon_tab_btn = '<button class="tab-btn" onclick="showTab(\'reconstruction\', this)">Reconstruction</button>'

    # ── 4. Metric cards HTML ───────────────────────────────────────────────────
    metrics_cards = ''
    metrics_tab_btn = ''
    metrics_tab_block = ''
    if metrics:
        def _cls(key, val):
            if key == 'directional_accuracy':
                return 'good' if val >= 55 else ('bad' if val < 45 else 'neutral')
            if key == 'sharpe_ratio':
                return 'good' if val >= 1 else ('bad' if val < 0 else 'neutral')
            if key == 'mape':
                return 'good' if val < 5 else ('bad' if val > 15 else 'neutral')
            return 'neutral'

        def _fmt(key, val):
            pct_keys = {'mape', 'directional_accuracy', 'max_drawdown',
                        'cumulative_return', 'volatility', 'win_rate'}
            if key in pct_keys:
                return f'{val:.1f}%'
            if key == 'sharpe_ratio':
                return f'{val:.3f}'
            return f'${val:.2f}'

        labels = {
            'rmse': 'RMSE', 'mae': 'MAE', 'mape': 'MAPE',
            'directional_accuracy': 'Dir. Accuracy',
            'sharpe_ratio': 'Sharpe', 'max_drawdown': 'Max Drawdown',
            'cumulative_return': 'Cum. Return', 'volatility': 'Volatility',
        }
        cards_html = ''
        for key, label in labels.items():
            if key in metrics:
                val = metrics[key]
                cls = _cls(key, val)
                cards_html += f'''
      <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{_fmt(key, val)}</div>
      </div>'''

        metrics_cards = f'<div class="metrics-grid">{cards_html}\n    </div>'
        metrics_tab_btn = '<button class="tab-btn" onclick="showTab(\'metrics\', this)">Metrics</button>'
        metrics_tab_block = f'''
  <div id="tab-metrics" class="tab-content">
    <div class="metrics-detail-grid">{cards_html}
    </div>
  </div>'''

    # ── 5. Info cards (trend + dominant cycles) ────────────────────────────────
    trend_card = ''
    if trend_info:
        slope = trend_info.get('trend_slope', 0)
        ttype = trend_info.get('trend_type', 'N/A')
        annual = slope * 252 * 100 if ttype == 'log' else None
        annual_str = f' ({annual:+.1f}%/yr)' if annual is not None else ''
        freqs = trend_info.get('dominant_frequencies', [])
        freq_pills = ''.join(
            f'<span class="pill">{1/f:.0f}d</span>' for f in freqs if f > 0
        )
        trend_card = f'''
      <div class="info-card">
        <h4>Trend Model</h4>
        <p><strong>{ttype}</strong> &nbsp;·&nbsp; slope: {slope:.5f}{annual_str}</p>
        <p style="margin-top:8px;color:var(--muted);">Top cycles: {freq_pills}</p>
      </div>'''

    last_price = prices[-1]
    pred_end = predicted_prices[-1] if predicted_prices else last_price
    pred_chg = (pred_end - last_price) / last_price * 100
    pred_class = 'up' if pred_chg >= 0 else 'down'
    pred_arrow = '▲' if pred_chg >= 0 else '▼'
    date_range = f'{dates[0]} → {dates[-1]}'

    cycles_card = ''
    if dominant_cycles:
        cycle_pills = ''.join(
            f'<span class="pill">{c["period_days"]:.0f}d</span>'
            for c in dominant_cycles[:6]
        )
        cycles_card = f'''
      <div class="info-card">
        <h4>Dominant Market Cycles</h4>
        <p>{cycle_pills}</p>
      </div>'''

    info_row = ''
    if trend_card or cycles_card:
        info_row = f'<div class="info-row">{trend_card}{cycles_card}\n    </div>'

    # ── 6. Serialize Plotly charts ─────────────────────────────────────────────
    cfg = {'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
    pred_div  = pred_fig.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    spec_div  = spec_fig.to_html(full_html=False, include_plotlyjs=False, config=cfg)

    # ── 7. Assemble HTML ───────────────────────────────────────────────────────
    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{ticker} — FFT Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-page:  {_BG_PAGE};
      --bg-card:  {_BG_CARD};
      --bg-chart: {_BG_CHART};
      --border:   {_BORDER};
      --text:     {_TEXT};
      --muted:    {_MUTED};
      --blue:     {_BLUE};
      --green:    {_GREEN};
      --red:      {_RED};
      --yellow:   {_YELLOW};
      --indigo:   {_INDIGO};
      --radius:   10px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-page);
      color: var(--text);
      font-family: Inter, system-ui, sans-serif;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}

    /* ── Header ── */
    .header {{
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      padding: 18px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(8px);
    }}
    .header-left {{ display: flex; align-items: center; gap: 12px; }}
    .ticker {{ font-size: 26px; font-weight: 700; letter-spacing: -0.5px; color: var(--text); }}
    .badge {{
      background: rgba(99,102,241,0.15);
      color: var(--indigo);
      border: 1px solid rgba(99,102,241,0.3);
      border-radius: 6px;
      padding: 3px 9px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }}
    .header-stats {{ display: flex; gap: 36px; align-items: center; }}
    .hstat {{ text-align: right; }}
    .hstat-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; }}
    .hstat-value {{ font-size: 17px; font-weight: 600; margin-top: 2px; color: var(--text); }}
    .hstat-value.up   {{ color: var(--green); }}
    .hstat-value.down {{ color: var(--red); }}

    /* ── Metric cards row ── */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      padding: 20px 32px 0;
    }}
    .metric-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 18px;
      transition: border-color 0.2s;
    }}
    .metric-card:hover {{ border-color: var(--indigo); }}
    .metric-label {{
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 6px;
    }}
    .metric-value {{ font-size: 20px; font-weight: 600; }}
    .metric-value.good    {{ color: var(--green); }}
    .metric-value.bad     {{ color: var(--red); }}
    .metric-value.neutral {{ color: var(--blue); }}

    /* ── Tabs ── */
    .tabs {{
      display: flex;
      gap: 2px;
      padding: 20px 32px 0;
      border-bottom: 1px solid var(--border);
    }}
    .tab-btn {{
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--muted);
      cursor: pointer;
      padding: 9px 18px;
      font-size: 13px;
      font-weight: 500;
      border-radius: 6px 6px 0 0;
      margin-bottom: -1px;
      transition: color 0.15s, background 0.15s;
      font-family: inherit;
    }}
    .tab-btn:hover {{ color: var(--text); background: rgba(255,255,255,0.04); }}
    .tab-btn.active {{
      color: var(--indigo);
      border-bottom-color: var(--indigo);
      background: rgba(99,102,241,0.08);
    }}

    /* ── Tab content ── */
    .tab-content {{ display: none; padding: 24px 32px; }}
    .tab-content.active {{ display: block; }}

    /* ── Chart card ── */
    .chart-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      padding: 4px;
    }}

    /* ── Info row below prediction chart ── */
    .info-row {{ display: flex; gap: 14px; margin-top: 16px; flex-wrap: wrap; }}
    .info-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 18px;
      flex: 1;
      min-width: 220px;
    }}
    .info-card h4 {{
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 8px;
    }}
    .info-card p {{ font-size: 13px; color: var(--text); line-height: 1.6; }}
    .pill {{
      display: inline-block;
      background: rgba(99,102,241,0.12);
      color: var(--indigo);
      border: 1px solid rgba(99,102,241,0.25);
      border-radius: 20px;
      padding: 2px 9px;
      font-size: 11px;
      font-weight: 500;
      margin: 2px 3px 2px 0;
    }}

    /* ── Metrics detail tab ── */
    .metrics-detail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }}
    .metrics-detail-grid .metric-card {{ padding: 20px 24px; }}
    .metrics-detail-grid .metric-value {{ font-size: 28px; }}
    .metrics-detail-grid .metric-label {{ font-size: 11px; margin-bottom: 8px; }}

    /* ── Footer ── */
    .footer {{
      text-align: center;
      padding: 20px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--border);
      margin-top: 24px;
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header class="header">
    <div class="header-left">
      <span class="ticker">{ticker}</span>
      <span class="badge">FFT Analysis</span>
    </div>
    <div class="header-stats">
      <div class="hstat">
        <div class="hstat-label">Period</div>
        <div class="hstat-value" style="font-size:13px;color:var(--muted)">{date_range}</div>
      </div>
      <div class="hstat">
        <div class="hstat-label">Last Price</div>
        <div class="hstat-value">${last_price:.2f}</div>
      </div>
      <div class="hstat">
        <div class="hstat-label">Forecast End</div>
        <div class="hstat-value {pred_class}">{pred_arrow} ${pred_end:.2f} ({pred_chg:+.1f}%)</div>
      </div>
    </div>
  </header>

  <!-- Metric cards summary row -->
  {metrics_cards}

  <!-- Tab navigation -->
  <nav class="tabs">
    <button class="tab-btn active" onclick="showTab('prediction', this)">Prediction</button>
    <button class="tab-btn" onclick="showTab('spectrum', this)">Frequency Spectrum</button>
    {recon_tab_btn}
    {metrics_tab_btn}
  </nav>

  <!-- Prediction tab -->
  <div id="tab-prediction" class="tab-content active">
    <div class="chart-card">{pred_div}</div>
    {info_row}
  </div>

  <!-- Spectrum tab -->
  <div id="tab-spectrum" class="tab-content">
    <div class="chart-card">{spec_div}</div>
  </div>

  <!-- Reconstruction tab (optional) -->
  {recon_html_block}

  <!-- Metrics detail tab (optional) -->
  {metrics_tab_block}

  <footer class="footer">
    FFT-Trading &nbsp;·&nbsp; Generated {generated}
  </footer>

  <script>
    function showTab(id, btn) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + id).classList.add('active');
      btn.classList.add('active');
    }}
  </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


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
