# FFT-Trading - Project Context for Claude

## Objetivo del Proyecto

**Predicción real de mercados** usando Transformada Rápida de Fourier (FFT) para analizar el espectro de frecuencias de precios y extrapolar tendencias futuras.

## Requisitos del Usuario

| Requisito | Configuración |
|-----------|---------------|
| Mercados | Índices (^GSPC, ^IXIC, ^DJI) y acciones individuales |
| Horizonte | Semanas a meses (medium-long term) |
| Datos | yfinance directo |
| Output | Guardar resultados (CSV/SQLite) + visualización HTML |

## Arquitectura del Sistema

```
src/fft_trading/
├── __init__.py              # API pública del paquete
├── data_fetcher.py          # Descarga datos de yfinance
├── fft_analysis.py          # Análisis FFT básico con scipy
├── spectral_analysis.py     # Análisis espectral avanzado (periodos, ciclos dominantes)
├── reconstruction.py        # Reconstrucción de señal desde frecuencias
├── prediction.py            # Train/test split + predicción FFT; find_optimal_n_components, compute_stability_weights
├── metrics.py               # Métricas de evaluación (RMSE, MAE, MAPE, directional accuracy)
├── backtest.py              # Backtesting de estrategia + walk-forward analysis
├── visualization.py         # Gráficos Plotly interactivos (incluye create_dashboard)
├── storage.py               # Exportación CSV/SQLite
├── logging_config.py        # Configuración de logging
├── window_optimizer.py      # Selección adaptativa de ventana histórica óptima (2yr–20yr)
├── ensemble.py              # Predicciones ensemble multi-ventana con bandas de confianza
└── benchmark.py             # Comparación sistemática de métodos (baseline/auto_n/stability/ensemble)

tests/
├── test_data_fetcher.py     # 4 tests
├── test_fft_analysis.py     # 7 tests
├── test_prediction.py       # 5 tests
└── test_visualization.py    # 4 tests

main.py                      # CLI principal - pipeline end-to-end
outputs/                     # Resultados (CSV, HTML, SQLite) - no commitear
```

## Estado Actual (2026-04-16)

### ✅ Completado

| Componente | Estado | Descripción |
|------------|--------|-------------|
| **Core** | ✅ | 4 módulos base (data_fetcher, fft_analysis, prediction, visualization) |
| **Extended** | ✅ | spectral_analysis, reconstruction, metrics, storage, logging |
| **P0 - Pipeline** | ✅ | `main.py` con CLI completo (fetch → analyze → predict → save → visualize) |
| **P0 - Backtest** | ✅ | `backtest.py` con métricas RMSE, MAE, MAPE, % aciertos, Sharpe, drawdown |
| **Tests** | ✅ | 20 tests unitarios pasando |
| **Git** | ✅ | `.gitignore` configurado (excluye `__pycache__`, `.pytest_cache`, `outputs/`) |
| **Window Optimizer** | ✅ | `window_optimizer.py` — evalúa ventanas 2yr–20yr vía walk-forward + estabilidad espectral |
| **Ensemble** | ✅ | `ensemble.py` — combina predicciones de múltiples ventanas (pesos por performance/stability/igual) |
| **Benchmark** | ✅ | `benchmark.py` — comparación sistemática: baseline vs auto_n vs stability vs ensemble |
| **Auto N components** | ✅ | `find_optimal_n_components()` en `prediction.py` — CV para elegir mejor N |
| **Stability weights** | ✅ | `compute_stability_weights()` — pondera componentes FFT por persistencia temporal |
| **Dashboard unificado** | ✅ | `create_dashboard()` en `visualization.py` — HTML único con todos los gráficos |

### ❌ Pendiente

| Prioridad | Tarea | Descripción |
|-----------|-------|-------------|
| **P1** | Manejo de errores robusto | Validación de inputs, manejo de excepciones y reintentos en descargas |
| **P2** | Documentación | README con ejemplos de uso y documentación de la API |
| **P2** | Análisis de múltiples tickers | Batch processing y comparación de predicciones |
| **P3** | Modelos alternativos | Comparar FFT vs ARIMA, LSTM, Prophet |

## Comandos Útiles

```bash
# Ejecutar tests
pytest tests/ -v

# Pipeline de predicción
python main.py AAPL --start 2024-01-01 --end 2025-03-01 --train-end 2024-10-01

# Backtest histórico
python main.py AAPL --backtest --start 2023-01-01 --end 2024-12-31

# Usar todos los datos disponibles con proyección suave
python main.py ^GSPC --all-data --soft-projection --reconstruct

# Guardar en base de datos
python main.py AAPL --save-to-db --db-path outputs/predictions.db

# Optimización automática de ventana histórica (evalúa 2yr–20yr)
python main.py ^GSPC --all-data --optimize-window --trend-type log

# Auto-selección de N componentes via cross-validation
python main.py AAPL --all-data --auto-components --trend-type log

# Componentes ponderados por estabilidad espectral
python main.py AAPL --all-data --stability-weighted --trend-type log

# Predicción ensemble multi-ventana (2yr, 3yr, 5yr, 10yr)
python main.py ^GSPC --all-data --use-ensemble --ensemble-weighting performance

# Benchmark de métodos (baseline vs auto_n vs stability vs ensemble)
python main.py --benchmark --tickers ^GSPC AAPL MSFT

# Ver estado git
git status

# Instalar dependencias
pip install -r requirements.txt
```

## Decisiones de Diseño

1. **FFT-based prediction:** Se extraen frecuencias dominantes del espectro y se extrapolan como componentes cosenoidales. El número de componentes es configurable (default: 5-10).

2. **yfinance:** Fuente de datos por ser gratuita, fácil de usar, y cubrir índices + acciones globales. Maneja automáticamente splits y dividendos.

3. **Plotly:** Visualización interactiva HTML para análisis exploratorio. Los gráficos incluyen: predicciones, espectro FFT, reconstrucción de señal, y forecast con bandas de confianza.

4. **Módulos separados:** Arquitectura modular donde cada responsabilidad está aislada. Las dataclasses (`StockData`, `FFTResult`, `PredictionResult`, `BacktestResult`) actúan como contratos entre módulos.

5. **Walk-forward analysis:** Además del backtest tradicional, se implementó análisis walk-forward para validar estabilidad del algoritmo en múltiples ventanas temporales.

6. **Window optimizer:** Testa ventanas de 2yr a 20yr en trading days [504, 756, 1260, 2520, 3780, 5040] y elige la mejor usando un composite score: `0.5*MAPE + 0.3*(100-DirAcc) + 0.2*(1-Stability)*100`.

7. **Ensemble multi-ventana:** Corre predicciones desde 4 ventanas (2yr, 3yr, 5yr, 10yr) y las combina. Las bandas de confianza vienen del spread entre ventanas (`± 1.5 * std`). Tres modos de weighting: `performance` (inverso-MAPE), `stability` (estabilidad espectral), `equal`.

8. **Benchmark framework:** Compara métodos side-by-side vía walk-forward con stride configurable. Métodos: `baseline`, `auto_n`, `stability`, `ensemble`. Genera tabla comparativa por ticker con MAPE, median MAPE, DirAcc.

## Notas Técnicas Importantes

### Algoritmo de Predicción
- Usa las frecuencias dominantes del FFT del período de entrenamiento
- Extrapola evaluando la serie de Fourier en puntos futuros: `x[t] = (1/n) * sum_k X[k] * exp(2*pi*i*k*t/n)`
- El componente DC (frecuencia cero) siempre se incluye para capturar el nivel de la serie
- `max_cycle_ratio` (default 0.33) filtra componentes con período > n*ratio para evitar ciclos espurios muy largos
- `_select_dominant_components()` es la función compartida entre predicción y reconstrucción para garantizar selección idéntica de índices FFT

### Métricas de Evaluación
| Métrica | Descripción | Rango |
|---------|-------------|-------|
| RMSE | Root Mean Square Error | >= 0 (menor = mejor) |
| MAE | Mean Absolute Error | >= 0 (menor = mejor) |
| MAPE | Mean Absolute Percentage Error | >= 0 (menor = mejor) |
| Directional Accuracy | % de aciertos en dirección (up/down) | 0-100% (50% = random) |
| Sharpe Ratio | Retorno ajustado por riesgo | >1 es bueno, >2 es excelente |
| Max Drawdown | Caída máxima desde pico | 0-100% (menor = mejor) |

### Limitaciones Conocidas
1. **Estacionalidad:** FFT asume periodicidad; puede fallar en rupturas estructurales del mercado
2. **Componentes:** 5-10 componentes dominantes pueden no capturar toda la dinámica del precio
3. **Tendencia:** El modelo no tiene tendencia inherente; predicciones largas pueden diverger
4. **Volatilidad:** No modela cambios en la volatilidad del mercado

## Próximos Pasos Recomendados

1. **P1 - Robustez:** Agregar manejo de errores y reintentos en `data_fetcher.py`
2. **P2 - Docs:** Crear README con ejemplos y documentación de API
3. **P2 - Batch:** Permitir procesar múltiples tickers en un solo comando (`--tickers AAPL MSFT ^GSPC`)
4. **P3 - Modelos alternativos:** Comparar FFT vs ARIMA, LSTM, Prophet usando el benchmark framework
