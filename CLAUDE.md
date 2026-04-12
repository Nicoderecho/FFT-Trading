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
├── prediction.py            # Train/test split + predicción FFT
├── metrics.py               # Métricas de evaluación (RMSE, MAE, MAPE, directional accuracy)
├── backtest.py              # Backtesting de estrategia + walk-forward analysis
├── visualization.py         # Gráficos Plotly interactivos
├── storage.py               # Exportación CSV/SQLite
└── logging_config.py        # Configuración de logging

tests/
├── test_data_fetcher.py     # 4 tests
├── test_fft_analysis.py     # 7 tests
├── test_prediction.py       # 5 tests
└── test_visualization.py    # 4 tests

main.py                      # CLI principal - pipeline end-to-end
outputs/                     # Resultados (CSV, HTML, SQLite) - no commitear
```

## Estado Actual (2026-04-12)

### ✅ Completado

| Componente | Estado | Descripción |
|------------|--------|-------------|
| **Core** | ✅ | 4 módulos base (data_fetcher, fft_analysis, prediction, visualization) |
| **Extended** | ✅ | spectral_analysis, reconstruction, metrics, storage, logging |
| **P0 - Pipeline** | ✅ | `main.py` con CLI completo (fetch → analyze → predict → save → visualize) |
| **P0 - Backtest** | ✅ | `backtest.py` con métricas RMSE, MAE, MAPE, % aciertos, Sharpe, drawdown |
| **Tests** | ✅ | 20 tests unitarios pasando |
| **Git** | ✅ | `.gitignore` configurado (excluye `__pycache__`, `.pytest_cache`, `outputs/`) |

### ❌ Pendiente

| Prioridad | Tarea | Descripción |
|-----------|-------|-------------|
| **P1** | Manejo de errores robusto | Validación de inputs, manejo de excepciones y reintentos en descargas |
| **P1** | Optimización de hiperparámetros | Grid search para n_components, hold_period, etc. |
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

## Notas Técnicas Importantes

### Algoritmo de Predicción
- Usa las frecuencias dominantes del FFT del período de entrenamiento
- Extrapola evaluando la serie de Fourier en puntos futuros: `x[t] = (1/n) * sum_k X[k] * exp(2*pi*i*k*t/n)`
- El componente DC (frecuencia cero) siempre se incluye para capturar el nivel de la serie

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
2. **P1 - Optimización:** Implementar grid search para encontrar mejores hiperparámetros
3. **P2 - Docs:** Crear README con ejemplos y documentación de API
4. **P2 - Batch:** Permitir procesar múltiples tickers en un solo comando
