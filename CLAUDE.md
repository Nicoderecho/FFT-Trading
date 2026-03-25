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
├── __init__.py          # (pendiente: exponer API pública)
├── data_fetcher.py      # Descarga datos de yfinance
├── fft_analysis.py      # Análisis FFT con scipy
├── prediction.py        # Train/test split + predicción
└── visualization.py     # Gráficos Plotly interactivos

tests/
├── test_data_fetcher.py
├── test_fft_analysis.py
├── test_prediction.py
└── test_visualization.py
```

## Estado Actual (2026-03-25)

### ✅ Completado
- 4 módulos principales implementados
- Tests unitarios (20 tests)
- `.gitignore` configurado (excluye `__pycache__`, `.pytest_cache`)

### ❌ Pendiente (Priorizado)

| Prioridad | Tarea | Descripción |
|-----------|-------|-------------|
| **P0** | `main.py` | Pipeline end-to-end: fetch → analyze → predict → save → visualize |
| **P0** | `backtest.py` | Validación con datos históricos: RMSE, MAE, % aciertos |
| **P1** | Manejo de errores | Validación de inputs, try/except, logs |
| **P1** | Almacenamiento | CSV/SQLite para predicciones |
| **P2** | Documentación | README con ejemplos de uso |

## Comandos Útiles

```bash
# Ejecutar tests
pytest tests/ -v

# Ver estado git
git status

# Instalar dependencias
pip install -r requirements.txt
```

## Decisiones de Diseño

1. **FFT-based prediction:** Se extraen frecuencias dominantes del espectro y se extrapolan como componentes cosenoidales
2. **yfinance:** Fuente de datos por ser gratuita, fácil de usar, y cubrir índices + acciones globales
3. **Plotly:** Visualización interactiva HTML para análisis exploratorio
4. **Módulos separados:** Cada responsabilidad aislada para facilitar testing y mantenimiento

## Archivos a Crear (Próxima Sesión)

1. `main.py` - Script principal con CLI para tickers
2. `backtest.py` - Validación de precisión del algoritmo
3. `outputs/` - Carpeta para CSVs y visualizaciones
4. `src/fft_trading/__init__.py` - Exponer API pública

## Notas para Futuras Sesiones

- El algoritmo de predicción actual es simplista (5 componentes dominantes)
- No hay validación de calidad de predicción implementada
- Se necesita backtesting para determinar si el enfoque tiene valor predictivo real
- El manejo de errores es inexistente - agregar antes de producción
