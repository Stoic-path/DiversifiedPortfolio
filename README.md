# Portfolio Optimizer

Optimización de portafolio mixto (acciones, criptomonedas y commodities) usando programación cuadrática con restricciones por clase de activo y análisis de condiciones de Karush-Kuhn-Tucker (KKT).

Proyecto académico para la asignatura **Modelos de Investigación de Operaciones**, Universidad Central del Ecuador, 2026-2026.

## Descripción

Este proyecto implementa el modelo clásico de Markowitz extendido con selección de activos:

- **Universo de selección amplio** distribuido en 3 clases:
  - Acciones: ~15 empresas del S&P 500 de distintos sectores.
  - Criptomonedas: top 10 por capitalización de mercado.
  - Commodities: ETFs de oro, plata, petróleo, gas, agricultura y cobre.
- **Selección automática**: el optimizador decide qué activos entran al portafolio (algunos quedarán con peso 0).
- **Restricciones por clase**: criptomonedas ≤ 33%, commodities ≤ 33%.
- **Análisis KKT**: extracción e interpretación de los multiplicadores de Lagrange para cada restricción activa.
- **Backtesting** contra benchmarks: S&P 500, Bitcoin puro, portafolio sin cripto.
- **App interactiva** con Streamlit para visualizar la frontera eficiente.

## Modelo matemático

Se minimiza la varianza del portafolio (medida estándar de riesgo financiero):

    min  wᵀΣw

    sujeto a:
      μᵀw ≥ r_target          (retorno mínimo deseado)
      Σ wᵢ = 1                (todo invertido)
      wᵢ ≥ 0  ∀i              (sin ventas en corto)
      Σᵢ∈C wᵢ ≤ 0.33          (criptomonedas ≤ 33%)
      Σᵢ∈M wᵢ ≤ 0.33          (commodities ≤ 33%)

Donde `w` es el vector de pesos, `Σ` la matriz de covarianza, `μ` los retornos esperados, `C` el conjunto de criptomonedas y `M` el conjunto de commodities.

La función objetivo wᵀΣw es cuadrática (programación cuadrática), convexa (garantiza óptimo global), y permite aplicar condiciones KKT como condiciones necesarias y suficientes.

## Stack tecnológico

- Python 3.11+
- yfinance para descarga de precios históricos.
- pandas / numpy para manipulación de datos.
- PyPortfolioOpt para la optimización cuadrática.
- scipy.optimize para validación manual del modelo.
- Streamlit + Plotly para la interfaz interactiva.

## Instalación

### Requisitos previos

- Python 3.11 o superior
- Git

### Pasos

1. Clonar el repositorio:

       git clone git@github.com:Stoic-path/DiversifiedPortfolio.git
       cd DiversifiedPortfolio

2. Crear y activar entorno virtual:

       python -m venv venv

   Activación en Windows:

       venv\Scripts\activate

   Activación en Linux / macOS:

       source venv/bin/activate

3. Instalar dependencias:

       pip install -r requirements.txt

## Uso

Ejecutar el optimizador:

    python src/main.py

Lanzar la app interactiva:

    streamlit run src/app.py

## Estructura del proyecto

    portfolio-optimizer/
    ├── src/
    │   ├── data_loader.py      # Descarga y limpieza de datos
    │   ├── optimizer.py        # Modelo Markowitz + restricciones
    │   ├── kkt_analysis.py     # Extracción de multiplicadores KKT
    │   ├── backtester.py       # Backtesting contra benchmarks
    │   └── app.py              # Interfaz Streamlit
    ├── data/                   # CSVs cacheados (no rastreados por git)
    ├── docs/                   # Documentación adicional
    ├── requirements.txt        # Dependencias Python
    ├── CLAUDE.md               # Contexto para Claude Code
    ├── LICENSE                 # Licencia MIT
    └── README.md               # Este archivo

## Equipo

- Stoic-path — Datos, backtesting y validación de resultados.
- NOMBRE COMPAÑERO 1 — Modelado matemático y análisis KKT.
- NOMBRE COMPAÑERO 2 — Interfaz y visualizaciones.

## Profesor responsable

Ing. X — Modelos de Investigación de Operaciones (TIP09BFT01).

## Licencia

Distribuido bajo licencia MIT. Ver LICENSE para más información.