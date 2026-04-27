# Contexto del proyecto: Portfolio Optimizer

## Resumen ejecutivo

Proyecto académico de optimización de portafolio de inversión usando programación cuadrática con restricciones por clase de activo y análisis de condiciones KKT.

Asignatura: **Modelos de Investigación de Operaciones (TIP09BFT01)**, 9° semestre, Universidad Central del Ecuador, 2026-2026.

Profesor: Ing. X.

## Objetivo del proyecto

Implementar el modelo de Markowitz extendido con selección automática de activos desde un universo amplio (no fijo) y restricciones por clase, para presentarlo en dos exposiciones académicas:

- **Exposición 1**: Tipos de problemas no lineales y optimización no restringida.
- **Exposición 2**: Optimización restringida con condiciones de Kuhn-Tucker, programación cuadrática.

El proyecto debe demostrar dominio de los conceptos del syllabus, no solo funcionar técnicamente.

## Modelo matemático

Función objetivo (minimizar varianza del portafolio):

    min  wᵀΣw

Sujeto a:

    μᵀw ≥ r_target          (retorno mínimo deseado)
    Σ wᵢ = 1                (todo el capital invertido)
    wᵢ ≥ 0  ∀i              (sin ventas en corto)
    Σᵢ∈C wᵢ ≤ 0.33          (criptomonedas ≤ 33%)
    Σᵢ∈M wᵢ ≤ 0.33          (commodities ≤ 33%)
    wᵢ ≤ 0.25  ∀i           (máximo 25% por activo individual)

Donde:
- `w` es el vector de pesos (variables de decisión)
- `Σ` es la matriz de covarianza de los retornos
- `μ` es el vector de retornos esperados anualizados
- `C` es el conjunto de criptomonedas
- `M` es el conjunto de commodities

Este es un problema de **programación cuadrática convexa**. La función objetivo wᵀΣw es cuadrática y convexa (porque Σ es semidefinida positiva). Esto garantiza que las condiciones KKT son **necesarias y suficientes** para encontrar el óptimo global.

## Universo de activos (selección amplia, no fija)

El optimizador decide cuáles entran al portafolio óptimo desde este universo:

**Acciones (~15 candidatos del S&P 500 por sector):**
- Tech: AAPL, MSFT, GOOGL, NVDA, META
- Salud: JNJ, UNH, PFE
- Finanzas: JPM, BAC, V
- Consumo: AMZN, WMT, KO, PG
- Energía: XOM, CVX

**Criptomonedas (top 10 por capitalización):**
BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD, AVAX-USD, DOT-USD, MATIC-USD, LINK-USD

**Commodities (ETFs disponibles en Yahoo Finance):**
GLD (oro), SLV (plata), USO (petróleo), UNG (gas natural), DBA (agricultura), CPER (cobre)

## División de trabajo del equipo

- **Persona 1 (compañero)**: modelado matemático, implementación del optimizador, extracción e interpretación de condiciones KKT. Archivos: `optimizer.py`, `kkt_analysis.py`.
- **Persona 2 (yo, líder del equipo)**: descarga y limpieza de datos, cálculo de μ y Σ, backtesting contra benchmarks, validación con datos reales. Archivos: `data_loader.py`, `backtester.py`. Coordina la integración del proyecto.
- **Persona 3 (compañera)**: app interactiva en Streamlit, visualizaciones con Plotly. Archivos: `app.py`.

## Stack tecnológico

- **Python 3.11+**
- **yfinance**: descarga de precios históricos.
- **pandas / numpy**: manipulación de datos y operaciones matriciales.
- **PyPortfolioOpt**: librería principal del optimizador (basada en cvxpy).
- **scipy.optimize**: validación manual del modelo.
- **plotly**: gráficos interactivos para la app.
- **matplotlib**: gráficos estáticos para la presentación.
- **streamlit**: interfaz web interactiva.

## Estructura del proyecto

    portfolio-optimizer/
    ├── src/
    │   ├── data_loader.py      # Persona 2: descarga, limpieza, μ y Σ
    │   ├── optimizer.py        # Persona 1: modelo Markowitz + restricciones
    │   ├── kkt_analysis.py     # Persona 1: extracción de multiplicadores
    │   ├── backtester.py       # Persona 2: backtesting contra benchmarks
    │   └── app.py              # Persona 3: interfaz Streamlit
    ├── data/                   # CSVs cacheados (gitignored)
    ├── docs/                   # Modelo matemático, presentación
    ├── requirements.txt
    ├── CLAUDE.md               # Este archivo
    ├── LICENSE                 # MIT
    └── README.md

## Convenciones de código

- **Idioma**: comentarios y docstrings en español, nombres de variables en inglés (estándar Python).
- **Estilo**: PEP 8. Líneas máximo 100 caracteres.
- **Type hints**: usarlos siempre que sea posible para que el código sea autodocumentado.
- **Docstrings**: estilo NumPy o Google. Incluir descripción, parámetros, retornos y ejemplos cuando aplique.
- **Tests**: no son prioridad, pero si agregas pruebas, usar `pytest`.

## Convenciones de datos

- **Período de datos**: 2-3 años de historia diaria.
- **Frecuencia**: diaria (`interval="1d"`).
- **Anualización**: retornos × 252 días hábiles para acciones, × 252 también para todo (es la convención mixta más usada en proyectos académicos cuando se mezclan clases).
- **Alineación de fechas**: las criptos tienen datos los 7 días, las acciones solo días hábiles. Usar `inner join` por fecha.
- **Cache**: guardar precios descargados como CSV en `data/` para no re-descargar en cada ejecución.

## Conceptos clave a destacar (para la exposición)

El proyecto debe mostrar conexión explícita con estos temas del syllabus:

1. **Programación cuadrática**: la función wᵀΣw es cuadrática.
2. **Programación convexa**: la matriz Σ es semidefinida positiva, lo que hace el problema convexo.
3. **Condiciones KKT**: cada restricción genera un multiplicador (λ₁, λ₂, λ₃, λ₄, νᵢ, ηᵢ). Su interpretación económica es central.
4. **Procesos estocásticos**: los retornos se modelan como variables aleatorias con μ y Σ estimados del histórico.

## Restricciones académicas

- El proyecto debe poder ejecutarse en una máquina nueva con solo `pip install -r requirements.txt`.
- El profesor debe poder clonar el repo y correrlo sin configuración adicional.
- La app interactiva (Streamlit) es valorada porque otros grupos también presentan demos en vivo.

## Notas sobre interacción con Claude Code

- Antes de implementar cualquier módulo, Claude Code debe explicar brevemente qué va a hacer y por qué.
- Después de implementar código, debe explicar las partes no triviales con comentarios claros.
- Si hay decisiones de diseño con trade-offs, mencionarlos explícitamente.
- Preferir código simple y legible sobre código corto y críptico.
- Si una librería ofrece una función que hace exactamente lo que necesitamos, usarla en vez de reimplementar.
- Validar resultados con casos de prueba simples antes de declarar terminado un módulo.

## Convención de mensajes de commit

A partir del segundo commit, usar **Conventional Commits** en inglés:

    <tipo>: <descripción corta en presente>

Tipos: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Ejemplos:
- `feat: add data_loader module with yfinance integration`
- `fix: align dates between stocks and crypto correctly`
- `docs: update README with usage instructions`
- `chore: bump pandas to 2.2.0`