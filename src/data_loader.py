"""
Módulo de carga, descarga y preprocesamiento de datos de mercado.

Responsable (Persona 2) del pipeline:
    universo -> descarga -> caché -> alineación -> retornos -> mu y sigma

Estos resultados (mu y sigma) son los inputs del optimizador de Markowitz
implementado en src/optimizer.py.

Convenciones:
- Frecuencia diaria, anualización con factor 252 (días hábiles) para todas las
  clases de activos (acciones, criptos y commodities) por consistencia
  académica al mezclar series.
- Alineación entre stocks (5 días/sem) y criptos (7 días/sem) por inner join
  de fechas; se descartan los días con cualquier NaN.
- Retornos simples (pct_change) por compatibilidad directa con la formulación
  de Markowitz: el retorno del portafolio es mu^T w solo bajo retornos simples.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# Rutas de proyecto resueltas relativas a la raíz del repo, no al cwd, para
# que el módulo funcione tanto al correrlo directamente como al importarlo.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_CONFIG_PATH: Path = _PROJECT_ROOT / "config" / "assets.yaml"
_DATA_DIR: Path = _PROJECT_ROOT / "data"
_CACHE_PATH: Path = _DATA_DIR / "prices.csv"

# Factor estándar de anualización a 252 días hábiles. Aunque las criptos cotizan
# 7 días/sem, al hacer el inner join con stocks quedan ~252 observaciones por
# año, por lo que el factor común es correcto.
TRADING_DAYS_PER_YEAR: int = 252


# =============================================================================
# Carga del universo de activos
# =============================================================================
def load_universe() -> dict[str, list[str]]:
    """
    Carga el universo de activos definido en ``config/assets.yaml``.

    Returns
    -------
    dict[str, list[str]]
        Diccionario con tres claves: ``stocks``, ``cryptos``, ``commodities``,
        cada una mapeando a una lista de tickers compatibles con yfinance.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe en la ruta esperada.
    ValueError
        Si el YAML no contiene las tres clases requeridas.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {_CONFIG_PATH}\n"
            "Debes crear 'config/assets.yaml' con las claves 'stocks', "
            "'cryptos' y 'commodities'."
        )

    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        universe = yaml.safe_load(f)

    required_keys = {"stocks", "cryptos", "commodities"}
    missing = required_keys - set(universe or {})
    if missing:
        raise ValueError(
            f"config/assets.yaml está incompleto. Faltan las claves: {missing}"
        )

    # Normalización defensiva: garantiza que cada valor sea lista de strings.
    return {key: list(universe[key]) for key in required_keys}


# =============================================================================
# Descarga de precios con caché incremental
# =============================================================================
def _download_yf(tickers: list[str], start: str | None, period: str | None) -> pd.DataFrame:
    """
    Wrapper sobre yfinance que devuelve precios de cierre ajustados.

    Maneja la diferencia de forma del DataFrame que devuelve yfinance al pedir
    1 vs N tickers, y emite un warning si algún ticker no devolvió datos.
    """
    # auto_adjust=True hace que la columna 'Close' ya venga ajustada por
    # splits/dividendos; es lo correcto para cálculo de retornos.
    if start is not None:
        raw = yf.download(
            tickers=tickers,
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
    else:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )

    if raw is None or raw.empty:
        warnings.warn(
            f"yfinance no devolvió datos para tickers={tickers}. "
            "Revisa conectividad o validez de los símbolos."
        )
        return pd.DataFrame()

    # Caso 1 ticker: yfinance devuelve columnas planas (Open, High, ..., Close).
    # Caso N tickers: devuelve MultiIndex (campo, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            warnings.warn("La descarga no contiene la columna 'Close'.")
            return pd.DataFrame()
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            warnings.warn("La descarga no contiene la columna 'Close'.")
            return pd.DataFrame()
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    # Tickers que vinieron 100% vacíos (símbolos delisted, mal escritos, etc.)
    fully_empty = [c for c in prices.columns if prices[c].isna().all()]
    if fully_empty:
        warnings.warn(
            f"Tickers sin datos, se descartan: {fully_empty}"
        )
        prices = prices.drop(columns=fully_empty)

    return prices


def _read_cache() -> pd.DataFrame | None:
    """
    Lee el CSV de caché. Si no existe o está corrupto, devuelve None.
    """
    if not _CACHE_PATH.exists():
        return None
    try:
        df = pd.read_csv(_CACHE_PATH, index_col=0, parse_dates=True)
        if df.empty or df.index.isna().any():
            raise ValueError("Caché vacía o con índice inválido")
        return df.sort_index()
    except Exception as exc:
        warnings.warn(
            f"Caché corrupto en {_CACHE_PATH} ({exc}). Se regenerará desde cero."
        )
        return None


def download_prices(
    tickers: list[str],
    period: str = "5y",
    force: bool = False,
) -> pd.DataFrame:
    """
    Descarga precios diarios de cierre ajustado con caché incremental.

    La estrategia de caché minimiza llamadas a yfinance:
    - Si no hay caché o ``force=True``, descarga el período completo.
    - Si hay caché y la última fecha es de hoy o ayer (calendario), usa la
      caché tal cual y no descarga nada.
    - Si la caché está desactualizada, descarga solo los días faltantes.
    - Si el universo creció (tickers nuevos), descarga esos completos para el
      período y los une por columna.

    Parameters
    ----------
    tickers : list[str]
        Lista de símbolos compatibles con yfinance.
    period : str, default "5y"
        Período histórico a descargar la primera vez. Acepta el formato de
        yfinance ("1y", "2y", "5y", "10y", "max").
    force : bool, default False
        Si es True, ignora la caché y re-descarga todo el período.

    Returns
    -------
    pd.DataFrame
        DataFrame de precios indexado por fecha, una columna por ticker, con
        las fechas alineadas (inner join por dropna).
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Caso 1: forzar descarga completa.
    if force:
        prices = _download_yf(tickers, start=None, period=period)
        prices = prices.dropna(how="any")
        prices.to_csv(_CACHE_PATH)
        return prices

    cached = _read_cache()

    # Caso 2: no hay caché válido -> descarga completa.
    if cached is None:
        prices = _download_yf(tickers, start=None, period=period)
        prices = prices.dropna(how="any")
        prices.to_csv(_CACHE_PATH)
        return prices

    # Caso 3: hay caché. Determinar qué falta.
    # 3a: tickers nuevos que no están en la caché.
    cached_tickers = set(cached.columns)
    requested_tickers = set(tickers)
    new_tickers = sorted(requested_tickers - cached_tickers)

    if new_tickers:
        new_data = _download_yf(new_tickers, start=None, period=period)
        if not new_data.empty:
            # Unión por columnas; el outer join puede generar NaN al inicio si
            # los nuevos tickers tienen historia más corta. Se limpian al final.
            cached = cached.join(new_data, how="outer")

    # 3b: días faltantes hasta hoy.
    last_date = cached.index.max().date()
    today = date.today()
    days_behind = (today - last_date).days

    # Yahoo cierra fines de semana y feriados; aceptamos hasta 1 día de delay
    # como "actualizado" para no descargar de más en weekends.
    if days_behind > 1:
        # Pedir desde el día siguiente al último cacheado.
        start = (last_date + timedelta(days=1)).isoformat()
        # Para la actualización pedimos todos los tickers actuales así también
        # extendemos las series de los nuevos.
        update = _download_yf(tickers, start=start, period=None)
        if not update.empty:
            # Concatenar y deduplicar por índice (por si yfinance devuelve
            # solapamiento con la última fila de la caché).
            cached = pd.concat([cached, update])
            cached = cached[~cached.index.duplicated(keep="last")].sort_index()

    # Filtrar a las columnas pedidas (en orden) y aplicar inner join por dropna.
    available = [t for t in tickers if t in cached.columns]
    prices = cached[available].dropna(how="any")

    # Persistimos el caché actualizado (con todas las columnas, no solo las
    # filtradas, para que llamadas futuras con universo distinto reutilicen).
    cached.to_csv(_CACHE_PATH)

    return prices


# =============================================================================
# Cálculo de retornos, mu y sigma
# =============================================================================
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos simples diarios a partir de un DataFrame de precios.

    Se usan retornos simples (no logarítmicos) por compatibilidad con la
    formulación de Markowitz: el retorno esperado del portafolio mu^T w y
    la varianza w^T Sigma w son exactos solo bajo retornos simples.

    Parameters
    ----------
    prices : pd.DataFrame
        Precios indexados por fecha, una columna por ticker.

    Returns
    -------
    pd.DataFrame
        Retornos diarios (r_t = P_t / P_{t-1} - 1), sin la primera fila.
    """
    return prices.pct_change().dropna(how="any")


def calculate_mu(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Calcula el vector de retornos esperados mu.

    mu_i se estima como la media muestral de los retornos diarios del activo i.
    Este es el estimador clásico; existen alternativas más robustas
    (James-Stein, Black-Litterman) que quedan fuera del alcance del proyecto.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios.
    annualize : bool, default True
        Si es True, multiplica por ``TRADING_DAYS_PER_YEAR`` (252).

    Returns
    -------
    pd.Series
        Retorno esperado por activo, indexado por ticker.
    """
    mu = returns.mean()
    if annualize:
        mu = mu * TRADING_DAYS_PER_YEAR
    return mu


def calculate_sigma(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Calcula la matriz de covarianza Sigma de los retornos.

    Sigma es semidefinida positiva por construcción, lo que garantiza que el
    problema cuadrático ``min w^T Sigma w`` es convexo. Esta convexidad es la
    que hace que las condiciones KKT sean necesarias y suficientes para el
    óptimo global (relevante para la Exposición 2).

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios.
    annualize : bool, default True
        Si es True, multiplica la matriz por ``TRADING_DAYS_PER_YEAR`` (252).

    Returns
    -------
    pd.DataFrame
        Matriz de covarianza N x N, indexada por ticker en filas y columnas.
    """
    sigma = returns.cov()
    if annualize:
        sigma = sigma * TRADING_DAYS_PER_YEAR
    return sigma


# =============================================================================
# Bloque de prueba: pipeline completo + health check
# =============================================================================
def _print_health_check(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: dict[str, list[str]],
) -> None:
    """Imprime un resumen legible del pipeline para validación visual."""
    print("=" * 78)
    print("HEALTH CHECK DEL PIPELINE DE DATOS")
    print("=" * 78)

    # --- Universo configurado ---
    print("\n[1] Universo de activos configurado")
    print(f"  - Stocks      : {len(universe['stocks']):3d}  -> {universe['stocks']}")
    print(f"  - Cryptos     : {len(universe['cryptos']):3d}  -> {universe['cryptos']}")
    print(f"  - Commodities : {len(universe['commodities']):3d}  -> {universe['commodities']}")
    total_requested = sum(len(v) for v in universe.values())
    print(f"  - Total solicitado : {total_requested}")

    # --- Estado del DataFrame de precios ---
    print("\n[2] DataFrame de precios (post inner join)")
    print(f"  - Filas (días)         : {prices.shape[0]}")
    print(f"  - Columnas (activos)   : {prices.shape[1]}")
    print(f"  - Rango de fechas      : {prices.index.min().date()}  ->  "
          f"{prices.index.max().date()}")
    print(f"  - Activos con datos    : {list(prices.columns)}")
    dropped = total_requested - prices.shape[1]
    if dropped > 0:
        missing = sorted(
            set(sum(universe.values(), [])) - set(prices.columns)
        )
        print(f"  - Activos descartados  : {dropped} -> {missing}")
    print(f"  - NaN restantes        : {int(prices.isna().sum().sum())}")

    # --- Cabecera de precios ---
    print("\n[3] Primeras filas de precios")
    with pd.option_context("display.width", 140, "display.max_columns", 12):
        print(prices.head())

    # --- mu ordenado ---
    print("\n[4] Retornos esperados mu (anualizados, %), ordenados desc")
    mu_pct = (mu * 100).sort_values(ascending=False)
    for ticker, value in mu_pct.items():
        print(f"  {ticker:10s}  {value:7.2f}%")

    # --- sigma ---
    print("\n[5] Matriz de covarianza Sigma (anualizada), primeras filas/cols")
    with pd.option_context("display.width", 140, "display.max_columns", 8,
                           "display.float_format", "{:.4f}".format):
        print(sigma.iloc[:5, :5])
    print(f"\n  - Forma de Sigma          : {sigma.shape}")
    print(f"  - Sigma simétrica         : {np.allclose(sigma.values, sigma.values.T)}")
    eigvals = np.linalg.eigvalsh(sigma.values)
    print(f"  - Sigma SDP (eigval min)  : {eigvals.min():.6e}")

    # --- Retornos ---
    print("\n[6] DataFrame de retornos diarios")
    print(f"  - Filas : {returns.shape[0]}   Columnas : {returns.shape[1]}")
    print(f"  - Rango : {returns.index.min().date()}  ->  "
          f"{returns.index.max().date()}")

    print("\n" + "=" * 78)
    print("OK  -  Pipeline ejecutado correctamente.")
    print("=" * 78)


if __name__ == "__main__":
    universe = load_universe()
    all_tickers = universe["stocks"] + universe["cryptos"] + universe["commodities"]

    prices = download_prices(all_tickers, period="5y", force=False)
    returns = calculate_returns(prices)
    mu = calculate_mu(returns, annualize=True)
    sigma = calculate_sigma(returns, annualize=True)

    _print_health_check(prices, returns, mu, sigma, universe)
