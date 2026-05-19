"""
Backtesting out-of-sample y métricas de desempeño.

Flujo:
    1. train_test_split  : separa retornos en ventana de entrenamiento (para
                           estimar mu y sigma) y de prueba (para validar).
    2. simulate_portfolio : aplica los pesos fijos w* sobre los retornos OOS.
    3. get_benchmark_returns / get_benchmark_cum : construye el benchmark
                           (SPY o equiponderado) sobre el mismo período OOS.
    4. plot_equity_curve  : genera la única gráfica que se lleva a la
                           presentación: portafolio vs benchmark, base 1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def train_test_split(
    returns: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los retornos en (train, test) usando una fecha de corte.

    La fecha pertenece a ambos lados solo nominalmente; los retornos del
    portafolio se simulan sobre `test`, así que el "leakage" potencial de un
    día se diluye.
    """
    train = returns.loc[:split_date]
    test = returns.loc[split_date:]
    return train, test


def simulate_portfolio(
    weights: dict | pd.Series,
    returns_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Simula el desempeño OOS con pesos fijos (buy-and-hold del vector w*).

    Returns
    -------
    portfolio_daily_returns : pd.Series
        Retornos diarios del portafolio en el período de prueba.
    cumulative_value : pd.Series
        Valor acumulado partiendo de 1.0 (curva de equity).
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    w_series = weights.reindex(returns_test.columns).fillna(0)
    portfolio_daily_returns = (returns_test * w_series).sum(axis=1)
    cumulative_value = (1 + portfolio_daily_returns).cumprod()
    return portfolio_daily_returns, cumulative_value


def calculate_metrics(portfolio_returns: pd.Series) -> dict[str, str]:
    """Métricas financieras clave: retorno acumulado, vol anual, Sharpe, MDD."""
    ann_factor = 252

    prod_val = cast(float, (1 + portfolio_returns).prod())
    cum_return = prod_val - 1
    volatility = portfolio_returns.std() * np.sqrt(ann_factor)
    sharpe = (
        (portfolio_returns.mean() * ann_factor) / volatility
        if volatility != 0
        else 0.0
    )

    cum_val = (1 + portfolio_returns).cumprod()
    running_max = cum_val.cummax()
    drawdown = (cum_val / running_max) - 1
    max_drawdown = drawdown.min()

    return {
        "Retorno Acumulado": f"{cum_return:.2%}",
        "Volatilidad Anual": f"{volatility:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
    }


# =============================================================================
# Benchmarks
# =============================================================================
def get_benchmark_returns(
    benchmark: str,
    returns_test: pd.DataFrame,
    period: str = "5y",
) -> pd.Series:
    """
    Construye la serie de retornos diarios del benchmark sobre el período OOS.

    Parameters
    ----------
    benchmark : str
        - "SPY"          : descarga SPY (proxy del S&P 500).
        - "equal_weight" : portafolio equiponderado sobre el universo.
    returns_test : pd.DataFrame
        Retornos diarios OOS. Define el índice temporal del benchmark.
    period : str
        Período de descarga para SPY (solo aplica a "SPY").

    Returns
    -------
    pd.Series
        Retornos diarios del benchmark alineados con `returns_test.index`.
    """
    if benchmark == "SPY":
        # Import local para no crear ciclos al cargar el módulo.
        from data_loader import download_prices, calculate_returns

        spy_prices = download_prices(["SPY"], period=period)
        spy_returns = calculate_returns(spy_prices)
        # Reindexamos al calendario de test (inner join por fecha disponible).
        aligned = spy_returns["SPY"].reindex(returns_test.index).dropna()
        return aligned

    if benchmark == "equal_weight":
        n = returns_test.shape[1]
        weights = pd.Series(1.0 / n, index=returns_test.columns)
        return (returns_test * weights).sum(axis=1)

    raise ValueError(
        f"Benchmark no soportado: {benchmark!r}. "
        "Opciones válidas: 'SPY', 'equal_weight'."
    )


def benchmark_cumulative(benchmark_returns: pd.Series) -> pd.Series:
    """Convierte retornos diarios del benchmark en curva de valor (base 1.0)."""
    return (1 + benchmark_returns).cumprod()


# =============================================================================
# Gráfica única para la presentación
# =============================================================================
def plot_equity_curve(
    portfolio_cum: pd.Series,
    benchmark_cum: pd.Series,
    benchmark_name: str = "Benchmark",
    title: str = "Equity curve out-of-sample",
    save_path: Path | str | None = None,
):
    """
    Una sola gráfica: portafolio óptimo vs benchmark, base 1.0.

    Diseño deliberadamente sobrio: dos líneas, una línea de referencia en 1.0,
    leyenda y grid suave. Pensada para imprimir en la diapositiva sin retoques.

    Returns
    -------
    matplotlib.figure.Figure
        Figura ya renderizada. Si se pasa `save_path`, también se guarda en PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    # Alinear índices: el portafolio se simuló desde una fecha; el benchmark
    # puede tener una fecha de inicio ligeramente distinta por feriados.
    common_index = portfolio_cum.index.intersection(benchmark_cum.index)
    p = portfolio_cum.loc[common_index]
    b = benchmark_cum.loc[common_index]

    # Re-normalizamos a 1.0 en el primer día común para que ambas curvas
    # arranquen en el mismo punto y la comparación sea visual e inmediata.
    p = p / p.iloc[0]
    b = b / b.iloc[0]

    ax.plot(p.index, p.values, label="Portafolio óptimo (w*)", linewidth=2.0)
    ax.plot(
        b.index, b.values,
        label=benchmark_name,
        linewidth=2.0,
        linestyle="--",
    )
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor acumulado (base 1.0)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig
