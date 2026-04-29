"""
Módulo de Backtesting y Métricas de Desempeño.
Responsable (Persona 2): Validación de resultados y benchmarks.
"""

import pandas as pd
import numpy as np
from typing import cast

def train_test_split(returns: pd.DataFrame, split_date: str):
    """Divide el histórico en entrenamiento (para optimizar) y prueba (para validar)."""
    train = returns.loc[:split_date]
    test = returns.loc[split_date:]
    return train, test

def simulate_portfolio(weights: dict, returns_test: pd.DataFrame):
    """Calcula el valor diario del portafolio basado en los pesos fijos."""
    # Convertir dict de pesos a serie alineada con las columnas de retornos
    w_series = pd.Series(weights).reindex(returns_test.columns).fillna(0)
    
    # Retorno diario del portafolio: R_p = sum(w_i * r_i)
    portfolio_daily_returns = (returns_test * w_series).sum(axis=1)
    
    # Valor acumulado (empezando en 1.0)
    cumulative_value = (1 + portfolio_daily_returns).cumprod()
    return portfolio_daily_returns, cumulative_value

def calculate_metrics(portfolio_returns: pd.Series):
    """Calcula métricas financieras clave."""
    ann_factor = 252
    
    # Usamos cast para decirle a Pylance que esto será un float, no un número complejo
    prod_val = cast(float, (1 + portfolio_returns).prod())
    cum_return = prod_val - 1
    
    volatility = portfolio_returns.std() * np.sqrt(ann_factor)
    sharpe = (portfolio_returns.mean() * ann_factor) / volatility if volatility != 0 else 0
    
    # Max Drawdown
    cum_val = (1 + portfolio_returns).cumprod()
    running_max = cum_val.cummax()
    drawdown = (cum_val / running_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        "Retorno Acumulado": f"{cum_return:.2%}",
        "Volatilidad Anual": f"{volatility:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}"
    }
