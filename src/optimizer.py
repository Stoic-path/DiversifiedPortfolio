"""
Módulo de optimización de portafolio de Markowitz.
Implementación del modelo cuadrático y restricciones.
"""

import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions

def build_problem(mu: pd.Series, sigma: pd.DataFrame, universe_dict: dict):
    """
    Construye el objeto de Frontera Eficiente con las restricciones del proyecto.
    """
    # 1. Inicializar la frontera eficiente (Restricciones implícitas: sum(w)=1 y w_i >= 0)
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0, 0.25)) # w_i <= 0.25 (Máximo por activo)

    # 2. Obtener índices de activos por categoría
    all_assets = mu.index.tolist()
    crypto_indices = [i for i, ticker in enumerate(all_assets) if ticker in universe_dict['cryptos']]
    commodity_indices = [i for i, ticker in enumerate(all_assets) if ticker in universe_dict['commodities']]

    # 3. Añadir restricciones de grupo (Límite del 33% por clase)
    # Criptomonedas <= 33%
    ef.add_constraint(lambda w: sum(w[i] for i in crypto_indices) <= 0.33)
    # Commodities <= 33%
    ef.add_constraint(lambda w: sum(w[i] for i in commodity_indices) <= 0.33)

    return ef

def solve(ef: EfficientFrontier, method: str = "max_sharpe", target_return: float | None = None):
    """
    Resuelve el problema de optimización.
    """
    # ... (el resto del código se mantiene igual)
    if method == "max_sharpe":
        # Maximizar Ratio de Sharpe (Tangencia)
        weights = ef.max_sharpe()
    elif method == "min_volatility":
        # Minimizar varianza w^T Σ w
        weights = ef.min_volatility()
    elif method == "efficient_return" and target_return is not None:
        # Minimizar riesgo para un retorno objetivo mu^T w >= r_target
        weights = ef.efficient_return(target_return)
    else:
        raise ValueError(f"Método {method} no soportado o falta target_return.")
    
    return ef.clean_weights()

def get_portfolio_performance(ef: EfficientFrontier):
    """Retorna (Retorno esperado, Volatilidad, Ratio de Sharpe) del portafolio óptimo."""
    return ef.portfolio_performance(verbose=True)