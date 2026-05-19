"""
Módulo de optimización de portafolio.

Modelo (Markowitz extendido, formulación oficial del grupo):

    min  f(w) = wᵀ Σ w                          (varianza del portafolio)
    s.a. μᵀ w ≥ r_objetivo                       (g_return)
         wᵢ ≥ 0,        ∀ i ∈ {1, ..., N}       (g_nonneg)
         wᵢ ≤ 0.25,     ∀ i ∈ {1, ..., N}       (g_max)
         Σ_{i ∈ C} wᵢ ≤ 0.33                    (g_crypto)
         Σ_{i ∈ M} wᵢ ≤ 0.33                    (g_commodity)
         Σ_i wᵢ = 1                             (h_budget)

Donde C y M son los índices de criptomonedas y commodities en el universo.

Notación de multiplicadores de Lagrange:
    λ_return, λ_nonneg, λ_max, λ_crypto, λ_commodity ≥ 0     (desigualdades)
    ν_budget ∈ ℝ                                              (igualdad)

La diapositiva del grupo usa μ_j para los multiplicadores de igualdad,
pero aquí se renombra a ν (nu) para no colisionar con μ = vector de
retornos esperados. Es la convención de Boyd & Vandenberghe.

Nota sobre la forma estándar QP (½) xᵀ Q x:
- En este proyecto el objetivo se escribe wᵀ Σ w (sin el factor ½),
  alineado con la definición financiera estándar de varianza.
- Si se quisiera reescribir en la forma estándar, Q = 2 Σ. Los duales
  reportados por CVXPY corresponden al objetivo wᵀ Σ w tal como está
  formulado, no a la forma con factor ½.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class QPSolution:
    """Resultado de resolver el problema cuadrático del portafolio.

    Attributes
    ----------
    weights : pd.Series
        Pesos óptimos w*, indexados por ticker.
    duals : dict[str, float | np.ndarray]
        Multiplicadores de Lagrange con nombres del modelo. Claves:
        - 'lambda_return'     : escalar, asociado a μᵀw ≥ r_obj.
        - 'lambda_nonneg'     : vector (N,), asociado a w_i ≥ 0.
        - 'lambda_max'        : vector (N,), asociado a w_i ≤ 0.25.
        - 'lambda_crypto'     : escalar, asociado al tope cripto.
        - 'lambda_commodity'  : escalar, asociado al tope commodities.
        - 'nu_budget'         : escalar, asociado a Σw = 1.
    objective_value : float
        Varianza óptima wᵀΣw evaluada en w*.
    target_return : float
        Retorno mínimo objetivo usado al resolver.
    status : str
        Estado devuelto por CVXPY ('optimal', 'optimal_inaccurate', ...).
    universe : dict[str, list[str]]
        Diccionario del universo (stocks, cryptos, commodities) usado.
    mu : pd.Series
        Vector de retornos esperados usado, alineado con weights.
    """
    weights: pd.Series
    duals: dict
    objective_value: float
    target_return: float
    status: str
    universe: dict
    mu: pd.Series

    @property
    def expected_return(self) -> float:
        """Retorno esperado del portafolio óptimo: μᵀ w*."""
        return float((self.mu * self.weights).sum())

    @property
    def volatility(self) -> float:
        """Volatilidad anual (raíz de la varianza óptima)."""
        return float(np.sqrt(self.objective_value))


def solve_qp(
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: dict[str, list[str]],
    target_return: float,
    max_weight: float = 0.25,
    crypto_cap: float = 0.33,
    commodity_cap: float = 0.33,
    solver: str | None = None,
) -> QPSolution:
    """
    Resuelve el problema cuadrático del portafolio en su forma canónica.

    Parameters
    ----------
    mu : pd.Series
        Retornos esperados anualizados, indexados por ticker.
    sigma : pd.DataFrame
        Matriz de covarianza anualizada N×N, alineada con mu en filas y cols.
    universe : dict[str, list[str]]
        Diccionario con claves 'stocks', 'cryptos', 'commodities'.
    target_return : float
        Retorno mínimo objetivo anualizado (ej. 0.15 = 15%).
    max_weight : float, default 0.25
        Peso máximo por activo individual.
    crypto_cap : float, default 0.33
        Suma máxima permitida para criptomonedas.
    commodity_cap : float, default 0.33
        Suma máxima permitida para commodities.
    solver : str | None
        Solver de CVXPY ('CLARABEL', 'ECOS', 'OSQP', ...). None deja
        que CVXPY elija (típicamente CLARABEL para QP convexa).

    Returns
    -------
    QPSolution
        Pesos óptimos, duales etiquetados, valor objetivo y metadatos.

    Raises
    ------
    ValueError
        Si mu y sigma no están alineados.
    RuntimeError
        Si el solver no encuentra óptimo (p. ej. target_return inalcanzable
        dadas las restricciones por clase y por activo).
    """
    if not sigma.index.equals(mu.index) or not sigma.columns.equals(mu.index):
        raise ValueError(
            "mu y sigma deben compartir el mismo índice de tickers "
            "(filas y columnas de sigma alineadas con mu)."
        )

    tickers = mu.index.tolist()
    n = len(tickers)

    crypto_idx = [i for i, t in enumerate(tickers) if t in universe["cryptos"]]
    commodity_idx = [i for i, t in enumerate(tickers) if t in universe["commodities"]]

    w = cp.Variable(n, name="w")

    # psd_wrap evita la verificación numérica de PSD-ness en cada llamada;
    # data_loader ya garantiza que Sigma es semidefinida positiva por
    # construcción (matriz de covarianza muestral).
    Sigma_np = sigma.values
    mu_np = mu.values

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma_np)))

    # Cada restricción se mantiene en una variable local para poder
    # consultar su .dual_value de forma legible después de resolver.
    g_return = mu_np @ w >= target_return
    g_nonneg = w >= 0
    g_max = w <= max_weight
    g_crypto = cp.sum(w[crypto_idx]) <= crypto_cap
    g_commodity = cp.sum(w[commodity_idx]) <= commodity_cap
    h_budget = cp.sum(w) == 1

    constraints = [g_return, g_nonneg, g_max, g_crypto, g_commodity, h_budget]

    problem = cp.Problem(objective, constraints)
    if solver is not None:
        problem.solve(solver=solver)
    else:
        problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"El solver no encontró óptimo. Estado: {problem.status}. "
            f"Causa probable: target_return={target_return:.2%} es inalcanzable "
            "dadas las cotas por clase y por activo."
        )

    # Redondeo defensivo: el solver puede dejar residuos numéricos -1e-10
    # que se ven feos al imprimir; no afecta la solución.
    w_vals = np.asarray(w.value, dtype=float)
    w_vals = np.where(np.abs(w_vals) < 1e-9, 0.0, w_vals)
    weights = pd.Series(w_vals, index=tickers, name="w_star")

    duals = {
        "lambda_return": float(g_return.dual_value),
        "lambda_nonneg": np.asarray(g_nonneg.dual_value, dtype=float),
        "lambda_max": np.asarray(g_max.dual_value, dtype=float),
        "lambda_crypto": float(g_crypto.dual_value),
        "lambda_commodity": float(g_commodity.dual_value),
        "nu_budget": float(h_budget.dual_value),
    }

    return QPSolution(
        weights=weights,
        duals=duals,
        objective_value=float(problem.value),
        target_return=target_return,
        status=problem.status,
        universe=universe,
        mu=mu,
    )


def exclude_assets(
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: dict[str, list[str]],
    to_exclude: list[str],
) -> tuple[pd.Series, pd.DataFrame, dict[str, list[str]]]:
    """
    Devuelve (mu, sigma, universe) recortados quitando los tickers indicados.

    Útil para la doble corrida 33 vs 31 (sin PFE, sin DOT-USD) requerida
    en la sección B del entregable.

    Parameters
    ----------
    mu, sigma : retornos y covarianza originales.
    universe : diccionario del universo original.
    to_exclude : lista de tickers a quitar (ignora los que no estén presentes).

    Returns
    -------
    tuple
        Versiones filtradas de mu, sigma y universe. Se preserva el
        orden relativo del índice original.
    """
    keep = [t for t in mu.index if t not in to_exclude]
    mu_f = mu.loc[keep]
    sigma_f = sigma.loc[keep, keep]
    universe_f = {
        cls: [t for t in members if t in keep]
        for cls, members in universe.items()
    }
    return mu_f, sigma_f, universe_f
