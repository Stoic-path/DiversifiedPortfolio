"""
Análisis de condiciones de Karush-Kuhn-Tucker (KKT) del problema cuadrático.

Recibe un QPSolution con los duales ya etiquetados por nombre de restricción
y produce las tablas y resúmenes que se llevan a la presentación.

Convenciones de interpretación
------------------------------
- Multiplicador de desigualdad λ > 0  ⇔  restricción ACTIVA en el óptimo.
- Multiplicador de desigualdad λ = 0  ⇔  restricción inactiva (con holgura).
- Multiplicador de igualdad ν puede tener cualquier signo (la restricción
  siempre está activa por definición).

En todos los casos, el multiplicador es la sensibilidad de la varianza
óptima ante un relajamiento marginal del lado derecho de la restricción
(teorema del valor envolvente / interpretación económica del dual).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer import QPSolution


_TOLERANCE = 1e-5


def multiplier_table(solution: QPSolution, tolerance: float = _TOLERANCE) -> pd.DataFrame:
    """
    Construye la tabla principal de interpretación económica.

    Cumple el formato exigido por el entregable D1:
        # | Restricción | Expresión | Multiplicador | Valor numérico | ¿Activa? | Lectura económica

    Para las restricciones vectoriales (no-negatividad y tope individual),
    el "valor numérico" reporta el conteo de componentes activas y el
    multiplicador máximo entre ellas, porque mostrar el vector completo
    de 33 entradas en la presentación no es legible.
    """
    duals = solution.duals
    n_assets = len(solution.weights)

    lam_ret = duals["lambda_return"]
    lam_nn = duals["lambda_nonneg"]
    lam_mx = duals["lambda_max"]
    lam_cr = duals["lambda_crypto"]
    lam_co = duals["lambda_commodity"]
    nu_bud = duals["nu_budget"]

    n_nn_active = int(np.sum(lam_nn > tolerance))
    n_mx_active = int(np.sum(lam_mx > tolerance))

    rows = [
        {
            "#": 1,
            "Restricción": "Retorno mínimo",
            "Expresión": "μᵀw ≥ r_obj",
            "Multiplicador": "λ_return",
            "Valor numérico": f"{lam_ret:.6f}",
            "¿Activa?": "Sí" if lam_ret > tolerance else "No",
            "Lectura económica": (
                "Costo marginal en varianza por pedir 1 pp adicional de retorno. "
                "Si está activa, exigir más rentabilidad obliga a aceptar más riesgo."
            ),
        },
        {
            "#": 2,
            "Restricción": "Presupuesto total",
            "Expresión": "Σ wᵢ = 1",
            "Multiplicador": "ν_budget",
            "Valor numérico": f"{nu_bud:.6f}",
            "¿Activa?": "Sí (igualdad)",
            "Lectura económica": (
                "Precio sombra del capital invertido. Cambiar el presupuesto "
                "de 1 a 1+ε desplaza la varianza óptima en ν_budget·ε."
            ),
        },
        {
            "#": 3,
            "Restricción": "No-negatividad",
            "Expresión": "wᵢ ≥ 0",
            "Multiplicador": "λ_nonneg (vector)",
            "Valor numérico": (
                f"{n_nn_active} de {n_assets} activos con λ>0  "
                f"(máx = {lam_nn.max():.6f})"
            ),
            "¿Activa?": "Sí" if n_nn_active > 0 else "No",
            "Lectura económica": (
                "Cada activo con peso 0 tiene un λ que mide cuánto bajaría "
                "la varianza si se permitiera venderlo en corto."
            ),
        },
        {
            "#": 4,
            "Restricción": "Tope criptomonedas",
            "Expresión": "Σ_{i∈C} wᵢ ≤ 0.33",
            "Multiplicador": "λ_crypto",
            "Valor numérico": f"{lam_cr:.6f}",
            "¿Activa?": "Sí" if lam_cr > tolerance else "No",
            "Lectura económica": (
                "Si λ_crypto > 0, el optimizador querría > 33% en cripto. "
                "El valor mide la mejora marginal de varianza por cada 1 pp "
                "de relajación de la cuota."
            ),
        },
        {
            "#": 5,
            "Restricción": "Tope commodities",
            "Expresión": "Σ_{i∈M} wᵢ ≤ 0.33",
            "Multiplicador": "λ_commodity",
            "Valor numérico": f"{lam_co:.6f}",
            "¿Activa?": "Sí" if lam_co > tolerance else "No",
            "Lectura económica": (
                "Si λ_commodity > 0, el portafolio está limitado por la cuota "
                "de commodities; cuantifica la reducción de varianza alcanzable "
                "al aflojarla."
            ),
        },
        {
            "#": 6,
            "Restricción": "Tope individual",
            "Expresión": "wᵢ ≤ 0.25",
            "Multiplicador": "λ_max (vector)",
            "Valor numérico": (
                f"{n_mx_active} de {n_assets} activos con λ>0  "
                f"(máx = {lam_mx.max():.6f})"
            ),
            "¿Activa?": "Sí" if n_mx_active > 0 else "No",
            "Lectura económica": (
                "Cada activo pegado al tope del 25% tiene un λ_max que "
                "mide cuánto bajaría la varianza si se le permitiera "
                "concentrarse más allá del límite."
            ),
        },
    ]

    return pd.DataFrame(rows)


def active_assets_at_bounds(
    solution: QPSolution,
    tolerance: float = _TOLERANCE,
) -> pd.DataFrame:
    """
    Lista los activos que están pegados a alguna cota (w = 0 o w = 0.25).

    Es la tabla complementaria de "restricciones activas" para las dos
    desigualdades vectoriales: identifica caso por caso qué activo está
    en qué cota y con qué multiplicador.
    """
    weights = solution.weights
    lam_nn = pd.Series(solution.duals["lambda_nonneg"], index=weights.index)
    lam_mx = pd.Series(solution.duals["lambda_max"], index=weights.index)

    rows = []
    for ticker in weights.index:
        w_i = float(weights.loc[ticker])
        if w_i <= tolerance and lam_nn[ticker] > tolerance:
            rows.append({
                "Activo": ticker,
                "Peso w*": f"{w_i:.4%}",
                "Cota activa": "wᵢ = 0",
                "λ asociado": f"{lam_nn[ticker]:.6f}",
            })
        elif w_i >= 0.25 - tolerance and lam_mx[ticker] > tolerance:
            rows.append({
                "Activo": ticker,
                "Peso w*": f"{w_i:.4%}",
                "Cota activa": "wᵢ = 0.25",
                "λ asociado": f"{lam_mx[ticker]:.6f}",
            })

    return pd.DataFrame(rows)


def interpret(solution: QPSolution, tolerance: float = _TOLERANCE) -> list[str]:
    """
    Genera explicaciones legibles, una por restricción del modelo.

    Útil para imprimir en consola; complementa la tabla retornada por
    `multiplier_table`.
    """
    duals = solution.duals
    lam_ret = duals["lambda_return"]
    lam_cr = duals["lambda_crypto"]
    lam_co = duals["lambda_commodity"]
    nu_bud = duals["nu_budget"]
    lam_nn = duals["lambda_nonneg"]
    lam_mx = duals["lambda_max"]

    n_nn = int(np.sum(lam_nn > tolerance))
    n_mx = int(np.sum(lam_mx > tolerance))

    out = []

    # Retorno
    if lam_ret > tolerance:
        out.append(
            f"[λ_return = {lam_ret:.4f}] La restricción de retorno mínimo "
            f"({solution.target_return:.2%}) está ACTIVA: el portafolio se ve "
            "obligado a asumir más varianza para alcanzarlo."
        )
    else:
        out.append(
            "[λ_return ≈ 0] La restricción de retorno está inactiva: el "
            "óptimo de mínima varianza ya supera el retorno objetivo por sí solo."
        )

    # Presupuesto
    out.append(
        f"[ν_budget = {nu_bud:.4f}] Precio sombra del capital. Una unidad "
        "extra de presupuesto desplazaría la varianza óptima en ese factor."
    )

    # Tope cripto
    if lam_cr > tolerance:
        out.append(
            f"[λ_crypto = {lam_cr:.4f}] Tope de criptomonedas ACTIVO: el "
            "portafolio querría asignar más del 33% a esta clase."
        )
    else:
        out.append("[λ_crypto ≈ 0] Tope de cripto inactivo (asignación < 33%).")

    # Tope commodities
    if lam_co > tolerance:
        out.append(
            f"[λ_commodity = {lam_co:.4f}] Tope de commodities ACTIVO."
        )
    else:
        out.append("[λ_commodity ≈ 0] Tope de commodities inactivo.")

    # No-negatividad
    if n_nn > 0:
        out.append(
            f"[λ_nonneg] {n_nn} activos quedaron con peso 0 (no rentables "
            "para el objetivo dado). Su multiplicador mide el costo de "
            "forzarlos a entrar."
        )
    else:
        out.append("[λ_nonneg ≈ 0] Todos los activos del universo entran al portafolio.")

    # Tope individual
    if n_mx > 0:
        out.append(
            f"[λ_max] {n_mx} activos están pegados al tope individual de 25%. "
            "Relajar ese tope les permitiría concentrarse más y bajaría la varianza."
        )
    else:
        out.append("[λ_max ≈ 0] Ningún activo alcanza el tope individual.")

    return out


def summary(solution: QPSolution) -> str:
    """Resumen tabular del óptimo para imprimir en consola."""
    lines = [
        "=" * 78,
        "ANÁLISIS KKT — Resultado de la optimización",
        "=" * 78,
        f"Estado del solver        : {solution.status}",
        f"Retorno objetivo r_obj   : {solution.target_return:.2%}",
        f"Retorno esperado μᵀw*    : {solution.expected_return:.2%}",
        f"Varianza óptima w*ᵀΣw*   : {solution.objective_value:.6f}",
        f"Volatilidad anual        : {solution.volatility:.2%}",
        "",
        "Pesos no nulos (w* > 0.01%):",
    ]
    for ticker, w in solution.weights.sort_values(ascending=False).items():
        if w > 1e-4:
            lines.append(f"  {ticker:10s}  {w:7.2%}")
    return "\n".join(lines)
