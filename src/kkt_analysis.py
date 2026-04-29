"""
Módulo de análisis de condiciones de Karush-Kuhn-Tucker (KKT).
Extrae e interpreta los multiplicadores de Lagrange (variables duales) del problema resuelto.
"""

import numpy as np
import pandas as pd

def extract_multipliers(ef) -> dict:
    """
    Extrae los multiplicadores de Lagrange (dual_value) de las restricciones de cvxpy
    desde el objeto EfficientFrontier ya resuelto.
    """
    multipliers = {}
    
    # ef._constraints almacena las restricciones de cvxpy. 
    # El orden depende de cómo se añadieron en optimizer.py.
    # Por defecto PyPortfolioOpt añade: 0: sum(w)=1, 1: w>=0 bounds.
    # Las siguientes son las custom (criptos, commodities).
    
    for i, constraint in enumerate(ef._constraints):
        # El dual_value puede ser un escalar o un array (ej. para w >= 0)
        dual = constraint.dual_value
        
        # Redondear valores muy cercanos a cero por tolerancia numérica
        if dual is not None:
            if isinstance(dual, np.ndarray):
                dual = np.round(dual, 5)
            else:
                dual = round(float(dual), 5)
                
        multipliers[f"Restricción_{i}"] = dual
        
    return multipliers

def interpret(multipliers: dict, tolerance: float = 1e-4) -> list:
    """
    Genera una interpretación en lenguaje natural de los multiplicadores.
    """
    interpretaciones = []
    
    for name, value in multipliers.items():
        if value is None:
            continue
            
        # Si es un array (como la restricción de positividad w >= 0)
        if isinstance(value, np.ndarray):
            activos_restringidos = (value > tolerance).sum()
            if activos_restringidos > 0:
                interpretaciones.append(
                    f"{name}: Hay {activos_restringidos} activos con peso 0. "
                    "El multiplicador KKT indica cuánto mejoraría el portafolio si permitiéramos ventas en corto (pesos negativos) para estos activos."
                )
        # Si es un escalar (como la suma de pesos = 1, o los topes de 33%)
        elif isinstance(value, float):
            if abs(value) > tolerance:
                interpretaciones.append(
                    f"{name}: Restricción ACTIVA. Multiplicador = {value}. "
                    f"Relajando esta restricción marginalmente (ej. de 33% a 34%), la varianza cambiaría en {-value}."
                )
            else:
                interpretaciones.append(
                    f"{name}: Restricción INACTIVA (holgura > 0). Multiplicador = 0. "
                    "No afecta el óptimo actual."
                )
                
    return interpretaciones

def active_constraints_table(weights: dict, constraints_config: dict) -> pd.DataFrame:
    """
    Genera una tabla booleana evaluando físicamente los pesos contra los límites.
    """
    data = []
    
    # Ejemplo de validación manual
    peso_maximo = max(weights.values())
    data.append({
        "Restricción": "Peso individual <= 25%",
        "Valor Actual": f"{peso_maximo:.2%}",
        "Activa (Límite alcanzado)": peso_maximo >= 0.2499
    })
    
    return pd.DataFrame(data)