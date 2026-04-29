"""
Punto de entrada principal para el optimizador de portafolios.
"""

from data_loader import load_universe, download_prices, calculate_returns, calculate_mu, calculate_sigma
from optimizer import build_problem, solve, get_portfolio_performance
from backtester import train_test_split, simulate_portfolio, calculate_metrics
import pandas as pd

def main():
    # 1. Cargar configuración y datos
    universe = load_universe()
    all_tickers = universe["stocks"] + universe["cryptos"] + universe["commodities"]
    
    print("Descargando datos...")
    prices = download_prices(all_tickers, period="5y")
    returns = calculate_returns(prices)
    
    # 2. Dividir datos para validación (ejemplo: último año para test)
    split_date = (returns.index.max() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    train_rets, test_rets = train_test_split(returns, split_date)
    
    # 3. Optimización (Fase de Entrenamiento)
    mu = calculate_mu(train_rets)
    sigma = calculate_sigma(train_rets)
    
    print(f"\nOptimizando portafolio (Entrenamiento hasta {split_date})...")
    ef = build_problem(mu, sigma, universe)
    weights = solve(ef, method="max_sharpe")

    # Líneas de prueba a insertar en src/main.py justo después de 'weights = solve(...)'
    from kkt_analysis import extract_multipliers, interpret
    print("\n--- Análisis KKT ---")
    multiplicadores = extract_multipliers(ef)
    interpretaciones = interpret(multiplicadores)
    for texto in interpretaciones:
        print(texto)
    
    print("\nPesos Óptimos Calculados:")
    for ticker, weight in weights.items():
        if weight > 0.01: # Mostrar solo activos relevantes
            print(f"  {ticker}: {weight:.2%}")
            
    # 4. Backtesting (Fase de Prueba)
    print("\nEjecutando Backtest en datos de prueba...")
    port_rets, port_val = simulate_portfolio(weights, test_rets)
    metrics = calculate_metrics(port_rets)
    
    print("\nDesempeño del Portafolio (Out-of-sample):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()