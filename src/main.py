"""
Punto de entrada principal del proyecto.

Orquesta el flujo completo que se lleva a la presentación:

    1. Descarga y prepara datos (data_loader).
    2. Exporta tabla de μ resaltando SOL/NVDA/PFE/DOT  (entregable D2).
    3. Optimiza el portafolio con los 33 activos       (entregable E1, parte A).
    4. Optimiza con 31 activos (sin PFE ni DOT-USD)    (entregable B).
    5. Imprime tabla KKT + interpretación              (entregables A2, D1, E2, E3).
    6. Backtesting OOS contra benchmark SPY            (entregable C1).
    7. Genera la única gráfica de equity curve         (entregable C2 / F).

Todos los artefactos exportables se guardan en ``reports/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Windows: forzar UTF-8 en stdout para imprimir μ, Σ, →, etc. sin que
# la consola cp1252 reviente con UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_loader import (
    calculate_mu,
    calculate_returns,
    calculate_sigma,
    download_prices,
    load_universe,
)
from optimizer import QPSolution, exclude_assets, solve_qp
from kkt_analysis import (
    active_assets_at_bounds,
    interpret,
    multiplier_table,
    summary,
)
from backtester import (
    benchmark_cumulative,
    calculate_metrics,
    get_benchmark_returns,
    plot_equity_curve,
    simulate_portfolio,
    train_test_split,
)


# =============================================================================
# Configuración del experimento
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# Activos que se excluyen en la segunda corrida por baja rentabilidad
# esperada (ver tabla de μ generada como entregable D2).
ASSETS_EXCLUDED = ["PFE", "DOT-USD"]

# Activos resaltados en la tabla de μ (justifican las decisiones de
# exclusión y de "candidatos fuertes").
ASSETS_HIGHLIGHTED = ["SOL-USD", "NVDA", "PFE", "DOT-USD"]

# Retorno objetivo anualizado. Se elige un valor moderadamente exigente
# pero alcanzable dado el histórico reciente.
TARGET_RETURN = 0.15

# Período de descarga y fecha de corte del backtest. El año más reciente
# se reserva para out-of-sample.
DATA_PERIOD = "5y"
OOS_WINDOW_DAYS = 365

# Benchmark contra el cual se mide la estrategia.
BENCHMARK = "SPY"


# =============================================================================
# Tabla de μ (entregable D2)
# =============================================================================
def export_mu_table(mu: pd.Series, highlighted: list[str]) -> pd.DataFrame:
    """
    Construye y exporta la tabla de retornos esperados anualizados.

    Las filas se ordenan de mayor a menor μ. Los activos resaltados
    (SOL/NVDA/PFE/DOT) se marcan en la columna "Destacado" para que en la
    diapositiva se puedan colorear sin recalcular nada.
    """
    df = mu.sort_values(ascending=False).to_frame(name="μ anualizado")
    df.insert(0, "Ticker", df.index)
    df["μ %"] = (df["μ anualizado"] * 100).round(2).astype(str) + "%"
    df["Destacado"] = df["Ticker"].apply(
        lambda t: "⬤" if t in highlighted else ""
    )
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "#"

    REPORTS_DIR.mkdir(exist_ok=True)
    csv_path = REPORTS_DIR / "expected_returns_mu.csv"
    md_path = REPORTS_DIR / "expected_returns_mu.md"
    df.to_csv(csv_path)
    df.to_markdown(md_path)

    return df


# =============================================================================
# Comparación de dos corridas (33 vs 31)
# =============================================================================
def compare_weights(sol_full: QPSolution, sol_reduced: QPSolution) -> pd.DataFrame:
    """Tabla lado a lado de los pesos óptimos de ambas corridas."""
    w_full = sol_full.weights.rename("w* (33 activos)")
    w_red = sol_reduced.weights.rename("w* (31 activos)")
    df = pd.concat([w_full, w_red], axis=1).fillna(0.0)
    # Solo mostramos los activos que aparecen en al menos una corrida con
    # peso no trivial; de otra manera la tabla se llena de ceros.
    mask = (df > 1e-4).any(axis=1)
    df = df[mask].sort_values("w* (33 activos)", ascending=False)
    # pandas 3.x: applymap fue removido; .map() sobre DataFrame es el reemplazo.
    df = df.map(lambda x: f"{x:.2%}")
    return df


def export_solution_artifacts(solution: QPSolution, suffix: str) -> None:
    """Persiste pesos, multiplicadores y activos en cotas en ``reports/``."""
    REPORTS_DIR.mkdir(exist_ok=True)

    mult_df = multiplier_table(solution)
    bounds_df = active_assets_at_bounds(solution)

    mult_df.to_csv(REPORTS_DIR / f"kkt_multipliers_{suffix}.csv", index=False)
    mult_df.to_markdown(REPORTS_DIR / f"kkt_multipliers_{suffix}.md", index=False)

    if not bounds_df.empty:
        bounds_df.to_csv(REPORTS_DIR / f"kkt_active_bounds_{suffix}.csv", index=False)
        bounds_df.to_markdown(REPORTS_DIR / f"kkt_active_bounds_{suffix}.md", index=False)

    solution.weights.to_csv(REPORTS_DIR / f"weights_{suffix}.csv", header=["w*"])


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # ---------------------------------------------------------------------
    # 1. Datos
    # ---------------------------------------------------------------------
    print("\n[1/6] Cargando universo y descargando precios...")
    universe = load_universe()
    all_tickers = universe["stocks"] + universe["cryptos"] + universe["commodities"]
    prices = download_prices(all_tickers, period=DATA_PERIOD)
    returns = calculate_returns(prices)
    print(f"      Activos disponibles : {prices.shape[1]} / {len(all_tickers)}")
    print(f"      Rango               : {returns.index.min().date()} → {returns.index.max().date()}")

    # ---------------------------------------------------------------------
    # 2. Split OOS y estimación de μ, Σ sobre la ventana de entrenamiento
    # ---------------------------------------------------------------------
    split_date = (returns.index.max() - pd.Timedelta(days=OOS_WINDOW_DAYS)).strftime("%Y-%m-%d")
    train_rets, test_rets = train_test_split(returns, split_date)
    mu = calculate_mu(train_rets)
    sigma = calculate_sigma(train_rets)
    print(f"      Corte train/test    : {split_date}")
    print(f"      Días train / test   : {len(train_rets)} / {len(test_rets)}")

    # ---------------------------------------------------------------------
    # 3. Tabla de μ (entregable D2)
    # ---------------------------------------------------------------------
    print(f"\n[2/6] Exportando tabla de μ con activos destacados {ASSETS_HIGHLIGHTED}...")
    mu_table = export_mu_table(mu, ASSETS_HIGHLIGHTED)
    print(f"      → reports/expected_returns_mu.{{csv,md}}")
    print("      Top 5 / Bottom 5 de μ:")
    print(mu_table.head(5).to_string())
    print("      ...")
    print(mu_table.tail(5).to_string())

    # ---------------------------------------------------------------------
    # 4. Optimización A: 33 activos
    # ---------------------------------------------------------------------
    print(f"\n[3/6] Optimización con 33 activos (r_obj = {TARGET_RETURN:.0%})...")
    sol_full = solve_qp(mu, sigma, universe, target_return=TARGET_RETURN)
    print(summary(sol_full))
    export_solution_artifacts(sol_full, suffix="33_activos")

    # ---------------------------------------------------------------------
    # 5. Optimización B: 31 activos (sin PFE ni DOT)
    # ---------------------------------------------------------------------
    print(f"\n[4/6] Optimización con 31 activos (excluidos {ASSETS_EXCLUDED})...")
    mu_r, sigma_r, universe_r = exclude_assets(mu, sigma, universe, ASSETS_EXCLUDED)
    sol_reduced = solve_qp(mu_r, sigma_r, universe_r, target_return=TARGET_RETURN)
    print(summary(sol_reduced))
    export_solution_artifacts(sol_reduced, suffix="31_activos")

    # ---------------------------------------------------------------------
    # 6. Comparación + análisis KKT
    # ---------------------------------------------------------------------
    print("\n[5/6] Comparación de pesos óptimos (33 vs 31)...")
    compare_df = compare_weights(sol_full, sol_reduced)
    print(compare_df.to_string())
    compare_df.to_markdown(REPORTS_DIR / "weights_comparison.md")

    print("\n      Interpretación KKT (corrida con 33 activos):")
    for line in interpret(sol_full):
        print(f"        • {line}")

    print("\n      Tabla de multiplicadores (corrida con 33 activos):")
    print(multiplier_table(sol_full).to_string(index=False))

    # ---------------------------------------------------------------------
    # 7. Backtest OOS y gráfica única
    # ---------------------------------------------------------------------
    print(f"\n[6/6] Backtest OOS vs benchmark '{BENCHMARK}'...")
    # Usamos la corrida reducida (31 activos) como portafolio "final" porque
    # PFE y DOT se excluyeron justamente por baja μ; si quisieran ver la otra,
    # basta cambiar sol_reduced → sol_full aquí.
    port_rets, port_cum = simulate_portfolio(sol_reduced.weights, test_rets)
    bench_rets = get_benchmark_returns(BENCHMARK, test_rets, period=DATA_PERIOD)
    bench_cum = benchmark_cumulative(bench_rets)

    print("      Métricas del portafolio óptimo (OOS):")
    for k, v in calculate_metrics(port_rets).items():
        print(f"        {k:20s} {v}")
    print(f"      Métricas del benchmark {BENCHMARK} (OOS):")
    for k, v in calculate_metrics(bench_rets).items():
        print(f"        {k:20s} {v}")

    REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = REPORTS_DIR / "equity_curve_oos.png"
    plot_equity_curve(
        portfolio_cum=port_cum,
        benchmark_cum=bench_cum,
        benchmark_name=f"Benchmark ({BENCHMARK})",
        title="Portafolio óptimo vs benchmark — out-of-sample",
        save_path=plot_path,
    )
    print(f"      → {plot_path.relative_to(PROJECT_ROOT)}")

    print("\nListo. Artefactos en:", REPORTS_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
