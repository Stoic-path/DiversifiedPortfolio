"""
Interfaz interactiva del optimizador.

Ejecución:
    streamlit run src/app.py

Diseñada para la demo en vivo: pesos óptimos, tabla KKT con interpretación
económica y equity curve out-of-sample contra benchmark.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    calculate_mu,
    calculate_returns,
    calculate_sigma,
    download_prices,
    load_universe,
)
from optimizer import solve_qp
from kkt_analysis import multiplier_table, summary
from backtester import (
    benchmark_cumulative,
    calculate_metrics,
    get_benchmark_returns,
    simulate_portfolio,
    train_test_split,
)


st.set_page_config(page_title="Portfolio Optimizer - UCE", layout="wide")
st.title("📈 Optimizador de Portafolio Diversificado")
st.markdown(
    "Modelo de Markowitz extendido con restricciones por clase de activo. "
    "Programación cuadrática convexa resuelta por CVXPY + análisis KKT."
)


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.header("Parámetros del modelo")
target_return = st.sidebar.slider(
    "Retorno objetivo anualizado", 0.05, 0.50, 0.15, 0.01
)
crypto_cap = st.sidebar.slider("Tope criptomonedas", 0.0, 1.0, 0.33, 0.01)
commodity_cap = st.sidebar.slider("Tope commodities", 0.0, 1.0, 0.33, 0.01)
max_weight = st.sidebar.slider("Tope por activo individual", 0.0, 1.0, 0.25, 0.01)

st.sidebar.markdown("---")
benchmark = st.sidebar.selectbox(
    "Benchmark para el backtest", options=["SPY", "equal_weight"], index=0
)


# =============================================================================
# Datos
# =============================================================================
@st.cache_data
def get_data():
    universe = load_universe()
    tickers = universe["stocks"] + universe["cryptos"] + universe["commodities"]
    prices = download_prices(tickers, period="5y")
    returns = calculate_returns(prices)
    return universe, returns


universe, returns = get_data()
split_date = (returns.index.max() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
train_rets, test_rets = train_test_split(returns, split_date)
mu = calculate_mu(train_rets)
sigma = calculate_sigma(train_rets)

st.caption(
    f"Universo: {len(mu)} activos | Entrenamiento: {len(train_rets)} días "
    f"hasta {split_date} | Test OOS: {len(test_rets)} días"
)


# =============================================================================
# Optimización
# =============================================================================
try:
    solution = solve_qp(
        mu=mu,
        sigma=sigma,
        universe=universe,
        target_return=target_return,
        max_weight=max_weight,
        crypto_cap=crypto_cap,
        commodity_cap=commodity_cap,
    )
except RuntimeError as e:
    st.error(
        f"El solver no encontró óptimo. {e}\n\n"
        "Sugerencia: baja el retorno objetivo o sube los topes por clase."
    )
    st.stop()

# ---- Panel superior: composición y métricas básicas ----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pesos óptimos w*")
    df_w = solution.weights[solution.weights > 1e-4].sort_values(ascending=False)
    fig_pie = px.pie(
        values=df_w.values,
        names=df_w.index,
        title="Composición del portafolio",
    )
    st.plotly_chart(fig_pie, width="stretch")

with col2:
    st.subheader("Desempeño del óptimo")
    st.metric("Retorno esperado (μᵀw*)", f"{solution.expected_return:.2%}")
    st.metric("Volatilidad anual", f"{solution.volatility:.2%}")
    st.metric("Varianza óptima", f"{solution.objective_value:.6f}")
    st.metric("Estado del solver", solution.status)

# ---- Análisis KKT ---------------------------------------------------------
st.subheader("Tabla de multiplicadores KKT")
st.markdown(
    "Cada multiplicador es la sensibilidad de la varianza óptima ante un "
    "relajamiento marginal del lado derecho de la restricción asociada."
)
st.dataframe(multiplier_table(solution), width="stretch")

with st.expander("Resumen detallado de la optimización"):
    st.text(summary(solution))


# =============================================================================
# Backtest OOS
# =============================================================================
st.subheader(f"Backtest out-of-sample vs benchmark ({benchmark})")

port_rets, port_cum = simulate_portfolio(solution.weights, test_rets)
bench_rets = get_benchmark_returns(benchmark, test_rets, period="5y")
bench_cum = benchmark_cumulative(bench_rets)

# Alinear y renormalizar a 1.0 en el primer día común.
common_idx = port_cum.index.intersection(bench_cum.index)
p = port_cum.loc[common_idx]
b = bench_cum.loc[common_idx]
p = p / p.iloc[0]
b = b / b.iloc[0]

df_curves = pd.DataFrame({"Portafolio óptimo": p, benchmark: b})
st.line_chart(df_curves)

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Métricas — Portafolio óptimo**")
    st.table(pd.Series(calculate_metrics(port_rets), name="Valor"))
with col4:
    st.markdown(f"**Métricas — Benchmark ({benchmark})**")
    st.table(pd.Series(calculate_metrics(bench_rets), name="Valor"))
