"""
Interfaz interactiva del optimizador usando Streamlit.
Ejecución: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_universe, download_prices, calculate_returns, calculate_mu, calculate_sigma
from optimizer import build_problem, solve
from kkt_analysis import active_constraints_table

# Configuración de página
st.set_page_config(page_title="Portfolio Optimizer - UCE", layout="wide")

st.title("📈 Optimizador de Portafolio Diversificado")
st.markdown("Modelo de Markowitz extendido con restricciones por clase de activo.")

# =======================
# Barra lateral (Sidebar)
# =======================
st.sidebar.header("Parámetros del Modelo")

target_return = st.sidebar.slider("Retorno Objetivo Anualizado", min_value=0.05, max_value=0.50, value=0.15, step=0.01)
crypto_limit = st.sidebar.slider("Límite Criptomonedas (%)", 0, 100, 33) / 100
commodity_limit = st.sidebar.slider("Límite Commodities (%)", 0, 100, 33) / 100
max_weight = st.sidebar.slider("Límite por activo individual (%)", 0, 100, 25) / 100

method = st.sidebar.radio("Método de Optimización", ("max_sharpe", "efficient_return", "min_volatility"))

# =======================
# Lógica Principal
# =======================
@st.cache_data
def get_data():
    universe = load_universe()
    tickers = universe["stocks"] + universe["cryptos"] + universe["commodities"]
    prices = download_prices(tickers, period="5y")
    returns = calculate_returns(prices)
    return universe, returns

universe, returns = get_data()
mu = calculate_mu(returns)
sigma = calculate_sigma(returns)

st.write("### Datos cargados correctamente. Calculando portafolio óptimo...")

# Resolver optimización (Aquí se conectarían los límites dinámicos al build_problem)
# Por simplicidad se llama a las funciones base
try:
    from pypfopt import EfficientFrontier
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))
    
    # Lógica simplificada para demostración en UI
    all_assets = mu.index.tolist()
    crypto_idx = [i for i, t in enumerate(all_assets) if t in universe['cryptos']]
    ef.add_constraint(lambda w: sum(w[i] for i in crypto_idx) <= crypto_limit)
    
    weights = solve(ef, method=method, target_return=target_return)
    
    # =======================
    # Visualización
    # =======================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pesos Óptimos")
        df_weights = pd.DataFrame.from_dict(weights, orient='index', columns=['Peso'])
        df_weights = df_weights[df_weights['Peso'] > 0.001].sort_values(by='Peso', ascending=False)
        
        # Gráfico circular
        fig = px.pie(df_weights, values='Peso', names=df_weights.index, title="Composición del Portafolio")
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("Análisis KKT y Restricciones")
        st.markdown("**Condiciones Activas Evaluadas:**")
        
        # Uso del módulo KKT
        df_kkt = active_constraints_table(weights, {})
        st.dataframe(df_kkt, use_container_width=True)
        
        st.info("Los multiplicadores KKT completos se calculan en consola mediante `kkt_analysis.py`.")

except Exception as e:
    st.error(f"Error en la optimización: {e}. Puede que el retorno objetivo sea inalcanzable con los activos actuales.")