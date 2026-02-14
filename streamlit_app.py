import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

from core.binomial import simulate_binomial
from core.geometric import simulate_geometric
from core.poisson import simulate_poisson

from theory.formulas import (
    standardize_binomial,
    geometric_pmf,
    poisson_pmf
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Quant Probability Lab",
    layout="wide"
)

# ============================================================
# DARK QUANT STYLING
# ============================================================

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #e6edf3;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00f5ff;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# ============================================================
# HEADER
# ============================================================

st.title("📈 Quant Probability Lab")
st.markdown(
    "Monte Carlo simulation engine for discrete probability distributions."
)

st.divider()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    ["CLT Engine", "Geometric Process", "Poisson Process"]
)

# ============================================================
# TAB 1 — CLT (Binomial → Normal)
# ============================================================

with tab1:

    st.subheader("Central Limit Theorem Engine")

    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.slider("Number of trials (n)", 10, 300, 50)

    with col2:
        p = st.slider("Success probability (p)", 0.1, 0.9, 0.5)

    with col3:
        simulations = st.slider("Monte Carlo runs", 1000, 10000, 4000)

    # Simulation
    binomial_data = simulate_binomial(n, p, simulations)

    # Standardize using theory function
    z_values = [
        standardize_binomial(x, n, p)
        for x in binomial_data
    ]

    # Plot
    fig, ax = plt.subplots()
    ax.hist(z_values, bins=40, density=True, alpha=0.5)

    x = np.linspace(-4, 4, 500)
    normal_curve = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    ax.plot(x, normal_curve, linewidth=2)

    ax.set_title("Standardized Binomial → Normal Convergence")
    ax.set_xlabel("Z")
    ax.set_ylabel("Density")

    st.pyplot(fig)

    # Metrics
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Theoretical Mean (μ)", round(mu, 4))
    with m2:
        st.metric("Theoretical Std Dev (σ)", round(sigma, 4))

    with st.expander("Theory Insight"):
        st.latex(r"\mu = np")
        st.latex(r"\sigma = \sqrt{np(1-p)}")
        st.write("As n increases, the Binomial distribution converges to Normal.")

# ============================================================
# TAB 2 — GEOMETRIC
# ============================================================

with tab2:

    st.subheader("Geometric Waiting Time Model")

    col1, col2 = st.columns(2)

    with col1:
        p_geo = st.slider("Success probability (p)", 0.01, 0.9, 0.2)

    with col2:
        simulations_geo = st.slider("Monte Carlo runs", 1000, 10000, 5000)

    geo_data = simulate_geometric(p_geo, simulations_geo)

    fig, ax = plt.subplots()
    ax.hist(geo_data, bins=50, density=True, alpha=0.5)

    k = np.arange(1, max(geo_data))
    pmf = [geometric_pmf(i, p_geo) for i in k]

    ax.plot(k, pmf, linewidth=2)
    ax.set_title("Trials Until First Success")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")

    st.pyplot(fig)

    mean_geo = 1 / p_geo
    var_geo = (1 - p_geo) / (p_geo**2)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Expected Value", round(mean_geo, 4))
    with m2:
        st.metric("Variance", round(var_geo, 4))

    with st.expander("Theory Insight"):
        st.latex(r"P(X = k) = (1-p)^{k-1}p")
        st.write("Models number of trials until first success.")

# ============================================================
# TAB 3 — POISSON
# ============================================================

with tab3:

    st.subheader("Poisson Event Arrival Model")

    col1, col2 = st.columns(2)

    with col1:
        lam = st.slider("Event rate (λ)", 1, 30, 7)

    with col2:
        simulations_poi = st.slider("Monte Carlo runs", 1000, 10000, 5000)

    poisson_data = simulate_poisson(lam, simulations_poi)

    fig, ax = plt.subplots()
    ax.hist(poisson_data, bins=40, density=True, alpha=0.5)

    k = np.arange(0, max(poisson_data))
    pmf = [poisson_pmf(i, lam) for i in k]

    ax.plot(k, pmf, linewidth=2)
    ax.set_title("Discrete Event Count Distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")

    st.pyplot(fig)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Mean (λ)", lam)
    with m2:
        st.metric("Variance (λ)", lam)

    with st.expander("Theory Insight"):
        st.latex(r"P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}")
        st.write(
            "Used in queueing theory, insurance risk modeling, and event arrival modeling."
        )