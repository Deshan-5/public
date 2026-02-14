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
# PAGE CONFIG
st.set_page_config(
    page_title="Quant Probability Lab",
    layout="wide"
)

plt.style.use("dark_background")


# DARK QUANT CSS

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
        padding: 12px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# HEADER

st.title("📈 Quant Probability Lab")
st.markdown("Monte Carlo simulation engine for discrete distributions.")

st.divider()


# TABS


tab1, tab2, tab3 = st.tabs(
    ["CLT Engine", "Geometric Process", "Poisson Process"]
)


# TAB 1 — CLT


with tab1:

    st.subheader("Central Limit Theorem")

    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.slider("Trials (n)", 10, 300, 50)

    with col2:
        p = st.slider("Success probability (p)", 0.1, 0.9, 0.5)

    with col3:
        simulations = st.slider(
            "Monte Carlo runs",
            1000,
            10000,
            4000,
            key="clt_sim"
        )

    binomial_data = simulate_binomial(n, p, simulations)
    z_values = [standardize_binomial(x, n, p) for x in binomial_data]

    left, right = st.columns([3, 2])

    # GRAPH
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(z_values, bins=40, density=True, alpha=0.5)

        x = np.linspace(-4, 4, 500)
        normal_curve = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        ax.plot(x, normal_curve, linewidth=2)

        ax.set_title("Standardized Binomial → Normal")
        fig.tight_layout()
        st.pyplot(fig)

    # THEORY PANEL
    with right:
        mu = n * p
        sigma = math.sqrt(n * p * (1 - p))

        st.markdown("### Theoretical Values")
        st.metric("μ = np", round(mu, 4))
        st.metric("σ = √np(1−p)", round(sigma, 4))

        st.markdown("### Formula")
        st.latex(r"\mu = np")
        st.latex(r"\sigma = \sqrt{np(1-p)}")

        st.markdown("As n increases, Binomial converges to Normal.")

# TAB 2 — GEOMETRIC


with tab2:

    st.subheader("Geometric Distribution")

    col1, col2 = st.columns(2)

    with col1:
        p_geo = st.slider("Success probability (p)", 0.01, 0.9, 0.2)

    with col2:
        simulations_geo = st.slider(
            "Monte Carlo runs",
            1000,
            10000,
            5000,
            key="geo_sim"
        )

    geo_data = simulate_geometric(p_geo, simulations_geo)

    left, right = st.columns([3, 2])

    # GRAPH
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(geo_data, bins=40, density=True, alpha=0.5)

        k = np.arange(1, max(geo_data))
        pmf = [geometric_pmf(i, p_geo) for i in k]
        ax.plot(k, pmf, linewidth=2)

        ax.set_title("Trials Until First Success")
        fig.tight_layout()
        st.pyplot(fig)

    # THEORY
    with right:
        mean_geo = 1 / p_geo
        var_geo = (1 - p_geo) / (p_geo**2)

        st.markdown("### Theoretical Values")
        st.metric("E[X]", round(mean_geo, 4))
        st.metric("Var(X)", round(var_geo, 4))

        st.markdown("### Formula")
        st.latex(r"P(X=k)=(1-p)^{k-1}p")

        st.markdown("Models waiting time until first success.")


# TAB 3 — POISSON


with tab3:

    st.subheader("Poisson Event Arrival Model")

    col1, col2 = st.columns(2)

    with col1:
        lam = st.slider("Event rate (λ)", 1, 30, 7)

    with col2:
        simulations_poi = st.slider(
            "Monte Carlo runs",
            1000,
            10000,
            5000,
            key="poi_sim"
        )

    poisson_data = simulate_poisson(lam, simulations_poi)

    left, right = st.columns([3, 2])

    # GRAPH
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(poisson_data, bins=40, density=True, alpha=0.5)

        k = np.arange(0, max(poisson_data))
        pmf = [poisson_pmf(i, lam) for i in k]
        ax.plot(k, pmf, linewidth=2)

        ax.set_title("Discrete Event Count Distribution")
        fig.tight_layout()
        st.pyplot(fig)

    # THEORY
    with right:
        st.markdown("### Theoretical Values")
        st.metric("Mean (λ)", lam)
        st.metric("Variance (λ)", lam)

        st.markdown("### Formula")
        st.latex(r"P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}")

        st.markdown(
            "Used in queueing systems, insurance risk and event modeling."
        )
