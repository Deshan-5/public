import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core.geometric import simulate_geometric
from core.binomial import simulate_binomial  # assuming you have this
from theory.formulas import geometric_pmf, standardize_binomial
st.set_page_config(page_title="Probability Lab", layout="wide")

st.title("Probability Distribution Lab")
st.markdown("Explore how probability theory behaves through simulation.")


section = st.sidebar.selectbox(
    "Choose Experiment",
    ["Central Limit Theorem", "Geometric Distribution"]
)

# --------------------------------------
# CLT SECTION
# --------------------------------------

if section == "Central Limit Theorem":

    st.header("Central Limit Theorem (Binomial → Normal)")

    n = st.slider("Number of trials (n)", 5, 200, 30)
    p = st.slider("Probability of success (p)", 0.1, 0.9, 0.5)
    simulations = st.slider("Number of simulations", 500, 5000, 2000)

    # Simulate Binomial row sums
    binomial_data = simulate_binomial(n, p, simulations)

    # Standardize
    z_values = [
        standardize_binomial(x, n, p)
        for x in binomial_data
    ]

    fig, ax = plt.subplots()
    ax.hist(z_values, bins=30, density=True, alpha=0.6)

    x = np.linspace(-4, 4, 500)
    normal_curve = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    ax.plot(x, normal_curve)

    ax.set_title("Standardized Binomial Distribution")
    st.pyplot(fig)
    st.metric("Theoretical Mean", n * p)
    st.metric("Theoretical Std Dev", (n * p * (1 - p))**0.5)


# --------------------------------------
# GEOMETRIC SECTION
# --------------------------------------

if section == "Geometric Distribution":

    st.header("Geometric Distribution Explorer")

    p = st.slider("Probability of success (p)", 0.01, 0.9, 0.2)
    simulations = st.slider("Number of simulations", 1000, 10000, 5000)

    geo_data = simulate_geometric(p, simulations)

    fig, ax = plt.subplots()
    ax.hist(geo_data, bins=50, density=True, alpha=0.6)

    k_values = np.arange(1, max(geo_data))
    pmf_values = [geometric_pmf(k, p) for k in k_values]

    ax.plot(k_values, pmf_values)
    ax.set_title(f"Geometric Distribution (p={p})")

    st.pyplot(fig)
    st.metric("Theoretical Mean", 1 / p)
    st.metric("Theoretical Variance", (1 - p) / (p**2))

