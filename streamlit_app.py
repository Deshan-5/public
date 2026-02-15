import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from core.binomial import simulate_binomial
from core.geometric import simulate_geometric
from core.poisson import simulate_poisson
from core.normal import simulate_normal, normal_pdf
from core.exponential import simulate_exponential, exponential_pdf
from theory.formulas import (
    standardize_binomial,
    geometric_pmf,
    poisson_pmf
)
# PAGE CONFIG
st.set_page_config(
    page_title="Probability Lab",
    layout="wide"
)

# CSS PART.
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #e6edf3;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        color: #00fff7;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #e6edf3;
        line-height: 1.2;
    }
    .author {
        font-size: 1rem;
        font-weight: 600;
        color: #FFD700;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<h1 style="color:#00fff7; font-size:3rem;">Probability Lab</h1>', unsafe_allow_html=True)

col_subtitle, col_author = st.columns([3, 1])
with col_subtitle:
    st.markdown('<div class="subtitle">Monte Carlo simulation engine for discrete distributions.</div>', unsafe_allow_html=True)
with col_author:
    st.markdown('<div class="author">Project by Deshan Gautam</div>', unsafe_allow_html=True)

st.divider()



# tabs.
tab1, tab2, tab3, tab4 , tab5= st.tabs(
    ["CLT Engine", "Geometric Distribution", "Poisson Distribution", "Normal Distribution", "Exponential Distribution"]
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
    
    show_normal = st.checkbox("Show Normal Approximation", key="norm_approx")

    if show_normal:
        mu = n * p
        sigma = np.sqrt(n * p * (1 - p))

        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        ax.plot(x, y)


    #THEORY PANEL
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

# Tab 2 — Geometric.
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


# TAB 3 — Poisson
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

    #Theory
    with right:
        st.markdown("### Theoretical Values")
        st.metric("Mean (λ)", lam)
        st.metric("Variance (λ)", lam)

        st.markdown("### Formula")
        st.latex(r"P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}")

        st.markdown(
            "Used in queueing systems, insurance risk and event modeling."
        )
# Normal distribution tab
with tab4:

    col_graph, col_theory = st.columns([2, 1])

    with col_graph:

        st.subheader("Normal Distribution Simulation")

        #Sliders and controls 
        mu = st.slider("Mean (μ)", -10.0, 10.0, 0.0, key="norm_mu")
        sigma = st.slider("Std Dev (σ)", 0.1, 10.0, 1.0, key="norm_sigma")
        simulations = st.slider("Number of Samples", 1000, 10000, 5000, key="norm_sim")
        animate = st.button("Animate Sampling", key="norm_anim")
        show_curve = st.checkbox("Show Theoretical Curve", key="norm_curve")

        #Generates simulated data
        samples = simulate_normal(mu, sigma, simulations)

        # Theoretical curve values 
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        y = normal_pdf(x, mu, sigma)

        #Animation path.  ->if someone interested make it better by adding a progress bar or smoother animation.
        if animate:
            placeholder = st.empty()
            for i in range(200, simulations, 400):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(samples[:i], bins=50, density=True, alpha=0.6)

                if show_curve:
                    ax.plot(x, y, color="FFD700", label="Theoretical Curve")

                ax.set_title("Normal Distribution")
                ax.set_xlabel("x")
                ax.set_ylabel("Density")
                ax.legend()
                placeholder.pyplot(fig)
                time.sleep(0.05)

        # Static plot path
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(samples, bins=50, density=True, alpha=0.6, label="Simulated Samples")

            if show_curve:
                ax.plot(x, y, color="#FFD700", label="Theoretical Curve")

            ax.set_title("Normal Distribution")
            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            ax.legend()
            st.pyplot(fig, use_container_width=False)

    #Theory panel
    with col_theory:
        st.subheader("Theory Insight")
        st.markdown(f"""
**Probability Density Function (PDF):**

f(x) = (1 / (σ√(2π))) exp(-(x - μ)² / (2σ²))

---

**Theoretical Mean:** {mu}  
**Theoretical Variance:** {sigma**2}

**Simulated Mean:** {samples.mean():.2f}  
**Simulated Std Dev:** {samples.std():.2f}

---

Normal distribution is symmetric and fully defined by:  
- μ → center  
- σ → spread  
- By the Central Limit Theorem, many distributions converge to Normal.
""")



    
with tab5:

    col_graph, col_theory = st.columns([2, 1])

    with col_graph:

        st.subheader("Exponential Distribution")

        lmbda = st.slider("Rate (λ)", 0.1, 5.0, 1.0, key="exp_lambda")
        simulations = st.slider("Samples", 1000, 10000, 5000, key="exp_sim")

        samples = simulate_exponential(lmbda, simulations)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(samples, bins=50, density=True, alpha=0.6)

        x = np.linspace(0, 5/lmbda, 500)
        y = exponential_pdf(x, lmbda)

        ax.plot(x, y)

        ax.set_title("Exponential Distribution")
        st.pyplot(fig)

    with col_theory:

        st.subheader("Theory Insight")

        st.markdown(f"""
        **PDF:** f(x) = λ e^(-λx)

        ---
        **Mean:** {1/lmbda:.3f}  
        **Variance:** {1/(lmbda**2):.3f}

        ---
        Memoryless property:
        P(X > s + t | X > s) = P(X > t)
        """)
