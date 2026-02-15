import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import binomial
from core.binomial import simulate_binomial
from theory.formulas import standardize_binomial
from core.geometric import simulate_geometric
from theory.formulas import geometric_pmf
from theory.formulas import geometric_variance



def clt_visualization(binomial_data, n, p):
    # Standardize data
    z_values = [standardize_binomial(x, n, p) for x in binomial_data]

    # Plot histogram
    plt.hist(z_values, bins=30, density=True, alpha=0.6)

    # Plot standard normal curve
    x = np.linspace(-4, 4, 500)
    normal_curve = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    plt.plot(x, normal_curve)

    plt.title("Central Limit Theorem: Binomial → Normal")
    plt.xlabel("Z value")
    plt.ylabel("Density")
    plt.show()

def plot_geometric_convergence(geo_data, p):
    cumulative_means = []
    running_sum = 0

    for i, value in enumerate(geo_data):
        running_sum += value
        cumulative_means.append(running_sum / (i + 1))

    theoretical = 1 / p

    plt.plot(cumulative_means)
    plt.axhline(y=theoretical)

    plt.title("Geometric Distribution: Cumulative Mean Convergence")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Mean")
    plt.show()

def rare_event_visualization():
    p = 0.01  # very small probability
    trials = 5000

    geo_data = simulate_geometric(p, trials)

    plt.hist(geo_data, bins=100, density=True)
    plt.title("Geometric Distribution - Rare Event (p=0.01)")
    plt.xlabel("Trials Until First Success")
    plt.ylabel("Density")
    plt.show()


def rare_event_with_theory(p=0.01, trials=5000):
    geo_data = simulate_geometric(p, trials)

    # Histogram--(normalized)
    plt.hist(geo_data, bins=100, density=True, alpha=0.6)

    # Theoretical curve.
    k_values = np.arange(1, max(geo_data))
    pmf_values = [geometric_pmf(k, p) for k in k_values]

    plt.plot(k_values, pmf_values)

    plt.title(f"Geometric Distribution (p={p})")
    plt.xlabel("Trials Until First Success")
    plt.ylabel("Probability")
    plt.show()

def compare_geometric_ps():
    trials = 5000

    for p in [0.01, 0.3]:
        geo_data = simulate_geometric(p, trials)

        plt.hist(geo_data, bins=100, density=True, alpha=0.5)
        plt.title(f"Geometric Distribution (p={p})")
        plt.xlabel("Trials Until First Success")
        plt.ylabel("Density")
        plt.show()


def geometric_convergence_with_variance(p=0.05, trials=5000):
    geo_data = simulate_geometric(p, trials)

    cumulative_means = []
    running_sum = 0

    for i, value in enumerate(geo_data):
        running_sum += value
        cumulative_means.append(running_sum / (i + 1))

    theoretical_mean = 1 / p
    variance = geometric_variance(p)
    std = variance**0.5

    plt.plot(cumulative_means)
    plt.axhline(theoretical_mean)
    plt.axhline(theoretical_mean + std)
    plt.axhline(theoretical_mean - std)

    plt.title("Geometric Convergence with Variance Bands")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Mean")
    plt.show()