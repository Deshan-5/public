import numpy as np


def simulate_normal(mu, sigma, simulations):
    samples = np.random.normal(mu, sigma, simulations)
    return samples

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)
           