import numpy as np

def simulate_exponential(lmbda, simulations):
    samples = np.random.exponential(1 / lmbda, simulations)
    return samples

def exponential_pdf(x, lmbda):
    return lmbda * np.exp(-lmbda * x)