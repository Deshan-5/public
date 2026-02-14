import numpy as np

def simulate_poisson(lam, simulations):
    """
    Simulates Poisson-distributed random variables.
    """
    return np.random.poisson(lam, simulations)
