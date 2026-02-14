import math

def binomial_mean(n, p):
    return n * p

def binomial_variance(n, p):
    return n * p * (1 - p)

def geometric_mean(p):
    return 1 / p

def geometric_variance(p):
    return (1 - p) / (p ** 2)

def standardize_binomial(x, n, p):
    mean = n * p
    std = math.sqrt(n * p * (1 - p))
    return (x - mean) / std

def geometric_pmf(k, p):
    return (1 - p)**(k - 1) * p

def geometric_variance(p):
    return (1 - p) / (p**2)

def poisson_pmf(k, lam):
    """
    Computes the Poisson probability mass function.
    
    P(X = k) = (e^(-λ) * λ^k) / k!
    """
    return (math.exp(-lam) * lam**k) / math.factorial(k)
