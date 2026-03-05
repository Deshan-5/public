import numpy as np
from scipy import stats

# ── Discrete 
def simulate_binomial(n, p, size=5000):
    return np.random.binomial(n, p, size)

def simulate_geometric(p, size=5000):
    return np.random.geometric(p, size)

def simulate_poisson(lam, size=5000):
    return np.random.poisson(lam, size)

def simulate_negative_binomial(r, p, size=5000):
    return np.random.negative_binomial(r, p, size)

# ── Continuous
def simulate_normal(mu, sigma, size=5000):
    return np.random.normal(mu, sigma, size)

def simulate_exponential(lam, size=5000):
    return np.random.exponential(1 / lam, size)

def simulate_gamma(k, theta, size=5000):
    return np.random.gamma(k, theta, size)

def simulate_beta(alpha, beta, size=5000):
    return np.random.beta(alpha, beta, size)

def simulate_uniform(a, b, size=5000):
    return np.random.uniform(a, b, size)

def simulate_chi_squared(df, size=5000):
    return np.random.chisquare(df, size)

def simulate_t_distribution(df, size=5000):
    return np.random.standard_t(df, size)

# ── Theory curves 
def binomial_pmf(n, p):
    k = np.arange(0, n + 1)
    return k, stats.binom.pmf(k, n, p)

def geometric_pmf(p, max_k=None):
    if max_k is None:
        max_k = int(np.ceil(5 / p)) + 1
    k = np.arange(1, max_k + 1)
    return k, stats.geom.pmf(k, p)

def poisson_pmf(lam):
    max_k = int(lam + 4 * np.sqrt(lam)) + 1
    k = np.arange(0, max_k + 1)
    return k, stats.poisson.pmf(k, lam)

def normal_pdf(mu, sigma):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    return x, stats.norm.pdf(x, mu, sigma)

def exponential_pdf(lam):
    x = np.linspace(0, 6 / lam, 400)
    return x, stats.expon.pdf(x, scale=1 / lam)

def gamma_pdf(k, theta):
    x = np.linspace(0, k * theta + 5 * np.sqrt(k) * theta, 400)
    return x, stats.gamma.pdf(x, k, scale=theta)

def beta_pdf(alpha, beta):
    x = np.linspace(0.001, 0.999, 400)
    return x, stats.beta.pdf(x, alpha, beta)

def chi_squared_pdf(df):
    x = np.linspace(0, df + 5 * np.sqrt(2 * df), 400)
    return x, stats.chi2.pdf(x, df)

def t_distribution_pdf(df):
    x = np.linspace(-5, 5, 400)
    return x, stats.t.pdf(x, df)
