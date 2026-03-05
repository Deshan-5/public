import numpy as np
from scipy import stats
import math

# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════

def z_test_one_sample(sample_mean, pop_mean, pop_std, n, alternative="two-sided"):
    """One-sample Z-test."""
    se = pop_std / np.sqrt(n)
    z_stat = (sample_mean - pop_mean) / se
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        p_value = stats.norm.cdf(z_stat)
    return {"statistic": z_stat, "p_value": p_value, "se": se}

def t_test_one_sample(data, pop_mean, alternative="two-sided"):
    """One-sample T-test from raw data."""
    t_stat, p_value = stats.ttest_1samp(data, pop_mean, alternative=alternative)
    n = len(data)
    df = n - 1
    se = np.std(data, ddof=1) / np.sqrt(n)
    return {"statistic": t_stat, "p_value": p_value, "df": df, "se": se,
            "sample_mean": np.mean(data), "sample_std": np.std(data, ddof=1)}

def t_test_two_sample(data1, data2, equal_var=True, alternative="two-sided"):
    """Two-sample T-test."""
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alternative)
    df = len(data1) + len(data2) - 2 if equal_var else None
    return {"statistic": t_stat, "p_value": p_value, "df": df,
            "mean1": np.mean(data1), "mean2": np.mean(data2),
            "std1": np.std(data1, ddof=1), "std2": np.std(data2, ddof=1)}

def chi_square_gof_test(observed, expected=None):
    """Chi-square goodness-of-fit test."""
    observed = np.array(observed, dtype=float)
    if expected is None:
        expected = np.full(len(observed), observed.sum() / len(observed))
    else:
        expected = np.array(expected, dtype=float)
    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
    df = len(observed) - 1
    return {"statistic": chi2_stat, "p_value": p_value, "df": df,
            "observed": observed, "expected": expected}

def chi_square_independence_test(contingency_table):
    """Chi-square test of independence."""
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {"statistic": chi2, "p_value": p, "df": dof, "expected": expected}

def get_critical_value(test_type, alpha, df=None, alternative="two-sided"):
    """Return critical value for a given test."""
    if test_type == "z":
        if alternative == "two-sided":
            return stats.norm.ppf(1 - alpha / 2)
        return stats.norm.ppf(1 - alpha)
    elif test_type == "t" and df is not None:
        if alternative == "two-sided":
            return stats.t.ppf(1 - alpha / 2, df)
        return stats.t.ppf(1 - alpha, df)
    elif test_type == "chi2" and df is not None:
        return stats.chi2.ppf(1 - alpha, df)
    return None

# ══════════════════════════════════════════════════════════════════
# CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════

def ci_mean_known_sigma(sample_mean, pop_std, n, confidence=0.95):
    """CI for mean with known population std (Z-interval)."""
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    me = z * pop_std / np.sqrt(n)
    return {"lower": sample_mean - me, "upper": sample_mean + me,
            "margin_of_error": me, "z_critical": z, "method": "Z-interval"}

def ci_mean_unknown_sigma(data, confidence=0.95):
    """CI for mean with unknown population std (T-interval)."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    df = n - 1
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    me = t_crit * se
    return {"lower": mean - me, "upper": mean + me, "margin_of_error": me,
            "t_critical": t_crit, "df": df, "sample_mean": mean, "se": se, "method": "T-interval"}

def ci_proportion(successes, n, confidence=0.95):
    """Wilson CI for a proportion."""
    p_hat = successes / n
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    center = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return {"lower": max(0, center - margin), "upper": min(1, center + margin),
            "p_hat": p_hat, "margin_of_error": margin, "z_critical": z, "method": "Wilson CI"}

def ci_variance(data, confidence=0.95):
    """CI for population variance using chi-square distribution."""
    n = len(data)
    s2 = np.var(data, ddof=1)
    df = n - 1
    alpha = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha / 2, df)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
    return {"lower": df * s2 / chi2_upper, "upper": df * s2 / chi2_lower,
            "sample_variance": s2, "df": df, "method": "Chi-square CI"}

# ══════════════════════════════════════════════════════════════════
# BAYES THEOREM
# ══════════════════════════════════════════════════════════════════

def bayes_theorem(prior, likelihood_given_true, likelihood_given_false):
    """
    P(H|E) = P(E|H)·P(H) / [P(E|H)·P(H) + P(E|¬H)·P(¬H)]
    """
    prior_complement = 1 - prior
    numerator = likelihood_given_true * prior
    denominator = (likelihood_given_true * prior) + (likelihood_given_false * prior_complement)
    posterior = numerator / denominator if denominator > 0 else 0
    return {
        "prior": prior,
        "prior_complement": prior_complement,
        "likelihood_given_true": likelihood_given_true,
        "likelihood_given_false": likelihood_given_false,
        "posterior": posterior,
        "bayes_factor": likelihood_given_true / likelihood_given_false if likelihood_given_false > 0 else float('inf'),
        "numerator": numerator,
        "denominator": denominator,
    }

def bayes_update_sequence(initial_prior, likelihoods_true, likelihoods_false):
    """Sequential Bayesian updating."""
    posteriors = [initial_prior]
    current = initial_prior
    for lt, lf in zip(likelihoods_true, likelihoods_false):
        result = bayes_theorem(current, lt, lf)
        current = result["posterior"]
        posteriors.append(current)
    return posteriors

def beta_binomial_update(alpha_prior, beta_prior, successes, failures):
    """Beta-Binomial conjugate update."""
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    post_mean = alpha_post / (alpha_post + beta_post)
    return {"alpha_prior": alpha_prior, "beta_prior": beta_prior,
            "alpha_posterior": alpha_post, "beta_posterior": beta_post,
            "prior_mean": prior_mean, "posterior_mean": post_mean,
            "successes": successes, "failures": failures}

# ══════════════════════════════════════════════════════════════════
# PERMUTATIONS & COMBINATIONS
# ══════════════════════════════════════════════════════════════════

def permutations(n, r):
    """nPr = n! / (n-r)!"""
    if r > n or n < 0 or r < 0:
        return 0
    return math.factorial(n) // math.factorial(n - r)

def combinations(n, r):
    """nCr = n! / (r! * (n-r)!)"""
    if r > n or n < 0 or r < 0:
        return 0
    return math.comb(n, r)

def multinomial_coefficient(n, groups):
    """n! / (k1! * k2! * ... * km!)"""
    if sum(groups) != n:
        return None
    result = math.factorial(n)
    for g in groups:
        result //= math.factorial(g)
    return result

def combinations_with_repetition(n, r):
    """C(n+r-1, r) — stars and bars."""
    return math.comb(n + r - 1, r)

def permutations_with_repetition(n, groups):
    """n! / (n1! * n2! * ...) for repeated elements."""
    result = math.factorial(n)
    for g in groups:
        result //= math.factorial(g)
    return result

def birthday_problem(n_people):
    """Probability that at least 2 people share a birthday."""
    prob_no_shared = 1.0
    for i in range(n_people):
        prob_no_shared *= (365 - i) / 365
    return 1 - prob_no_shared

def generate_combinations_list(items, r):
    """Generate all combinations as a list (for small n)."""
    from itertools import combinations as itertools_combinations
    return list(itertools_combinations(items, r))

def generate_permutations_list(items, r):
    """Generate all permutations as a list (for small n)."""
    from itertools import permutations as itertools_permutations
    return list(itertools_permutations(items, r))
