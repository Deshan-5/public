import numpy as np

def count_successes_per_session(bernoulli_matrix: np.ndarray) -> np.ndarray:
    """
    For each session (row),
    count how many successes occurred.
    """

    total_successes = []

    for session in bernoulli_matrix:
        successes = 0

        for outcome in session:
            if outcome == 1:
                successes += 1

        total_successes.append(successes)

    return np.array(total_successes)

def simulate_binomial(n, p, simulations):
    return np.random.binomial(n, p, simulations)
    # will optimize this later using numpy's vectorized operations, but for now, this is straightforward.
