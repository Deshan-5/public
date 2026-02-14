import numpy as np
import random

def trials_until_first_success(bernoulli_matrix: np.ndarray) -> np.ndarray:
    """
    For each session,
    count how many trials it takes to get the first success.
    """

    waiting_times = []

    for session in bernoulli_matrix:
        trials_count = 0

        for outcome in session:
            trials_count += 1

            if outcome == 1:
                break

        waiting_times.append(trials_count)

    return np.array(waiting_times)

def simulate_geometric(p, trials):
    results = []

    for _ in range(trials):
        count = 0
        while True:
            count += 1
            if random.random() < p:
                break
        results.append(count)

    return results
