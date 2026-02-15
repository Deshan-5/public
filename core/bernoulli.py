import numpy as np

def generate_bernoulli_matrix(number_of_sessions: int,
                              trials_per_session: int,
                              success_probability: float) -> np.ndarray:
    """
    Simulates repeated Bernoulli experiments.
    Each row represents one session of repeated trials.
    Each entry is:
        1 → success
        0 → failure
    """

    matrix = []

    for session in range(number_of_sessions):
        current_session = []

        for trial in range(trials_per_session):
            random_value = np.random.rand()  # uniform between 0 and 1

            if random_value < success_probability:
                current_session.append(1)
            else:
                current_session.append(0)

        matrix.append(current_session)

    return np.array(matrix)
# will optimize this later using numpy's vectorized operations, but for now, this is straightforward.
