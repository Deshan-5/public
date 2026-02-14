import numpy as np

def calculate_summary_statistics(data: np.ndarray) -> dict:
    """
    Compute descriptive statistics for simulated data.

    Returns:
        A dictionary containing:
        - sample size
        - mean
        - variance
        - standard deviation
        - minimum
        - maximum
    """

    sample_size = len(data)

    sample_mean = np.mean(data)

    # ddof=1 gives sample variance instead of population variance
    sample_variance = np.var(data, ddof=1) #ddof ->delta degrees of freedom/correction factor.

    sample_std_dev = np.std(data, ddof=1)

    minimum_value = np.min(data)
    maximum_value = np.max(data)

    return {
        "sample_size": sample_size,
        "mean": sample_mean,
        "variance": sample_variance,
        "std_dev": sample_std_dev,
        "min": minimum_value,
        "max": maximum_value
    }
