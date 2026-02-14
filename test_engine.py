import matplotlib.pyplot as plt
import numpy as np
from core.bernoulli import generate_bernoulli_matrix
from core.binomial import count_successes_per_session
from core.diagnostics import calculate_summary_statistics
from theory.formulas import binomial_mean, binomial_variance

# Parameters
p = 0.5
n = 10
rows = 10000

# Generate Bernoulli matrix
matrix = generate_bernoulli_matrix(rows, n, p)
binomial_data = count_successes_per_session(matrix)

# Cumulative mean calculation
cumulative_mean = np.cumsum(binomial_data) / np.arange(1, rows + 1)

# Theoretical mean line
theoretical_mean = binomial_mean(n, p)

# Plot convergence
plt.figure(figsize=(12, 6))
plt.plot(cumulative_mean, label="Simulated Cumulative Mean", color="blue")
plt.hlines(theoretical_mean, 0, rows, colors="red", linestyles="dashed", label="Theoretical Mean")
plt.title(f"Convergence of Simulated Binomial Mean to Theoretical Mean (n={n}, p={p})")
plt.xlabel("Number of Sessions")
plt.ylabel("Mean Number of Successes")
plt.legend()
plt.grid(True)
plt.show()
