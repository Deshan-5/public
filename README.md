# 🎲 Probability Lab v3
### Monte Carlo Simulation Engine · Statistical Toolkit

---

## What's New in v2.0

- **Landing page** with animated moving-gradient navbar
- **Discrete Lab** — Binomial, Geometric, Poisson (with animate button)
- **Continuous Lab** — Normal, Exponential, Gamma, Beta, Uniform, Chi-Squared, Student-t
- **Hypothesis Testing** — One-sample Z/T, Two-sample T, Chi-Square (GoF + Independence)
- **Confidence Intervals** — Z-interval, T-interval, Wilson Proportion CI, Variance CI, Coverage Simulation
- **Bayes Theorem** — Classic, Sequential updating, Beta-Binomial conjugate
- **Combinatorics** — nPr, nCr, Multinomial, Birthday Problem, List Generator

---

## Project Structure

```
probability_lab/
├── app_sr.py        ← main app (run this)
├── requirements.txt
├── core/
│   ├── __init__.py
│   └── distributions.py    ← all simulators + theory curves
└── tools/
    ├── __init__.py
    └── stats_tools.py       ← hypothesis tests, CIs, Bayes, combinatorics
```

---

## Quick Start

```bash
git clone <your-repo>
cd probability_lab

python3 -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
streamlit run app_sr.py
```

---

## Features


### ⚀ Discrete Distributions
| Distribution | Parameters | Formula |
|---|---|---|
| Binomial | n, p | P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ |
| Geometric | p | P(X=k) = (1-p)ᵏ⁻¹p |
| Poisson | λ | P(X=k) = e⁻λλᵏ/k! |

### 〰️ Continuous Distributions
| Distribution | Parameters |
|---|---|
| Normal | μ, σ |
| Exponential | λ |
| Gamma | k, θ |
| Beta | α, β |
| Uniform | a, b |
| Chi-Squared | df |
| Student-t | df |

### 🔬 Hypothesis Testing
- **Z-test** — one-sample with known σ
- **T-test** — one-sample and two-sample
- **Chi-Square** — goodness of fit + independence
- Rejection region visualizations, p-values, critical values

### 📏 Confidence Intervals
- Z-interval (known σ)
- T-interval (unknown σ)
- Wilson CI for proportions
- Chi-square CI for variance
- Coverage probability simulation

### 🎲 Bayes Theorem
- Classic P(H|E) calculator
- Sequential Bayesian updating animation
- Beta-Binomial conjugate prior/posterior

### 🧮 Combinatorics
- nPr, nCr with/without repetition
- Pascal's Triangle visualization
- Multinomial coefficients
- Birthday problem curve
- Full combination/permutation list generator

---

Built with Python · Streamlit · NumPy · SciPy · Matplotlib
