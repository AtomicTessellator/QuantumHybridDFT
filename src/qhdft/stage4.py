import numpy as np
from scipy.stats import norm

# Stage 4: Electron Density Estimation via Quantum Measurements
# Estimates selected components of the density F(n) = diagonals of Γ = f(H), where H is from current n.
# In a real quantum setting, this would use the QSVT circuit U to prepare states and measure probabilities
# via amplitude estimation for efficiency O(1/ε). Here, we simulate classically: compute exact density,
# project to coarse grid, add Gaussian noise mimicking measurement variance, for selected block indices.
# Provides estimates hat_f, confidence intervals, and query complexity estimate.
# Block selection allows randomized updates in SCF, reducing per-iteration cost to O(B / ε) instead of O(NI / ε).


def estimate_density(
    H, norm_factor, beta, mu, D, D_delta, N, indices, M, delta, epsilon_est
):
    # Simulates estimation of coarse density on selected indices (block of size B).
    # Classically computes exact fine density n_fine = sum occ * |ψ_i|^2, with occ = f(eigs).
    # Projects to coarse n_coarse_true via least-squares with shape matrix N.
    # Adds noise ~ Normal(0, sigma^2), where sigma tuned for L2 error < ε_est with high probability.
    # Confidence intervals assume normality. Complexity rough O(sparsity * B / ε * M).
    # Classically compute true density on fine grid for simulation
    eigs, evecs = np.linalg.eigh(H.toarray())
    occ = 1 / (1 + np.exp(beta * (eigs - mu)))
    n_fine = np.sum(occ[:, None] * (evecs**2), axis=0)
    # Interpolate to coarse (though for estimation, we simulate on fine then project, but simplify)
    diffs = D[:, None, :] - D_delta[None, :, :]
    N_matrix = N(diffs)  # (Ng, NI)
    # True coarse density by least squares or projection
    n_coarse_true = np.linalg.lstsq(N_matrix, n_fine, rcond=None)[0]
    # For selected indices (subset of 0 to NI-1)
    B = len(indices)
    # Simulated estimation: true value + Gaussian noise with variance based on M
    # Assume variance sigma^2 = (true_val * (1 - true_val)) / M or something; for simplicity, use fixed sigma
    # From plan, variance < sigma^2, e.g. sigma=1e-3
    sigma = epsilon_est / (3 * np.sqrt(B))  # To have low prob of exceeding epsilon_est
    hat_f = n_coarse_true[indices] + np.random.normal(0, sigma, B)
    # Confidence intervals, e.g. 95% CI assuming normal
    z = norm.ppf(1 - delta / 2)
    ci = np.column_stack((hat_f - z * sigma, hat_f + z * sigma))
    # Query complexity O(s * NI / epsilon_est), but per block B
    complexity = int(H.nnz / H.shape[0] * B / epsilon_est) * M  # Rough
    return hat_f, ci, complexity
