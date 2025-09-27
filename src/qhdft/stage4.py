import numpy as np
from scipy.stats import norm


def estimate_density(
    H, norm_factor, beta, mu, D, D_delta, N, indices, M, delta, epsilon_est
):
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
