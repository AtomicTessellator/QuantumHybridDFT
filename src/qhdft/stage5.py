import numpy as np
from scipy.optimize import bisect

from .stage2 import build_hamiltonian
from .stage4 import estimate_density


def find_mu(eigs, beta, Ne):
    def func(mu):
        return np.sum(1 / (1 + np.exp(beta * (eigs - mu)))) - Ne

    mu_min = np.min(eigs) - 1
    mu_max = np.max(eigs) + 1
    return bisect(func, mu_min, mu_max)


def run_scf(
    initial_n,
    D,
    D_delta,
    N,
    params,
    beta,
    alpha,
    B,
    K,
    epsilon_scf,
    delta,
    epsilon_est,
    M,
):
    n_k = initial_n.copy()
    NI = len(D_delta)
    Ne = sum(params["Z"])
    residuals = []
    total_complexity = 0

    for k in range(K):
        H, norm, _, _ = build_hamiltonian(n_k, D, D_delta, N, params)
        eigs = np.linalg.eigh(H.toarray())[0]  # For mu finding, classical
        mu = find_mu(eigs, beta, Ne)

        # Select random block
        indices = np.random.choice(NI, B, replace=False)

        # Estimate on block
        hat_f, _, complexity = estimate_density(
            H, norm, beta, mu, D, D_delta, N, indices, M, delta, epsilon_est
        )
        total_complexity += complexity

        # Mixing update only on block
        n_new = n_k.copy()
        n_new[indices] = alpha * hat_f + (1 - alpha) * n_k[indices]

        # Residual
        res = np.linalg.norm(n_new - n_k)
        residuals.append(res)
        n_k = n_new

        if res < epsilon_scf:
            break

    return n_k, np.array(residuals), total_complexity
