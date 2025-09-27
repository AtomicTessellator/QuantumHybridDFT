import numpy as np
from scipy.optimize import bisect

from .stage2 import build_hamiltonian
from .stage4 import estimate_density

# Stage 5: Self-Consistent Field (SCF) Iterations with Randomized Block Coordinates
# Solves the DFT self-consistency n = F(n) iteratively: start with initial n0, update blocks of n using estimates of F.
# Uses Anderson mixing: n_{k+1} = α hat{F}(n_k) + (1-α) n_k on selected block, for stability with noisy estimates.
# Random block selection (size B) reduces cost per iteration to O(B / ε), with total iterations O(log(1/ε_scf)).
# Converges when residual ||n_{k+1} - n_k|| < ε_scf, yielding approximate ground state density.
# μ is adjusted each iteration to enforce ∫ n ≈ Ne via bisection.


def find_mu(eigs, beta, Ne):
    # Finds chemical potential μ such that sum 1/(1+exp(β(e_i - μ))) = Ne.
    # Uses bisection on [min(e)-1, max(e)+1]; clips large exponents to avoid overflow.
    def func(mu):
        occ = np.zeros_like(eigs)
        for i, e in enumerate(eigs):
            arg = beta * (e - mu)
            if arg > 100:
                occ[i] = 0
            elif arg < -100:
                occ[i] = 1
            else:
                occ[i] = 1 / (1 + np.exp(arg))
        return np.sum(occ) - Ne

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
    # Performs K iterations (or until converged) of hybrid SCF.
    # Each iteration: build H from current n_k, find μ, select random block indices,
    # estimate hat_f = F(n_k) on block via stage4, mix to update n_new on block.
    # Tracks residuals and total query complexity summed over estimates.
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
