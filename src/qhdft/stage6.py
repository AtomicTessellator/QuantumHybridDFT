import numpy as np
from scipy.stats import linregress

from .discretization import setup_discretization
from .stage2 import build_hamiltonian
from .stage5 import find_mu, run_scf

# Stage 6: Validation and Numerical Results
# Validates the converged density from SCF by computing energy, plotting density, analyzing scaling with system size Na,
# and breaking down errors (polynomial approx, statistical, iteration). Compares to classical baselines.


def compute_energy(hat_n, D, D_delta, N, params, beta):
    # Computes ground state energy E = sum occ_i * e_i, where occ_i = f(e_i), from H built on converged hat_n.
    # Uses exact diagonalization here for validation; in quantum setting, could use similar estimation.
    H, _, _, _ = build_hamiltonian(hat_n, D, D_delta, N, params)
    eigs, _ = np.linalg.eigh(H.toarray())
    Ne = sum(params["Z"])
    mu = find_mu(eigs, beta, Ne)
    occ = 1 / (1 + np.exp(beta * (eigs - mu)))
    E = np.sum(occ * eigs)
    return E


def generate_density_data(hat_n, D, D_delta, N):
    # Generates data for plotting: fine grid positions r and interpolated density n(r) from coarse hat_n.
    # Uses shape functions to reconstruct n_fine = N_matrix @ hat_n.
    diffs = D[:, None, :] - D_delta[None, :, :]
    N_matrix = N(diffs)
    n_fine = N_matrix @ hat_n
    return D[:, 0], n_fine


def run_scaling_test(
    base_params, beta, alpha, K, epsilon_scf, delta, epsilon_est, M, Na_range
):
    # Tests scaling: runs SCF for varying Na (system size), measures total queries (proxy for time).
    # Fits linear model queries ~ slope * Na + intercept, checks R^2 > 0.95 for linear scaling.
    # Uses simple H chain (Z=1) with positions spread in domain.
    queries = []
    times = []  # Placeholder, since no actual time measurement
    for Na in Na_range:
        params = base_params.copy()
        params["atomic_positions"] = np.linspace(0, base_params["domain"][1], Na + 1)[
            1:
        ]
        params["Z"] = [1] * Na  # Simple H chain
        D, D_delta, n0, N = setup_discretization(params)
        _, _, complexity = run_scf(  # Assuming run_scf from stage5
            n0,
            D,
            D_delta,
            N,
            params,
            beta,
            alpha,
            Na,  # Full update
            K,
            1e-10,  # Small to run full K
            delta,
            epsilon_est,
            M,
        )
        queries.append(complexity)
        times.append(complexity)  # Proxy
    # Linear fit
    slope, intercept, r_value, _, _ = linregress(Na_range, queries)
    return dict(Na=Na_range, queries=queries, times=times), r_value**2


def compute_error_breakdown(
    hat_n, n_star, E_hat, E_star, poly_max_err, epsilon_est, num_iters
):
    # Quantifies errors: polynomial approximation error, statistical fluctuation bound,
    # number of iterations, L2 density error, absolute energy error vs. classical baseline.
    errors = {
        "poly_approx": poly_max_err,
        "stat_fluct": epsilon_est,
        "iter_complex": num_iters,
        "density_l2": np.linalg.norm(hat_n - n_star),
        "energy_err": abs(E_hat - E_star),
    }
    return errors
