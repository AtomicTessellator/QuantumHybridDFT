import numpy as np
from scipy.stats import linregress

from .stage1 import setup_discretization
from .stage2 import build_hamiltonian
from .stage5 import find_mu, run_scf


def compute_energy(hat_n, D, D_delta, N, params, beta):
    H, _, _, _ = build_hamiltonian(hat_n, D, D_delta, N, params)
    eigs, _ = np.linalg.eigh(H.toarray())
    Ne = sum(params["Z"])
    mu = find_mu(eigs, beta, Ne)
    occ = 1 / (1 + np.exp(beta * (eigs - mu)))
    E = np.sum(occ * eigs)
    return E


def generate_density_data(hat_n, D, D_delta, N):
    diffs = D[:, None, :] - D_delta[None, :, :]
    N_matrix = N(diffs)
    n_fine = N_matrix @ hat_n
    return D[:, 0], n_fine


def run_scaling_test(
    base_params, beta, alpha, K, epsilon_scf, delta, epsilon_est, M, Na_range
):
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
    errors = {
        "poly_approx": poly_max_err,
        "stat_fluct": epsilon_est,
        "iter_complex": num_iters,
        "density_l2": np.linalg.norm(hat_n - n_star),
        "energy_err": abs(E_hat - E_star),
    }
    return errors
