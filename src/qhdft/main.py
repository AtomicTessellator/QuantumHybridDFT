import numpy as np

from qhdft.stage1 import setup_discretization
from qhdft.stage2 import build_hamiltonian
from qhdft.stage3 import build_qsvt
from qhdft.stage5 import run_scf
from qhdft.stage6 import compute_energy, compute_error_breakdown, run_scaling_test

# Main Pipeline: Quantum Hybrid DFT
# Integrates all stages to solve DFT for a given system.
# 1. Setup discretization and initial density (stage1).
# 2. Run SCF iterations (stage5, which calls stage2,4 internally; stage3 is placeholder).
# 3. Validate with energy, scaling, errors (stage6).
# Note: stage3 QSVT is not fully integrated in simulation; used for poly approx in tests.


def main():
    # System parameters for 1D Li-H chain example.
    params = {
        "dim": 1,
        "domain": [0, 10.0],
        "m": 5,  # Ng=32
        "atomic_positions": [3.0, 7.0],
        "Z": [3, 1],
        "sigma": 0.5,
        "epsilon": 0.1,
    }
    beta = 10.0
    alpha = 0.5
    B = 2
    K = 100
    epsilon_scf = 1e-4
    delta = 0.01
    epsilon_est = 1e-4
    M = 100000
    d = 1500
    epsilon_poly = 1e-4

    # Stage 1: Discretization
    D, D_delta, n0, N = setup_discretization(params)

    # Run SCF (stages 2,4,5)
    hat_n, residuals, total_complexity = run_scf(
        n0, D, D_delta, N, params, beta, alpha, B, K, epsilon_scf, delta, epsilon_est, M
    )

    # Stage 6: Validation
    E = compute_energy(hat_n, D, D_delta, N, params, beta)
    print(f"Ground state energy: {E}")

    # Scaling test
    Na_range = np.arange(2, 21, 2)
    metrics, r2 = run_scaling_test(
        params, beta, alpha, K, epsilon_scf, delta, epsilon_est, M, Na_range
    )
    print(f"Scaling R^2: {r2}")
    print(f"Queries vs Na: {metrics['queries']}")

    # Error breakdown (using placeholder poly_err; in full, from stage3)
    # For demo, compute classical n_star, E_star
    from qhdft.stage5 import find_mu  # For classical

    n_star = n0.copy()  # Placeholder; actual would run classical SCF
    E_star = E  # Placeholder
    poly_max_err = epsilon_poly
    errors = compute_error_breakdown(
        hat_n, n_star, E, E_star, poly_max_err, epsilon_est, len(residuals)
    )
    print(f"Error breakdown: {errors}")

    # Note: To include stage3, could build H from hat_n, normalize, call build_qsvt, but not simulated here.


if __name__ == "__main__":
    main()
