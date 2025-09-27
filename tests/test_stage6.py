import unittest

import numpy as np

from qhdft.stage1 import setup_discretization
from qhdft.stage2 import build_hamiltonian
from qhdft.stage5 import find_mu, run_scf
from qhdft.stage6 import (
    compute_energy,
    compute_error_breakdown,
    generate_density_data,
    run_scaling_test,
)


class TestStage6(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dim": 1,
            "domain": [0, 10.0],
            "m": 5,
            "atomic_positions": [3.0, 7.0],
            "Z": [3, 1],
            "sigma": 0.5,
            "epsilon": 0.1,
        }
        self.beta = 10.0
        self.alpha = 0.5
        self.B = 2
        self.K = 100
        self.epsilon_scf = 1e-4
        self.delta = 0.01
        self.epsilon_est = 1e-4
        self.M = 100000
        self.D, self.D_delta, self.n0, self.N = setup_discretization(self.params)
        self.Ne = sum(self.params["Z"])
        # Run SCF for converged n
        self.hat_n, self.residuals, _ = run_scf(
            self.n0,
            self.D,
            self.D_delta,
            self.N,
            self.params,
            self.beta,
            self.alpha,
            self.B,
            self.K,
            self.epsilon_scf,
            self.delta,
            self.epsilon_est,
            self.M,
        )
        # Classical converged for comparison
        self.n_star, self.E_star = self.classical_scf(self.n0)

    def classical_scf(self, n0, tol=1e-6, max_iter=100):
        n_k = n0.copy()
        for _ in range(max_iter):
            H, _, _, _ = build_hamiltonian(
                n_k, self.D, self.D_delta, self.N, self.params
            )
            eigs, evecs = np.linalg.eigh(H.toarray())
            mu = find_mu(eigs, self.beta, self.Ne)
            occ = 1 / (1 + np.exp(self.beta * (eigs - mu)))
            n_fine = np.sum(occ[:, None] * (evecs**2), axis=0)
            diffs = self.D[:, None, :] - self.D_delta[None, :, :]
            N_matrix = self.N(diffs)
            n_new = np.linalg.lstsq(N_matrix, n_fine, rcond=None)[0]
            res = np.linalg.norm(n_new - n_k)
            n_k = n_new
            if res < tol:
                break
        energy = np.sum(occ * eigs)
        return n_k, energy

    def test_compute_energy(self):
        E_hat = compute_energy(
            self.hat_n, self.D, self.D_delta, self.N, self.params, self.beta
        )
        self.assertLess(abs(E_hat - self.E_star), 1e-2)

    def test_generate_density_data(self):
        r, n_plot = generate_density_data(self.hat_n, self.D, self.D_delta, self.N)
        _, n_star_plot = generate_density_data(
            self.n_star, self.D, self.D_delta, self.N
        )
        l2_diff = np.linalg.norm(n_plot - n_star_plot)
        self.assertLess(l2_diff, 1e-2)
        self.assertEqual(len(r), len(self.D))
        self.assertEqual(len(n_plot), len(self.D))

    def test_run_scaling_test(self):
        Na_range = np.arange(2, 21, 2)  # 2 to 20 step 2
        metrics, r2 = run_scaling_test(
            self.params,
            self.beta,
            self.alpha,
            self.K,
            self.epsilon_scf,
            self.delta,
            self.epsilon_est,
            self.M,
            Na_range,
        )
        self.assertGreater(r2, 0.95)
        self.assertEqual(len(metrics["Na"]), len(Na_range))
        self.assertTrue(all(q > 0 for q in metrics["queries"]))

    def test_compute_error_breakdown(self):
        E_hat = compute_energy(
            self.hat_n, self.D, self.D_delta, self.N, self.params, self.beta
        )
        poly_max_err = 1e-4  # From stage3
        errors = compute_error_breakdown(
            self.hat_n,
            self.n_star,
            E_hat,
            self.E_star,
            poly_max_err,
            self.epsilon_est,
            len(self.residuals),
        )
        self.assertIn("poly_approx", errors)
        self.assertIn("stat_fluct", errors)
        self.assertIn("iter_complex", errors)
        self.assertIn("density_l2", errors)
        self.assertIn("energy_err", errors)
        self.assertLess(errors["density_l2"], 0.01)
        self.assertLess(errors["energy_err"], 0.01)


if __name__ == "__main__":
    unittest.main()
