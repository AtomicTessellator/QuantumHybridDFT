import unittest

import numpy as np

from qhdft.stage1 import setup_discretization
from qhdft.stage2 import build_hamiltonian
from qhdft.stage5 import find_mu, run_scf


class TestStage5(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dim": 1,
            "domain": [0, 10.0],
            "m": 5,  # Ng=32
            "atomic_positions": [3.0, 7.0],
            "Z": [3, 1],
            "sigma": 0.5,
            "epsilon": 0.1,
        }
        self.beta = 10.0
        self.alpha = 0.5
        self.B = 2  # Update all for small NI=2
        self.K = 100
        self.epsilon_scf = 1e-4
        self.delta = 0.01
        self.epsilon_est = 1e-4
        self.M = 100000
        self.D, self.D_delta, self.n0, self.N = setup_discretization(self.params)
        self.Ng = len(self.D)
        self.Ne = sum(self.params["Z"])

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
        # Compute energy: sum occ * eigs
        energy = np.sum(occ * eigs)
        return n_k, energy

    def test_scf(self):
        n_star, E_star = self.classical_scf(self.n0)
        np.random.seed(42)
        num_trials = 50
        iterations_list = []
        energy_errors = []
        norm_errors = []
        for _ in range(num_trials):
            hat_n, residuals, complexity = run_scf(
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
            self.assertLessEqual(len(residuals), self.K)
            self.assertGreater(complexity, 0)

            norm_err = np.linalg.norm(hat_n - n_star)
            norm_errors.append(norm_err)

            # Compute energy for hat_n
            H, _, _, _ = build_hamiltonian(
                hat_n, self.D, self.D_delta, self.N, self.params
            )
            eigs, evecs = np.linalg.eigh(H.toarray())
            mu = find_mu(eigs, self.beta, self.Ne)
            occ = 1 / (1 + np.exp(self.beta * (eigs - mu)))
            E_hat = np.sum(occ * eigs)
            energy_err = abs(E_hat - E_star)
            energy_errors.append(energy_err)
            iterations_list.append(len(residuals))

        avg_iters = np.mean(iterations_list)
        self.assertLess(avg_iters, 25)  # Adjusted bound
        self.assertLess(np.mean(norm_errors), 0.005)
        self.assertLess(np.mean(energy_errors), 0.002)
        success_rate = np.sum(np.array(energy_errors) < 0.005) / num_trials
        self.assertGreater(success_rate, 0.9)


if __name__ == "__main__":
    unittest.main()
