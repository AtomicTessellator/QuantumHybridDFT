import unittest

import numpy as np
from scipy.optimize import bisect
from scipy.sparse import csr_matrix

from qhdft.stage1 import setup_discretization
from qhdft.stage2 import build_hamiltonian
from qhdft.stage3 import build_qsvt
from qhdft.stage4 import estimate_density


class TestStage4(unittest.TestCase):
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
        self.d = 1500
        self.epsilon_poly = 1e-4
        self.D, self.D_delta, self.n0, self.N = setup_discretization(self.params)
        self.Ng = len(self.D)
        self.m = int(np.log2(self.Ng))
        self.Ne = sum(self.params["Z"])
        self.H, self.norm, _, _ = build_hamiltonian(
            self.n0, self.D, self.D_delta, self.N, self.params
        )
        self.eigs = np.linalg.eigh(self.H.toarray())[0]

    def find_mu(self, eigs, beta, Ne):
        def func(mu):
            return np.sum(1 / (1 + np.exp(beta * (eigs - mu)))) - Ne

        mu_min = np.min(eigs) - 1
        mu_max = np.max(eigs) + 1
        return bisect(func, mu_min, mu_max)

    def test_estimate_density(self):
        mu = self.find_mu(self.eigs, self.beta, self.Ne)
        H_norm = csr_matrix(
            (1 / self.norm)
            * build_hamiltonian(self.n0, self.D, self.D_delta, self.N, self.params)[0]
        )
        U, P, _ = build_qsvt(
            H_norm, self.norm, self.beta, mu, self.d, self.epsilon_poly, self.m
        )
        # Parameters
        NI = len(self.D_delta)
        indices = np.arange(NI)  # All for simplicity, B=NI=2
        M = 1000
        delta = 0.01
        epsilon_est = 1e-2
        # Compute true coarse density classically for verification
        evecs = np.linalg.eigh(self.H.toarray())[1]
        occ = 1 / (1 + np.exp(self.beta * (self.eigs - mu)))
        n_fine = np.sum(occ[:, None] * (evecs**2), axis=0)
        diffs = self.D[:, None, :] - self.D_delta[None, :, :]
        N_matrix = self.N(diffs)
        n_coarse_true = np.linalg.lstsq(N_matrix, n_fine, rcond=None)[0]
        # Run multiple trials
        num_trials = 100
        successes = 0
        for _ in range(num_trials):
            hat_f, ci, complexity = estimate_density(
                self.H,
                self.norm,
                self.beta,
                mu,
                self.D,
                self.D_delta,
                self.N,
                indices,
                M,
                delta,
                epsilon_est,
            )
            # Check L2 error
            l2_error = np.linalg.norm(hat_f - n_coarse_true)
            if l2_error < epsilon_est:
                successes += 1
        # Verify in >95% trials error < epsilon_est
        self.assertGreater(successes / num_trials, 0.95)
        self.assertGreater(complexity, 0)


if __name__ == "__main__":
    unittest.main()
