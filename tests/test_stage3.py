import unittest

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import bisect
from scipy.sparse import csr_matrix

from qhdft.stage1 import setup_discretization
from qhdft.stage2 import build_hamiltonian
from qhdft.stage3 import build_qsvt


class TestStage3(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dim": 1,
            "domain": [0, 10.0],
            "m": 5,  # Ng=32 for small simulation
            "atomic_positions": [3.0, 7.0],
            "Z": [3, 1],
            "sigma": 0.5,
            "epsilon": 0.1,
        }
        self.beta = 10.0  # Inverse temperature
        self.d = 1500  # Polynomial degree
        self.epsilon_poly = 1e-4
        self.D, self.D_delta, self.n0, self.N = setup_discretization(self.params)
        self.Ng = len(self.D)
        self.m = int(np.log2(self.Ng))
        self.Ne = sum(self.params["Z"])
        H, self.norm, _, _ = build_hamiltonian(
            self.n0, self.D, self.D_delta, self.N, self.params
        )
        # Compute eigenvalues classically
        self.eigs = np.linalg.eigh(H.toarray())[0]

    def find_mu(self, eigs, beta, Ne):
        def func(mu):
            return np.sum(1 / (1 + np.exp(beta * (eigs - mu)))) - Ne

        mu_min = np.min(eigs) - 1
        mu_max = np.max(eigs) + 1
        return bisect(func, mu_min, mu_max)

    def test_qsvt(self):
        mu = self.find_mu(self.eigs, self.beta, self.Ne)
        H_norm = csr_matrix(
            (1 / self.norm)
            * build_hamiltonian(self.n0, self.D, self.D_delta, self.N, self.params)[0]
        )
        U, P, complexity = build_qsvt(
            H_norm, self.norm, self.beta, mu, self.d, self.epsilon_poly, self.m
        )
        # Check outputs
        self.assertIsInstance(U, QuantumCircuit)
        self.assertTrue(callable(P))
        self.assertIsInstance(complexity, int)
        self.assertGreater(complexity, 0)
        # Classical verification of polynomial approximation
        true_f = 1 / (1 + np.exp(self.beta * (self.eigs - mu)))
        true_trace = np.sum(true_f)
        self.assertLess(abs(true_trace - self.Ne), 1e-6)
        norm_eigs = self.eigs / self.norm
        approx_f = P(norm_eigs)
        approx_trace = np.sum(approx_f)
        self.assertLess(abs(approx_trace - self.Ne), 1e-3)
        # Note: Actual circuit simulation omitted as QSVT circuit is placeholder; classical approx verifies the polynomial.


if __name__ == "__main__":
    unittest.main()
