import unittest

import numpy as np
from scipy.sparse import issparse

from qhdft.discretization import setup_discretization
from qhdft.stage2 import build_hamiltonian


class TestStage2(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dim": 1,
            "domain": [0, 10.0],
            "m": 6,  # Small Ng=64 for tests
            "atomic_positions": [3.0, 7.0],
            "Z": [3, 1],
            "sigma": 0.5,
            "epsilon": 0.1,
        }
        self.fineGrid, self.coarsePoints, self.coarseDensity, self.shapeFunction = (
            setup_discretization(self.params)
        )
        self.Ng = len(self.fineGrid)

    def test_build_hamiltonian(self):
        H, norm, L, V = build_hamiltonian(
            self.coarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
            self.params,
        )
        # Check sparse
        self.assertTrue(issparse(H))
        # Check shape
        self.assertEqual(H.shape, (self.Ng, self.Ng))
        # Check sparsity: approx 3 nonzeros per row (tridiagonal)
        nnz_per_row = H.nnz / self.Ng
        self.assertAlmostEqual(nnz_per_row, 3, delta=0.1)  # boundary rows have 2
        # Check Hermitian
        self.assertLess(np.max(np.abs(H - H.T)), 1e-10)
        # Check normalization positive
        self.assertGreater(norm, 0)
        # Check oracles for row 0: should have col 0 and 1
        self.assertEqual(L(0, 0), 0)
        self.assertEqual(L(0, 1), 1)
        # Check values non-zero
        self.assertNotEqual(V(0, 0), 0)
        self.assertNotEqual(V(0, 1), 0)
        # Check index error
        with self.assertRaises(IndexError):
            L(0, 2)
        # TODO: Add more checks, e.g., compare to known simple case


if __name__ == "__main__":
    unittest.main()
