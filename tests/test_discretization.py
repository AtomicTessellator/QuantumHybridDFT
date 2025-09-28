import unittest

import numpy as np

from qhdft.discretization import gaussian, setup_discretization


class TestDiscretization(unittest.TestCase):
    def test_discretization(self):
        params = {
            "dim": 1,
            "domain": [0, 10.0],
            "m": 10,
            "atomic_positions": [3.0, 7.0],
            "Z": [3, 1],
            "sigma": 0.5,
        }
        fineGrid, coarsePoints, coarseDensity, shapeFunction = setup_discretization(
            params
        )
        Ng = 2**10
        NI = 2
        Ne = 4
        self.assertEqual(fineGrid.shape, (Ng, 1))
        self.assertEqual(coarsePoints.shape, (NI, 1))
        self.assertEqual(coarseDensity.shape, (NI,))
        np.testing.assert_allclose(coarseDensity, [3, 1], atol=1e-10)
        # Reconstruct
        diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]
        N_matrix = shapeFunction(diffs)
        n_recon = np.dot(N_matrix, coarseDensity)
        # Compute n_fine
        sigma = 0.5
        n_fine = np.zeros(Ng)
        for pos, z in zip(params["atomic_positions"], params["Z"]):
            d = fineGrid[:, 0] - pos
            n_fine += z * gaussian(d, sigma)
        # L2 error
        l2_error = np.sqrt(np.mean((n_recon - n_fine) ** 2))
        self.assertLess(l2_error, 1e-4)
        # Integral
        integral = np.trapezoid(n_recon, fineGrid[:, 0])
        self.assertLess(abs(integral - Ne), 1e-6)


if __name__ == "__main__":
    unittest.main()
