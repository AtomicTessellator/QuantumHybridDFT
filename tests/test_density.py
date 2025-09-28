import unittest

import numpy as np
from scipy.optimize import bisect
from scipy.sparse import csr_matrix

from qhdft.density import estimate_density
from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.qsvt import build_qsvt


class TestDensity(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dimension": 1,
            "computational_domain": [0, 10.0],
            "grid_exponent": 5,  # Ng=32
            "atomic_positions": [3.0, 7.0],
            "atomic_numbers": [3, 1],
            "Z": [3, 1],  # Still needed for build_hamiltonian
            "gaussian_width": 0.5,
            "interpolation_tolerance": 0.1,
            "epsilon": 0.1,  # Still needed for build_hamiltonian
        }
        self.inverse_temperature = 10.0
        self.polynomial_degree = 1500
        self.polynomial_error_tolerance = 1e-4
        self.fineGrid, self.coarsePoints, self.coarseDensity, self.shapeFunction = (
            setup_discretization(self.params)
        )
        self.num_grid_points = len(self.fineGrid)
        self.num_qubits = int(np.log2(self.num_grid_points))
        self.num_electrons = sum(self.params["Z"])
        self.hamiltonian, self.normalization_factor, _, _ = build_hamiltonian(
            self.coarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
            self.params,
        )
        self.eigenvalues = np.linalg.eigh(self.hamiltonian.toarray())[0]

    def find_mu(self, eigenvalues, inverse_temperature, num_electrons):
        def occupation_difference(mu):
            return (
                np.sum(1 / (1 + np.exp(inverse_temperature * (eigenvalues - mu))))
                - num_electrons
            )

        mu_min = np.min(eigenvalues) - 1
        mu_max = np.max(eigenvalues) + 1
        return bisect(occupation_difference, mu_min, mu_max)

    def test_estimate_density(self):
        chemical_potential = self.find_mu(
            self.eigenvalues, self.inverse_temperature, self.num_electrons
        )
        normalized_hamiltonian = csr_matrix(
            (1 / self.normalization_factor)
            * build_hamiltonian(
                self.coarseDensity,
                self.fineGrid,
                self.coarsePoints,
                self.shapeFunction,
                self.params,
            )[0]
        )
        quantum_circuit, polynomial_approximation, _ = build_qsvt(
            normalized_hamiltonian,
            self.normalization_factor,
            self.inverse_temperature,
            chemical_potential,
            self.polynomial_degree,
            self.polynomial_error_tolerance,
            self.num_qubits,
        )
        # Parameters
        num_interpolation_points = len(self.coarsePoints)
        indices = np.arange(
            num_interpolation_points
        )  # All for simplicity, B=num_interpolation_points=2
        num_quantum_samples = 1000
        confidence_level = 0.01
        estimation_error_tolerance = 1e-2
        # Compute true coarse density classically for verification
        eigenvectors = np.linalg.eigh(self.hamiltonian.toarray())[1]
        occupations = 1 / (
            1
            + np.exp(self.inverse_temperature * (self.eigenvalues - chemical_potential))
        )
        fine_density = np.sum(occupations[:, None] * (eigenvectors**2), axis=0)
        diffs = self.fineGrid[:, None, :] - self.coarsePoints[None, :, :]
        interpolation_matrix = self.shapeFunction(diffs)
        coarse_density_true = np.linalg.lstsq(
            interpolation_matrix, fine_density, rcond=None
        )[0]
        # Run multiple trials
        num_trials = 100
        successes = 0
        for _ in range(num_trials):
            estimated_density, confidence_intervals, query_complexity = (
                estimate_density(
                    self.hamiltonian,
                    self.normalization_factor,
                    self.inverse_temperature,
                    chemical_potential,
                    self.fineGrid,
                    self.coarsePoints,
                    self.shapeFunction,
                    indices,
                    num_quantum_samples,
                    confidence_level,
                    estimation_error_tolerance,
                )
            )
            # Check L2 error
            l2_error = np.linalg.norm(estimated_density - coarse_density_true)
            if l2_error < estimation_error_tolerance:
                successes += 1
        # Verify in >95% trials error < estimation_error_tolerance
        self.assertGreater(successes / num_trials, 0.95)
        self.assertGreater(query_complexity, 0)


if __name__ == "__main__":
    unittest.main()
