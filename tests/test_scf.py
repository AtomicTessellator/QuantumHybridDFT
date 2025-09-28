import unittest

import numpy as np

from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.scf import find_mu, run_scf


class TestScf(unittest.TestCase):
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
        self.mixing_parameter = 0.5
        self.block_size = 2  # Update all for small numInterpolationPoints=2
        self.max_iterations = 100
        self.convergence_threshold = 1e-4
        self.confidence_level = 0.01
        self.estimation_error_tolerance = 1e-4
        self.num_quantum_samples = 100000
        (
            self.fineGrid,
            self.coarsePoints,
            self.initialCoarseDensity,
            self.shapeFunction,
        ) = setup_discretization(self.params)
        self.num_grid_points = len(self.fineGrid)
        self.num_electrons = sum(self.params["Z"])

    def classical_scf(self, initial_coarse_density, tolerance=1e-6, max_iterations=100):
        current_coarse_density = initial_coarse_density.copy()
        for _ in range(max_iterations):
            hamiltonian, _, _, _ = build_hamiltonian(
                current_coarse_density,
                self.fineGrid,
                self.coarsePoints,
                self.shapeFunction,
                self.params,
            )
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
            chemical_potential = find_mu(
                eigenvalues, self.inverse_temperature, self.num_electrons
            )
            occupations = 1 / (
                1
                + np.exp(self.inverse_temperature * (eigenvalues - chemical_potential))
            )
            fine_density = np.sum(occupations[:, None] * (eigenvectors**2), axis=0)
            diffs = self.fineGrid[:, None, :] - self.coarsePoints[None, :, :]
            interpolation_matrix = self.shapeFunction(diffs)
            new_coarse_density = np.linalg.lstsq(
                interpolation_matrix, fine_density, rcond=None
            )[0]
            residual = np.linalg.norm(new_coarse_density - current_coarse_density)
            current_coarse_density = new_coarse_density
            if residual < tolerance:
                break
        # Compute energy: sum occupations * eigenvalues
        energy = np.sum(occupations * eigenvalues)
        return current_coarse_density, energy

    def test_scf(self):
        exact_coarse_density, exact_energy = self.classical_scf(
            self.initialCoarseDensity
        )
        np.random.seed(42)
        num_trials = 50
        iterations_list = []
        energy_errors = []
        norm_errors = []
        for _ in range(num_trials):
            estimated_coarse_density, residuals, query_complexity = run_scf(
                self.initialCoarseDensity,
                self.fineGrid,
                self.coarsePoints,
                self.shapeFunction,
                self.params,
                self.inverse_temperature,
                self.mixing_parameter,
                self.block_size,
                self.max_iterations,
                self.convergence_threshold,
                self.confidence_level,
                self.estimation_error_tolerance,
                self.num_quantum_samples,
            )
            self.assertLessEqual(len(residuals), self.max_iterations)
            self.assertGreater(query_complexity, 0)

            norm_error = np.linalg.norm(estimated_coarse_density - exact_coarse_density)
            norm_errors.append(norm_error)

            # Compute energy for estimated_coarse_density
            hamiltonian, _, _, _ = build_hamiltonian(
                estimated_coarse_density,
                self.fineGrid,
                self.coarsePoints,
                self.shapeFunction,
                self.params,
            )
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
            chemical_potential = find_mu(
                eigenvalues, self.inverse_temperature, self.num_electrons
            )
            occupations = 1 / (
                1
                + np.exp(self.inverse_temperature * (eigenvalues - chemical_potential))
            )
            estimated_energy = np.sum(occupations * eigenvalues)
            energy_error = abs(estimated_energy - exact_energy)
            energy_errors.append(energy_error)
            iterations_list.append(len(residuals))

        average_iterations = np.mean(iterations_list)
        self.assertLess(average_iterations, 25)  # Adjusted bound
        self.assertLess(np.mean(norm_errors), 0.005)
        self.assertLess(np.mean(energy_errors), 0.002)
        success_rate = np.sum(np.array(energy_errors) < 0.005) / num_trials
        self.assertGreater(success_rate, 0.9)


if __name__ == "__main__":
    unittest.main()
