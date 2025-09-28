import unittest

import numpy as np

from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.scf import (
    Discretization,
    EstimationControls,
    SCFConfig,
    SCFControls,
    find_mu,
    run_scf_configured,
)


class TestScf(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dimension": 1,
            "computational_domain": [0, 10.0],
            "grid_exponent": 5,  # Ng=32
            "atomic_positions": [3.0, 7.0],
            "atomic_numbers": [3, 1],
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
        self.num_electrons = sum(self.params["atomic_numbers"])

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
            scf_config = SCFConfig(
                initial_coarse_density=self.initialCoarseDensity,
                discretization=Discretization(
                    fine_grid=self.fineGrid,
                    coarse_points=self.coarsePoints,
                    shape_function=self.shapeFunction,
                    system_params=self.params,
                ),
                scf=SCFControls(
                    inverse_temperature=self.inverse_temperature,
                    mixing_parameter=self.mixing_parameter,
                    block_size=self.block_size,
                    max_iterations=self.max_iterations,
                    convergence_threshold=self.convergence_threshold,
                ),
                estimation=EstimationControls(
                    confidence_level=self.confidence_level,
                    estimation_error_tolerance=self.estimation_error_tolerance,
                    num_quantum_samples=self.num_quantum_samples,
                ),
            )
            result = run_scf_configured(scf_config)
            estimated_coarse_density = result.converged_coarse_density
            residuals = result.residuals
            query_complexity = result.total_complexity
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
