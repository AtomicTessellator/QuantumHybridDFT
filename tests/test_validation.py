import unittest

import numpy as np

from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.scf import find_mu, run_scf
from qhdft.validation import (
    compute_energy,
    compute_error_breakdown,
    generate_density_data,
    run_scaling_test,
)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dimension": 1,
            "computational_domain": [0, 10.0],
            "grid_exponent": 5,
            "atomic_positions": [3.0, 7.0],
            "atomic_numbers": [3, 1],
            "Z": [3, 1],  # Still needed for build_hamiltonian
            "gaussian_width": 0.5,
            "interpolation_tolerance": 0.1,
            "epsilon": 0.1,  # Still needed for build_hamiltonian
        }
        self.inverse_temperature = 10.0
        self.mixing_parameter = 0.5
        self.block_size = 2
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
        self.num_electrons = sum(self.params["Z"])
        # Run SCF for converged density
        self.convergedCoarseDensity, self.residuals, _ = run_scf(
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
        # Classical converged for comparison
        self.exactCoarseDensity, self.exactEnergy = self.classical_scf(
            self.initialCoarseDensity
        )

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
        energy = np.sum(occupations * eigenvalues)
        return current_coarse_density, energy

    def test_compute_energy(self):
        estimated_energy = compute_energy(
            self.convergedCoarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
            self.params,
            self.inverse_temperature,
        )
        self.assertLess(abs(estimated_energy - self.exactEnergy), 1e-2)

    def test_generate_density_data(self):
        r, densityPlot = generate_density_data(
            self.convergedCoarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
        )
        _, exactDensityPlot = generate_density_data(
            self.exactCoarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
        )
        l2Difference = np.linalg.norm(densityPlot - exactDensityPlot)
        self.assertLess(l2Difference, 1e-2)
        self.assertEqual(len(r), len(self.fineGrid))
        self.assertEqual(len(densityPlot), len(self.fineGrid))

    def test_run_scaling_test(self):
        atom_count_range = np.arange(2, 21, 2)  # 2 to 20 step 2
        metrics, r_squared = run_scaling_test(
            self.params,
            self.inverse_temperature,
            self.mixing_parameter,
            self.max_iterations,
            self.convergence_threshold,
            self.confidence_level,
            self.estimation_error_tolerance,
            self.num_quantum_samples,
            atom_count_range,
        )
        self.assertGreater(r_squared, 0.95)
        self.assertEqual(len(metrics["atomCounts"]), len(atom_count_range))
        self.assertTrue(all(q > 0 for q in metrics["queries"]))

    def test_compute_error_breakdown(self):
        estimated_energy = compute_energy(
            self.convergedCoarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
            self.params,
            self.inverse_temperature,
        )
        polynomial_max_error = 1e-4  # From stage3
        errors = compute_error_breakdown(
            self.convergedCoarseDensity,
            self.exactCoarseDensity,
            estimated_energy,
            self.exactEnergy,
            polynomial_max_error,
            self.estimation_error_tolerance,
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
