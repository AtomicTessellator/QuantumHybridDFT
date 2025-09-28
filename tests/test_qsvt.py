import unittest

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import bisect
from scipy.sparse import csr_matrix

from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.qsvt import build_qsvt


class TestQsvt(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dimension": 1,
            "computational_domain": [0, 10.0],
            "grid_exponent": 5,  # numGridPoints=32 for small simulation
            "atomic_positions": [3.0, 7.0],
            "atomic_numbers": [3, 1],
            "Z": [3, 1],  # Still needed for build_hamiltonian
            "gaussian_width": 0.5,
            "interpolation_tolerance": 0.1,
            "epsilon": 0.1,  # Still needed for build_hamiltonian
        }
        self.inverse_temperature = 10.0  # Beta parameter
        self.polynomial_degree = 1500  # Degree of Chebyshev polynomial
        self.polynomial_error_tolerance = 1e-4
        self.fineGrid, self.coarsePoints, self.coarseDensity, self.shapeFunction = (
            setup_discretization(self.params)
        )
        self.num_grid_points = len(self.fineGrid)
        self.num_qubits = int(np.log2(self.num_grid_points))
        self.num_electrons = sum(self.params["Z"])
        hamiltonian, self.normalization_factor, _, _ = build_hamiltonian(
            self.coarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
            self.params,
        )
        # Compute eigenvalues classically
        self.eigenvalues = np.linalg.eigh(hamiltonian.toarray())[0]

    def find_mu(self, eigenvalues, inverse_temperature, num_electrons):
        def occupation_difference(mu):
            return (
                np.sum(1 / (1 + np.exp(inverse_temperature * (eigenvalues - mu))))
                - num_electrons
            )

        mu_min = np.min(eigenvalues) - 1
        mu_max = np.max(eigenvalues) + 1
        return bisect(occupation_difference, mu_min, mu_max)

    def test_qsvt(self):
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
        quantum_circuit, polynomial_approximation, gate_complexity = build_qsvt(
            normalized_hamiltonian,
            self.normalization_factor,
            self.inverse_temperature,
            chemical_potential,
            self.polynomial_degree,
            self.polynomial_error_tolerance,
            self.num_qubits,
        )
        # Check outputs
        self.assertIsInstance(quantum_circuit, QuantumCircuit)
        self.assertTrue(callable(polynomial_approximation))
        self.assertIsInstance(gate_complexity, int)
        self.assertGreater(gate_complexity, 0)
        # Classical verification of polynomial approximation
        exact_fermi_dirac = 1 / (
            1
            + np.exp(self.inverse_temperature * (self.eigenvalues - chemical_potential))
        )
        exact_trace = np.sum(exact_fermi_dirac)
        self.assertLess(abs(exact_trace - self.num_electrons), 1e-6)
        normalized_eigenvalues = self.eigenvalues / self.normalization_factor
        approximate_fermi_dirac = polynomial_approximation(normalized_eigenvalues)
        approximate_trace = np.sum(approximate_fermi_dirac)
        self.assertLess(abs(approximate_trace - self.num_electrons), 1e-3)
        # Note: Actual circuit simulation omitted as QSVT circuit is placeholder; classical approx verifies the polynomial.


if __name__ == "__main__":
    unittest.main()
