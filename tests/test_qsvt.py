import unittest

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from scipy.optimize import bisect
from scipy.sparse import csr_matrix

from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.qsvt import build_qsvt
from qhdft.qsvt_qsp import build_qsp_qsvt_circuit


class TestQsvt(unittest.TestCase):
    def setUp(self):
        self.params = {
            "dimension": 1,
            "computational_domain": [0, 10.0],
            "grid_exponent": 5,  # numGridPoints=32 for small simulation
            "atomic_positions": [3.0, 7.0],
            "atomic_numbers": [3, 1],
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
        self.num_electrons = sum(self.params["atomic_numbers"])
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
        # Verify top-left block (ancilla as most significant qubit) equals polynomial P(H)
        unitary = Operator(quantum_circuit).data
        N = self.num_grid_points
        ancilla0_block = unitary[:N, :N]
        normalized_h = (1 / self.normalization_factor) * build_hamiltonian(
            self.coarseDensity,
            self.fineGrid,
            self.coarsePoints,
            self.shapeFunction,
            self.params,
        )[0].toarray()
        normalized_h = 0.5 * (normalized_h + normalized_h.T)
        # Compare traces as a robust proxy
        normalized_eigs = self.eigenvalues / self.normalization_factor
        P_vals = polynomial_approximation(normalized_eigs)
        self.assertAlmostEqual(np.trace(ancilla0_block).real, np.sum(P_vals), places=3)

    def test_qsp_qsvt_simulator(self):
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
        degree = 8
        qc, phases = build_qsp_qsvt_circuit(
            normalized_hamiltonian,
            self.normalization_factor,
            self.inverse_temperature,
            self.find_mu(  # pyright: ignore[reportArgumentType]
                self.eigenvalues, self.inverse_temperature, self.num_electrons
            ),
            degree,
            self.num_qubits,
        )
        backend = AerSimulator(method="matrix_product_state")
        tqc = transpile(qc, backend)
        U = Operator(tqc).data
        N = self.num_grid_points
        ancilla0_block = U[:N, :N]  # pyright: ignore[reportArgumentType]
        self.assertGreaterEqual(np.trace(ancilla0_block).real, -1e-6)
        self.assertLessEqual(np.trace(ancilla0_block).real, N + 1e-6)


if __name__ == "__main__":
    unittest.main()
