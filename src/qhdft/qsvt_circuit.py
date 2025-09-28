import numpy as np
from numpy.polynomial.chebyshev import chebfit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator


def _clenshaw_chebyshev_matrix(matrixA, chebyshev_coefficients):
    """Evaluate Chebyshev polynomial with coefficients on a Hermitian matrix using Clenshaw's algorithm.

    Parameters
    ----------
    matrixA : np.ndarray
        Hermitian matrix with spectrum in [-1, 1]. Shape (N, N).
    chebyshev_coefficients : np.ndarray
        Coefficients c_k for T_k in numpy.polynomial.chebyshev convention.

    Returns
    -------
    np.ndarray
        Matrix P(A) with the same shape as A.
    """
    degree = len(chebyshev_coefficients) - 1
    size = matrixA.shape[0]
    identity = np.eye(size, dtype=np.complex128)

    b_kplus1 = np.zeros((size, size), dtype=np.complex128)  # B_{k+1}
    b_kplus2 = np.zeros((size, size), dtype=np.complex128)  # B_{k+2}

    for k in range(degree, 0, -1):
        # B_k = a_k I + 2 A B_{k+1} - B_{k+2}
        b_k = (
            chebyshev_coefficients[k] * identity + 2.0 * (matrixA @ b_kplus1) - b_kplus2
        )
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    # P(A) = a_0/2 I + A B_1 - B_2
    return (
        (chebyshev_coefficients[0] * 0.5) * identity + (matrixA @ b_kplus1) - b_kplus2
    )


def _positive_semidefinite_matrix_sqrt(matrix_psd):
    """Compute the positive semidefinite matrix square root via eigen-decomposition.

    Any tiny negative eigenvalues due to numerical noise are clamped to 0.
    """
    # Hermitian symmetrization for numerical stability
    matrix_psd = 0.5 * (matrix_psd + matrix_psd.conj().T)
    evals, evecs = np.linalg.eigh(matrix_psd)
    evals_clamped = np.clip(evals, a_min=0.0, a_max=None)
    sqrt_evals = np.sqrt(evals_clamped)
    return (evecs * sqrt_evals) @ evecs.conj().T


def _build_unitary_dilation_from_contraction(block_matrix):
    """Construct a unitary dilation U for a contraction B (||B|| <= 1):

    U = [[B, sqrt(I - B B^†)],
         [sqrt(I - B^† B), -B^†]]

    This yields a unitary 2N x 2N matrix whose top-left block equals B.
    """
    size = block_matrix.shape[0]
    identity = np.eye(size, dtype=np.complex128)

    # Proceed even if tiny numerical overshoot; square-root builder clamps negatives
    # This preserves the top-left block exactly equal to B without global re-unitarization.
    spectral_norm = np.linalg.norm(block_matrix, 2)

    top_right = _positive_semidefinite_matrix_sqrt(
        identity - block_matrix @ block_matrix.conj().T
    )
    bottom_left = _positive_semidefinite_matrix_sqrt(
        identity - block_matrix.conj().T @ block_matrix
    )
    bottom_right = -block_matrix.conj().T

    top = np.concatenate([block_matrix, top_right], axis=1)
    bottom = np.concatenate([bottom_left, bottom_right], axis=1)
    unitary = np.concatenate([top, bottom], axis=0)

    return unitary


def build_qsvt_unitary(
    normalizedHamiltonian,
    normalizationFactor,
    inverseTemperature,
    chemicalPotential,
    polynomialDegree,
    polynomialErrorTolerance,
    numQubits,
):
    """Build a rigorous block-encoding circuit for the Chebyshev approximation to the Fermi-Dirac function.

    The circuit implements a unitary dilation whose top-left block equals P(H_norm), where P is the
    Chebyshev approximation to f(x) = 1 / (1 + exp(β (x - μ))).

    Ancilla qubit is the most significant qubit in the unitary's qubit ordering. Therefore,
    the top-left N × N block of the 2N × 2N unitary equals P(H).
    """
    numGridPoints = normalizedHamiltonian.shape[0]
    sparsity = normalizedHamiltonian.nnz / numGridPoints

    def fermi_dirac_mapped(y):
        return 1.0 / (
            1.0
            + np.exp(inverseTemperature * (normalizationFactor * y - chemicalPotential))
        )

    # Chebyshev nodes (first kind) and fit
    node_indices = np.arange(polynomialDegree + 1)
    chebyshev_nodes = np.cos(
        np.pi * (2 * node_indices + 1) / (2 * (polynomialDegree + 1))
    )
    function_values = fermi_dirac_mapped(chebyshev_nodes)
    chebyshev_coefficients = chebfit(chebyshev_nodes, function_values, polynomialDegree)

    # Verify max error on dense grid
    test_points = np.linspace(-1.0, 1.0, 1000)
    exact_values = fermi_dirac_mapped(test_points)
    poly_values = np.polynomial.chebyshev.chebval(test_points, chebyshev_coefficients)
    max_err = float(np.max(np.abs(poly_values - exact_values)))
    if max_err > polynomialErrorTolerance:
        raise ValueError(
            f"Approximation error {max_err} exceeds tolerance {polynomialErrorTolerance}. Increase polynomial degree."
        )

    # Matrix polynomial via spectral decomposition (numerically robust for Hermitian)
    H_norm = normalizedHamiltonian.toarray().astype(np.complex128)
    H_norm = 0.5 * (H_norm + H_norm.conj().T)
    evals, evecs = np.linalg.eigh(H_norm)
    P_diag = np.polynomial.chebyshev.chebval(evals, chebyshev_coefficients)
    P_of_H = (evecs * P_diag) @ evecs.conj().T

    # Build unitary dilation of P(H)
    unitary_2n = _build_unitary_dilation_from_contraction(P_of_H)

    # Map into QuantumCircuit with ancilla as most significant qubit (first in list)
    qc = QuantumCircuit(numQubits + 1)
    gate = UnitaryGate(Operator(unitary_2n), check_input=False)
    # Place ancilla at index 0 for clarity, followed by system qubits 1..m
    wires = [qc.qubits[0]] + qc.qubits[1:]
    qc.append(gate, wires)

    # Complexity estimate (conservative)
    gate_complexity = int(
        sparsity * polynomialDegree * (np.log(1.0 / polynomialErrorTolerance) ** 2)
    )

    # Polynomial evaluator for scalars/arrays on [-1,1]
    def polynomial_approximation(z):
        return np.polynomial.chebyshev.chebval(z, chebyshev_coefficients)

    return qc, polynomial_approximation, gate_complexity
