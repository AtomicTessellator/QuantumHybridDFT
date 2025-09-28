from typing import Callable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from scipy.optimize import least_squares

from .qsvt_circuit import _build_unitary_dilation_from_contraction


def _su2_signal_oracle(x: float) -> np.ndarray:
    """Single-qubit SU(2) signal oracle for QSP at scalar x in [-1, 1].

    W(x) = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]]
    """
    s = np.sqrt(max(0.0, 1.0 - float(x) ** 2))
    return np.array([[x, 1j * s], [1j * s, x]], dtype=np.complex128)


def _qsp_top_left(x: float, phases: np.ndarray) -> complex:
    """Evaluate the QSP sequence's (0,0) entry for scalar x with given phases.

    Implements U_Φ(x) = Φ(φ_0) Π_{k=1..d} [ W(x) Φ(φ_k) ]
    with Φ(φ) = diag(e^{i φ}, e^{-i φ}). Returns the (0,0) element.
    """
    d = len(phases) - 1
    phi = phases

    def phase_gate(angle: float) -> np.ndarray:
        return np.diag([np.exp(1j * angle), np.exp(-1j * angle)])

    U = phase_gate(phi[0])
    Wx = _su2_signal_oracle(x)
    for k in range(1, d + 1):
        U = U @ Wx @ phase_gate(phi[k])
    return U[0, 0]


def synthesize_qsp_phases(
    target_function: Callable[[np.ndarray], np.ndarray],
    degree: int,
    samples: int = 129,
    max_iter: int = 200,
) -> np.ndarray:
    """Numerically synthesize QSP phases for a bounded real target on [-1,1].

    Solves an unconstrained least-squares fit of Re(U_Φ(x)_{00}) to target(x) at sample points.

    Parameters
    ----------
    target_function : callable
        Maps numpy array in [-1,1] to target values in [-1,1].
    degree : int
        QSP sequence degree (number of W applications). Number of phases = degree + 1.
    samples : int
        Number of grid samples in [-1, 1] for the fit.
    max_iter : int
        Maximum iterations for least squares solver.
    """
    xgrid = np.linspace(-1.0, 1.0, samples)
    y = target_function(xgrid).astype(np.float64)
    y = np.clip(y, -1.0, 1.0)

    # Initial guess: zeros; small random jitter to break symmetry
    rng = np.random.default_rng(1234)
    phi0 = 1e-3 * rng.standard_normal(degree + 1)

    def residuals(phi: np.ndarray) -> np.ndarray:
        vals = np.empty_like(y, dtype=np.float64)
        for i, xv in enumerate(xgrid):
            vals[i] = float(np.real(_qsp_top_left(float(xv), phi)))
        return vals - y

    res = least_squares(residuals, phi0, max_nfev=max_iter, xtol=1e-10, ftol=1e-10)
    return res.x


def build_qsp_qsvt_circuit(
    normalizedHamiltonian,
    normalizationFactor: float,
    inverseTemperature: float,
    chemicalPotential: float,
    degree: int,
    numQubits: int,
) -> Tuple[QuantumCircuit, np.ndarray]:
    """Construct a QSP-based QSVT circuit using a block-encoding of H and synthesized phases.

    This uses the dilation unitary as the block-encoding of H and interleaves ancilla Z-phase
    rotations with applications of the block-encoding, implementing the QSP sequence.

    Returns the circuit and the phases used.
    """

    def fermi_dirac_mapped(y):
        return 1.0 / (
            1.0
            + np.exp(inverseTemperature * (normalizationFactor * y - chemicalPotential))
        )

    phases = synthesize_qsp_phases(fermi_dirac_mapped, degree)

    # Build block-encoding U from normalized Hamiltonian
    H_norm = normalizedHamiltonian
    U = _build_unitary_dilation_from_contraction(H_norm.toarray().astype(np.complex128))

    qc = QuantumCircuit(numQubits + 1)
    U_gate = UnitaryGate(Operator(U), check_input=False)

    # Apply Φ(φ_0)
    qc.rz(2.0 * float(phases[0]), 0)
    # Interleave W≈U and Φ(φ_k)
    for k in range(1, degree + 1):
        qc.append(U_gate, qc.qubits)
        qc.rz(2.0 * float(phases[k]), 0)

    return qc, phases
