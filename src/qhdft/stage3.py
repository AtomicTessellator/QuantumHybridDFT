import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from qiskit import QuantumCircuit

# Stage 3: Quantum Singular Value Transformation (QSVT) for Density-Matrix Encoding
# Applies QSVT to block-encode an approximation of the density matrix Γ = f(H), where f(x) is the Fermi-Dirac function
# f(x) = 1 / (1 + exp(β (x - μ))), encoding thermal occupations at inverse temperature β and chemical potential μ.
# This avoids explicit diagonalization of H, enabling linear scaling O(Na) instead of O(Na^3) for large systems.
# We approximate f with a Chebyshev polynomial P_d of degree d, ensuring error < ε_poly on [-1,1].
# The QSVT circuit U acts on normalized H (eigenvalues in [-1,1]), producing a block encoding of P_d(H_norm).
# Gate complexity is O(s d log(1/ε)), where s is sparsity of H.


def build_qsvt(H_norm, norm_factor, beta, mu, d, epsilon_poly, m):
    # Builds QSVT circuit for approximating f on normalized H_norm = H / norm_factor.
    # Inputs: sparse H_norm, norm_factor, β, μ, polynomial degree d, error tolerance ε_poly, qubit count m = log2(Ng).
    # Outputs: placeholder circuit U, polynomial P, gate complexity estimate.
    # Sparsity s used for complexity estimate.
    Ng = H_norm.shape[0]
    s = H_norm.nnz / Ng

    # Target function g(y) = f(norm_factor * y) for y in [-1,1].
    def g(y):
        return 1 / (1 + np.exp(beta * (norm_factor * y - mu)))

    # Fit Chebyshev polynomial at Chebyshev nodes for stable approximation.
    k = np.arange(d + 1)
    x = np.cos(np.pi * (2 * k + 1) / (2 * (d + 1)))
    y = g(x)
    coeffs = chebfit(x, y, d)
    # Verify max error on test points.
    x_test = np.linspace(-1, 1, 1000)
    g_test = g(x_test)
    p_test = chebval(x_test, coeffs)
    max_err = np.max(np.abs(p_test - g_test))
    if max_err > epsilon_poly:
        raise ValueError(
            f"Approximation error {max_err} exceeds epsilon_poly {epsilon_poly}. Increase degree d."
        )

    # Polynomial evaluator using coefficients.
    def P(z):
        return chebval(z, coeffs)

    # Placeholder for actual QSVT circuit; in practice, would compute phase factors for QSVT sequence.
    # Uses m qubits for system + 1 ancillary for block encoding.
    qc = QuantumCircuit(m + 1)
    qc.barrier()  # Placeholder for QSVT gates
    # Gate complexity estimate: O(s * d * log(1/ε)^2), conservative.
    complexity = int(s * d * (np.log(1 / epsilon_poly) ** 2))
    return qc, P, complexity
