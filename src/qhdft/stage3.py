import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from qiskit import QuantumCircuit


def build_qsvt(H_norm, norm_factor, beta, mu, d, epsilon_poly, m):
    # Sparsity s
    Ng = H_norm.shape[0]
    s = H_norm.nnz / Ng

    # Function g(y) = 1 / (1 + exp(beta * (norm_factor * y - mu)))
    def g(y):
        return 1 / (1 + np.exp(beta * (norm_factor * y - mu)))

    # Chebyshev nodes for fitting
    k = np.arange(d + 1)
    x = np.cos(np.pi * (2 * k + 1) / (2 * (d + 1)))
    y = g(x)
    # Fit Chebyshev coefficients
    coeffs = chebfit(x, y, d)
    # Verify approximation error
    x_test = np.linspace(-1, 1, 1000)
    g_test = g(x_test)
    p_test = chebval(x_test, coeffs)
    max_err = np.max(np.abs(p_test - g_test))
    if max_err > epsilon_poly:
        raise ValueError(
            f"Approximation error {max_err} exceeds epsilon_poly {epsilon_poly}. Increase degree d."
        )

    # Polynomial callable
    def P(z):
        return chebval(z, coeffs)

    # Placeholder QSVT circuit (actual implementation would use phases from QSVT algorithm)
    # Here, we create a dummy circuit with m system qubits + 1 ancillary
    qc = QuantumCircuit(m + 1)
    qc.barrier()  # Placeholder for QSVT gates
    # Estimated gate complexity O(s * d * log(1/eps)^2) - conservative estimate
    complexity = int(s * d * (np.log(1 / epsilon_poly) ** 2))
    return qc, P, complexity
