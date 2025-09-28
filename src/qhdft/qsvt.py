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


def build_qsvt(
    normalizedHamiltonian,
    normalizationFactor,
    inverseTemperature,
    chemicalPotential,
    polynomialDegree,
    polynomialErrorTolerance,
    numQubits,
):
    # Builds QSVT circuit for approximating Fermi-Dirac function on normalized Hamiltonian = H / normalizationFactor.
    # Inputs: sparse normalizedHamiltonian, normalizationFactor, inverseTemperature (β), chemicalPotential (μ),
    #         polynomialDegree, error tolerance polynomialErrorTolerance, qubit count numQubits = log2(numGridPoints).
    # Outputs: placeholder circuit, polynomial approximation, gate complexity estimate.
    # Sparsity used for complexity estimate.
    numGridPoints = normalizedHamiltonian.shape[0]
    sparsity = normalizedHamiltonian.nnz / numGridPoints

    # Target function: Fermi-Dirac on original scale, mapped to [-1,1] domain.
    def fermiDiracMapped(y):
        return 1 / (
            1
            + np.exp(inverseTemperature * (normalizationFactor * y - chemicalPotential))
        )

    # Fit Chebyshev polynomial at Chebyshev nodes for stable approximation.
    nodeIndices = np.arange(polynomialDegree + 1)
    chebyshevNodes = np.cos(
        np.pi * (2 * nodeIndices + 1) / (2 * (polynomialDegree + 1))
    )
    functionValues = fermiDiracMapped(chebyshevNodes)
    chebyshevCoefficients = chebfit(chebyshevNodes, functionValues, polynomialDegree)
    # Verify max error on test points.
    testPoints = np.linspace(-1, 1, 1000)
    exactValues = fermiDiracMapped(testPoints)
    polynomialValues = chebval(testPoints, chebyshevCoefficients)
    maxApproximationError = np.max(np.abs(polynomialValues - exactValues))
    if maxApproximationError > polynomialErrorTolerance:
        raise ValueError(
            f"Approximation error {maxApproximationError} exceeds tolerance {polynomialErrorTolerance}. Increase polynomial degree."
        )

    # Polynomial evaluator using Chebyshev coefficients.
    def polynomialApproximation(z):
        return chebval(z, chebyshevCoefficients)

    # Placeholder for actual QSVT circuit; in practice, would compute phase factors for QSVT sequence.
    # Uses numQubits for system + 1 ancillary for block encoding.
    quantumCircuit = QuantumCircuit(numQubits + 1)
    quantumCircuit.barrier()  # Placeholder for QSVT gates
    # Gate complexity estimate: O(sparsity * polynomialDegree * log(1/ε)^2), conservative.
    gateComplexity = int(
        sparsity * polynomialDegree * (np.log(1 / polynomialErrorTolerance) ** 2)
    )
    return quantumCircuit, polynomialApproximation, gateComplexity
