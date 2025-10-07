import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.stats import norm

from qhdft.qsvt import build_qsvt

# Stage 4: Electron Density Estimation via Quantum Measurements
# Estimates selected components of the density F(n) = diagonals of Γ = f(H), where H is from current n.
# In a real quantum setting, this would use the QSVT circuit U to prepare states and measure probabilities
# via amplitude estimation for efficiency O(1/ε). Here, we simulate classically: compute exact density,
# project to coarse grid, add Gaussian noise mimicking measurement variance, for selected block indices.
# Provides estimates hat_f, confidence intervals, and query complexity estimate.
# Block selection allows randomized updates in SCF, reducing per-iteration cost to O(B / ε) instead of O(NI / ε).


def estimate_density(
    hamiltonian,
    normalizationFactor,
    inverseTemperature,
    chemicalPotential,
    fineGrid,
    coarsePoints,
    shapeFunction,
    blockIndices,
    numQuantumSamples,
    confidenceLevel,
    estimationErrorTolerance,
):
    # Simulates estimation of coarse density on selected block indices.
    # Classically computes exact fine density fineDensity = sum occupations * |ψ_i|^2.
    # Projects to coarse truCoarseDensity via least-squares with interpolation matrix.
    # Adds noise ~ Normal(0, sigma^2), where sigma tuned for L2 error < estimationErrorTolerance with high probability.
    # Confidence intervals assume normality. Complexity rough O(sparsity * blockSize / ε * numQuantumSamples).
    # Classically compute true density on fine grid for simulation
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
    occupations = 1 / (
        1 + np.exp(inverseTemperature * (eigenvalues - chemicalPotential))
    )
    fineDensity = np.sum(occupations[:, None] * (eigenvectors**2), axis=0)
    # Interpolate to coarse (though for estimation, we simulate on fine then project, but simplify)
    diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]
    interpolationMatrix = shapeFunction(
        diffs
    )  # (numGridPoints, numInterpolationPoints)
    # True coarse density by least squares or projection
    trueCoarseDensity = np.linalg.lstsq(interpolationMatrix, fineDensity, rcond=None)[0]
    # For selected block indices (subset of 0 to numInterpolationPoints-1)
    blockSize = len(blockIndices)
    # Simulated estimation: true value + Gaussian noise with variance based on numQuantumSamples
    # Assume variance sigma^2 = (true_val * (1 - true_val)) / numQuantumSamples; for simplicity, use fixed sigma
    # From plan, variance < sigma^2 to ensure low probability of exceeding estimationErrorTolerance
    standardDeviation = estimationErrorTolerance / (
        3 * np.sqrt(blockSize)
    )  # To have low prob of exceeding estimationErrorTolerance
    estimatedDensityBlock = trueCoarseDensity[blockIndices] + np.random.normal(
        0, standardDeviation, blockSize
    )
    # Confidence intervals, e.g. 95% CI assuming normal distribution
    zScore = norm.ppf(1 - confidenceLevel / 2)
    confidenceIntervals = np.column_stack(
        (
            estimatedDensityBlock - zScore * standardDeviation,
            estimatedDensityBlock + zScore * standardDeviation,
        )
    )
    # Query complexity O(sparsity * numInterpolationPoints / estimationErrorTolerance), but per block blockSize
    sparsity = hamiltonian.nnz / hamiltonian.shape[0]
    queryComplexity = (
        int(sparsity * blockSize / estimationErrorTolerance) * numQuantumSamples
    )  # Rough estimate
    return estimatedDensityBlock, confidenceIntervals, queryComplexity


def estimate_density_quantum(
    hamiltonian,
    normalizationFactor,
    inverseTemperature,
    chemicalPotential,
    fineGrid,
    coarsePoints,
    shapeFunction,
    blockIndices,
    numQuantumSamples,
    confidenceLevel,
    estimationErrorTolerance,
):
    """Quantum density estimation using QSVT circuit and quantum simulator.

    This function computes density estimates ab initio using quantum circuits:
    1. Build QSVT circuit to block-encode Fermi-Dirac function
    2. Prepare computational basis states
    3. Apply QSVT unitary
    4. Measure probabilities via statevector simulation
    5. Project to coarse grid via interpolation matrix

    NO classical eigendecomposition is used - all computation via quantum simulator.
    """
    numGridPoints = hamiltonian.shape[0]
    numQubits = int(np.log2(numGridPoints))
    blockSize = len(blockIndices)

    # Normalize Hamiltonian for QSVT (eigenvalues to [-1,1])
    from scipy.sparse import csr_matrix

    normalizedHamiltonian = csr_matrix((1.0 / normalizationFactor) * hamiltonian)

    # Build QSVT circuit - this creates block encoding of Fermi-Dirac function
    polynomialDegree = 1500
    polynomialErrorTolerance = 1e-4
    quantumCircuit, polynomialApproximation, gateComplexity = build_qsvt(
        normalizedHamiltonian,
        normalizationFactor,
        inverseTemperature,
        chemicalPotential,
        polynomialDegree,
        polynomialErrorTolerance,
        numQubits,
    )

    # Extract fine density from quantum circuit via amplitude measurements
    # The QSVT circuit has ancilla as first qubit, system qubits follow
    # Top-left block (ancilla=0) encodes P(H) ≈ Fermi-Dirac function

    # Use statevector simulator - single execution for all grid points
    backend = AerSimulator(method="statevector")

    # Prepare uniform superposition: |+⟩^⊗n on system qubits
    # Extract all density values from one circuit execution
    prepCircuit = QuantumCircuit(numQubits + 1)
    for qubitIdx in range(1, numQubits + 1):
        prepCircuit.h(qubitIdx)

    # Apply QSVT circuit once
    prepCircuit.compose(quantumCircuit, inplace=True)
    prepCircuit.save_statevector()

    # Single execution
    compiled = transpile(prepCircuit, backend)
    job = backend.run(compiled, shots=1)
    result = job.result()
    statevec = result.get_statevector(prepCircuit)
    probabilities = np.abs(statevec.data) ** 2

    # Extract density from ancilla=0 subspace
    fineDensityQuantum = np.zeros(numGridPoints)
    for gridIdx in range(numGridPoints):
        fineDensityQuantum[gridIdx] = probabilities[gridIdx]

    # Project to coarse grid via interpolation matrix
    diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]
    interpolationMatrix = shapeFunction(diffs)
    coarseDensityQuantum = np.linalg.lstsq(
        interpolationMatrix, fineDensityQuantum, rcond=None
    )[0]

    # Extract block indices
    estimatedDensityBlock = coarseDensityQuantum[blockIndices]

    # Add measurement noise based on shot noise
    shotNoise = 1.0 / np.sqrt(numQuantumSamples)
    estimatedDensityBlock += np.random.normal(0, shotNoise * 0.1, blockSize)

    # Confidence intervals based on shot noise
    zScore = norm.ppf(1 - confidenceLevel / 2)
    confidenceIntervals = np.column_stack(
        (
            estimatedDensityBlock - zScore * shotNoise,
            estimatedDensityBlock + zScore * shotNoise,
        )
    )

    # Query complexity: grid points × samples × gate complexity
    sparsity = hamiltonian.nnz / hamiltonian.shape[0]
    queryComplexity = numGridPoints * numQuantumSamples * gateComplexity

    return estimatedDensityBlock, confidenceIntervals, queryComplexity
