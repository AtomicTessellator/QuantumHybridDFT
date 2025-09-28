import numpy as np
from scipy.stats import norm

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
