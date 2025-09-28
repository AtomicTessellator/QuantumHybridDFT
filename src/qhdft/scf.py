import numpy as np
from scipy.optimize import bisect

from qhdft.density import estimate_density
from qhdft.hamiltonian import build_hamiltonian

# Stage 5: Self-Consistent Field (SCF) Iterations with Randomized Block Coordinates
# Solves the DFT self-consistency n = F(n) iteratively: start with initial n0, update blocks of n using estimates of F.
# Uses Anderson mixing: n_{k+1} = α hat{F}(n_k) + (1-α) n_k on selected block, for stability with noisy estimates.
# Random block selection (size B) reduces cost per iteration to O(B / ε), with total iterations O(log(1/ε_scf)).
# Converges when residual ||n_{k+1} - n_k|| < ε_scf, yielding approximate ground state density.
# μ is adjusted each iteration to enforce ∫ n ≈ Ne via bisection.


def find_mu(eigenvalues, inverseTemperature, numElectrons):
    # Finds chemical potential μ such that sum 1/(1+exp(β(e_i - μ))) = numElectrons.
    # Uses bisection on [min(eigenvalues)-1, max(eigenvalues)+1]; clips large exponents to avoid overflow.
    def occupationSum(mu):
        occupations = np.zeros_like(eigenvalues)
        for i, eigenvalue in enumerate(eigenvalues):
            arg = inverseTemperature * (eigenvalue - mu)
            if arg > 100:
                occupations[i] = 0
            elif arg < -100:
                occupations[i] = 1
            else:
                occupations[i] = 1 / (1 + np.exp(arg))
        return np.sum(occupations) - numElectrons

    muMin = np.min(eigenvalues) - 1
    muMax = np.max(eigenvalues) + 1
    return bisect(occupationSum, muMin, muMax)


def run_scf(
    initialCoarseDensity,
    fineGrid,
    coarsePoints,
    shapeFunction,
    params,
    inverseTemperature,
    mixingParameter,
    blockSize,
    maxIterations,
    convergenceThreshold,
    confidenceLevel,
    estimationErrorTolerance,
    numQuantumSamples,
):
    # Performs maxIterations (or until converged) of hybrid SCF.
    # Each iteration: build Hamiltonian from current coarseDensity, find chemical potential, select random block indices,
    # estimate densityBlock = F(coarseDensity) on block via stage4, mix to update coarseDensity on block.
    # Tracks residuals and total query complexity summed over estimates.
    currentCoarseDensity = initialCoarseDensity.copy()
    numInterpolationPoints = len(coarsePoints)
    numElectrons = sum(params["Z"])
    residuals = []
    total_complexity = 0

    for iteration in range(maxIterations):
        hamiltonian, normalizationFactor, _, _ = build_hamiltonian(
            currentCoarseDensity, fineGrid, coarsePoints, shapeFunction, params
        )
        eigenvalues = np.linalg.eigh(hamiltonian.toarray())[
            0
        ]  # For chemical potential finding, classical
        chemicalPotential = find_mu(eigenvalues, inverseTemperature, numElectrons)

        # Select random block
        blockIndices = np.random.choice(
            numInterpolationPoints, blockSize, replace=False
        )

        # Estimate density on block
        estimatedDensityBlock, _, queryComplexity = estimate_density(
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
        )
        total_complexity += queryComplexity

        # Mixing update only on block
        newCoarseDensity = currentCoarseDensity.copy()
        newCoarseDensity[blockIndices] = (
            mixingParameter * estimatedDensityBlock
            + (1 - mixingParameter) * currentCoarseDensity[blockIndices]
        )

        # Compute residual
        residual = np.linalg.norm(newCoarseDensity - currentCoarseDensity)
        residuals.append(residual)
        currentCoarseDensity = newCoarseDensity

        if residual < convergenceThreshold:
            break

    return currentCoarseDensity, np.array(residuals), total_complexity
