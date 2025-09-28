import numpy as np
from scipy.optimize import bisect

from qhdft.density import estimate_density
from qhdft.hamiltonian import build_hamiltonian

# Self-Consistent Field (SCF) Iterations with Randomized Block Coordinates
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
    """Performs hybrid SCF iterations until convergence or maximum iterations reached.

    Executes the self-consistent field iteration loop where each iteration:
    1. Builds Hamiltonian from current coarse density
    2. Finds chemical potential
    3. Selects random block indices
    4. Estimates density block F(coarseDensity) on block via stage4
    5. Mixes to update coarse density on block

    The function tracks residuals and total query complexity summed over all estimates.

    Parameters
    ----------
    initialCoarseDensity : array
        Initial coarse density values
    fineGrid : Grid
        Fine discretization grid
    coarsePoints : array
        Coarse interpolation points
    shapeFunction : callable
        Shape function for interpolation
    params : dict
        System parameters including atomic charges
    inverseTemperature : float
        Inverse temperature (beta) parameter
    mixingParameter : float
        Mixing parameter for density updates
    blockSize : int
        Size of blocks for quantum estimation
    maxIterations : int
        Maximum number of SCF iterations
    convergenceThreshold : float
        Convergence threshold for residuals
    confidenceLevel : float
        Confidence level for quantum estimation
    estimationErrorTolerance : float
        Error tolerance for density estimation
    numQuantumSamples : int
        Number of quantum samples for estimation

    Returns
    -------
    dict
        Dictionary containing:
        - 'converged': bool indicating if convergence was achieved
        - 'coarseDensity': final coarse density array
        - 'residuals': list of residuals per iteration
        - 'total_complexity': total query complexity
    """
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
