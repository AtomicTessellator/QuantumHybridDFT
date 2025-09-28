from dataclasses import dataclass
from typing import Any, Callable, Dict

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


def find_mu(
    eigenvalues: np.ndarray, inverseTemperature: float, numElectrons: int
) -> float:
    # Finds chemical potential μ such that sum 1/(1+exp(β(e_i - μ))) = numElectrons.
    # Uses bisection on [min(eigenvalues)-1, max(eigenvalues)+1]; clips large exponents to avoid overflow.
    def occupationSum(mu: float) -> float:
        occupations = np.zeros_like(eigenvalues)
        for i, eigenvalue in enumerate(eigenvalues):
            arg = inverseTemperature * (eigenvalue - mu)
            if arg > 100:
                occupations[i] = 0
            elif arg < -100:
                occupations[i] = 1
            else:
                occupations[i] = 1 / (1 + np.exp(arg))
        return float(np.sum(occupations) - numElectrons)

    muMin = np.min(eigenvalues) - 1
    muMax = np.max(eigenvalues) + 1
    return bisect(occupationSum, muMin, muMax)


@dataclass
class Discretization:
    """Holds discretization-related inputs for SCF.

    - fine_grid: fine discretization positions
    - coarse_points: interpolation points for coarse density representation
    - shape_function: kernel mapping pairwise diffs to interpolation weights
    - system_params: system parameters (e.g., atomic numbers/positions)
    """

    fine_grid: np.ndarray
    coarse_points: np.ndarray
    shape_function: Callable[[np.ndarray], np.ndarray]
    system_params: Dict[str, Any]


@dataclass
class SCFControls:
    """Algorithmic controls for the SCF iteration."""

    inverse_temperature: float
    mixing_parameter: float
    block_size: int
    max_iterations: int
    convergence_threshold: float


@dataclass
class EstimationControls:
    """Controls for the quantum/classical density estimation subroutine."""

    confidence_level: float
    estimation_error_tolerance: float
    num_quantum_samples: int


@dataclass
class SCFConfig:
    """Complete configuration for running SCF."""

    initial_coarse_density: np.ndarray
    discretization: Discretization
    scf: SCFControls
    estimation: EstimationControls


@dataclass
class SCFResult:
    """Results of an SCF run."""

    converged_coarse_density: np.ndarray
    residuals: np.ndarray
    total_complexity: int


def _validate_scf_config(config: SCFConfig) -> None:
    """Basic validation of SCF configuration values."""
    if not (0 < config.scf.mixing_parameter <= 1):
        raise ValueError("mixing_parameter must be in (0, 1]")
    if config.scf.block_size <= 0:
        raise ValueError("block_size must be positive")
    if config.scf.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.scf.convergence_threshold <= 0:
        raise ValueError("convergence_threshold must be positive")
    if config.estimation.confidence_level <= 0:
        raise ValueError("confidence_level must be positive")
    if config.estimation.estimation_error_tolerance <= 0:
        raise ValueError("estimation_error_tolerance must be positive")
    if config.estimation.num_quantum_samples <= 0:
        raise ValueError("num_quantum_samples must be positive")


def run_scf_configured(config: SCFConfig) -> SCFResult:
    """Performs hybrid SCF iterations using a structured configuration.

    Subcalculations mapping:
    - Hamiltonian construction: uses `config.discretization` and current density
    - Chemical potential search: uses `config.scf.inverse_temperature`
    - Block selection and mixing: uses `config.scf.block_size` and `config.scf.mixing_parameter`
    - Density estimation: uses `config.estimation` controls
    """
    _validate_scf_config(config)

    currentCoarseDensity = config.initial_coarse_density.copy()
    fineGrid = config.discretization.fine_grid
    coarsePoints = config.discretization.coarse_points
    shapeFunction = config.discretization.shape_function
    params = config.discretization.system_params

    inverseTemperature = config.scf.inverse_temperature
    mixingParameter = config.scf.mixing_parameter
    blockSize = config.scf.block_size
    maxIterations = config.scf.max_iterations
    convergenceThreshold = config.scf.convergence_threshold

    confidenceLevel = config.estimation.confidence_level
    estimationErrorTolerance = config.estimation.estimation_error_tolerance
    numQuantumSamples = config.estimation.num_quantum_samples

    numInterpolationPoints = len(coarsePoints)
    numElectrons = sum(params["atomic_numbers"])
    residuals = []
    total_complexity = 0

    for iteration in range(maxIterations):
        hamiltonian, normalizationFactor, _, _ = build_hamiltonian(
            currentCoarseDensity, fineGrid, coarsePoints, shapeFunction, params
        )
        eigenvalues = np.linalg.eigh(hamiltonian.toarray())[0]
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

    return SCFResult(
        converged_coarse_density=currentCoarseDensity,
        residuals=np.array(residuals),
        total_complexity=total_complexity,
    )
