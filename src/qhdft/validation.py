import numpy as np
from scipy.stats import linregress

from qhdft.discretization import setup_discretization
from qhdft.hamiltonian import build_hamiltonian
from qhdft.scf import find_mu, run_scf

# Stage 6: Validation and Numerical Results
# Validates the converged density from SCF by computing energy, plotting density, analyzing scaling with system size Na,
# and breaking down errors (polynomial approx, statistical, iteration). Compares to classical baselines.


def compute_energy(
    convergedCoarseDensity,
    fineGrid,
    coarsePoints,
    shapeFunction,
    params,
    inverseTemperature,
):
    # Computes ground state energy E = sum occ_i * e_i, where occ_i = f(e_i), from H built on converged coarse density.
    # Uses exact diagonalization here for validation; in quantum setting, could use similar estimation.
    H, _, _, _ = build_hamiltonian(
        convergedCoarseDensity, fineGrid, coarsePoints, shapeFunction, params
    )
    eigenvalues, _ = np.linalg.eigh(H.toarray())
    numElectrons = sum(params["Z"])
    chemicalPotential = find_mu(eigenvalues, inverseTemperature, numElectrons)
    occupations = 1 / (
        1 + np.exp(inverseTemperature * (eigenvalues - chemicalPotential))
    )
    totalEnergy = np.sum(occupations * eigenvalues)
    return totalEnergy


def generate_density_data(coarseDensity, fineGrid, coarsePoints, shapeFunction):
    # Generates data for plotting: fine grid positions r and interpolated density n(r) from coarse density.
    # Uses shape functions to reconstruct fineDensity = interpolationMatrix @ coarseDensity.
    diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]
    interpolationMatrix = shapeFunction(diffs)
    fineDensity = interpolationMatrix @ coarseDensity
    return fineGrid[:, 0], fineDensity


def run_scaling_test(
    baseParams,
    inverseTemperature,
    mixingParameter,
    maxIterations,
    convergenceThreshold,
    confidenceLevel,
    estimationErrorTolerance,
    numQuantumSamples,
    atomCountRange,
):
    # Tests scaling: runs SCF for varying atom count (system size), measures total queries (proxy for time).
    # Fits linear model queries ~ slope * atomCount + intercept, checks R^2 > 0.95 for linear scaling.
    # Uses simple H chain (Z=1) with positions spread in domain.
    queryComplexities = []
    timings = []  # Placeholder, since no actual time measurement
    for atomCount in atomCountRange:
        params = baseParams.copy()
        params["atomic_positions"] = np.linspace(
            0, baseParams["computational_domain"][1], atomCount + 1
        )[1:]
        params["Z"] = [1] * atomCount  # Simple H chain
        fineGrid, coarsePoints, initialCoarseDensity, shapeFunction = (
            setup_discretization(params)
        )
        _, _, queryComplexity = run_scf(  # Assuming run_scf from stage5
            initialCoarseDensity,
            fineGrid,
            coarsePoints,
            shapeFunction,
            params,
            inverseTemperature,
            mixingParameter,
            atomCount,  # Full update
            maxIterations,
            1e-10,  # Small to run full maxIterations
            confidenceLevel,
            estimationErrorTolerance,
            numQuantumSamples,
        )
        queryComplexities.append(queryComplexity)
        timings.append(queryComplexity)  # Proxy
    # Linear fit
    slope, intercept, rValue, _, _ = linregress(atomCountRange, queryComplexities)
    return (
        dict(atomCounts=atomCountRange, queries=queryComplexities, times=timings),
        rValue**2,
    )


def compute_error_breakdown(
    estimatedDensity,
    exactDensity,
    estimatedEnergy,
    exactEnergy,
    polynomialMaxError,
    estimationErrorTolerance,
    numIterations,
):
    # Quantifies errors: polynomial approximation error, statistical fluctuation bound,
    # number of iterations, L2 density error, absolute energy error vs. classical baseline.
    errors = {
        "poly_approx": polynomialMaxError,
        "stat_fluct": estimationErrorTolerance,
        "iter_complex": numIterations,
        "density_l2": np.linalg.norm(estimatedDensity - exactDensity),
        "energy_err": abs(estimatedEnergy - exactEnergy),
    }
    return errors
