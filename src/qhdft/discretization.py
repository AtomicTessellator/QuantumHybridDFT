from typing import Any, Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import lstsq

# Stage 1: System Discretization and Setup
# This stage establishes the numerical grid for representing the quantum system in Density Functional Theory (DFT).
# We create a fine grid for accurate Hamiltonian discretization and a coarse set of interpolation points centered at atoms
# to reduce computational complexity from O(Ng) to O(NI), where Ng is the number of grid points (~ number of electrons Ne)
# and NI is the number of interpolation points (~ number of atoms Na). This enables linear scaling in quantum algorithms.
# The initial electron density is approximated as a superposition of Gaussians around atomic positions, normalized to integrate to Ne.
# Shape functions (Gaussians) are used for interpolation between coarse and fine grids, ensuring accurate density reconstruction.


def gaussian(
    distance: npt.NDArray[np.float64], sigma: float
) -> npt.NDArray[np.float64]:
    # Gaussian function for initial density and shape functions.
    # Used to model localized electron density around atoms and for interpolation.
    # Sigma controls the spread; chosen to balance localization and smoothness.
    return np.exp(-(distance**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def setup_discretization(
    params: Dict[str, Any],
) -> Tuple[
    npt.NDArray[np.float64],  # fineGrid
    npt.NDArray[np.float64],  # coarsePoints
    npt.NDArray[np.float64],  # coarseDensity
    Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],  # shapeFunction
]:
    # Sets up the fine grid (fineGrid), coarse points (coarsePoints), initial coarse density (coarseDensity), and shape function (shapeFunction).
    # Parameters include dimension (currently 1D), domain [0, L], grid exponent m (numGridPoints=2^m for qubit mapping),
    # atomic positions and charges (atomicCharges), and Gaussian width sigma.
    # Fine grid (fineGrid) is uniform in [0, L] with numGridPoints points.
    # Coarse points (coarsePoints) are at atomic positions (NI = Na).
    # Initial fine density (fineDensity) is sum of Gaussians centered at atoms, scaled by atomicCharges (atomic number, approximating electron count).
    # Coarse density (coarseDensity) is obtained by least-squares fitting fineDensity = interpolationMatrix @ coarseDensity, minimizing reconstruction error.
    # This setup reduces the DFT problem size while preserving accuracy for potentials and densities.
    dim = params.get("dim", 1)
    if dim != 1:
        raise NotImplementedError("Only 1D supported currently.")
    domain = params["domain"]  # [0, L]
    m = params["m"]
    numGridPoints = 2**m
    fineGrid = np.linspace(domain[0], domain[1], numGridPoints)[
        :, None
    ]  # (numGridPoints, dim)
    atomic_positions = params["atomic_positions"]
    atomicCharges = params["Z"]
    coarsePoints = np.array(atomic_positions)[:, None]  # (NI, dim)
    sigma = params.get("sigma", 0.5)

    def shapeFunction(diff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Shape function: Gaussian of distance, for interpolating from coarse to fine grid.
        # shapeFunction(diff) gives the weight for each coarse point's contribution to a fine point.
        distance = np.abs(diff[..., 0])
        return gaussian(distance, sigma)

    # Compute initial fineDensity on fine grid as sum of atomic Gaussians.
    fineDensity = np.zeros(numGridPoints)
    for pos, charge in zip(atomic_positions, atomicCharges):
        distance = fineGrid[:, 0] - pos
        fineDensity += charge * gaussian(distance, sigma)
    # Build interpolationMatrix (numGridPoints, NI) for interpolation.
    diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]  # (numGridPoints, NI, dim)
    interpolationMatrix = shapeFunction(diffs)  # (numGridPoints, NI)
    # Solve for coarseDensity such that interpolationMatrix @ coarseDensity ≈ fineDensity (least squares).
    coarseDensity, _, _, _ = lstsq(interpolationMatrix, fineDensity)
    return fineGrid, coarsePoints, coarseDensity, shapeFunction
