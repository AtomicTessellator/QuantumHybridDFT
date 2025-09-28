from typing import Any, Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import lstsq

# System Discretization and Setup
# This stage establishes the numerical grid for representing the quantum system in Density Functional Theory (DFT).
# We create a fine grid for accurate Hamiltonian discretization and a coarse set of interpolation points centered at atoms
# to reduce computational complexity from
#
#  O(Ng) to O(NI),
#
#  Ng is the number of grid points (~ number of electrons)
#  NI is the number of interpolation points (~ number of atoms).
#
# This enables linear scaling in quantum algorithms.
# The initial electron density is approximated as a superposition of Gaussians around atomic positions,
# normalized to integrate to Ng.
#
# Shape functions (Gaussians) are used for interpolation between coarse and fine grids,
# ensuring accurate density reconstruction.


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
    """Set up the discretization for the quantum hybrid DFT calculation.

    This function establishes the fine grid, coarse points, initial coarse density,
    and shape function for the DFT calculation. It reduces the problem size while
    preserving accuracy for potentials and densities.

    Parameters
    ----------
    params : Dict[str, Any]
        Configuration parameters including:
        - dimension : int (currently only 1D supported)
        - domain : [0, L]
        - gridExponent : int (m, where numGridPoints = 2^m for qubit mapping)
        - atomicPositions : array of atomic positions
        - atomicCharges : array of atomic charges (atomic numbers, approximating electron count)
        - sigma : float (Gaussian width parameter)

    Returns
    -------
    fineGrid : npt.NDArray[np.float64]
        Uniform grid in [0, L] with numGridPoints points
    coarsePoints : npt.NDArray[np.float64]
        Points at atomic positions (NI = Na)
    coarseDensity : npt.NDArray[np.float64]
        Coarse density obtained by least-squares fitting fineDensity = interpolationMatrix @ coarseDensity,
        minimizing reconstruction error
    shapeFunction : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        Function for interpolation between grids

    Notes
    -----
    The initial fine density (fineDensity) is computed as the sum of Gaussians centered
    at atoms, scaled by atomicCharges (atomic number, approximating electron count).
    """

    dim = params.get("dimension")
    if dim != 1:
        raise NotImplementedError("Only 1D supported currently.")

    domain = params.get("computational_domain")  # [0, L]

    m = params.get("grid_exponent")
    numGridPoints = 2**m
    fineGrid = np.linspace(domain[0], domain[1], numGridPoints)[
        :, None
    ]  # (numGridPoints, dim)

    atomic_positions = params.get("atomic_positions")
    atomicCharges = params.get("atomic_numbers")
    coarsePoints = np.array(atomic_positions)[:, None]  # (NI, dim)
    sigma = params.get("gaussian_width")

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
