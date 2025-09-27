import numpy as np
from scipy.linalg import lstsq

# Stage 1: System Discretization and Setup
# This stage establishes the numerical grid for representing the quantum system in Density Functional Theory (DFT).
# We create a fine grid for accurate Hamiltonian discretization and a coarse set of interpolation points centered at atoms
# to reduce computational complexity from O(Ng) to O(NI), where Ng is the number of grid points (~ number of electrons Ne)
# and NI is the number of interpolation points (~ number of atoms Na). This enables linear scaling in quantum algorithms.
# The initial electron density is approximated as a superposition of Gaussians around atomic positions, normalized to integrate to Ne.
# Shape functions (Gaussians) are used for interpolation between coarse and fine grids, ensuring accurate density reconstruction.


def gaussian(d, sigma):
    # Gaussian function for initial density and shape functions.
    # Used to model localized electron density around atoms and for interpolation.
    # Sigma controls the spread; chosen to balance localization and smoothness.
    return np.exp(-(d**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def setup_discretization(params):
    # Sets up the fine grid D, coarse points D_delta, initial coarse density n0, and shape function N.
    # Parameters include dimension (currently 1D), domain [0, L], grid exponent m (Ng=2^m for qubit mapping),
    # atomic positions and charges Z, and Gaussian width sigma.
    # Fine grid D is uniform in [0, L] with Ng points.
    # Coarse points D_delta are at atomic positions (NI = Na).
    # Initial fine density n_fine is sum of Gaussians centered at atoms, scaled by Z (atomic number, approximating electron count).
    # Coarse n0 is obtained by least-squares fitting n_fine = N_matrix @ n0, minimizing reconstruction error.
    # This setup reduces the DFT problem size while preserving accuracy for potentials and densities.
    dim = params.get("dim", 1)
    if dim != 1:
        raise NotImplementedError("Only 1D supported currently.")
    domain = params["domain"]  # [0, L]
    m = params["m"]
    Ng = 2**m
    D = np.linspace(domain[0], domain[1], Ng)[:, None]  # (Ng, dim)
    atomic_positions = params["atomic_positions"]
    Z = params["Z"]
    D_delta = np.array(atomic_positions)[:, None]  # (NI, dim)
    sigma = params.get("sigma", 0.5)

    def N(diff):
        # Shape function: Gaussian of distance, for interpolating from coarse to fine grid.
        # N(diff) gives the weight for each coarse point's contribution to a fine point.
        d = np.abs(diff[..., 0])
        return gaussian(d, sigma)

    # Compute initial n_fine on fine grid as sum of atomic Gaussians.
    n_fine = np.zeros(Ng)
    for pos, z in zip(atomic_positions, Z):
        d = D[:, 0] - pos
        n_fine += z * gaussian(d, sigma)
    # Build N_matrix (Ng, NI) for interpolation.
    diffs = D[:, None, :] - D_delta[None, :, :]  # (Ng, NI, dim)
    N_matrix = N(diffs)  # (Ng, NI)
    # Solve for n0 such that N_matrix @ n0 ≈ n_fine (least squares).
    n0, _, _, _ = lstsq(N_matrix, n_fine)
    return D, D_delta, n0, N
