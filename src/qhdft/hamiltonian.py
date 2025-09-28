import numpy as np
from scipy.sparse import csr_matrix, diags

# Stage 2: Hamiltonian Construction
# Builds the sparse Hamiltonian matrix H for the current electron density n in DFT.
# H = T + V_eff, where T is the kinetic energy (discretized Laplacian), and V_eff = V_ext + V_H + V_xc.
# - V_ext is the external potential from nuclei (softened Coulomb for numerical stability).
# - V_H is the Hartree potential (classical electrostatic repulsion of electrons).
# - V_xc is the exchange-correlation potential (approximating quantum effects; here a simple LDA placeholder).
# The Hamiltonian is sparse (banded due to finite differences), enabling efficient oracle access for quantum algorithms.
# Normalization factor scales eigenvalues to [-1,1] for polynomial approximations in QSVT.
# Oracles provide locations and values of nonzeros, used in quantum block encoding.


def build_hamiltonian(coarseDensity, fineGrid, coarsePoints, shapeFunction, params):
    # Constructs sparse H (Ng x Ng) from coarse density (NI,).
    # First interpolates coarseDensity to fine grid fineDensity using shape functions.
    # Computes potentials on fine grid, adds to kinetic matrix.
    # Ensures H is symmetric (Hermitian for real matrices).
    # Returns H, norm factor, and oracles for sparse access.
    numGridPoints = len(fineGrid)
    assert fineGrid.shape == (numGridPoints, 1)
    gridSpacing = fineGrid[1, 0] - fineGrid[0, 0]  # uniform grid spacing
    # Kinetic: -1/2 Laplacian, second-order finite difference approximation.
    # Diagonal: 1/gridSpacing^2, off-diagonals: -0.5/gridSpacing^2, for d^2/dx^2.
    kinetic_diag = np.full(numGridPoints, 1.0 / gridSpacing**2)
    kinetic_off = np.full(numGridPoints - 1, -0.5 / gridSpacing**2)
    kinetic = diags([kinetic_off, kinetic_diag, kinetic_off], [-1, 0, 1])
    # Interpolate coarseDensity to fine grid.
    diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]
    interpolationMatrix = shapeFunction(diffs)  # (numGridPoints, NI)
    fineDensity = interpolationMatrix @ coarseDensity  # (numGridPoints,)
    # V_H: Hartree potential, discretized Poisson: sum_k |x_i - x_k| n_k gridSpacing (for 1D).
    # Approximates ∫ |r - r'| n(r') dr'.
    V_H = np.zeros(numGridPoints)
    for i in range(numGridPoints):
        V_H[i] = (
            np.sum(np.abs(fineGrid[i, 0] - fineGrid[:, 0]) * fineDensity) * gridSpacing
        )
    # V_ext: attractive nuclear potential, softened 1/sqrt(r^2 + ε^2) to avoid singularities.
    epsilon = params.get("epsilon", 0.1)
    atomic_positions = params["atomic_positions"]
    atomicCharges = params["Z"]
    V_ext = np.zeros(numGridPoints)
    for pos, charge in zip(atomic_positions, atomicCharges):
        dist = np.sqrt((fineGrid[:, 0] - pos) ** 2 + epsilon**2)
        V_ext += -charge / dist
    # V_xc: exchange-correlation, simple local density approximation -n^{1/3} (placeholder).
    # In real DFT, this would use a functional like LDA or GGA for electron correlation.
    # Use maximum to avoid issues with negative densities.
    V_xc = -(np.maximum(fineDensity, 0) ** (1 / 3))
    # Effective potential V_eff = V_ext + V_H + V_xc.
    V_eff = V_ext + V_H + V_xc
    # H = kinetic + diag(V_eff)
    H = kinetic + diags([V_eff], [0])
    H = csr_matrix(H)
    # Verify symmetry (essential for real eigenvalues in DFT).
    assert np.max(np.abs(H - H.T)) < 1e-8
    # Normalization: upper bound on ||H|| to scale to [-1,1] for QSVT.
    norm = np.max(np.abs(V_eff)) + 2.0 / gridSpacing**2

    # Location oracle: for row, returns column of l-th nonzero.
    def location_oracle(row, index):
        start = H.indptr[row]
        end = H.indptr[row + 1]
        if index >= end - start:
            raise IndexError("Index out of range for row")
        return H.indices[start + index]

    # Value oracle: for row, returns value of l-th nonzero.
    def value_oracle(row, index):
        start = H.indptr[row]
        end = H.indptr[row + 1]
        if index >= end - start:
            raise IndexError("Index out of range for row")
        return H.data[start + index]

    return H, norm, location_oracle, value_oracle
