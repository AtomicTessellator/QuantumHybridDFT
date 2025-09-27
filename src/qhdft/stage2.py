import numpy as np
from scipy.sparse import csr_matrix, diags


def build_hamiltonian(n, D, D_delta, N, params):
    Ng = len(D)
    assert D.shape == (Ng, 1)
    h = D[1, 0] - D[0, 0]  # uniform grid
    # Kinetic: -1/2 Laplacian, finite difference
    kinetic_diag = np.full(Ng, 1.0 / h**2)
    kinetic_off = np.full(Ng - 1, -0.5 / h**2)
    kinetic = diags([kinetic_off, kinetic_diag, kinetic_off], [-1, 0, 1])
    # Interpolate n to fine grid
    diffs = D[:, None, :] - D_delta[None, :, :]
    N_matrix = N(diffs)  # (Ng, NI)
    n_fine = N_matrix @ n  # (Ng,)
    # V_H: 1D Hartree potential ∫ |x - x'| n(x') dx' ≈ sum |x_i - x_k| n_k h
    V_H = np.zeros(Ng)
    for i in range(Ng):
        V_H[i] = np.sum(np.abs(D[i, 0] - D[:, 0]) * n_fine) * h
    # V_ext: softened Coulomb
    epsilon = params.get("epsilon", 0.1)
    atomic_positions = params["atomic_positions"]
    Z = params["Z"]
    V_ext = np.zeros(Ng)
    for pos, z in zip(atomic_positions, Z):
        dist = np.sqrt((D[:, 0] - pos) ** 2 + epsilon**2)
        V_ext += -z / dist
    # V_xc: simple placeholder LDA-like, - (n)^{1/3}
    # TODO: Implement proper 1D LDA
    V_xc = -(np.maximum(n_fine, 0) ** (1 / 3))
    # V_eff
    V_eff = V_ext + V_H + V_xc
    # H = kinetic + diag(V_eff)
    H = kinetic + diags([V_eff], [0])
    H = csr_matrix(H)
    # Ensure Hermitian (real symmetric)
    assert np.max(np.abs(H - H.T)) < 1e-8
    # Normalization factor (rough estimate for [-1,1])
    norm = np.max(np.abs(V_eff)) + 2.0 / h**2

    # Oracles
    def location_oracle(row, index):
        start = H.indptr[row]
        end = H.indptr[row + 1]
        if index >= end - start:
            raise IndexError("Index out of range for row")
        return H.indices[start + index]

    def value_oracle(row, index):
        start = H.indptr[row]
        end = H.indptr[row + 1]
        if index >= end - start:
            raise IndexError("Index out of range for row")
        return H.data[start + index]

    return H, norm, location_oracle, value_oracle
