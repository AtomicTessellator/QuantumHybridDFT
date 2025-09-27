import numpy as np
from scipy.linalg import lstsq


def gaussian(d, sigma):
    return np.exp(-(d**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def setup_discretization(params):
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
        d = np.abs(diff[..., 0])
        return gaussian(d, sigma)

    # Compute initial n_fine
    n_fine = np.zeros(Ng)
    for pos, z in zip(atomic_positions, Z):
        d = D[:, 0] - pos
        n_fine += z * gaussian(d, sigma)
    # Build N_matrix (Ng, NI)
    diffs = D[:, None, :] - D_delta[None, :, :]  # (Ng, NI, dim)
    N_matrix = N(diffs)  # (Ng, NI)
    # Least squares
    n0, _, _, _ = lstsq(N_matrix, n_fine)
    return D, D_delta, n0, N
