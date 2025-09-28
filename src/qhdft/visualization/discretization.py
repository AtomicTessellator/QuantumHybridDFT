from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def visualize_discretization(
    fineGrid: npt.NDArray[np.float64],
    coarsePoints: npt.NDArray[np.float64],
    coarseDensity: npt.NDArray[np.float64],
    shapeFunction: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    system_params: Dict[str, Any],
    output_folder: str,
) -> None:
    """
    Visualize the discretization data structures from stage 1.

    Args:
        fineGrid: Fine grid points for Hamiltonian discretization
        coarsePoints: Coarse interpolation points (atomic positions)
        coarseDensity: Density values at coarse points
        shapeFunction: Gaussian shape function for interpolation
        system_params: System parameters dictionary
        output_folder: Folder to save visualization PNG files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Extract parameters
    atomic_positions = system_params.get("atomic_positions", [])
    atomic_numbers = system_params.get("atomic_numbers", [])
    sigma = system_params.get("gaussian_width", 0.5)
    domain = system_params.get("computational_domain", [0, 10])

    # 1. Visualize the grids and atomic positions
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot fine grid points
    ax.scatter(
        fineGrid[:, 0],
        np.zeros_like(fineGrid[:, 0]),
        alpha=0.3,
        s=10,
        color="blue",
        label=f"Fine grid ({len(fineGrid)} points)",
    )

    # Plot coarse points (atomic positions)
    ax.scatter(
        coarsePoints[:, 0],
        np.zeros_like(coarsePoints[:, 0]),
        s=200,
        color="red",
        marker="*",
        label=f"Coarse points/Atoms ({len(coarsePoints)} atoms)",
    )

    # Add atomic labels
    for i, (pos, z) in enumerate(zip(atomic_positions, atomic_numbers)):
        element = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O"}.get(z, f"Z={z}")
        ax.annotate(
            element,
            (pos, 0.01),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Position (Bohr)", fontsize=12)
    ax.set_title("Grid Points and Atomic Positions", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02, 0.05)
    ax.set_xlim(domain[0] - 0.5, domain[1] + 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "grids_and_atoms.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Visualize the initial density (fine grid)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Compute initial fine density (sum of atomic Gaussians)
    from qhdft.discretization import gaussian

    fineDensity = np.zeros(len(fineGrid))
    for pos, charge in zip(atomic_positions, atomic_numbers):
        distance = fineGrid[:, 0] - pos
        fineDensity += charge * gaussian(distance, sigma)

    ax.plot(
        fineGrid[:, 0],
        fineDensity,
        "b-",
        linewidth=2,
        label="Initial density (fine grid)",
    )

    # Mark atomic positions
    for pos, z in zip(atomic_positions, atomic_numbers):
        ax.axvline(x=pos, color="red", linestyle="--", alpha=0.5)
        element = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O"}.get(z, f"Z={z}")
        ax.text(pos, ax.get_ylim()[1] * 0.9, element, ha="center", fontsize=12)

    ax.set_xlabel("Position (Bohr)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Initial Electron Density on Fine Grid", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "initial_density.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Visualize coarse density and interpolation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Coarse density values at atomic positions
    markerline, stemlines, baseline = ax1.stem(
        coarsePoints[:, 0],
        coarseDensity,
        basefmt=" ",
        label="Coarse density values",
        markerfmt="ro",
    )
    plt.setp(stemlines, linewidth=2, color="r")
    plt.setp(markerline, markersize=10)

    for i, (pos, val) in enumerate(zip(coarsePoints[:, 0], coarseDensity)):
        ax1.text(pos, val + 0.1, f"{val:.2f}", ha="center", fontsize=10)

    ax1.set_xlabel("Position (Bohr)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(
        "Coarse Density Values at Atomic Positions", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom panel: Interpolated density from coarse to fine grid
    # Build interpolation matrix
    diffs = fineGrid[:, None, :] - coarsePoints[None, :, :]
    interpolationMatrix = shapeFunction(diffs)
    interpolatedDensity = interpolationMatrix @ coarseDensity

    ax2.plot(
        fineGrid[:, 0],
        fineDensity,
        "b-",
        linewidth=2,
        alpha=0.7,
        label="Original fine density",
    )
    ax2.plot(
        fineGrid[:, 0],
        interpolatedDensity,
        "g--",
        linewidth=2,
        label="Interpolated from coarse",
    )
    ax2.scatter(
        coarsePoints[:, 0],
        coarseDensity,
        color="red",
        s=100,
        zorder=5,
        label="Coarse points",
    )

    ax2.set_xlabel("Position (Bohr)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(
        "Density Interpolation: Coarse to Fine Grid", fontsize=14, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path / "coarse_density_interpolation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Visualize shape functions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Individual shape functions centered at each atom
    x_range = np.linspace(domain[0], domain[1], 500)
    for i, atom_pos in enumerate(coarsePoints[:, 0]):
        diff = x_range - atom_pos
        shape_vals = shapeFunction(diff[:, None])
        ax1.plot(
            x_range, shape_vals, linewidth=2, label=f"Atom {i+1} at {atom_pos:.1f}"
        )
        ax1.axvline(x=atom_pos, color="gray", linestyle=":", alpha=0.5)

    ax1.set_xlabel("Position (Bohr)", fontsize=12)
    ax1.set_ylabel("Shape function value", fontsize=12)
    ax1.set_title(
        f"Gaussian Shape Functions (σ={sigma})", fontsize=14, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Interpolation matrix visualization
    im = ax2.imshow(
        interpolationMatrix.T,
        aspect="auto",
        cmap="viridis",
        extent=[domain[0], domain[1], len(coarsePoints) - 0.5, -0.5],
    )
    ax2.set_xlabel("Fine grid position (Bohr)", fontsize=12)
    ax2.set_ylabel("Coarse point index", fontsize=12)
    ax2.set_title("Interpolation Matrix", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Weight", fontsize=12)

    # Label coarse points
    for i, pos in enumerate(coarsePoints[:, 0]):
        element = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O"}.get(
            atomic_numbers[i], f"Z={atomic_numbers[i]}"
        )
        ax2.text(-0.5, i, f"{element}", ha="right", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / "shape_functions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Summary statistics
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis("off")

    # Calculate statistics
    reconstruction_error = np.linalg.norm(fineDensity - interpolatedDensity)
    total_electrons_fine = np.trapz(fineDensity, fineGrid[:, 0])
    total_electrons_interp = np.trapz(interpolatedDensity, fineGrid[:, 0])

    stats_text = f"""
Discretization Summary
======================

Grid Information:
- Fine grid points: {len(fineGrid)}
- Grid spacing: {(domain[1] - domain[0]) / (len(fineGrid) - 1):.4f} Bohr
- Domain: [{domain[0]}, {domain[1]}] Bohr

Atomic System:
- Number of atoms: {len(atomic_positions)}
- Atomic positions: {atomic_positions}
- Atomic numbers: {atomic_numbers}

Interpolation:
- Coarse points: {len(coarsePoints)}
- Gaussian width (σ): {sigma}
- Reconstruction error: {reconstruction_error:.6f}

Density Integration:
- Total electrons (fine): {total_electrons_fine:.4f}
- Total electrons (interpolated): {total_electrons_interp:.4f}
- Difference: {abs(total_electrons_fine - total_electrons_interp):.6f}
"""

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        output_path / "discretization_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Visualizations saved to: {output_path}")
    print("Generated files:")
    print("  - discretization/grids_and_atoms.png")
    print("  - discretization/initial_density.png")
    print("  - discretization/coarse_density_interpolation.png")
    print("  - discretization/shape_functions.png")
    print("  - discretization/discretization_summary.png")
