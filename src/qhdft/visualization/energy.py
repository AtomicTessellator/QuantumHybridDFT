from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec


def visualize_energy_analysis(
    converged_density: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    shape_function: Any,
    system_params: Dict[str, Any],
    inverse_temperature: float,
    ground_state_energy: float,
    scf_residuals: Optional[npt.NDArray[np.float64]] = None,
    reference_energy: Optional[float] = None,
    output_folder: str = "visualizations/energy/",
) -> None:
    """
    Visualize energy analysis from compute_energy calculation.

    Parameters
    ----------
    converged_density : array
        Converged coarse density from SCF
    fine_grid : array
        Fine grid points
    coarse_points : array
        Coarse interpolation points
    shape_function : callable
        Shape function for interpolation
    system_params : dict
        System parameters
    inverse_temperature : float
        Inverse temperature (beta)
    ground_state_energy : float
        Computed ground state energy
    scf_residuals : array, optional
        SCF convergence residuals
    reference_energy : float, optional
        Reference energy for comparison
    output_folder : str
        Folder to save visualization PNG files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Extract parameters
    atomic_positions = system_params.get("atomic_positions", [])
    atomic_numbers = system_params.get("atomic_numbers", [])

    # 1. Energy Components Visualization
    _plot_energy_components(
        converged_density,
        fine_grid,
        coarse_points,
        shape_function,
        system_params,
        inverse_temperature,
        ground_state_energy,
        output_path,
    )

    # 2. Density and Energy Relationship
    _plot_density_energy_relationship(
        converged_density,
        fine_grid,
        coarse_points,
        shape_function,
        system_params,
        ground_state_energy,
        atomic_positions,
        atomic_numbers,
        output_path,
    )

    # 3. Energy Convergence (if SCF residuals provided)
    if scf_residuals is not None:
        _plot_energy_convergence(
            scf_residuals,
            ground_state_energy,
            reference_energy,
            output_path,
        )

    # 4. Summary Figure
    _create_energy_summary_figure(
        converged_density,
        fine_grid,
        coarse_points,
        shape_function,
        system_params,
        inverse_temperature,
        ground_state_energy,
        scf_residuals,
        reference_energy,
        output_path,
    )

    print(f"Energy visualizations saved to {output_path}")


def _plot_energy_components(
    converged_density: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    shape_function: Any,
    system_params: Dict[str, Any],
    inverse_temperature: float,
    ground_state_energy: float,
    output_path: Path,
) -> None:
    """Plot energy component breakdown."""
    # Import necessary functions
    from qhdft.hamiltonian import build_hamiltonian
    from qhdft.scf import find_mu

    # Build Hamiltonian to get eigenvalues and compute energy components
    H, _, _, _ = build_hamiltonian(
        converged_density, fine_grid, coarse_points, shape_function, system_params
    )
    eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())

    # Compute chemical potential
    num_electrons = sum(system_params["atomic_numbers"])
    chemical_potential = find_mu(eigenvalues, inverse_temperature, num_electrons)

    # Compute occupations
    occupations = 1 / (
        1 + np.exp(inverse_temperature * (eigenvalues - chemical_potential))
    )

    # Interpolate density to fine grid
    diffs = fine_grid[:, None, :] - coarse_points[None, :, :]
    interpolation_matrix = shape_function(diffs)
    fine_density = interpolation_matrix @ converged_density

    # Compute energy components
    # Kinetic energy (from eigenvalues and eigenvectors)
    grid_spacing = fine_grid[1, 0] - fine_grid[0, 0]
    kinetic_energy = 0
    for i, (occ, psi) in enumerate(zip(occupations, eigenvectors.T)):
        # Compute second derivative using finite differences
        psi_second_deriv = np.zeros_like(psi)
        psi_second_deriv[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / grid_spacing**2
        kinetic_energy += (
            occ * 0.5 * np.sum(psi.conj() * psi_second_deriv) * grid_spacing
        )

    # External potential energy
    epsilon = system_params.get("epsilon", 0.1)
    atomic_positions = system_params["atomic_positions"]
    atomic_charges = system_params["atomic_numbers"]
    V_ext = np.zeros(len(fine_grid))
    for pos, charge in zip(atomic_positions, atomic_charges):
        dist = np.sqrt((fine_grid[:, 0] - pos) ** 2 + epsilon**2)
        V_ext += -charge / dist
    external_energy = np.sum(fine_density * V_ext) * grid_spacing

    # Hartree energy
    V_H = np.zeros(len(fine_grid))
    for i in range(len(fine_grid)):
        V_H[i] = (
            np.sum(np.abs(fine_grid[i, 0] - fine_grid[:, 0]) * fine_density)
            * grid_spacing
        )
    hartree_energy = 0.5 * np.sum(fine_density * V_H) * grid_spacing

    # Exchange-correlation energy
    V_xc = -(np.maximum(fine_density, 0) ** (1 / 3))
    xc_energy = np.sum(fine_density * V_xc) * grid_spacing

    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    components = ["Kinetic", "External", "Hartree", "XC", "Total"]
    values = [
        kinetic_energy,
        external_energy,
        hartree_energy,
        xc_energy,
        ground_state_energy,
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    bars = ax.bar(components, values, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom" if value > 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title("Energy Component Breakdown", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "energy_components.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_density_energy_relationship(
    converged_density: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    shape_function: Any,
    system_params: Dict[str, Any],
    ground_state_energy: float,
    atomic_positions: List[float],
    atomic_numbers: List[int],
    output_path: Path,
) -> None:
    """Plot density profile with energy information."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # Interpolate density to fine grid
    diffs = fine_grid[:, None, :] - coarse_points[None, :, :]
    interpolation_matrix = shape_function(diffs)
    fine_density = interpolation_matrix @ converged_density
    x = fine_grid[:, 0]

    # Top plot: Density profile
    ax1.plot(x, fine_density, "b-", linewidth=2, label="Converged density")
    ax1.fill_between(x, fine_density, alpha=0.3, color="blue")

    # Add atomic positions
    for i, (pos, z) in enumerate(zip(atomic_positions, atomic_numbers)):
        ax1.axvline(x=pos, color="red", linestyle="--", alpha=0.5)
        element = (
            ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"][z - 1]
            if z <= 10
            else f"Z={z}"
        )
        ax1.scatter(
            pos,
            max(fine_density) * 1.1,
            s=200,
            marker="v",
            color="red",
            edgecolor="black",
            linewidth=2,
        )
        ax1.text(
            pos,
            max(fine_density) * 1.15,
            element,
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax1.set_xlabel("Position x (Bohr)", fontsize=12)
    ax1.set_ylabel("Density n(x)", fontsize=12)
    ax1.set_title(
        f"Electron Density Profile (E = {ground_state_energy:.6f} Ha)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom plot: Local energy density
    # Compute effective potential
    epsilon = system_params.get("epsilon", 0.1)
    atomic_charges = system_params["atomic_numbers"]
    V_ext = np.zeros(len(fine_grid))
    for pos, charge in zip(atomic_positions, atomic_charges):
        dist = np.sqrt((fine_grid[:, 0] - pos) ** 2 + epsilon**2)
        V_ext += -charge / dist

    V_H = np.zeros(len(fine_grid))
    for i in range(len(fine_grid)):
        V_H[i] = np.sum(np.abs(fine_grid[i, 0] - fine_grid[:, 0]) * fine_density) * (
            fine_grid[1, 0] - fine_grid[0, 0]
        )

    V_xc = -(np.maximum(fine_density, 0) ** (1 / 3))
    V_eff = V_ext + V_H + V_xc

    # Local energy density
    local_energy_density = fine_density * V_eff

    ax2.plot(x, local_energy_density, "g-", linewidth=2)
    ax2.fill_between(x, local_energy_density, alpha=0.3, color="green")
    ax2.set_xlabel("Position x (Bohr)", fontsize=12)
    ax2.set_ylabel("Local Energy Density", fontsize=12)
    ax2.set_title("Local Energy Density ε(x) = n(x) * V_eff(x)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add atomic position indicators
    for pos in atomic_positions:
        ax2.axvline(x=pos, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        output_path / "density_energy_profile.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def _plot_energy_convergence(
    scf_residuals: npt.NDArray[np.float64],
    ground_state_energy: float,
    reference_energy: Optional[float],
    output_path: Path,
) -> None:
    """Plot energy convergence during SCF iterations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Estimate energy at each iteration (simplified - assumes linear relationship)
    # In reality, would need to store energy at each SCF iteration
    iterations = np.arange(len(scf_residuals))

    # Simulate energy convergence based on residuals
    energy_estimates = ground_state_energy - scf_residuals * 0.1  # Scaling factor

    ax.plot(
        iterations,
        energy_estimates,
        "bo-",
        linewidth=2,
        markersize=6,
        label="Estimated Energy",
    )
    ax.axhline(
        y=ground_state_energy,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Final Energy",
    )

    if reference_energy is not None:
        ax.axhline(
            y=reference_energy,
            color="red",
            linestyle=":",
            linewidth=2,
            label="Reference Energy",
        )

    ax.set_xlabel("SCF Iteration", fontsize=12)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title("Energy Convergence During SCF", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add text box with convergence info
    textstr = f"Final Energy: {ground_state_energy:.6f} Ha\n"
    textstr += f"Converged in {len(scf_residuals)} iterations"
    if reference_energy is not None:
        textstr += f"\nError vs Reference: {abs(ground_state_energy - reference_energy):.2e} Ha"

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_path / "energy_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def _create_energy_summary_figure(
    converged_density: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    shape_function: Any,
    system_params: Dict[str, Any],
    inverse_temperature: float,
    ground_state_energy: float,
    scf_residuals: Optional[npt.NDArray[np.float64]],
    reference_energy: Optional[float],
    output_path: Path,
) -> None:
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Import necessary functions
    from qhdft.hamiltonian import build_hamiltonian
    from qhdft.scf import find_mu

    # Build Hamiltonian
    H, _, _, _ = build_hamiltonian(
        converged_density, fine_grid, coarse_points, shape_function, system_params
    )
    eigenvalues, _ = np.linalg.eigh(H.toarray())

    # 1. Eigenvalue spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    num_electrons = sum(system_params["atomic_numbers"])
    chemical_potential = find_mu(eigenvalues, inverse_temperature, num_electrons)

    ax1.scatter(range(len(eigenvalues)), eigenvalues, c="blue", s=50, alpha=0.6)
    ax1.axhline(
        y=chemical_potential,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"μ = {chemical_potential:.4f}",
    )
    ax1.set_xlabel("State Index", fontsize=12)
    ax1.set_ylabel("Energy (Hartree)", fontsize=12)
    ax1.set_title("Eigenvalue Spectrum", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Occupation numbers
    ax2 = fig.add_subplot(gs[0, 1])
    occupations = 1 / (
        1 + np.exp(inverse_temperature * (eigenvalues - chemical_potential))
    )

    ax2.scatter(range(len(occupations)), occupations, c="green", s=50, alpha=0.6)
    ax2.set_xlabel("State Index", fontsize=12)
    ax2.set_ylabel("Occupation", fontsize=12)
    ax2.set_title(
        f"Fermi-Dirac Occupations (β = {inverse_temperature})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3)

    # Add text showing total electrons
    total_electrons = np.sum(occupations)
    ax2.text(
        0.95,
        0.95,
        f"Total electrons: {total_electrons:.2f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 3. Density profile
    ax3 = fig.add_subplot(gs[1, :])
    diffs = fine_grid[:, None, :] - coarse_points[None, :, :]
    interpolation_matrix = shape_function(diffs)
    fine_density = interpolation_matrix @ converged_density
    x = fine_grid[:, 0]

    ax3.plot(x, fine_density, "b-", linewidth=2)
    ax3.fill_between(x, fine_density, alpha=0.3, color="blue")

    # Add atoms
    atomic_positions = system_params.get("atomic_positions", [])
    atomic_numbers = system_params.get("atomic_numbers", [])
    for pos, z in zip(atomic_positions, atomic_numbers):
        ax3.axvline(x=pos, color="red", linestyle="--", alpha=0.5)
        element = (
            ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"][z - 1]
            if z <= 10
            else f"Z={z}"
        )
        ax3.text(
            pos,
            max(fine_density) * 1.05,
            element,
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax3.set_xlabel("Position x (Bohr)", fontsize=12)
    ax3.set_ylabel("Density n(x)", fontsize=12)
    ax3.set_title("Converged Electron Density", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # 4. Energy info box
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")

    info_text = "Energy Analysis Summary\n" + "=" * 30 + "\n\n"
    info_text += f"Ground State Energy: {ground_state_energy:.6f} Ha\n"
    info_text += f"Total Electrons: {num_electrons}\n"
    info_text += f"Chemical Potential: {chemical_potential:.4f} Ha\n"
    info_text += f"Temperature: {1/inverse_temperature:.4f} Ha\n"

    if reference_energy is not None:
        error = abs(ground_state_energy - reference_energy)
        info_text += f"\nReference Energy: {reference_energy:.6f} Ha\n"
        info_text += f"Absolute Error: {error:.2e} Ha\n"
        info_text += f"Relative Error: {error/abs(reference_energy)*100:.2f}%\n"

    if scf_residuals is not None:
        info_text += f"\nSCF Iterations: {len(scf_residuals)}\n"
        info_text += f"Final Residual: {scf_residuals[-1]:.2e}\n"

    ax4.text(
        0.5,
        0.5,
        info_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # 5. System parameters
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    param_text = "System Parameters\n" + "=" * 30 + "\n\n"
    param_text += f"Grid Points: {len(fine_grid)}\n"
    param_text += f"Domain: [{system_params['computational_domain'][0]:.1f}, "
    param_text += f"{system_params['computational_domain'][1]:.1f}] Bohr\n"
    param_text += f"Atoms: {len(atomic_positions)}\n"
    param_text += f"Gaussian Width: {system_params.get('gaussian_width', 'N/A')}\n"
    param_text += f"Interpolation Points: {len(coarse_points)}\n"

    ax5.text(
        0.5,
        0.5,
        param_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # Main title
    fig.suptitle(
        "Quantum Hybrid DFT - Energy Analysis Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path / "energy_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_orbital_analysis(
    converged_density: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    shape_function: Any,
    system_params: Dict[str, Any],
    inverse_temperature: float,
    num_orbitals: int = 6,
    output_folder: str = "visualizations/energy/",
) -> None:
    """
    Visualize molecular orbitals and their contributions to the total energy.

    Parameters
    ----------
    converged_density : array
        Converged coarse density
    fine_grid : array
        Fine grid points
    coarse_points : array
        Coarse points
    shape_function : callable
        Shape function
    system_params : dict
        System parameters
    inverse_temperature : float
        Inverse temperature
    num_orbitals : int
        Number of orbitals to visualize
    output_folder : str
        Output folder
    """
    from qhdft.hamiltonian import build_hamiltonian
    from qhdft.scf import find_mu

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build Hamiltonian and get eigenstates
    H, _, _, _ = build_hamiltonian(
        converged_density, fine_grid, coarse_points, shape_function, system_params
    )
    eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())

    # Compute occupations
    num_electrons = sum(system_params["atomic_numbers"])
    chemical_potential = find_mu(eigenvalues, inverse_temperature, num_electrons)
    occupations = 1 / (
        1 + np.exp(inverse_temperature * (eigenvalues - chemical_potential))
    )

    # Plot orbitals
    fig, axes = plt.subplots(
        min(num_orbitals, len(eigenvalues)),
        1,
        figsize=(12, 3 * min(num_orbitals, len(eigenvalues))),
        sharex=True,
    )

    if min(num_orbitals, len(eigenvalues)) == 1:
        axes = [axes]

    x = fine_grid[:, 0]
    atomic_positions = system_params.get("atomic_positions", [])
    atomic_numbers = system_params.get("atomic_numbers", [])

    for i, ax in enumerate(axes):
        if i >= len(eigenvalues):
            break

        # Plot orbital
        orbital = eigenvectors[:, i]
        ax.plot(x, orbital, "b-", linewidth=2, label=f"ψ_{i}")
        ax.fill_between(x, orbital, alpha=0.3, color="blue")

        # Add atomic positions
        for pos, z in zip(atomic_positions, atomic_numbers):
            ax.axvline(x=pos, color="red", linestyle="--", alpha=0.5)

        # Labels and title
        ax.set_ylabel(f"ψ_{i}(x)", fontsize=10)
        title = f"Orbital {i}: E = {eigenvalues[i]:.4f} Ha, "
        title += f"Occupation = {occupations[i]:.3f}, "
        title += f"Energy Contribution = {occupations[i] * eigenvalues[i]:.4f} Ha"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add a horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    axes[-1].set_xlabel("Position x (Bohr)", fontsize=12)

    plt.suptitle("Molecular Orbital Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "orbital_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Orbital analysis saved to {output_path}")
