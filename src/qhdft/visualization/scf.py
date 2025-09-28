from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


def visualize_scf_convergence(
    scf_residuals: npt.NDArray[np.float64],
    converged_density: npt.NDArray[np.float64],
    initial_density: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    shape_function: Any,
    system_params: Dict[str, Any],
    computational_complexity: Optional[int] = None,
    output_folder: str = "visualizations/scf/",
    save_animation: bool = False,
) -> None:
    """
    Visualize the SCF convergence process and final results.

    Parameters
    ----------
    scf_residuals : array
        Array of residuals from each SCF iteration
    converged_density : array
        Final converged coarse density
    initial_density : array
        Initial coarse density
    coarse_points : array
        Coarse interpolation points (atomic positions)
    fine_grid : array
        Fine grid points
    shape_function : callable
        Shape function for interpolation
    system_params : dict
        System parameters
    computational_complexity : int, optional
        Total query complexity
    output_folder : str
        Folder to save visualization PNG files
    save_animation : bool
        Whether to save an animation of density evolution
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Extract parameters
    atomic_positions = system_params.get("atomic_positions", [])
    atomic_numbers = system_params.get("atomic_numbers", [])

    # 1. SCF Convergence Plot (Residuals vs Iterations)
    _plot_convergence(scf_residuals, computational_complexity, output_path)

    # 2. Initial vs Final Density Comparison
    _plot_density_comparison(
        initial_density,
        converged_density,
        coarse_points,
        fine_grid,
        shape_function,
        atomic_positions,
        atomic_numbers,
        output_path,
    )

    # 3. Convergence Analysis Plot
    _plot_convergence_analysis(scf_residuals, output_path)

    # 4. Summary Figure combining all elements
    _create_summary_figure(
        scf_residuals,
        initial_density,
        converged_density,
        coarse_points,
        fine_grid,
        shape_function,
        atomic_positions,
        atomic_numbers,
        computational_complexity,
        output_path,
    )

    print(f"SCF visualizations saved to {output_path}")


def _plot_convergence(
    residuals: npt.NDArray[np.float64], complexity: Optional[int], output_path: Path
) -> None:
    """Plot SCF convergence: residuals vs iterations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    iterations = np.arange(1, len(residuals) + 1)

    # Plot residuals on log scale
    ax.semilogy(iterations, residuals, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("SCF Iteration", fontsize=14)
    ax.set_ylabel("Residual ||n_{k+1} - n_k||", fontsize=14)
    ax.set_title("SCF Convergence", fontsize=16, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    # Add convergence info
    converged = len(residuals) < 100  # Assuming max_iterations = 100
    final_residual = residuals[-1]

    info_text = f"Iterations: {len(residuals)}\nFinal residual: {final_residual:.2e}"
    if complexity:
        info_text += f"\nTotal queries: {complexity:,}"
    if converged:
        info_text += "\nStatus: Converged"
    else:
        info_text += "\nStatus: Max iterations reached"

    ax.text(
        0.95,
        0.95,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path / "convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_density_comparison(
    initial_density: npt.NDArray[np.float64],
    final_density: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    shape_function: Any,
    atomic_positions: List[float],
    atomic_numbers: List[int],
    output_path: Path,
) -> None:
    """Plot initial vs final density on fine grid."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Interpolate densities to fine grid
    diffs = fine_grid[:, None, :] - coarse_points[None, :, :]
    interpolation_matrix = shape_function(diffs)

    initial_fine = interpolation_matrix @ initial_density
    final_fine = interpolation_matrix @ final_density

    x = fine_grid[:, 0]

    # Plot initial density
    ax1.plot(x, initial_fine, "b-", linewidth=2, label="Initial density")
    ax1.fill_between(x, initial_fine, alpha=0.3, color="blue")
    _add_atoms_to_plot(ax1, atomic_positions, atomic_numbers)
    ax1.set_ylabel("Density n(x)", fontsize=12)
    ax1.set_title("Initial Density", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot final density
    ax2.plot(x, final_fine, "r-", linewidth=2, label="Converged density")
    ax2.fill_between(x, final_fine, alpha=0.3, color="red")
    _add_atoms_to_plot(ax2, atomic_positions, atomic_numbers)
    ax2.set_ylabel("Density n(x)", fontsize=12)
    ax2.set_title("Converged Density", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot difference
    density_diff = final_fine - initial_fine
    ax3.plot(x, density_diff, "g-", linewidth=2)
    ax3.fill_between(x, density_diff, alpha=0.3, color="green")
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    _add_atoms_to_plot(ax3, atomic_positions, atomic_numbers)
    ax3.set_xlabel("Position x (Bohr)", fontsize=12)
    ax3.set_ylabel("Δn(x)", fontsize=12)
    ax3.set_title("Density Change (Final - Initial)", fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "density_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_convergence_analysis(
    residuals: npt.NDArray[np.float64], output_path: Path
) -> None:
    """Analyze convergence behavior with additional metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    iterations = np.arange(1, len(residuals) + 1)

    # 1. Residuals with exponential fit (if converging)
    ax1.semilogy(iterations, residuals, "bo", markersize=6, label="Residuals")

    # Try to fit exponential decay for converged part
    if len(residuals) > 5:
        try:
            # Fit log(residual) = log(a) - b*iteration
            coeffs = np.polyfit(iterations[5:], np.log(residuals[5:]), 1)
            fit_residuals = np.exp(coeffs[1]) * np.exp(coeffs[0] * iterations)
            ax1.semilogy(
                iterations,
                fit_residuals,
                "r--",
                linewidth=2,
                label=f"Exp fit: rate={-coeffs[0]:.3f}",
            )
            ax1.legend()
        except Exception:
            pass

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual")
    ax1.set_title("Convergence Rate Analysis")
    ax1.grid(True, which="both", alpha=0.3)

    # 2. Convergence ratio
    if len(residuals) > 1:
        conv_ratio = residuals[1:] / residuals[:-1]
        ax2.plot(iterations[1:], conv_ratio, "go-", linewidth=2, markersize=6)
        ax2.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="No improvement")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("R[k+1]/R[k]")
        ax2.set_title("Convergence Ratio")
        ax2.set_ylim([0, 1.2])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Cumulative reduction
    initial_residual = residuals[0]
    cumulative_reduction = 1 - residuals / initial_residual
    ax3.plot(iterations, cumulative_reduction * 100, "mo-", linewidth=2, markersize=6)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Reduction (%)")
    ax3.set_title("Cumulative Residual Reduction")
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)

    # 4. Iterations to convergence thresholds
    thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    iters_to_threshold = []

    for threshold in thresholds:
        idx = np.where(residuals < threshold)[0]
        if len(idx) > 0:
            iters_to_threshold.append(idx[0] + 1)
        else:
            iters_to_threshold.append(np.nan)

    valid_mask = ~np.isnan(iters_to_threshold)
    ax4.semilogx(
        np.array(thresholds)[valid_mask],
        np.array(iters_to_threshold)[valid_mask],
        "co-",
        linewidth=2,
        markersize=8,
    )
    ax4.set_xlabel("Convergence Threshold")
    ax4.set_ylabel("Iterations Required")
    ax4.set_title("Iterations vs Convergence Threshold")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path / "convergence_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def _create_summary_figure(
    residuals: npt.NDArray[np.float64],
    initial_density: npt.NDArray[np.float64],
    final_density: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    shape_function: Any,
    atomic_positions: List[float],
    atomic_numbers: List[int],
    complexity: Optional[int],
    output_path: Path,
) -> None:
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Main convergence plot (top left, 2x2)
    ax_conv = fig.add_subplot(gs[0:2, 0:2])
    iterations = np.arange(1, len(residuals) + 1)
    ax_conv.semilogy(iterations, residuals, "bo-", linewidth=2, markersize=8)
    ax_conv.set_xlabel("SCF Iteration", fontsize=12)
    ax_conv.set_ylabel("Residual", fontsize=12)
    ax_conv.set_title("SCF Convergence", fontsize=14, fontweight="bold")
    ax_conv.grid(True, which="both", alpha=0.3)

    # Info box (top right)
    ax_info = fig.add_subplot(gs[0:2, 2])
    ax_info.axis("off")

    info_lines = [
        "SCF Summary",
        "-" * 20,
        f"Total iterations: {len(residuals)}",
        f"Final residual: {residuals[-1]:.2e}",
        f"Initial residual: {residuals[0]:.2e}",
        f"Reduction: {(1 - residuals[-1]/residuals[0])*100:.1f}%",
    ]

    if complexity:
        info_lines.append(f"Total queries: {complexity:,}")
        info_lines.append(f"Queries/iter: {complexity/len(residuals):.0f}")

    info_lines.extend(
        [
            "",
            "System Info",
            "-" * 20,
            f"Atoms: {len(atomic_positions)}",
            f"Grid points: {len(fine_grid)}",
            f"Interpolation points: {len(coarse_points)}",
        ]
    )

    info_text = "\n".join(info_lines)
    ax_info.text(
        0.1,
        0.9,
        info_text,
        transform=ax_info.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Density plots (bottom row)
    # Interpolate densities
    diffs = fine_grid[:, None, :] - coarse_points[None, :, :]
    interpolation_matrix = shape_function(diffs)
    initial_fine = interpolation_matrix @ initial_density
    final_fine = interpolation_matrix @ final_density
    x = fine_grid[:, 0]

    # Initial density
    ax_init = fig.add_subplot(gs[2, 0])
    ax_init.plot(x, initial_fine, "b-", linewidth=2)
    ax_init.fill_between(x, initial_fine, alpha=0.3, color="blue")
    _add_atoms_to_plot(ax_init, atomic_positions, atomic_numbers, small=True)
    ax_init.set_xlabel("x (Bohr)", fontsize=10)
    ax_init.set_ylabel("n(x)", fontsize=10)
    ax_init.set_title("Initial Density", fontsize=12)
    ax_init.grid(True, alpha=0.3)

    # Final density
    ax_final = fig.add_subplot(gs[2, 1])
    ax_final.plot(x, final_fine, "r-", linewidth=2)
    ax_final.fill_between(x, final_fine, alpha=0.3, color="red")
    _add_atoms_to_plot(ax_final, atomic_positions, atomic_numbers, small=True)
    ax_final.set_xlabel("x (Bohr)", fontsize=10)
    ax_final.set_ylabel("n(x)", fontsize=10)
    ax_final.set_title("Converged Density", fontsize=12)
    ax_final.grid(True, alpha=0.3)

    # Density difference
    ax_diff = fig.add_subplot(gs[2, 2])
    density_diff = final_fine - initial_fine
    ax_diff.plot(x, density_diff, "g-", linewidth=2)
    ax_diff.fill_between(x, density_diff, alpha=0.3, color="green")
    ax_diff.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    _add_atoms_to_plot(ax_diff, atomic_positions, atomic_numbers, small=True)
    ax_diff.set_xlabel("x (Bohr)", fontsize=10)
    ax_diff.set_ylabel("Δn(x)", fontsize=10)
    ax_diff.set_title("Density Change", fontsize=12)
    ax_diff.grid(True, alpha=0.3)

    plt.suptitle("Quantum Hybrid DFT - SCF Results", fontsize=16, fontweight="bold")
    plt.savefig(output_path / "scf_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def _add_atoms_to_plot(
    ax: plt.Axes, positions: List[float], atomic_numbers: List[int], small: bool = False
) -> None:
    """Add atomic position markers to a plot."""
    element_symbols = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O"}
    colors = {1: "white", 3: "purple", 6: "black", 7: "blue", 8: "red"}
    sizes = {1: 100, 3: 200, 6: 150, 7: 150, 8: 150}

    if small:
        sizes = {k: v * 0.6 for k, v in sizes.items()}

    y_min, y_max = ax.get_ylim()
    y_pos = y_min - 0.05 * (y_max - y_min)

    for pos, z in zip(positions, atomic_numbers):
        symbol = element_symbols.get(z, f"Z={z}")
        color = colors.get(z, "gray")
        size = sizes.get(z, 150)

        ax.scatter(
            pos, y_pos, s=size, c=color, edgecolors="black", linewidth=1.5, zorder=10
        )
        ax.text(
            pos,
            y_pos,
            symbol,
            ha="center",
            va="center",
            fontsize=8 if small else 10,
            fontweight="bold",
        )

    # Add vertical lines at atomic positions
    for pos in positions:
        ax.axvline(x=pos, color="gray", linestyle=":", alpha=0.5)


def create_density_evolution_animation(
    density_history: List[npt.NDArray[np.float64]],
    residuals: npt.NDArray[np.float64],
    coarse_points: npt.NDArray[np.float64],
    fine_grid: npt.NDArray[np.float64],
    shape_function: Any,
    atomic_positions: List[float],
    atomic_numbers: List[int],
    output_path: Path,
    fps: int = 5,
) -> None:
    """
    Create an animation showing density evolution during SCF iterations.

    Note: This requires density_history to be collected during SCF iterations,
    which would need modification to the run_scf function.
    """
    if not density_history:
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Interpolation matrix
    diffs = fine_grid[:, None, :] - coarse_points[None, :, :]
    interpolation_matrix = shape_function(diffs)
    x = fine_grid[:, 0]

    # Initial setup
    initial_fine = interpolation_matrix @ density_history[0]
    (line_density,) = ax1.plot(x, initial_fine, "b-", linewidth=2)
    ax1.fill_between(x, initial_fine, alpha=0.3, color="blue")

    _add_atoms_to_plot(ax1, atomic_positions, atomic_numbers)
    ax1.set_xlabel("Position x (Bohr)", fontsize=12)
    ax1.set_ylabel("Density n(x)", fontsize=12)
    ax1.set_ylim(
        [0, max([np.max(interpolation_matrix @ d) for d in density_history]) * 1.1]
    )
    ax1.grid(True, alpha=0.3)

    # Residual plot
    (line_residual,) = ax2.semilogy([], [], "ro-", linewidth=2, markersize=6)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_xlim([0, len(residuals)])
    ax2.set_ylim([min(residuals) * 0.5, max(residuals) * 2])
    ax2.grid(True, which="both", alpha=0.3)

    title = ax1.text(
        0.5,
        1.05,
        "",
        transform=ax1.transAxes,
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    def animate(frame):
        # Update density
        density_fine = interpolation_matrix @ density_history[frame]
        line_density.set_ydata(density_fine)

        # Update fill
        ax1.collections.clear()
        ax1.fill_between(x, density_fine, alpha=0.3, color="blue")
        _add_atoms_to_plot(ax1, atomic_positions, atomic_numbers)

        # Update residuals
        if frame > 0:
            line_residual.set_data(range(frame), residuals[:frame])

        # Update title
        title.set_text(f"SCF Iteration {frame + 1}")

        return line_density, line_residual, title

    anim = FuncAnimation(
        fig, animate, frames=len(density_history), interval=1000 // fps, blit=False
    )

    anim.save(output_path / "density_evolution.gif", writer="pillow", fps=fps)
    plt.close()

    print(f"Animation saved to {output_path / 'density_evolution.gif'}")
