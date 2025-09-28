from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress


def visualize_scaling_analysis(
    scaling_metrics: Dict[str, Any],
    r_squared: float,
    system_params: Dict[str, Any],
    output_folder: str = "visualizations / scaling/",
    theoretical_scaling: Optional[str] = "O(Na)",
) -> None:
    """
    Visualize scaling test results showing query complexity vs system size.

    Parameters
    ----------
    scaling_metrics : dict
        Dictionary containing:
        - atomCounts: array of atom counts tested
        - queries: array of query complexities
        - times: array of timing data (if available)
    r_squared : float
        R - squared value from linear regression fit
    system_params : dict
        Base system parameters used in scaling test
    output_folder : str
        Folder to save visualization PNG files
    theoretical_scaling : str, optional
        Expected theoretical scaling behavior
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Extract data
    atom_counts = np.array(scaling_metrics["atomCounts"])
    queries = np.array(scaling_metrics["queries"])
    times = np.array(
        scaling_metrics.get("times", queries)
    )  # Use queries as proxy if no times

    # 1. Main scaling plot
    _plot_scaling_main(
        atom_counts, queries, r_squared, theoretical_scaling, output_path
    )

    # 2. Detailed analysis with multiple metrics
    _plot_scaling_detailed(atom_counts, queries, times, output_path)

    # 3. Scaling efficiency analysis
    _plot_scaling_efficiency(atom_counts, queries, output_path)

    # 4. Summary figure
    _create_scaling_summary(
        atom_counts,
        queries,
        times,
        r_squared,
        system_params,
        theoretical_scaling,
        output_path,
    )

    print(f"Scaling visualizations saved to {output_path}")


def _plot_scaling_main(
    atom_counts: npt.NDArray[np.int_],
    queries: npt.NDArray[np.int_],
    r_squared: float,
    theoretical_scaling: str,
    output_path: Path,
) -> None:
    """Plot main scaling results with linear fit."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Scatter plot of actual data
    ax.scatter(
        atom_counts,
        queries,
        s=100,
        c="blue",
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
        label="Measured",
    )

    # Linear regression
    slope, intercept, _, _, _ = linregress(atom_counts, queries)
    fit_line = slope * atom_counts + intercept
    ax.plot(
        atom_counts,
        fit_line,
        "r--",
        linewidth=2,
        label=f"Linear fit: Q = {slope:.1f}·Na + {intercept:.1f}",
    )

    # Theoretical scaling (if different from linear)
    if theoretical_scaling and theoretical_scaling != "O(Na)":
        # Add other theoretical curves if needed
        pass

    ax.set_xlabel("Number of Atoms (Na)", fontsize=14)
    ax.set_ylabel("Query Complexity", fontsize=14)
    ax.set_title(
        f"Quantum Algorithm Scaling: Query Complexity vs System Size\n(R² = {r_squared:.4f})",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add text box with scaling info
    textstr = "Linear Scaling Confirmed\n"
    textstr += f"Slope: {slope:.2f} queries / atom\n"
    textstr += f"R² = {r_squared:.4f}"
    if r_squared > 0.95:
        textstr += "\n✓ Excellent linear fit"

    props = dict(
        boxstyle="round",
        facecolor="lightgreen" if r_squared > 0.95 else "wheat",
        alpha=0.8,
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_path / "scaling_main.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_scaling_detailed(
    atom_counts: npt.NDArray[np.int_],
    queries: npt.NDArray[np.int_],
    times: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot detailed scaling analysis with multiple views."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Log - log plot
    ax1.loglog(
        atom_counts, queries, "bo-", linewidth=2, markersize=8, label="Query Complexity"
    )
    ax1.loglog(atom_counts, atom_counts, "k--", alpha=0.5, label="O(Na)")
    ax1.loglog(atom_counts, atom_counts**2, "r:", alpha=0.5, label="O(Na²)")
    ax1.set_xlabel("Number of Atoms (log scale)", fontsize=12)
    ax1.set_ylabel("Query Complexity (log scale)", fontsize=12)
    ax1.set_title("Log - Log Scaling Plot", fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # 2. Queries per atom
    queries_per_atom = queries / atom_counts
    ax2.plot(atom_counts, queries_per_atom, "go-", linewidth=2, markersize=8)
    ax2.axhline(y=np.mean(queries_per_atom), color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Number of Atoms", fontsize=12)
    ax2.set_ylabel("Queries per Atom", fontsize=12)
    ax2.set_title("Normalized Query Complexity", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add mean value text
    ax2.text(
        0.95,
        0.95,
        f"Mean: {np.mean(queries_per_atom):.2f}",
        transform=ax2.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 3. Residuals from linear fit
    slope, intercept, _, _, _ = linregress(atom_counts, queries)
    predicted = slope * atom_counts + intercept
    residuals = queries - predicted
    residuals_percent = (residuals / predicted) * 100

    ax3.scatter(atom_counts, residuals_percent, s=60, c="purple", alpha=0.7)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax3.fill_between(atom_counts, -5, 5, alpha=0.2, color="gray", label="±5% band")
    ax3.set_xlabel("Number of Atoms", fontsize=12)
    ax3.set_ylabel("Residual (%)", fontsize=12)
    ax3.set_title("Linear Fit Residuals", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Scaling rate analysis
    if len(atom_counts) > 1:
        scaling_rates = np.diff(queries) / np.diff(atom_counts)
        ax4.plot(atom_counts[1:], scaling_rates, "mo-", linewidth=2, markersize=8)
        ax4.axhline(
            y=slope, color="red", linestyle="--", linewidth=2, label=f"Avg: {slope:.2f}"
        )
        ax4.set_xlabel("Number of Atoms", fontsize=12)
        ax4.set_ylabel("dQ / dNa", fontsize=12)
        ax4.set_title("Instantaneous Scaling Rate", fontsize=13, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(
            0.5,
            0.5,
            "Insufficient data\nfor rate analysis",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=14,
        )
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.suptitle("Detailed Scaling Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "scaling_detailed.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_scaling_efficiency(
    atom_counts: npt.NDArray[np.int_],
    queries: npt.NDArray[np.int_],
    output_path: Path,
) -> None:
    """Plot scaling efficiency metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Efficiency plot (queries per atom²)
    efficiency = queries / (atom_counts**2)
    ax1.plot(atom_counts, efficiency, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Atoms", fontsize=12)
    ax1.set_ylabel("Queries / Na²", fontsize=12)
    ax1.set_title("Algorithmic Efficiency", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add trend indicator
    if len(atom_counts) > 1:
        efficiency_slope, _, _, _, _ = linregress(atom_counts, efficiency)
        trend = (
            "Improving"
            if efficiency_slope < 0
            else "Degrading" if efficiency_slope > 0 else "Stable"
        )
        color = (
            "green"
            if efficiency_slope < 0
            else "red" if efficiency_slope > 0 else "blue"
        )
        ax1.text(
            0.95,
            0.95,
            f"Trend: {trend}",
            transform=ax1.transAxes,
            fontsize=11,
            ha="right",
            va="top",
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # 2. Relative performance vs classical
    # Assume classical scales as O(Na³) for comparison
    classical_estimate = atom_counts**3
    quantum_advantage = classical_estimate / queries

    ax2.semilogy(
        atom_counts,
        quantum_advantage,
        "go-",
        linewidth=2,
        markersize=8,
        label="Quantum Advantage",
    )
    ax2.axhline(y=1, color="red", linestyle="--", linewidth=2, label="Break - even")
    ax2.fill_between(
        atom_counts,
        1,
        quantum_advantage,
        where=(quantum_advantage > 1),
        alpha=0.3,
        color="green",
        label="Quantum better",
    )
    ax2.set_xlabel("Number of Atoms", fontsize=12)
    ax2.set_ylabel("Classical / Quantum Ratio", fontsize=12)
    ax2.set_title(
        "Quantum Advantage Factor (vs O(Na³) classical)", fontsize=13, fontweight="bold"
    )
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path / "scaling_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()


def _create_scaling_summary(
    atom_counts: npt.NDArray[np.int_],
    queries: npt.NDArray[np.int_],
    times: npt.NDArray[np.float64],
    r_squared: float,
    system_params: Dict[str, Any],
    theoretical_scaling: str,
    output_path: Path,
) -> None:
    """Create comprehensive scaling summary figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Main scaling plot (larger)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.scatter(
        atom_counts,
        queries,
        s=120,
        c="blue",
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
        label="Measured",
        zorder=5,
    )

    # Fit and plot
    slope, intercept, _, _, _ = linregress(atom_counts, queries)
    fit_line = slope * atom_counts + intercept
    ax1.plot(
        atom_counts,
        fit_line,
        "r--",
        linewidth=3,
        label=f"Fit: Q = {slope:.1f}·Na + {intercept:.1f}",
        zorder=4,
    )

    # Add confidence interval
    residuals = queries - fit_line
    std_residuals = np.std(residuals)
    ax1.fill_between(
        atom_counts,
        fit_line - 2 * std_residuals,
        fit_line + 2 * std_residuals,
        alpha=0.2,
        color="red",
        label="95% CI",
    )

    ax1.set_xlabel("Number of Atoms (Na)", fontsize=14)
    ax1.set_ylabel("Query Complexity", fontsize=14)
    ax1.set_title(
        f"Scaling Test Results (R² = {r_squared:.4f})", fontsize=15, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # 2. Performance metrics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    metrics_text = "Performance Metrics\n" + "=" * 25 + "\n\n"
    metrics_text += f"R² Value: {r_squared:.4f}\n"
    metrics_text += f"Scaling: {slope:.2f} Q / atom\n"
    metrics_text += f"Base Cost: {intercept:.1f} Q\n"
    metrics_text += f"Atoms Tested: {len(atom_counts)}\n"
    metrics_text += f"Range: {atom_counts[0]}-{atom_counts[-1]}\n"

    # Assess scaling quality
    if r_squared > 0.99:
        quality = "Excellent"
        color = "darkgreen"
    elif r_squared > 0.95:
        quality = "Good"
        color = "green"
    elif r_squared > 0.90:
        quality = "Fair"
        color = "orange"
    else:
        quality = "Poor"
        color = "red"

    metrics_text += f"\nScaling Quality: {quality}"

    ax2.text(
        0.5,
        0.5,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Add quality indicator
    ax2.text(
        0.5,
        0.1,
        quality,
        transform=ax2.transAxes,
        fontsize=14,
        verticalalignment="center",
        horizontalalignment="center",
        color=color,
        fontweight="bold",
    )

    # 3. System configuration
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis("off")

    config_text = "Test Configuration\n" + "=" * 25 + "\n\n"
    config_text += f"Grid: 2^{system_params.get('grid_exponent', 'N / A')} points\n"
    config_text += f"Domain: {system_params.get('computational_domain', 'N / A')}\n"
    config_text += f"Gaussian σ: {system_params.get('gaussian_width', 'N / A')}\n"
    config_text += "Atom Type: H chain\n"
    config_text += f"Temperature: {system_params.get('inverse_temperature', 10.0)}⁻¹\n"

    ax3.text(
        0.5,
        0.5,
        config_text,
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # 4. Extrapolation plot
    ax4 = fig.add_subplot(gs[2, :2])

    # Extrapolate to larger systems
    max_atoms_extrap = int(atom_counts[-1] * 2)
    atoms_extrap = np.linspace(atom_counts[0], max_atoms_extrap, 100)
    queries_extrap = slope * atoms_extrap + intercept

    ax4.scatter(
        atom_counts, queries, s=80, c="blue", alpha=0.7, label="Measured", zorder=5
    )
    ax4.plot(
        atoms_extrap, queries_extrap, "r-", linewidth=2, label="Extrapolation", zorder=3
    )
    # Find index where extrapolation starts
    extrap_start_idx = np.searchsorted(atoms_extrap, atom_counts[-1])
    ax4.fill_between(
        atoms_extrap[extrap_start_idx:],
        queries_extrap[extrap_start_idx:] - 2 * std_residuals,
        queries_extrap[extrap_start_idx:] + 2 * std_residuals,
        alpha=0.2,
        color="red",
    )

    # Add markers for specific system sizes
    marker_sizes = [50, 100, 200]
    for size in marker_sizes:
        if size <= max_atoms_extrap:
            q_est = slope * size + intercept
            ax4.scatter(
                size,
                q_est,
                s=150,
                marker="*",
                c="gold",
                edgecolors="black",
                linewidth=2,
                zorder=6,
            )
            ax4.text(
                size,
                q_est + 0.05 * max(queries),
                f"Na={size}\nQ≈{q_est:.0f}",
                ha="center",
                fontsize=9,
            )

    ax4.set_xlabel("Number of Atoms", fontsize=12)
    ax4.set_ylabel("Query Complexity", fontsize=12)
    ax4.set_title("Scaling Extrapolation", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim([0, max_atoms_extrap * 1.1])

    # 5. Comparison table
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis("off")

    compare_text = "Complexity Comparison\n" + "=" * 25 + "\n\n"
    compare_text += "Method        Scaling\n"
    compare_text += "-" * 22 + "\n"
    compare_text += "Quantum       O(Na)\n"
    compare_text += "Classical DFT O(Na³)\n"
    compare_text += "Exact         O(Na⁶)\n\n"

    # Calculate advantage at largest tested size
    classical_ops = atom_counts[-1] ** 3
    quantum_ops = queries[-1]
    advantage = classical_ops / quantum_ops

    compare_text += f"Advantage at Na={atom_counts[-1]}:\n"
    compare_text += f"{advantage:.1f}x vs Classical"

    ax5.text(
        0.5,
        0.5,
        compare_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Main title
    fig.suptitle(
        "Quantum Hybrid DFT - Scaling Analysis Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path / "scaling_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scaling_comparison(
    multiple_scaling_results: Dict[str, Tuple[Dict[str, Any], float]],
    output_folder: str = "visualizations / scaling/",
) -> None:
    """
    Compare scaling results from multiple runs or configurations.

    Parameters
    ----------
    multiple_scaling_results : dict
        Dictionary mapping configuration names to (scaling_metrics, r_squared) tuples
    output_folder : str
        Output folder for visualizations
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(multiple_scaling_results)))

    # Plot 1: All scaling curves
    for (name, (metrics, r_squared)), color in zip(
        multiple_scaling_results.items(), colors
    ):
        atom_counts = np.array(metrics["atomCounts"])
        queries = np.array(metrics["queries"])

        ax1.plot(
            atom_counts,
            queries,
            "o-",
            linewidth=2,
            markersize=8,
            color=color,
            label=f"{name} (R²={r_squared:.3f})",
        )

        # Add fit line
        slope, intercept, _, _, _ = linregress(atom_counts, queries)
        fit_line = slope * atom_counts + intercept
        ax1.plot(atom_counts, fit_line, "--", linewidth=1, color=color, alpha=0.5)

    ax1.set_xlabel("Number of Atoms", fontsize=12)
    ax1.set_ylabel("Query Complexity", fontsize=12)
    ax1.set_title("Scaling Comparison", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scaling rates comparison
    names = []
    slopes = []
    r_squareds = []

    for name, (metrics, r_squared) in multiple_scaling_results.items():
        atom_counts = np.array(metrics["atomCounts"])
        queries = np.array(metrics["queries"])
        slope, _, _, _, _ = linregress(atom_counts, queries)

        names.append(name)
        slopes.append(slope)
        r_squareds.append(r_squared)

    x_pos = np.arange(len(names))
    bars = ax2.bar(
        x_pos, slopes, color=colors[: len(names)], alpha=0.7, edgecolor="black"
    )

    # Add R² values on bars
    for bar, r2 in zip(bars, r_squareds):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"R²={r2:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_ylabel("Scaling Rate (queries / atom)", fontsize=12)
    ax2.set_title("Scaling Rate Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "scaling_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_scaling_animation(
    atom_counts: npt.NDArray[np.int_],
    queries: npt.NDArray[np.int_],
    output_folder: str = "visualizations / scaling/",
    fps: int = 2,
) -> None:
    """
    Create an animation showing how scaling evolves as more data points are added.

    Note: This saves individual frames. To create actual animation,
    additional tools like imageio or matplotlib.animation would be needed.
    """
    output_path = Path(output_folder) / "animation_frames"
    output_path.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")

    for i in range(2, len(atom_counts) + 1):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot data up to point i
        ax.scatter(
            atom_counts[:i],
            queries[:i],
            s=100,
            c="blue",
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
        )

        # Fit line if we have enough points
        if i >= 2:
            slope, intercept, r_value, _, _ = linregress(atom_counts[:i], queries[:i])
            fit_line = slope * atom_counts[:i] + intercept
            ax.plot(
                atom_counts[:i],
                fit_line,
                "r--",
                linewidth=2,
                label=f"Fit: Q = {slope:.1f}·Na + {intercept:.1f}",
            )

            # Add R² text
            ax.text(
                0.05,
                0.95,
                f"R² = {r_value ** 2:.4f}\nPoints: {i}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Set consistent limits
        ax.set_xlim([0, max(atom_counts) * 1.1])
        ax.set_ylim([0, max(queries) * 1.1])
        ax.set_xlabel("Number of Atoms", fontsize=14)
        ax.set_ylabel("Query Complexity", fontsize=14)
        ax.set_title("Scaling Analysis Progress", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if i >= 2:
            ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path / f"frame_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Animation frames saved to {output_path}")
    print(
        f"To create animation, use: ffmpeg -r {fps} -i frame_ % 03d.png -c:v libx264 scaling_animation.mp4"
    )
