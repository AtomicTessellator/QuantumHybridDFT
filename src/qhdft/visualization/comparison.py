import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_classical_vs_quantum(
    scf_result,
    coarse_points,
    output_folder: str = "visualizations/comparison/",
):
    """Generate comparison visualizations between classical and quantum SCF results.
    
    Creates PNG plots showing:
    - Convergence comparison
    - Density evolution comparison
    - Final density difference
    - Complexity comparison
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    classical_densities = scf_result.classical_densities
    quantum_densities = scf_result.quantum_densities
    classical_residuals = scf_result.classical_residuals
    quantum_residuals = scf_result.quantum_residuals
    
    # 1. Convergence Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations_classical = np.arange(len(classical_residuals))
    iterations_quantum = np.arange(len(quantum_residuals))
    
    ax.semilogy(iterations_classical, classical_residuals, 'o-', 
                label='Classical', linewidth=2, markersize=6)
    ax.semilogy(iterations_quantum, quantum_residuals, 's-', 
                label='Quantum', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Residual (log scale)', fontsize=12)
    ax.set_title('SCF Convergence: Classical vs Quantum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'convergence_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Density Evolution Comparison
    num_plots = min(5, len(classical_densities))
    indices = np.linspace(0, len(classical_densities)-1, num_plots, dtype=int)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Classical evolution
    for idx in indices:
        axes[0].plot(coarse_points, classical_densities[idx], 
                    label=f'Iter {idx}', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Position (Bohr)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Classical Density Evolution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Quantum evolution
    for idx in indices:
        axes[1].plot(coarse_points, quantum_densities[idx], 
                    label=f'Iter {idx}', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Position (Bohr)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Quantum Density Evolution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'density_evolution.png'), dpi=150)
    plt.close()
    
    # 3. Final Density Comparison
    final_classical = classical_densities[-1]
    final_quantum = quantum_densities[-1]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Overlay comparison
    axes[0].plot(coarse_points, final_classical, 'o-', 
                label='Classical', linewidth=2, markersize=8)
    axes[0].plot(coarse_points, final_quantum, 's-', 
                label='Quantum', linewidth=2, markersize=6, alpha=0.7)
    axes[0].set_xlabel('Position (Bohr)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Final Converged Density', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Difference plot
    difference = final_quantum - final_classical
    axes[1].plot(coarse_points, difference, 'o-', color='red', 
                linewidth=2, markersize=8)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Position (Bohr)', fontsize=12)
    axes[1].set_ylabel('Density Difference\n(Quantum - Classical)', fontsize=12)
    axes[1].set_title('Density Difference', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add error metrics
    l2_error = np.linalg.norm(difference)
    max_error = np.max(np.abs(difference))
    axes[1].text(0.02, 0.98, f'L2 Error: {l2_error:.6f}\nMax Error: {max_error:.6f}',
                transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'final_density_comparison.png'), dpi=150)
    plt.close()
    
    # 4. Iteration-by-iteration density difference
    num_iters = min(len(classical_densities), len(quantum_densities))
    diff_norms = []
    for i in range(num_iters):
        diff = quantum_densities[i] - classical_densities[i]
        diff_norms.append(np.linalg.norm(diff))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(num_iters), diff_norms, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('L2 Norm of Density Difference', fontsize=12)
    ax.set_title('Evolution of Classical-Quantum Density Difference', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'density_difference_evolution.png'), dpi=150)
    plt.close()
    
    # 5. Complexity Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Classical', 'Quantum']
    complexities = [scf_result.classical_complexity, scf_result.quantum_complexity]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, complexities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Query Complexity', fontsize=12)
    ax.set_title('Computational Complexity Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, complexities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'complexity_comparison.png'), dpi=150)
    plt.close()
    
    # 6. Summary heatmap of density evolution
    classical_matrix = np.array(classical_densities).T
    quantum_matrix = np.array(quantum_densities).T
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = axes[0].imshow(classical_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Coarse Grid Index', fontsize=12)
    axes[0].set_title('Classical Density Evolution Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Density')
    
    im2 = axes[1].imshow(quantum_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Coarse Grid Index', fontsize=12)
    axes[1].set_title('Quantum Density Evolution Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'density_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"Comparison visualizations saved to {output_folder}")
    print(f"  - Classical final L2 residual: {classical_residuals[-1]:.6e}")
    print(f"  - Quantum final L2 residual: {quantum_residuals[-1]:.6e}")
    print(f"  - Classical complexity: {scf_result.classical_complexity:,}")
    print(f"  - Quantum complexity: {scf_result.quantum_complexity:,}")
    print(f"  - Final density L2 difference: {l2_error:.6e}")
