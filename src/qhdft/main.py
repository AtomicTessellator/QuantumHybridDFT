from typing import Any, Dict

import numpy as np

from qhdft.discretization import setup_discretization
from qhdft.stage5 import run_scf
from qhdft.stage6 import compute_energy, compute_error_breakdown, run_scaling_test
from qhdft.visualization.discretization import visualize_discretization

#
# Quantum Hybrid DFT
# ┌─────────────────────────────────────────┐
# │ Stage 1: Discretization initial density │
# └───────────────────┬─────────────────────┘
#                     │
#    ┌────────────────▼──────────────────┐
#    │ Stage 2: Run SCF iterations       │
#    └───────────────────────────────────┘


#
# 1. Setup discretization and initial density (stage1).
# 2. Run SCF iterations (stage5, which calls stage2,4 internally; stage3 is placeholder).
# 3. Validate with energy, scaling, errors (stage6).
# Note: stage3 QSVT is not fully integrated in simulation; used for poly approx in tests.


def main(visualize: bool = True, visualization_folder: str = "visualizations") -> None:
    # System parameters for 1D Li-H chain example.
    system_params: Dict[str, Any] = {
        "dimension": 1,
        "computational_domain": [0, 10.0],  # Bohr units
        "grid_exponent": 5,  # Number of grid points = 2^5 = 32
        "atomic_positions": [3.0, 7.0],  # Li at 3.0, H at 7.0 Bohr
        "atomic_numbers": [3, 1],  # Li=3, H=1
        "Z": [3, 1],  # Same as atomic_numbers, for compatibility with stage5
        "gaussian_width": 0.5,  # Width for density approximation
        "interpolation_tolerance": 0.1,
    }
    # Quantum algorithm parameters
    inverse_temperature = 10.0  # Beta parameter for thermal state
    mixing_parameter = 0.5  # Alpha: controls SCF convergence
    chebyshev_degree = 2  # B: degree of Chebyshev expansion
    max_scf_iterations = 100  # Maximum SCF iterations
    scf_convergence_threshold = 1e-4  # SCF convergence criterion
    phase_estimation_precision = 0.01  # Delta: phase estimation precision
    estimation_error_tolerance = 1e-4  # Error tolerance for quantum estimation
    num_quantum_samples = 100000  # M: number of quantum samples
    polynomial_approximation_error = 1e-4  # Error in polynomial approximation

    # Stage 1: Discretization
    fine_grid, coarse_interpolation_points, initial_density_coarse, shape_function = (
        setup_discretization(system_params)
    )

    # Visualize discretization if requested
    if visualize:
        visualize_discretization(
            fine_grid,
            coarse_interpolation_points,
            initial_density_coarse,
            shape_function,
            system_params,
            visualization_folder + "/discretization/",
        )

    return
    # Run SCF (stages 2,4,5)
    converged_density, scf_residuals, computational_complexity = run_scf(
        initial_density_coarse,
        fine_grid,
        coarse_interpolation_points,
        shape_function,
        system_params,
        inverse_temperature,
        mixing_parameter,
        chebyshev_degree,
        max_scf_iterations,
        scf_convergence_threshold,
        phase_estimation_precision,
        estimation_error_tolerance,
        num_quantum_samples,
    )

    # Stage 6: Validation
    ground_state_energy = compute_energy(
        converged_density,
        fine_grid,
        coarse_interpolation_points,
        shape_function,
        system_params,
        inverse_temperature,
    )
    print(f"Ground state energy: {ground_state_energy}")

    # Scaling test
    num_atoms_range = np.arange(2, 21, 2)
    scaling_metrics, r_squared = run_scaling_test(
        system_params,
        inverse_temperature,
        mixing_parameter,
        max_scf_iterations,
        scf_convergence_threshold,
        phase_estimation_precision,
        estimation_error_tolerance,
        num_quantum_samples,
        num_atoms_range,
    )
    print(f"Scaling R^2: {r_squared}")
    print(f"Queries vs Na: {scaling_metrics['queries']}")

    # Error breakdown (using placeholder poly_err; in full, from stage3)
    # For demo, compute classical reference values
    reference_density = (
        initial_density_coarse.copy()
    )  # Placeholder; actual would run classical SCF
    reference_energy = ground_state_energy  # Placeholder
    polynomial_max_error = polynomial_approximation_error
    error_analysis = compute_error_breakdown(
        converged_density,
        reference_density,
        ground_state_energy,
        reference_energy,
        polynomial_max_error,
        estimation_error_tolerance,
        len(scf_residuals),
    )
    print(f"Error breakdown: {error_analysis}")

    # Note: To include stage3, could build H from hat_n, normalize, call build_qsvt, but not simulated here.


if __name__ == "__main__":
    main()
