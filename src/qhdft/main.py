from typing import Any, Dict

import numpy as np

from qhdft.discretization import setup_discretization
from qhdft.scf import (
    Discretization,
    EstimationControls,
    SCFConfig,
    SCFControls,
    run_scf_configured,
)
from qhdft.validation import compute_energy, compute_error_breakdown, run_scaling_test
from qhdft.visualization.discretization import visualize_discretization
from qhdft.visualization.energy import (
    visualize_energy_analysis,
    visualize_orbital_analysis,
)
from qhdft.visualization.scaling import visualize_scaling_analysis
from qhdft.visualization.scf import visualize_scf_convergence

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
    # System parameters for 1D Li - H chain example.
    system_params: Dict[str, Any] = {
        "dimension": 1,
        "computational_domain": [0, 10.0],  # Bohr units
        "grid_exponent": 5,  # Number of grid points = 2 ^ 5 = 32
        "atomic_positions": [3.0, 7.0],  # Li at 3.0, H at 7.0 Bohr
        "atomic_numbers": [3, 1],  # Li=3, H=1
        "gaussian_width": 0.5,  # Width for density approximation
        "interpolation_tolerance": 0.1,
    }
    # Quantum algorithm parameters
    inverse_temperature = 10.0  # Beta parameter for thermal state
    mixing_parameter = 0.5  # Alpha: controls SCF convergence
    block_size = 2  # Size of random block for updates
    max_scf_iterations = 100  # Maximum SCF iterations
    scf_convergence_threshold = 1e-4  # SCF convergence criterion
    confidence_level = 0.01  # Confidence parameter for estimation routine
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

    # Run SCF (stages 2,4,5) — using structured config for clarity
    scf_config = SCFConfig(
        initial_coarse_density=initial_density_coarse,
        discretization=Discretization(
            fine_grid=fine_grid,
            coarse_points=coarse_interpolation_points,
            shape_function=shape_function,
            system_params=system_params,
        ),
        scf=SCFControls(
            inverse_temperature=inverse_temperature,
            mixing_parameter=mixing_parameter,
            block_size=block_size,
            max_iterations=max_scf_iterations,
            convergence_threshold=scf_convergence_threshold,
        ),
        estimation=EstimationControls(
            confidence_level=confidence_level,
            estimation_error_tolerance=estimation_error_tolerance,
            num_quantum_samples=num_quantum_samples,
        ),
    )
    scf_result = run_scf_configured(scf_config)
    converged_density = scf_result.converged_coarse_density
    scf_residuals = scf_result.residuals
    computational_complexity = scf_result.total_complexity

    # Visualize SCF convergence if requested
    if visualize:
        visualize_scf_convergence(
            scf_residuals,
            converged_density,
            initial_density_coarse,
            coarse_interpolation_points,
            fine_grid,
            shape_function,
            system_params,
            computational_complexity,
            visualization_folder + "/scf/",
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

    # Visualize energy analysis if requested
    if visualize:
        visualize_energy_analysis(
            converged_density,
            fine_grid,
            coarse_interpolation_points,
            shape_function,
            system_params,
            inverse_temperature,
            ground_state_energy,
            scf_residuals,
            None,  # No reference energy in this example
            visualization_folder + "/energy/",
        )

        # Also create orbital analysis
        visualize_orbital_analysis(
            converged_density,
            fine_grid,
            coarse_interpolation_points,
            shape_function,
            system_params,
            inverse_temperature,
            num_orbitals=6,
            output_folder=visualization_folder + "/energy/",
        )

    # Scaling test
    num_atoms_range = np.arange(2, 21, 2)
    scaling_metrics, r_squared = run_scaling_test(
        system_params,
        inverse_temperature,
        mixing_parameter,
        max_scf_iterations,
        scf_convergence_threshold,
        confidence_level,
        estimation_error_tolerance,
        num_quantum_samples,
        num_atoms_range,
    )
    print(f"Scaling R ^ 2: {r_squared}")
    print(f"Queries vs Na: {scaling_metrics['queries']}")

    # Visualize scaling analysis if requested
    if visualize:
        visualize_scaling_analysis(
            scaling_metrics,
            r_squared,
            system_params,
            visualization_folder + "/scaling/",
            theoretical_scaling="O(Na)",
        )

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


if __name__ == "__main__":
    main()
