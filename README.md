# QuantumHybridDFT

QuantumHybridDFT is a research-oriented Python package that prototypes a hybrid quantum-classical Density Functional Theory (DFT) workflow. It follows a staged plan:
- Stage 1: Discretization and initial density
- Stage 2: Hamiltonian construction
- Stage 3: Quantum Singular Value Transformation (QSVT) for density-matrix encoding
- Stage 4: Electron density estimation (simulated)
- Stage 5: Self-Consistent Field (SCF) iterations with randomized block updates
- Stage 6: Validation, scaling, and visualizations

The code is organized under `src/qhdft`, with unit tests in `tests/`

## Quick Start

### Prerequisites
- Python 3.11+
- OS: Linux/macOS/Windows (tests are developed on Linux)

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
# or, to use requirements directly
pip install -r requirements.txt
```


### Running the demo pipeline
The entry point script `qhdft.main` runs the full example (1D Li–H chain), performs SCF, computes energy, and generates visualizations under `visualizations/`.

```bash
python -m qhdft.main
# or
python src/qhdft/main.py
```

Arguments: the default `main()` accepts two parameters:
- `visualize: bool = True` – generate figures into `visualizations/`
- `visualization_folder: str = "visualizations"` – base folder for outputs

You can modify system parameters inside `qhdft.main` (grid size, atom positions, temperature, etc.).

## Project Structure

```
QuantumHybridDFT/
  pyproject.toml          # Package metadata
  requirements.txt        # Python dependencies
  src/qhdft/
    __init__.py
    main.py               # Orchestrates the staged pipeline and visualizations
    discretization.py     # Stage 1: grid, interpolation points, initial density, shape function
    hamiltonian.py        # Stage 2: sparse Hamiltonian, normalization, sparse oracles
    density.py            # Stage 4: simulated blockwise density estimation
    scf.py                # Stage 5: SCF dataclasses and run_scf_configured()
    validation.py         # Stage 6: energy computation, scaling tests, error breakdown
    qsvt.py               # Stage 3: Chebyshev/QSVT wrapper API
    qsvt_circuit.py       # Stage 3: unitary dilation block-encoding utilities
    qsvt_qsp.py           # Stage 3: QSP-based circuit synthesis (phases)
    visualization/
      discretization.py   # Figures for grids, densities, interpolation matrix
      scf.py              # Convergence, comparisons, summaries
      energy.py           # Energy components, profiles, orbital analysis
      scaling.py          # Scaling analysis figures
  tests/
    test_discretization.py
    test_hamiltonian.py
    test_density.py
    test_qsvt.py
    test_scf.py
    test_validation.py
  visualizations/         # Generated figures (created at runtime)
```

## Usage Guide

### 1) Discretization
Create 1D grids and initial densities via:
```python
from qhdft.discretization import setup_discretization
params = {
    "dimension": 1,
    "computational_domain": [0, 10.0],
    "grid_exponent": 5,             # Ng = 2^m
    "atomic_positions": [3.0, 7.0],
    "atomic_numbers": [3, 1],
    "gaussian_width": 0.5,
    "interpolation_tolerance": 0.1,
}
fine_grid, coarse_points, coarse_density, shape_function = setup_discretization(params)
```

### 2) Hamiltonian
Build sparse Hamiltonian from coarse density:
```python
from qhdft.hamiltonian import build_hamiltonian
H, norm, L, V = build_hamiltonian(coarse_density, fine_grid, coarse_points, shape_function, params)
```
`H` is `scipy.sparse.csr_matrix`, `norm` scales spectrum to [-1,1], `L`/`V` are sparse oracles.

### 3) QSVT (polynomial approximation and circuits)
The `qsvt.py` and `qsvt_circuit.py` modules construct a block-encoded unitary whose top-left block approximates the Fermi–Dirac function of the normalized Hamiltonian. See `tests/test_qsvt.py` for usage.

### 4) Density estimation (simulated)
`qhdft.density.estimate_density(...)` emulates measurement-based estimation with controlled noise and returns confidence intervals and an estimated query complexity.

### 5) SCF iterations
Run the hybrid SCF with dataclass configs:
```python
from qhdft.scf import Discretization, EstimationControls, SCFConfig, SCFControls, run_scf_configured
scf_cfg = SCFConfig(
  initial_coarse_density=coarse_density,
  discretization=Discretization(
    fine_grid=fine_grid,
    coarse_points=coarse_points,
    shape_function=shape_function,
    system_params=params,
  ),
  scf=SCFControls(
    inverse_temperature=10.0,
    mixing_parameter=0.5,
    block_size=2,
    max_iterations=100,
    convergence_threshold=1e-4,
  ),
  estimation=EstimationControls(
    confidence_level=0.01,
    estimation_error_tolerance=1e-4,
    num_quantum_samples=100000,
  ),
)
result = run_scf_configured(scf_cfg)
```

### 6) Validation and visualizations
- Energy: `qhdft.validation.compute_energy(...)`
- Scaling tests: `qhdft.validation.run_scaling_test(...)`
- Figures under `qhdft.visualization.*` are produced automatically by `main()`.

## Running Tests
Tests use `pytest` and standard unittest-style assertions.

```bash
# Activate your virtual environment first
pytest -q
# or to run a specific test file
pytest tests/test_qsvt.py -q
```

Notes:
- Quantum circuit simulations use Qiskit (Aer). Ensure the dependencies in `requirements.txt` are installed.
- Tests are configured for small problem sizes (e.g., Ng=32) for quick runs.

## Dependencies
See `requirements.txt`. Core libraries include:
- numpy, scipy, matplotlib, seaborn
- qiskit, qiskit-aer (for circuit construction and simulation)

If Qiskit is not already present, install with:
```bash
pip install qiskit qiskit-aer
```

## Reproducibility & Figures
Running `python -m qhdft.main` will generate figures under `visualizations/`:
- `discretization/`: grids, initial density, interpolation summary
- `scf/`: convergence curves and summaries
- `energy/`: energy components, profiles, orbital analysis
- `scaling/`: scaling analyses
