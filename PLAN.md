## Project Plan Overview
The plan is structured into stages that align with the paper's logical flow: 
setup and discretization (Section II), quantum algorithm components (Section III), 
self-consistent iterations (Sections II and III), and validation/numerical testing 
(Section IV). 


### Stage 1: System Discretization and Setup
---
#### Purpose:

Establish the numerical representation of the DFT problem, including grid discretization, interpolation for reduced density representation, and initial electron density. This matches Section II's real-space discretization and problem setup, reducing the problem size from Ng (grid points ~ Ne) to NI (interpolation points ~ Na) for linear scaling.
#### Goals:

Generate a fine grid D (size Ng = 2^m for qubit mapping) and coarse interpolation points D∆ (size NI << Ng).
Compute initial density n0 on D∆, ensuring it satisfies ∫ n0(r) dr ≈ Ne with error < 1e-6.
Verify interpolation accuracy: Reconstruct full density on D from n0 on D∆ with L2 error < 1e-4.

Inputs:

System parameters: Number of atoms Na, electrons Ne, domain D (e.g., 1D interval [0, L]), grid size δ, inverse temperature β, chemical potential µ (initial guess).
Atomic positions and types (e.g., Li-H chain).

Outputs:

Fine grid points D (array of shape (Ng, dim), dim=1 or 3).
Interpolation points D∆ (array of shape (NI, dim)).
Initial density vector n0 (array of shape (NI,)).
Shape functions N for interpolation (callable: N(rk - r) for rk in D, r in D∆).

### Stage 2: Hamiltonian Construction
---
#### Purpose: 

Build the sparse Hamiltonian matrix H from the current electron density n, incorporating kinetic energy, Hartree potential, exchange-correlation, and external potential. This aligns with Section II's DFT formulations and discretization, enabling oracle access for quantum algorithms.

#### Goals:

Construct sparse H (Ng x Ng) with sparsity s (e.g., s=7 for 3D second-order finite difference).
Compute potentials: Solve Poisson for VH with error < 1e-5, evaluate Vxc (e.g., LDA functional) accurately to 1e-6, add Vext based on atomic positions.
Ensure H is Hermitian: Check max(|H - H.T.conj()|) < 1e-10.
Provide oracles: Location oracle L(j, l) returning column of l-th nonzero in row j, value oracle V(j, l) returning the nonzero value.

Inputs:

Current density n on D∆ (array of shape (NI,)).
Grid points D, D∆, shape functions N.
System parameters (Na, Ne, β, µ, atomic positions).

Outputs:

Sparse Hamiltonian matrix H (scipy.sparse.csr_matrix of shape (Ng, Ng)).
Normalization factor for H (to ensure eigenvalues in [-1,1] for QSVT).
Oracle functions: L(row, index) -> col, V(row, index) -> value (both callables).

### Stage 3: Quantum Singular Value Transformation (QSVT) for Density-Matrix Encoding
--- 
#### Purpose: 
Use QSVT to construct a quantum circuit that encodes the density-matrix Γ = f(H), where f is the Fermi-Dirac function. This is the core quantum component from Section III, avoiding explicit diagonalization for linear scaling.
#### Goals:

Approximate f(x) with a polynomial P_d(x) of degree d, ensuring |P_d(x) - f(x)| < ε_poly for x in [-1,1], with ε_poly = 1e-4.
Implement block encoding of normalized H, then apply QSVT to get U such that <0| U |0> ≈ Γ / ||Γ|| (up to normalization).
Simulate circuit on a classical quantum simulator: For a small Ng (e.g., 32), verify trace(Γ) ≈ Ne with error < 1e-3.

Inputs:

Normalized sparse H (from Stage 2).
Fermi-Dirac parameters β, µ.
Approximation parameters: Polynomial degree d, error tolerance ε_poly.
Qubit count m = log2(Ng).

Outputs:

QSVT circuit U (e.g., Qiskit QuantumCircuit object).
Polynomial approximation P_d (callable or coefficients array).
Gate complexity estimate (int, should be O(s * d * log(1/ε_poly))).

### Stage 4: Electron Density Estimation via Quantum Measurements
---

#### Purpose: 

Estimate selected diagonals of Γ (i.e., updated density F(n)) using amplitude amplification and measurements, incorporating statistical error analysis. This matches Section III's estimation method, with randomized block selection for efficiency.

#### Goals:

For a block of size B (<= NI), estimate F(n)[j] for j in block with variance < σ^2 (e.g., σ=1e-3) using M measurements per component.
Achieve overall L2 error ||hat{F}(n) - F(n)|| < ε_est (e.g., 1e-2) with probability > 1-δ (δ=0.01).
Simulate noise: Add Gaussian noise to measurements and verify error bounds hold in 95% of 100 trials.

Inputs:

QSVT circuit U (from Stage 3).
Interpolation points indices to estimate (array of shape (B,)).
Estimation parameters: Number of shots M, failure probability δ, error ε_est.

Outputs:

Estimated density components hat{F}(n) (array of shape (B,)).
Statistical confidence intervals (array of shape (B, 2)).
Query complexity (int, should be O(s * NI / ε_est)).

### Stage 5: Self-Consistent Field (SCF) Iterations with Randomized Block Coordinates
---
#### Purpose: 
Perform hybrid iterations to solve n = F(n), using mixing schemes and randomized block updates to handle measurement noise and accelerate convergence. This integrates Sections II and III, with convergence analysis.

#### Goals:

Update density: For each iteration k, select random block, compute hat{F}, mix with α (e.g., n_{k+1} = α * hat{F} + (1-α) * n_k), ensuring monotonic decrease in ||n_{k+1} - n_k||.
Converge to ||n* - hat{n}|| < ε_scf (e.g., 1e-4) in < K iterations (K ~ log(1/ε_scf)).
Test robustness: Run 50 trials with noise, verify average iterations < theoretical bound O(log(1/ε_scf)), and final energy error < 1e-3 Ha.

Inputs:

Initial n0 (from Stage 1).
Components from previous stages: Hamiltonian builder, QSVT, estimator.
Iteration parameters: Mixing factor α, block size B, max iterations K, tolerance ε_scf, initial γ (||n* - n0|| < γ).

Outputs:

Converged density hat{n} (array of shape (NI,)).
Iteration history: Array of residuals ||n_{k+1} - n_k|| (shape (num_iters,)).
Total query complexity (int, should be O(s * NI / ε_scf)).

### Stage 6: Validation and Numerical Results
---

#### Purpose: 

Reproduce numerical experiments (e.g., 1D Li-H chain), compute energies, and analyze scaling/complexity. This matches Section IV, verifying the implementation against classical DFT baselines.

#### Goals:

Compute ground state energy E = min eigenvalues sum with occupations, error < 1e-2 vs. classical diagonalization.
Plot density like Fig. 1, visual match (L2 difference < 1e-2).
Scaling test: Vary Na from 2 to 20, measure runtime/queries, fit to linear model with R^2 > 0.95.
Error analysis: Quantify function approx, statistical, and iteration errors separately, matching Theorems 1/2 bounds.

Inputs:

Converged density hat{n} (from Stage 5).
Full system parameters.
Baseline classical DFT results (e.g., from SciPy eigh).

Outputs:

Ground state energy E (float).
Density plot data (arrays for r, n(r)).
Complexity metrics: Table of Na vs. queries/time.
Error breakdown: Dict with keys 'poly_approx', 'stat_fluct', 'iter_complex', values as floats.
Outputs:

Fine grid points D (array of shape (Ng, dim), dim=1 or 3).
Interpolation points D∆ (array of shape (NI, dim)).
Initial density vector n0 (array of shape (NI,)).
Shape functions N for interpolation (callable: N(rk - r) for rk in D, r in D∆).

