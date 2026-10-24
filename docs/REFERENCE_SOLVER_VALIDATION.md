# Validation of the finite-difference reference solver

Subject: [`src/pinn_cavity/reference.py`](../src/pinn_cavity/reference.py) (`SOLVER_VERSION = 1`), the solution the network errors are measured against. Stored evidence: [`results/reference_check.json`](../results/reference_check.json), produced by `scripts/verify_reference.py`. Status labels as in [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md#classification).

## 1. Equations

On the unit square with unit lid speed, the steady Navier-Stokes equations in streamfunction-vorticity form are

$$\Delta\psi = -\omega, \qquad u = \psi_y, \quad v = -\psi_x, \qquad u\,\omega_x + v\,\omega_y = \mathrm{Re}^{-1}\,\Delta\omega,$$

with $\omega = v_x - u_y$. Then $\Delta\psi = \psi_{xx} + \psi_{yy} = -v_x + u_y = -\omega$, which is the sign used in the code. The transport residual in the code is `(dy @ psi) * (dx @ omega) - (dx @ psi) * (dy @ omega) - (lap @ omega) / reynolds`, that is $\psi_y\omega_x - \psi_x\omega_y - \mathrm{Re}^{-1}\Delta\omega$. It equals the vorticity equation with $u = \psi_y$, $v = -\psi_x$. Taking the curl of the momentum equation removes the pressure, so the formulation contains exactly the information of the primitive-variable problem except the pressure.

Status: **VERIFIED** by derivation.

## 2. Discretisation

- Uniform grid of $n$ nodes per side, $h = 1/(n-1)$, node index `i * n + j` for $(x_i, y_j)$.
- Second-order central differences for $\partial_x$, $\partial_y$ and the five-point Laplacian, built as Kronecker products.
- The operator rows at boundary nodes are truncated, but they are only used at interior nodes, whose stencils lie within the grid. The corner nodes are not in the stencil of any interior node: the five-point and central-difference stencils are crosses.
- Boundary rows: $\psi = 0$ on all walls, and the vorticity is given by Thom's formula.

Status: **VERIFIED** by reading. `tests/test_reference.py::test_discrete_continuity` confirms that the discrete velocity is divergence-free to $10^{-9}$.

## 3. Wall vorticity (Thom's formula)

On a wall, $\psi = 0$ and the tangential derivative vanishes, so $\omega = -\psi_{nn}$. A Taylor expansion towards the first interior node, at distance $h$ along the inward normal, with $\partial_n\psi = \pm U_{\text{tangential}}$, gives

$$\psi_1 = \psi_0 + h\,\partial_n\psi + \tfrac{h^2}{2}\,\partial_{nn}\psi + O(h^3).$$

On the lid ($y = 1$, inward normal $-y$, $\psi_y = u = 1$) this is $\psi_{N-1} = -h - \tfrac{h^2}{2}\omega_N + O(h^3)$, so

$$\omega_{\text{lid}} = -\frac{2\psi_{N-1}}{h^2} - \frac{2U}{h}.$$

On the fixed walls the same expansion gives $\omega = -2\psi_1/h^2$. These are the expressions in `_wall_vorticity`: entries $-2/h^2$ to the adjacent node, and $-2/h$ added on the lid rows only.

The formula is first order locally. It is known to give second-order accuracy of the interior solution (Thom 1933; the standard analyses of the streamfunction-vorticity method). The observed order in the interior (section 6) is 2.0.

The wall loop runs over the inner nodes of each wall only. The four corner nodes get the row $\omega = 0$, which does not enter any interior equation (section 2).

Status: **VERIFIED**.

## 4. Velocity at the walls and the lid corners

After the solve, $u = \psi_y$ and $v = -\psi_x$ are formed by central differences at interior nodes, and the wall values are overwritten with the boundary conditions. The assignment order `u[0, :] = u[-1, :] = u[:, 0] = 0`, then `u[:, -1] = 1`, gives the two lid corners the lid value $u = 1$.

This is a convention: the boundary velocity has no value at a point where it jumps from 1 to 0. It affects two nodes of 263 169. `tests/test_reference.py::test_boundary_conditions` records it. Its influence on the network metrics is quantified in [SCIENTIFIC_AUDIT.md, F10](SCIENTIFIC_AUDIT.md#f10-lid-corners).

Status: **VERIFIED**.

## 5. Newton iteration

The unknowns are $(\psi, \omega)$ at all $n^2$ nodes. The residual is $F = (F_\psi, F_\omega)$, with interior rows $\Delta_h\psi + \omega$ and $\psi_y\omega_x - \psi_x\omega_y - \mathrm{Re}^{-1}\Delta_h\omega$, and boundary rows $\psi$ and $\omega - A\psi - b$.

Exact Jacobian blocks:

$$J_{\psi\psi} = P_b + P_i\,\Delta_h, \quad J_{\psi\omega} = P_i, \quad J_{\omega\psi} = -A + P_i\bigl(\mathrm{diag}(\omega_x) D_y - \mathrm{diag}(\omega_y) D_x\bigr), \quad J_{\omega\omega} = P_b + P_i\bigl(\mathrm{diag}(\psi_y) D_x - \mathrm{diag}(\psi_x) D_y - \mathrm{Re}^{-1}\Delta_h\bigr),$$

with $P_i$, $P_b$ the projections on interior and boundary rows. These are the blocks `j_pp`, `j_po`, `j_op`, `j_oo` in the code. Each step solves $J\,\delta = -F$ with a sparse LU factorisation (SuperLU). Iteration stops when $\lVert F\rVert_\infty < 10^{-8}$.

For Re above 100, the solve is continued from the solution at 100 (and 400): the sequence 100 → 400 → 1000 keeps every Newton start inside the basin of quadratic convergence.

Observed on 513 nodes (`results/reference_check.json`):

| Re | Newton iterations per continuation stage | Final $\lVert F\rVert_\infty$ | Solve time (s), shared machine |
|---|---|---|---|
| 10 | 5 | $4.2 \times 10^{-9}$ | 52 |
| 100 | 6 | $2.7 \times 10^{-10}$ | 66 |
| 400 | 6, 6 | $8.7 \times 10^{-11}$ | 262 |
| 1000 | 6, 6, 8 | $3.6 \times 10^{-11}$ | 200 |

The residual reduction is quadratic: `tests/test_reference.py::test_newton_converges_quadratically` requires at most 8 iterations to $10^{-8}$ on 65 nodes. The algebraic error is therefore negligible against the discretisation error.

Status: **VERIFIED**.

## 6. Grid convergence

Let $d_k$ be the relative $L^2$ velocity difference between the grid with $k$ nodes and the 513-node grid, on the nodes of the coarser grid. If the field error behaves like $C h^p$, then $d_{129}/d_{257} = 2^p + 1$, which gives the observed order $p$. The error of the 513-node solution is then about $d_{257}/(2^p - 1)$.

| Re | Region | $d_{129}$ | $d_{257}$ | Ratio | Observed order $p$ | Estimated error of the 513 grid |
|---|---|---|---|---|---|---|
| 10 | whole cavity | 1.02% | 0.384% | 2.67 | 0.74 | 0.58% |
| 10 | core (0.1 to 0.9) | 0.111% | 0.022% | 5.06 | 2.02 | 0.007% |
| 100 | whole cavity | 1.04% | 0.383% | 2.71 | 0.78 | 0.54% |
| 100 | core | 0.285% | 0.056% | 5.04 | 2.01 | 0.019% |
| 400 | whole cavity | 1.52% | 0.429% | 3.54 | 1.35 | 0.28% |
| 400 | core | 1.36% | 0.269% | 5.05 | 2.02 | 0.088% |
| 1000 | whole cavity | 3.09% | 0.703% | 4.39 | 1.76 | 0.29% |
| 1000 | core | 3.08% | 0.614% | 5.02 | 2.01 | 0.20% |

- **Core.** The ratio is 5.0 at every Reynolds number, the value $2^2 + 1$ of a second-order method in the asymptotic range. Status: **VERIFIED**. (The README states this as "shrinks by a factor of 5.0 to 5.1", without naming the order; it is exactly the second-order signature.)
- **Whole cavity.** The order is reduced to 0.7 to 1.8 by the singularity at the lid corners, where the vorticity behaves like $1/r$ and the velocity is discontinuous. A single-order Richardson estimate is approximate here.
  - At Re = 10 and 100 the estimated error of the 513 grid is 0.54% to 0.58%, above the README's "about 0.4%", which quotes $d_{257}$.
  - At Re = 400 and 1000 the estimate is below $d_{257}$.

  Status of the README's figure: **SUPPORTED BUT LIMITED**. A defensible statement is "below 0.6% over the whole cavity and below 0.1% in the core at Re ≤ 400".
- **Relevance.** The smallest network error is 38.3%, at least 60 times the largest reference error, so the reference error changes no reported number by more than about 0.2 percentage points.

## 7. Comparison with published benchmarks

Extrema of the centreline profiles are found by cubic splines through the nodal values, then Richardson-extrapolated from the 257 and 513 grids with $p = 2$, the observed interior order. The published values are:

- Re = 1000: the Chebyshev spectral solution of Botella and Peyret (1998);
- Re = 100: the extrapolated 1024 × 1024 finite-volume solution of Marchi, Suero and Araki (2009).

| Re | Quantity | 513 grid minus published | Extrapolated minus published |
|---|---|---|---|
| 100 | $\min u$ on the vertical centreline | $3.6 \times 10^{-5}$ | $-7.4 \times 10^{-7}$ |
| 100 | $\max v$ on the horizontal centreline | $-4.2 \times 10^{-5}$ | $1.7 \times 10^{-7}$ |
| 100 | $\min v$ on the horizontal centreline | $5.8 \times 10^{-5}$ | $-2.7 \times 10^{-7}$ |
| 1000 | $\min u$ | $8.1 \times 10^{-4}$ | $-5.1 \times 10^{-6}$ |
| 1000 | position of $\min u$ | $2.6 \times 10^{-4}$ | $-4.2 \times 10^{-6}$ |
| 1000 | $\max v$ | $-8.3 \times 10^{-4}$ | $7.8 \times 10^{-6}$ |
| 1000 | $\min v$ | $1.1 \times 10^{-3}$ | $-5.0 \times 10^{-6}$ |
| 1000 | $\min \psi$ | $2.1 \times 10^{-4}$ | $-1.7 \times 10^{-6}$ |

Agreement to $10^{-5}$ after extrapolation, with the error of the unextrapolated 513 grid falling by two orders of magnitude, is strong evidence that the solver converges to the correct solution of the continuous problem.

Status: **VERIFIED**. The published values in `scripts/verify_reference.py` were checked against the values commonly cited for these benchmarks; they were not re-read from the original articles in this audit.

**Ghia, Ghia and Shin (1982).** The 17 tabulated values per profile at Re = 100, 400, 1000 agree with the 513-node solution:

- $u$: RMS 0.0012 to 0.0029, maximum 0.0053;
- $v$: RMS 0.0028 to 0.0087, maximum 0.017, with one value excluded;
- the excluded value, $v(0.9063)$ at Re = 400, differs by 0.151 on every grid and from its neighbours in the table;
- the minimum of $\psi$ agrees to $9 \times 10^{-5}$ at Re = 100 and $5 \times 10^{-6}$ at Re = 400; at Re = 1000 Ghia's value is $1.0 \times 10^{-3}$ above Botella and Peyret's, and the 513-node minimum lies between the two, with the extrapolated value within $2 \times 10^{-6}$ of Botella and Peyret.

Ghia's multigrid solution on 129 × 129 nodes is less accurate than the spectral benchmark. The repository correctly uses it as a secondary check only. Status: **VERIFIED**.

## 8. Mapping to the cavity of the example

The dimensionless solution is mapped by $x = (x^\ast - \tfrac12)L$, $y = (y^\ast - \tfrac12)L$, $\mathbf{u} = U\mathbf{u}^\ast$, $\psi = UL\psi^\ast$. The dimensionless problem depends only on $\mathrm{Re} = UL/\nu$, so this is the solution of the dimensional problem of the example with $\nu = UL/\mathrm{Re}$. The lid of the reference ($j = n - 1$) maps to $y = +L/2$, the lid of the PINN. `tests/test_reference.py::test_cache_and_scaling` checks the coordinates and the velocity scaling with $U = 2$.

The cache key covers the solver version, Re, $n$, $L$ and $U$, but not $H$; a non-square cavity is rejected. Status: **VERIFIED**.

## 9. Reproduction in this audit

- **513-node references.** Recomputed from scratch for Re = 10, 100 and 400. They are bitwise identical to the files the reported runs were scored against: maximum absolute difference 0 in $u$, $v$ and $\psi$. The Newton residuals are identical too.
- **Verification script.** `scripts/verify_reference.py` (Re = 10, 100, 400, 1000 on 129, 257 and 513 nodes) was re-run into a scratch directory. Every value of the new `reference_check.json` except the timings equals the stored file exactly: maximum absolute difference 0 over the grid convergence, extrema, Richardson values, Ghia comparisons and profiles. The Newton iteration counts are identical.

Status: **VERIFIED**. The stored evidence reproduces exactly on the same machine and environment.

## 10. What the reference does not provide

- **No pressure.** The network pressure is not evaluated. A pressure could be recovered from $\psi$ by solving the pressure Poisson equation $\Delta p = 2(u_x v_y - u_y v_x)$ with Neumann data from the momentum equation (see [EXPERIMENT_GAPS.md](EXPERIMENT_GAPS.md)).
- **No regularised-lid comparison.** The PINN loss effectively regularises the lid condition near the corners, through the weight $w$. The reference solves the classical discontinuous problem. Both converge to the same solution away from the corners, which F10 of the audit confirms for the error measure; a regularised-lid reference was not computed.
- **No independent second solver.** Independence rests on the agreement with the spectral benchmark at Re = 1000 and with the finite-volume benchmark at Re = 100, not on a second implementation for Re = 10 and 400.
  - At Re = 400 the solver lies between two verified Reynolds numbers, and the grid convergence there is clean.
  - At Re = 10 no high-accuracy published benchmark was used.
