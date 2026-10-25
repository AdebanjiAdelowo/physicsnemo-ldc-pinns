# Scientific audit of the Reynolds-number study

Scope: the mathematics, the numerics and the software of this repository at commit `5a3c584` (the state in which the results were published), the NVIDIA upstream example at commit `b45a5c8`, the ten trained networks of the two reported studies and the conclusions drawn from them. Companion documents:

- [UPSTREAM_COMPARISON.md](UPSTREAM_COMPARISON.md): the recipe against the upstream `train.py`, element by element;
- [REFERENCE_SOLVER_VALIDATION.md](REFERENCE_SOLVER_VALIDATION.md): the finite-difference reference;
- [EXPERIMENT_GAPS.md](EXPERIMENT_GAPS.md): what has not been run, with commands and resource estimates;
- [FULL_BUDGET_STUDY.md](FULL_BUDGET_STUDY.md): the study at the upstream budget, run after this audit (see the [addendum](#addendum-the-study-at-the-upstream-budget)).

## Classification

| Label | Meaning |
|---|---|
| VERIFIED | Checked independently in this audit, by derivation, by recomputation or by a test that passes. |
| SUPPORTED BUT LIMITED | Correct as stated, but the evidence is narrower than a reader might assume (one seed, one budget, a noisy estimate, a definition that matters). |
| UNVERIFIED | Not established by the available evidence. It may be true. |
| INCORRECT | Contradicted by the code, the data or a derivation. Fixed where it was a software defect. |

## How the audit was carried out

1. Every source file, configuration, test and result file was read. The upstream example was read from a clean local clone at the pinned commit, and the adapted parts were compared mechanically (`NavierStokes` body, upstream part of `configs/official.yaml`).
2. A project-local environment was built with `uv` at the package versions recorded in `results/*/study_metadata.json` (Python 3.12.2, PyTorch 2.14.1, NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0, PhysicsNeMo at `b45a5c8`); `uv.lock` pins it.
3. The test suite was run before (46 passed) and after the changes (52 passed), on CPU, without network access or GPU.
4. The ten checkpoints of the reported runs (`official/re0010_seed0` and the nine `local` runs) were evaluated on CPU from their original files, read in place and not modified. The three 513-node reference solutions were recomputed from scratch. `scripts/audit.py` does both and writes [`results/audit/audit.json`](../results/audit/audit.json).
5. The reference verification of `scripts/verify_reference.py` was re-run into a scratch directory and compared with the stored `results/reference_check.json`.
6. No training was run. No CUDA workload was started.

```bash
uv sync
uv run python -m pytest -q
uv run python scripts/audit.py --runs-root <directory with official/ and local/ run directories>
```

## Summary of findings

| # | Finding | Status |
|---|---|---|
| F1 | Under the loss of the example, a hand-written field that is not the cavity flow (a shear layer under the lid above fluid at rest) has a lower training loss than every trained network at Re = 100 and 400, and a lower velocity error. The cost of such a field falls like $\nu^2$. Low loss is not evidence of a correct solution under this formulation. | VERIFIED |
| F2 | The run `re0100_seed2` collapsed to the constant field $(u, v) = (1/3, 0)$: the last hidden layer is saturated (509 of 512 neurons output exactly zero), the residuals vanish identically, and the loss sits at the analytic optimum $1/3$ for a constant field. It is a degenerate state of the optimisation, not an evaluation error. | VERIFIED |
| F3 | Every stored metric of the ten runs is reproduced from the checkpoints on CPU (largest relative difference $2.4 \times 10^{-5}$); the reference solutions recompute bitwise identically. | VERIFIED |
| F4 | Reduced-study mean velocity errors 0.462, 1.072, 0.946 at Re = 10, 100, 400. | VERIFIED arithmetic; SUPPORTED BUT LIMITED as a summary (F4b) |
| F5 | Governing equations, viscosity-Reynolds relation, domain, boundary conditions and nondimensionalisation of residuals. | VERIFIED |
| F6 | Network, residuals, sampling, loss weights, optimiser and scheduler of `official` equal the upstream example. | VERIFIED ([UPSTREAM_COMPARISON.md](UPSTREAM_COMPARISON.md)) |
| F7 | The reference solver: equations, Thom wall vorticity, Newton iteration, second-order convergence in the interior, agreement with Botella and Peyret (1998) and Marchi et al. (2009) to $10^{-5}$ after extrapolation. | VERIFIED ([REFERENCE_SOLVER_VALIDATION.md](REFERENCE_SOLVER_VALIDATION.md)) |
| F8 | "The error of the reference itself is about 0.4%." | SUPPORTED BUT LIMITED: 0.3% to 0.6% over the whole cavity by a Richardson estimate with the observed order; at least 60 times below every network error |
| F9 | Evaluation on the 513-node grid without interpolation of the network; centreline and Ghia comparisons at the right coordinates. | VERIFIED |
| F10 | Lid corners: the stored boundary errors count the two corner nodes both as lid and as wall. The velocity errors are not driven by the corners. | Metric definition SUPPORTED BUT LIMITED (fixed by added metrics); error conclusions VERIFIED |
| F11 | The loss values in the tables and the "jumps by factors of 6 to 47" are single minibatch draws; the low-variance loss differs from them by up to a factor of 1.8. | SUPPORTED BUT LIMITED |
| F12 | The networks at Re = 100 and 400 are not at a good minimum of their own loss: their loss is 30 to 660 times that of the lid-layer field and is dominated by the wall terms. | VERIFIED |
| F13 | The Kaggle launcher cloned `physicsnemo-pinn-cavity` instead of this repository. | INCORRECT (fixed) |
| F14 | "Python 3.11 to 3.14": the recorded environment requires Python 3.12 or later (NumPy 2.5.3). | INCORRECT (fixed) |
| F15 | That the failure at Re = 100 and 400 persists at the upstream budget. | VERIFIED for the nine runs of `configs/full.yaml` (addendum, F18) |
| F16 | Which feature of the recipe causes the failure (distance weighting, unnormalised inputs, loss balance, optimiser). | UNVERIFIED as a cause; F1 shows that the loss formulation admits the observed failure mode at any budget |
| F17 | Pressure. | UNVERIFIED: not evaluated, the reference does not compute it |
| F18 | At the upstream budget no run at Re = 100 or 400 recovers the cavity flow (errors 0.66 to 0.90; primary vortex absent or at most 19% of the reference strength). At Re = 10 the error falls in two of three seeds. | VERIFIED (addendum) |
| F19 | The lid-layer counterexample of F1 still matches or beats every trained network at Re = 100 and 400 at the upstream budget. | VERIFIED (addendum) |
| F20 | The constant-field collapse of F2 recurs at the upstream budget. | Not reproduced: 0 of 9 runs (addendum) |
| F21 | The Kaggle launcher ran on Kaggle as written. | INCORRECT (fixed): three environment and memory defects, found and fixed while running the study (addendum) |

## F1. Low weighted loss with large physical error

**Setting.** The training loss of the example is

$$\mathcal{L} = \sum_{r} \overline{(d\,r)^2} + \overline{u^2}\big|_{\text{walls}} + \overline{v^2}\big|_{\text{walls}} + \overline{w\,(u - U)^2}\big|_{\text{lid}} + \overline{v^2}\big|_{\text{lid}},$$

with $d$ the distance to the nearest wall and $w = 1 - 2|x|/L$. The exact solution has $\mathcal{L} = 0$. The question is whether fields far from it can have a small $\mathcal{L}$.

**Construction.** Take, with $\eta = L/2 - y$ the distance below the lid,

$$u = U\,g(x)\,e^{-\eta/\delta}, \qquad v = 0, \qquad p = 0, \qquad g(x) = \bigl(1 - e^{-(L/2 - x)/\varepsilon}\bigr)\bigl(1 - e^{-(L/2 + x)/\varepsilon}\bigr).$$

It satisfies $u = 0$ on the side walls, $u \approx 0$ on the bottom, $u = U g(x)$ on the lid, and $v = 0$ everywhere. Below a layer of thickness $\delta$ the fluid is at rest: there is no vortex and no return flow. It is not the cavity flow at any Reynolds number. ([`src/pinn_cavity/diagnostics.py`](../src/pinn_cavity/diagnostics.py), `LidLayer`.)

**Cost.** Away from the side walls, $g = 1$, $u_x = 0$, so continuity holds and the $x$ momentum residual is the viscous term alone, $-\nu U \delta^{-2} e^{-\eta/\delta}$. With $d = \eta$ and uniform points,

$$\overline{(d\,r_x)^2} \approx \frac{1}{L}\int_0^\infty \eta^2\,\frac{\nu^2 U^2}{\delta^4}\,e^{-2\eta/\delta}\,d\eta = \frac{\nu^2 U^2}{4 L \delta}.$$

The factor $d^2$ cancels the $\delta^{-4}$ growth of the squared viscous term near the lid up to one power of $\delta$. The lid term is about $\varepsilon^2 / L^2$, and the remaining terms live in the $\varepsilon \times \delta$ corner regions. The cost of the field therefore falls like $\nu^2 = (UL/\mathrm{Re})^2$, while its velocity error does not depend on $\nu$ at all.

**Check of the estimate.** At $\delta = \varepsilon = L/20$ the measured $x$ momentum term is $3.8 \times 10^{-2}$, $3.9 \times 10^{-4}$ and $3.5 \times 10^{-5}$ at Re = 10, 100 and 400. The ratio from Re = 10 to 100 is 97, against $10^2$ predicted. The formula gives $5.0 \times 10^{-4}$ at Re = 100, against $3.9 \times 10^{-4}$ measured; the difference is the region near the side walls where $g < 1$. From Re = 100 to 400 the ratio is 11 instead of 16, because a $\nu$-independent convective term from the corner regions starts to dominate.

**Measurement.** Each field is scored with `engine.loss_terms`, the upstream loss, at 400 000 interior and 200 000 boundary points, and its velocity error is computed on the 513-node reference. The best of a $6 \times 5$ grid over $\delta/L \in [0.002, 0.1]$ and $\varepsilon/L \in [0.002, 0.05]$:

| Re | Lid-layer field: loss | Lid-layer field: velocity error | Trained networks: loss (low-variance estimate) | Trained networks: velocity error |
|---|---|---|---|---|
| 10 | $2.0 \times 10^{-2}$ | 0.725 | $1.3 \times 10^{-3}$ (upstream budget); $2.2$ to $4.3 \times 10^{-3}$ (reduced) | 0.383; 0.456 to 0.473 |
| 100 | $7.0 \times 10^{-4}$ | 0.747 | $2.1$ to $2.6 \times 10^{-2}$; collapsed run 0.334 | 0.802, 0.805; 1.607 |
| 400 | $9.5 \times 10^{-5}$ | 0.834 | $2.0 \times 10^{-2}$ to $6.3 \times 10^{-2}$ | 0.871 to 1.078 |

**Conclusions.**

- At Re = 10 the trained networks have a loss 5 to 15 times lower than the best lid-layer field, and a lower error: the loss separates them.
- At Re = 100 and 400 the order is reversed. A field with no recirculation has a loss 30 to 37 times (Re = 100) and 210 to 660 times (Re = 400) below the non-collapsed trained networks, and a velocity error below theirs. The trained networks at these Reynolds numbers are no better than this two-parameter field.
- Because the exact solution has loss 0 and fields with 75% to 85% error have loss $10^{-4}$ to $10^{-3}$, a training loss in that range certifies nothing about the velocity field. The discriminating power of the loss falls like $\mathrm{Re}^{-2}$ for a fixed layer thickness.
- This is a property of the loss formulation of the example (the distance factor on the residuals together with unit-weight boundary terms), not of the reduced budget. It holds for any optimiser and any number of steps.

`tests/test_audit.py::test_a_field_that_is_not_the_cavity_flow_beats_the_trained_networks_at_re100_and_re400` keeps this result.

Status: **VERIFIED**. This answers task 8 of the audit brief: under the chosen loss, low weighted loss and large physical error coexist, and the effect grows with the Reynolds number.

What F1 does not show: that gradient descent is attracted to this particular field, or that a better optimiser would find the exact solution. Both are open (F16).

## F2. The collapsed run

Run `re0100_seed2` follows the other seeds until step 4000 (tracked error 0.75 to 0.82). Between steps 4000 and 4100 the recorded loss jumps from 0.020 to 0.46 and the residual terms drop to $10^{-27}$, then to exactly zero.

Evidence from the checkpoint at step 5000:

- **Constant output.** Over the $129 \times 129$ nodes the spread of $u$ is zero and the mean is $0.33300$. The spread of $v$ and $p$ is about $10^{-24}$.
- **Saturated last hidden layer.** In the sixth hidden layer, 509 of 512 neurons output exactly zero at every node, and the other three are constant. Layers 1 to 5 still vary over the cavity. With SiLU, $x\,\sigma(x)$ underflows to zero for large negative pre-activations. No gradient then reaches the earlier layers, and only the bias of the output layer still learns.
- **Analytic optimum.** For a constant field $(c, 0)$ all residuals vanish identically. The loss is $c^2 + (c - 1)^2\,\overline{w} = c^2 + (c-1)^2/2$, minimal at $c = 1/3$ with value $1/3$. The low-variance loss of the checkpoint is $0.3338 \pm 0.0010$: $u_{\text{wall}}$ term $0.111 = 1/9$, lid term $0.221 = 2/9$. The trained output bias has converged to the optimum of the only family the network can still represent.
- **Error.** The velocity error of the constant field $(0.333, 0)$ equals that of the run to $10^{-5}$ relative. Both are 1.607 on 513 nodes and 1.582 on every fourth node.

The exactly zero residuals are therefore genuine: they are the residuals of a constant function, computed correctly by autograd. They are not a bug in `PhysicsInformer` or in the evaluation. The residual code is checked against an analytic field in `tests/test_engine.py`.

What is not established: why the layer saturated. The likely mechanism is a large Adam update during a loss spike, which drove the last-layer pre-activations strongly negative. The checkpoint holds the optimiser state at step 5000 only, so the event cannot be replayed exactly. Status of the mechanism: **UNVERIFIED**.

Status of the diagnosis: **VERIFIED**. The run is correctly kept in the tables. Its inclusion is what makes the Re = 100 mean exceed the Re = 400 mean (F4b).

## F3. Recomputation of the stored results

All metrics of `eval_metrics.json`, except inference time, were recomputed on CPU from the original checkpoints against freshly solved references.

| Study | Runs | Largest relative difference over all metrics |
|---|---|---|
| official | 1 | $1.5 \times 10^{-6}$ |
| local | 9 | $2.4 \times 10^{-5}$ (`centreline_u_min`, Re = 100, seed 1) |

Differences are those of float32 inference on MPS against CPU. `results/audit/audit.json` was produced from a clean checkout of commit `ebfbab3` (`git_dirty: false` in its metadata); a second, independent execution gave identical values in every field. The three reference solutions (Re = 10, 100, 400 on 513 nodes) recompute with a maximum absolute difference of exactly 0 from the files the runs were scored against. Status: **VERIFIED**.

## F4. The reported means

| Re | Seeds | Mean | Median | Sample standard deviation | Range | Mean without the collapsed run |
|---|---|---|---|---|---|---|
| 10 | 0.473, 0.456, 0.456 | 0.462 | 0.456 | 0.010 | 0.456 to 0.473 | 0.462 |
| 100 | 0.802, 0.805, 1.607 | 1.072 | 0.805 | 0.464 | 0.802 to 1.607 | 0.804 |
| 400 | 1.078, 0.871, 0.889 | 0.946 | 0.889 | 0.115 | 0.871 to 1.078 | 0.946 |

**F4a. Arithmetic.** VERIFIED.

**F4b. As a summary.** SUPPORTED BUT LIMITED:

- With three seeds and one collapsed run, the Re = 100 mean is set by that run. The ordering of the means (100 above 400) does not hold for the medians. No statement about how the error depends on Re between 100 and 400 should rest on the means.
- A velocity error of 1.0 is that of the zero field. All six runs at Re = 100 and 400 are within 0.80 to 1.61 of the reference, and the errors in the central part of the cavity are 1.00 to 2.15. These runs carry no more information about the interior flow than the zero field. The defensible statement is: "at Re = 100 and 400 the reduced runs did not find the recirculating flow", not "the error was about 100%".
- Three seeds give a sample standard deviation with very wide uncertainty. The spread over seeds is a description of these runs, not an estimate of a population variance.

## F5. Equations, parameters and nondimensionalisation

| Item | Check | Status |
|---|---|---|
| Momentum | $(\mathbf{u}\cdot\nabla)\mathbf{u} + \rho^{-1}\nabla p - \nu\Delta\mathbf{u}$, steady, $\rho = 1$ (symbolic test and analytic-field test) | VERIFIED |
| Continuity | $u_x + v_y$ | VERIFIED |
| Reynolds number | $\mathrm{Re} = UL/\nu$ with $L$ the side length 0.1 and $U = 1$, the definition of Ghia et al. and Botella and Peyret. $\nu$ = 0.01, 0.001, 0.00025 exactly in floating point. | VERIFIED |
| Domain | $(-L/2, L/2)^2$, lid at $y = +L/2$ in both the PINN and the mapped reference | VERIFIED |
| Boundary conditions | $\mathbf{u} = (U, 0)$ on the lid, $\mathbf{u} = 0$ on the other walls; the lid corners are left to the weight $w$ in the loss and assigned the lid value in the reference | VERIFIED; corners discussed in F10 |
| Mapping of the reference | $x = (x^\ast - 1/2)L$, $\mathbf{u} = U\mathbf{u}^\ast$, $\psi = UL\psi^\ast$. The dimensionless problem depends on Re only, so the reference at given Re is the solution of the dimensional problem with $\nu = UL/\mathrm{Re}$ | VERIFIED |
| Residual units | continuity scaled by $L/U$, momentum by $L/U^2$ | VERIFIED |
| Training variables | the network is trained in dimensional variables on $L = 0.1$ without input normalisation. The PDE is invariant under rescaling, but the loss is not: the residual weights, the input range and the relative size of the terms change with $L$ and $U$. This is a property of the upstream recipe and is reported, not changed | VERIFIED as described |

## F6. Agreement with the upstream example

See [UPSTREAM_COMPARISON.md](UPSTREAM_COMPARISON.md). Status: **VERIFIED**.

## F7, F8. Reference solver

See [REFERENCE_SOLVER_VALIDATION.md](REFERENCE_SOLVER_VALIDATION.md). Summary:

- The streamfunction-vorticity equations, Thom's formula and the Newton Jacobian were derived again, and the code agrees with them.
- Newton converges in 5 to 8 iterations to $\le 4.3 \times 10^{-9}$.
- In the core of the cavity the observed order is 2.01 to 2.02 at every Reynolds number.
- Richardson-extrapolated extrema agree with Botella and Peyret (Re = 1000) and Marchi et al. (Re = 100) to $\le 8 \times 10^{-6}$.

Over the whole cavity the observed order is 0.7 to 1.8 because of the corner singularity. The error of the 513-node solution in the norm used for the networks is estimated at 0.28% to 0.58%, slightly above the README's "about 0.4%" at Re = 10 and 100 (F8: SUPPORTED BUT LIMITED). It is at least 60 times smaller than every network error, so no conclusion depends on it.

## F9. Evaluation grid and interpolation

- The network is evaluated at the 513 × 513 reference nodes (263 169 points, `indexing="ij"` on both sides). Neither field is interpolated. The prediction and the reference share the layout `[i, j]` for $(x_i, y_j)$; `grid_points` and the reshape in `evaluate_run` agree.
- Centrelines are the node lines `n // 2` (n = 513 is odd), at $x = 0$ and $y = 0$.
- The comparisons with Ghia et al. evaluate the network at the 17 tabulated coordinates mapped by $(t - 1/2)L$, without interpolation. The reference is compared with the same table by linear interpolation between 513 nodes. The interpolation error is $O(h^2 \lvert u'' \rvert / 8) \approx 10^{-5}$, two orders below the reported differences.
- The error tracked during training uses every fourth node (129 × 129). The last tracked value equals the final evaluation in `tests/test_pipeline.py` up to that subsampling.
- `fields.npz` stores every fourth node for plotting only.
- The relative $L^2$ error weights all nodes equally. Trapezoidal weights, the quadrature of the continuous norm, change the official error from 0.3833 to 0.3857 and no reported error by more than 0.008.

Status: **VERIFIED**.

## F10. Lid corners

The boundary velocity jumps from $U$ to 0 at the two lid corners. Three components treat the corners differently:

- the loss gives the lid condition weight $w = 0$ there, while the side-wall condition keeps weight 1 up to the corner;
- the reference assigns the lid value $u = U$ to the corner nodes;
- the stored metrics `lid_u_rms_error` and `no_slip_rms_error` both include the two corner nodes, with the opposite targets $u = U$ and $u = 0$.

Effect on the boundary metrics. A network that is exact everywhere except the corners would show a lid error of $\sqrt{2/513} = 0.062$. The added metrics leave both corner nodes out. For the trained runs, the lid error changes by at most 0.009 (official: 0.129 to 0.120) and the wall error by at most 0.007.

At Re = 100 and 400 the non-collapsed networks have $u = 0.72$ to 1.14 at the corners; at Re = 10 they have 0.06 to 0.51. The higher-Re networks keep the lid velocity along the top of the side walls, which violates the no-slip condition there.

Effect on the velocity errors. Removing every node within $0.05L$ of a lid corner changes the official error from 0.383 to 0.375. The largest change over all runs is 0.012, and for the collapsed run the error without the corners is larger. The large errors are not a corner artefact. The reported "core" error (at least $0.1L$ from every wall) already shows this.

Fix. `engine.corner_robust_metrics` adds `velocity_rel_l2_trapezoid`, `velocity_rel_l2_without_lid_corners_r{0.02,0.05,0.1}`, `no_slip_rms_error_without_corners` and `lid_u_rms_error_without_corners` to every evaluation and to the summary. The stored records are unchanged. The audit values for the reported runs are in `results/audit/audit.json`.

## F11. Single-batch loss records

The training loop records the loss of the current minibatch: 1000 interior and 500 boundary points in the reduced study. `audit.py` estimates the expected loss of each final network from 25 independent batches at the upstream point counts.

| Run | Recorded final loss | Low-variance estimate |
|---|---|---|
| official, Re 10 | $1.33 \times 10^{-3}$ | $(1.289 \pm 0.012) \times 10^{-3}$ |
| local, Re 10, seeds 0, 1, 2 | 4.95, 2.81, 4.44 $\times 10^{-3}$ | 4.31, 2.16, 3.02 $\times 10^{-3}$ |
| local, Re 100, seeds 0, 1, 2 | 0.0246, 0.0350, 0.348 | 0.0260, 0.0213, 0.334 |
| local, Re 400, seeds 0, 1, 2 | 0.112, 0.0255, 0.0518 | 0.0626, 0.0203, 0.0382 |

The recorded values are unbiased draws, but individually they are off by up to a factor of 1.8; the cause of the large variance at these point counts was not analysed. Two README statements rest on these records:

- "losses jump by factors of 6 to 47 between consecutive records": measures increases between single draws, so part of it is sampling noise (SUPPORTED BUT LIMITED);
- "the upstream run ends with a loss 3.7 times lower than the reduced run": the ratio is 3.3 with the low-variance estimates; the conclusion does not change.

## F12. Where the loss sits at Re = 100 and 400

The low-variance loss of the non-collapsed runs at Re = 100 and 400 is 0.020 to 0.063:

- 94% to 99% of it is the $u$ wall and lid terms;
- the three residual terms together are $8 \times 10^{-6}$ to $1.4 \times 10^{-3}$.

The lid-layer field of F1 reaches $7 \times 10^{-4}$ and $1 \times 10^{-4}$. So the trained networks fail twice:

1. they do not reach the low-loss region their own loss admits (an optimisation or representation failure, plausibly related to the thin, high-gradient layer the boundary conditions demand from a network with unnormalised inputs in $[-0.05, 0.05]$);
2. even that region does not contain only the correct solution (F1).

The first point is VERIFIED as a measurement. Its cause is UNVERIFIED.

## F13, F14. Software defects

- **Kaggle launcher (F13, INCORRECT, fixed).** `kaggle/run_cuda.ipynb` cloned `https://github.com/AdebanjiAdelowo/physicsnemo-pinn-cavity.git`, a different public repository with a different history, and checked out the SHA given as `REF`. For a commit that exists only in this repository the checkout fails. For a shared older commit it would run code that is not this repository's. The launcher now clones this repository. `CITATION.cff` records `repository-code`, and `tests/test_notebook.py::test_clones_this_repository` ties the two together.
- **Python version (F14, INCORRECT, fixed).** NumPy 2.5.3, the recorded version, requires Python 3.12 or later. `pyproject.toml` now declares the dependencies at the recorded versions with `requires-python >= 3.12`, and `uv.lock` pins the environment.

## Items checked without a finding

- **Seeding and reproducibility.** The network is initialised on CPU, then moved, and the collocation points are drawn on the device. Error tracking draws no random numbers, so it does not perturb the trajectory.
- **Learning rate.** It decays to $0.8796 \times 10^{-3}$ after 10 000 steps and $0.9379 \times 10^{-3}$ after 5000, as stated.
- **Residual evaluation.** It uses the centres of 128 × 128 cells, without the distance weight, which avoids the walls and the corner singularity of the reference.
- **Ghia et al. tables.** All values agree with the solver to 0.017 or better, except $v(x = 0.9063)$ at Re = 400, which differs by 0.151. An isolated transcription error would show up as such an outlier, so the rest of the table is consistent. That one value matches the commonly reproduced table, and it could not be checked against the original article in this audit. The repository treats it correctly: kept, shown, and excluded from the summary range.
- **README history claims.** The tracked error falls from 64% at step 1000 to 37% at step 4000 and stays between 31% and 38%. The 1000-step mean loss is $3.8 \times 10^{-3}$ for steps 3001 to 4000 and $1.6 \times 10^{-3}$ for steps 9001 to 10000. The wall terms at Re = 100 and 400 are 15 to 72 times those at Re = 10. All reproduced from `history.csv`.

## Assessment of the stated conclusion

> These results suggest limitations of the tested training recipe, not a general failure of PINNs.

**SUPPORTED BUT LIMITED**, and stronger than before in one respect:

- **Correctly scoped.** Only one network, one loss and one optimiser were tested, at two budgets, and one of them was reduced. No general statement about PINNs follows, and none is made.
- **Strengthened.** F1 identifies a mechanism inside the tested recipe. The distance-weighted residual makes thin lid layers cheap, by a factor $\nu^2$, so the loss cannot distinguish the cavity flow from a non-recirculating field at Re of order 100. This is an analytic property of the loss, independent of budget, and it is consistent with the observed failure.
- **Still open.**
  - Whether the upstream budget changes the outcome at Re = 100 and 400 (F15).
  - Which change of recipe would fix it (F16).
  - Whether the trained networks are drawn towards lid-layer states or fail for another reason (F12).

None of these has been tested, and the conclusion should not imply that they have.

## Changes made in this audit

| Change | Reason |
|---|---|
| `src/pinn_cavity/engine.py`: `corner_robust_metrics`, called by `evaluate_run` | F10; new keys only, stored records unchanged |
| `src/pinn_cavity/study.py`: the new keys in the summary | F10 |
| `src/pinn_cavity/diagnostics.py` | analytic fields and low-variance loss for F1, F2, F11 |
| `scripts/audit.py`, `results/audit/audit.json` | F1 to F3, F9 to F12 |
| `kaggle/run_cuda.ipynb`, `CITATION.cff`, `tests/test_notebook.py` | F13 |
| `pyproject.toml`, `uv.lock` | F14 and the uv environment |
| `tests/test_audit.py` | regression tests for F1, F2, F10 and the distance weight |
| `README.md` | installation, reference error wording, links to these documents, summary of F1 and F2 |
| `docs/` | this audit |

The original evidence in `results/official`, `results/local`, `results/reference_check.json` and `figures/` was not modified.

## Addendum: the study at the upstream budget

Added after the audit, when `configs/full.yaml` was run on CUDA. Full report: [FULL_BUDGET_STUDY.md](FULL_BUDGET_STUDY.md). Records: `results/full/` (`analysis.json` for the diagnostics). The sections above are unchanged and describe the evidence at the audit commit.

**Runs.** Nine runs: Re ∈ {10, 100, 400} × seeds {0, 1, 2}, 10 000 steps, 4000 interior and 2000 boundary points per step, the upstream network, loss and optimiser. They ran on one Tesla T4 per Kaggle session from clean checkouts (commits `3ab0cca` and `36c76aa`, identical training code), with the environment of `uv.lock`: 7.8 GPU-hours of training. The checkpoints were evaluated on CPU. Where the GPU also evaluated, the two agree to $5.5 \times 10^{-8}$ relative.

**F15, F18. The failure persists.**

| Re | Velocity error, seeds 0, 1, 2 | Median: reduced budget, upstream budget | $\psi_{\min}$ / reference |
|---|---|---|---|
| 10 | 0.350, 0.349, 0.505 | 0.456, 0.350 | 0.63, 0.64, 0.58 |
| 100 | 0.837, 0.669, 0.655 | 0.805, 0.669 | 0.00, 0.19, 0.19 |
| 400 | 0.904, 0.881, 0.860 | 0.889, 0.881 | 0.01, 0.03, 0.19 |

- At Re = 400 no network has a primary vortex.
- At Re = 100 one has none, and two have a weak vortex displaced by $0.18L$ to $0.24L$ towards the downstream lid corner.
- The Re = 10 seed 2 value is that of a loss spike on the last step: its error was 0.30 at step 9000.
- The same configuration and seed gives 0.383 on MPS (`official`) and 0.350 on CUDA. Differences of that size between the two budgets are within device variation.

**F19. The counterexample holds.** Scored with the same objective, the best lid-layer field (no recirculation) has loss $7.0 \times 10^{-4}$ at Re = 100 and $9.5 \times 10^{-5}$ at Re = 400.

- The validation losses of the trained networks, over 25 independent batches, are 1.05 to 14.7 times the lid-layer value at Re = 100 and 59 to 181 times it at Re = 400.
- At Re = 100 the two best networks tie with a field that has no vortex.

**F20. Collapse not reproduced.** No hidden neuron of any of the nine final networks is constant or identically zero over the cavity, and no training record has vanishing residuals. Zero events in nine runs bounds the per-run probability at 0.28 (one-sided 95%). The cause of the collapse in F2 remains UNVERIFIED and is not attributed to budget or device.

**F16, revisited.** The study separates two weaknesses but does not rank them:

- **Objective.** The loss does not discriminate fields with no recirculation at the values reached at Re = 100 (F1, F19).
- **Optimisation.** At Re = 400 the optimiser stops 59 to 181 times above a loss that a two-parameter field reaches, with the loss dominated by the boundary terms (F12 at the upstream budget).

Which one controls the outcome is UNVERIFIED (gaps G2, G3).

**F21. Launcher defects (fixed).**

1. `pip install` into the Kaggle image mixed package versions, and the image's RAPIDS `cuml`, imported by PhysicsNeMo when present, rejected NumPy 2.5. The launcher now installs `uv.lock` with `uv sync --frozen`.
2. The image's `MPLBACKEND` named a backend absent from that environment. The launcher now sets `Agg`.
3. On a 16 GB GPU the evaluation ran out of memory in the inference timing, because the residual graphs were still held. They are now freed, with no metric change.

Defects 1 and 2 stopped sessions before training. Defect 3 occurred after the six Re = 100 and 400 runs had saved their checkpoints, which were then evaluated on CPU.
