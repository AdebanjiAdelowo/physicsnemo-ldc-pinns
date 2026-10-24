# Comparison with the NVIDIA upstream example

Upstream: [`examples/cfd/ldc_pinns`](https://github.com/NVIDIA/physicsnemo/tree/b45a5c810c741e6b41f8515be24c51121f8fc21f/examples/cfd/ldc_pinns) of NVIDIA PhysicsNeMo, commit `b45a5c810c741e6b41f8515be24c51121f8fc21f` (2 October 2026). The comparison was made against a local clone checked out at exactly that commit with a clean working tree; the installed package in `.venv` is built from the same commit (`uv.lock`).

Status labels follow [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md#classification).

## What the upstream example claims

The upstream `README.md` presents the example as a demonstration of the interoperability of `physicsnemo`, `physicsnemo.sym` and PyTorch: how to sample geometry, form residual and boundary losses and train. It reports no accuracy, no reference solution and no error metric, and it uses one viscosity. The study in this repository therefore does not test an accuracy claim made by NVIDIA; it measures what the demonstration recipe produces. Results should be described in those terms.

One inconsistency in the upstream README: it says points are sampled with "the `GeometryDatapipe` utility", while `train.py` samples with `physicsnemo.mesh.sampling.sample_random_points_on_cells`. This repository follows `train.py`.

## Element-by-element comparison

| Element | Upstream `train.py` | This repository | Status |
|---|---|---|---|
| Domain | $[-0.05, 0.05]^2$ (`height = width = 0.1`) | `problem.width = problem.height = 0.1` | VERIFIED identical |
| Interior mesh | `load_structured_grid`, 50 × 50 points (4802 triangles) | same call, `geometry.mesh_points_per_side = 50` | VERIFIED identical |
| Boundary mesh | `interior_mesh.get_boundary_mesh()` (196 segments) | same call | VERIFIED identical |
| Boundary sampling | uniform cell index, `sample_random_points_on_cells` | same code | VERIFIED identical |
| Interior sampling and distance $d$ | same, $d = \min(x - x_{\min}, x_{\max} - x, y - y_{\min}, y_{\max} - y)$ | same code | VERIFIED identical; sampling measured uniform (below) |
| Network | `FullyConnected(2, 3, num_layers=6, layer_size=512)`, SiLU, no skip connections, no adaptive activations | same call through the config | VERIFIED identical, 1 316 355 parameters |
| Input normalisation | none, inputs in $[-0.05, 0.05]$ | none | VERIFIED identical |
| PDE class | `NavierStokes` in `train.py` | `src/pinn_cavity/equations.py` | VERIFIED: class body identical (mechanical diff) |
| Residual evaluation | `PhysicsInformer(..., grad_method="autodiff")` | same | VERIFIED identical |
| Viscosity | constant `nu = 0.01` | $\nu = U L / \mathrm{Re}$; Re = 10 gives exactly `0.01` in floating point | VERIFIED |
| Density | `rho = 1.0` | `problem.rho = 1.0` | VERIFIED identical |
| Lid / wall split | `y >= height/2 - 1e-7` | same expression | VERIFIED identical |
| Lid weight | `1 - 20 * abs(x)` | `1 - (2 / width) * abs(x)`, equal for width 0.1 | VERIFIED identical |
| Loss | 7 unweighted means: 3 residuals multiplied by $d$ and squared, 4 boundary terms | same; `tests/test_engine.py` recomputes it with the upstream lines | VERIFIED identical to $10^{-5}$ relative |
| Optimiser | `Adam(lr=cfg.scheduler.initial_lr)`, `initial_lr = 1e-3` | same | VERIFIED identical |
| Scheduler | `LambdaLR`, factor `0.9999871767586216 ** step`, stepped every step | same constant from config; equals $0.95^{1/4000}$ to 16 digits | VERIFIED identical |
| Points per step | 4000 interior, 2000 boundary | `official`, `full`: same; `local`: 1000 / 500 | VERIFIED; `local` is a stated reduction |
| Steps | 10 000 (`range(10000)`) | `official`, `full`: 10 000 (`range(1, steps + 1)`); `local`: 5000 | VERIFIED; same number of optimiser steps |
| Order of random draws | boundary, then interior, every step | same; error tracking draws no random numbers | VERIFIED by reading |
| Precision | float32 | float32 | VERIFIED |
| Seeding | none | `np.random.seed`, `torch.manual_seed`; network initialised on CPU, then moved | Extension |
| Device | `DistributedManager` (CUDA or CPU) | config key (`auto`, `cuda`, `mps`, `cpu`) | Extension; reported runs used MPS |
| Unused upstream config | `arch` (FNO) block and `scheduler.decay_rate = 0.9995` are never read | kept verbatim above a marker in `configs/official.yaml`, not used | VERIFIED: upstream part byte-identical except a trailing newline |
| Output | a figure of $u$, $v$, $p$, $\lvert\mathbf{u}\rvert$ every 1000 steps; no checkpoint, no error | checkpoint, history, metrics, reference error | Extension |

The training recipe of `configs/official.yaml` is the upstream recipe plus a seed and an explicit device. The scientific content of the example (network, residuals, sampling, loss, optimiser, schedule, budget) is unchanged.

## Measured properties of the upstream recipe

These follow from the upstream code and were measured in this audit (`tests/test_audit.py`, `results/audit/audit.json`).

- **Sampling is uniform.** With $2 \times 10^6$ interior points, the fraction within $t$ of the lid matches $t / L$ to within 1% for $t$ from $10^{-4}$ to $10^{-2}$, and the position within a mesh cell is uniform.
- **Effective residual weight.** Multiplying the residual by $d$ before squaring weights the squared residual by $d^2$. For uniform points, $\mathbb{E}[d^2] = (L/2)^2 / 6 = 4.17 \times 10^{-4}$ (measured $4.164 \times 10^{-4}$), while each boundary term has weight 1. In units of the problem the momentum residual scales as $U^2/L = 10$, so a dimensionless residual $r^*$ that is uniform over the cavity costs about $0.042\,{r^*}^2$, and much less if it is concentrated near the walls, where $d \to 0$.
- **Corner weight.** The lid weight $w = 1 - 2|x|/L$ is zero at the lid corners and averages $1/2$ over the lid. It is non-negative on the lid: in float32, `1 - 20 * abs(x)` evaluates to exactly 0 at $x = \pm 0.05$.
- **The loss is zero at the exact solution.** The exact solution satisfies the PDE and the wall conditions, and $u = U$ on the open lid, so every term vanishes. The discontinuity at the corners does not enter, because $w = 0$ there and the side walls see $u = 0$.

## Differences that could affect a comparison with upstream runs

| Difference | Effect | Status |
|---|---|---|
| Apple MPS instead of CUDA | different kernels, not bitwise reproducible across devices | SUPPORTED BUT LIMITED: no CUDA run exists |
| Seeded runs | upstream runs are unseeded; any single upstream run is one draw from the same distribution | no bias, spread unknown at the upstream budget |
| `local` budget | a quarter of the points and half the steps | stated in every table; never mixed with `official` |
| Error tracking every 500/1000 steps | runs under `torch.no_grad()` and draws no random numbers | no effect on the trajectory |

## Conclusion

VERIFIED: the `official` configuration reproduces the upstream example's network, PDE residuals, geometry sampling, loss weighting, optimiser, scheduler and budget, and differs only by seeding, explicit device selection and the absence of in-loop figures. The `local` configuration is a documented reduction of the budget only.
