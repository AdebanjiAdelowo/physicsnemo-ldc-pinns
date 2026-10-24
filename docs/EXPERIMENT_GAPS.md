# Experiment gaps and execution plan

What the evidence in this repository does not yet cover, ordered by how much each gap limits the conclusions. Findings F1 to F17 refer to [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md).

Status: **PENDING** means fully specified, with commands that exist in this repository. **NOT IMPLEMENTED** means a design only, with no code. None of these experiments has been run.

## Budgets that must not be conflated

| Study | Steps | Points per step (interior / boundary) | Point evaluations | Re | Seeds | Status |
|---|---|---|---|---|---|---|
| `official` | 10 000 | 4000 / 2000 | $6.0 \times 10^7$ | 10 | 1 | run |
| `local` | 5000 | 1000 / 500 | $7.5 \times 10^6$ (1/8 of `official`) | 10, 100, 400 | 3 | run |
| `full` | 10 000 | 4000 / 2000 | $6.0 \times 10^7$ per run | 10, 100, 400 | 3 | PENDING |

The only pair with the same Reynolds number and seed at both budgets is Re = 10, seed 0: error 0.473 at the reduced budget, 0.383 at the upstream budget. Every statement about the Reynolds-number trend rests on the reduced budget. A statement that "the upstream recipe fails at Re = 100" is not supported by any run (F15).

## G1. The study at the upstream budget (`configs/full.yaml`): PENDING, highest priority

**Question.** Does the failure at Re = 100 and 400 persist at the budget of the example? What is the seed spread at that budget?

F1 predicts that a larger budget cannot remove the degeneracy of the loss. It does not predict whether the optimiser finds the flow.

**Why not locally.** The recorded step time of the upstream recipe on the Apple M3 Pro (MPS) is 0.5 s isolated and 0.73 s on the shared machine. Nine runs of 10 000 steps would take 12 to 18 hours of continuous MPS training while other experiments share the machine. That conflicts with the machine policy, and the brief asks for CUDA instead.

**Resource estimate.** The GPU time is a projection; the probe in step A of the notebook measures it before anything long starts.

| Resource | Estimate | Basis |
|---|---|---|
| GPU time, 9 runs | 3 to 10 h on a T4 or P100 (0.1 to 0.4 s per step) | arithmetic of the 6 × 512 MLP with second derivatives at 4000 points, relative to 0.5 s per step on MPS; not measured |
| GPU memory | under 4 GB | 1.3 M parameters, 4000 interior points with second-order autograd graphs; the probe records the peak |
| CPU time for the three 513-node references | 5 to 15 min | 52 to 262 s each on the M3 Pro |
| Disk, outputs | about 160 MB | 16 MB checkpoint and optimiser state per run, 6 MB per reference, small records |
| Disk, environment | 3 to 5 GB | PhysicsNeMo with the `sym` extra and its dependencies, installed by `pip` in the session |
| Download | about 2 to 3 GB of packages, from PyPI and GitHub | no datasets or pretrained models |

**Kaggle (preferred: the launcher exists and is tested statically).**

1. Push the commit to run and note its SHA (`git rev-parse HEAD`).
2. Create a Kaggle notebook from [`kaggle/run_cuda.ipynb`](../kaggle/run_cuda.ipynb). Settings: Accelerator GPU (T4 ×2 or P100), Internet on.
3. In the first cell set `REF` to the SHA and `SESSION_BUDGET_HOURS` to the GPU time you accept (at most the session limit; Kaggle's limits change, so check them). Leave `RUN_FULL = False` for a first session.
4. Run all. Step A runs the tests, a CUDA smoke study and a 200-step probe at Re = 400, then prints `PROJECTION for step B: X h`.
5. If X fits the budget, set `RUN_FULL = True` and run all again. The notebook refuses to start step B if the projection does not fit, and it does not reduce the configuration.
6. Download `physicsnemo-pinn-cavity-full.zip` (records) and `physicsnemo-pinn-cavity-full-eval-bundle.zip` (checkpoints) from `/kaggle/working`.

If one session is too short for nine runs, split by Reynolds number. Use one session per value, with these overrides in the step B command of the notebook:

```text
python scripts/train.py --config full --resume device=cuda study.reynolds=[10]
python scripts/train.py --config full --resume device=cuda study.reynolds=[100]
python scripts/train.py --config full --resume device=cuda study.reynolds=[400]
```

Copy the nine run directories into one `runs/full/` locally, then run `python scripts/evaluate.py --config full` there. Inference times then belong to the local machine; training times stay those of the GPU.

**Colab (alternative, no launcher).** On a GPU runtime:

```bash
git clone https://github.com/AdebanjiAdelowo/physicsnemo-ldc-pinns.git && cd physicsnemo-ldc-pinns
git checkout <SHA>
pip install -q -r requirements.txt
python -m pytest -q
python scripts/train.py --config full --no-evaluate name=full_probe device=cuda study.reynolds=[400] study.seeds=[0] training.steps=200 training.error_every=200
python scripts/train.py --config full --resume device=cuda      # only if 9 x 10000 x seconds_per_step fits the runtime
python scripts/results.py package runs/full && python scripts/results.py bundle runs/full
```

Colab runtimes can be disconnected. `--resume` restarts at the first unfinished run, but not within a run, so copy `runs/full` to Drive after every run.

**Integration and analysis (local, CPU).**

```bash
python scripts/results.py publish runs/full.zip
unzip <eval bundle> -d .
python scripts/evaluate.py --config full --verify device=cpu
python scripts/audit.py --runs-root runs --studies full --out results/audit/audit_full.json
python scripts/plot_results.py --study results/full --out figures
python -m pytest -q
```

`audit.py --studies full` gives the low-variance loss and the corner-robust metrics of every run. Compare those losses with the lid-layer losses in `results/audit/audit.json` (F1) to see whether the upstream budget reaches the low-loss region at all.

## G2. Optimisation failure or loss degeneracy: NOT IMPLEMENTED

**Question.** At Re = 100 and 400, is the exact solution reachable and stable under this loss with this network, or does training from it drift away?

**Design.**

1. Fit the network by supervised regression to the 513-node reference velocity. The reference has no pressure, so fit $p$ through the momentum residual, or add the pressure of G5.
2. Record the PINN loss of the fitted network with `diagnostics.sampled_loss`.
3. Continue training with the PINN loss of the example for 5000 steps, tracking the velocity error.

Interpretation:

- If the fitted network has a loss below the trained networks and stays near the reference, the failure is one of optimisation from a random start.
- If it drifts towards a lid-layer state, the loss landscape itself favours the degenerate fields at this Reynolds number.

**Cost.** 2 runs per Reynolds number at the reduced point counts; about 30 to 60 min of MPS each. It is a lightweight experiment, but it needs new code (a supervised fitting loop) and a decision on how to fit the pressure.

## G3. Causes in the recipe: NOT IMPLEMENTED (methodological changes, need approval)

F1 isolates one mechanism in the loss: the distance factor $d$ on the residuals with unit-weight boundary terms. Candidate changes, each a departure from the upstream recipe:

| Ablation | Change | What it tests |
|---|---|---|
| A1 | residuals unweighted ($d \to 1$), or weighted by $d / \sqrt{\mathbb{E}[d^2]}$ | whether the near-wall discount causes the lid-layer degeneracy |
| A2 | inputs scaled to $[-1, 1]$ (equivalently, training on the unit square with $\nu = 1/\mathrm{Re}$) | spectral bias from inputs in $[-0.05, 0.05]$ |
| A3 | balanced loss terms (fixed weights from the term magnitudes, or adaptive weighting) | the 1 : $4 \times 10^{-4}$ imbalance of boundary and residual weights |
| A4 | velocity from a streamfunction output, $u = \psi_y$, $v = -\psi_x$; hard no-slip by a distance factor | exact continuity and boundary conditions |
| A5 | Reynolds-number continuation (initialise Re = 100 from Re = 10, Re = 400 from Re = 100) | whether a good initial state avoids the degenerate states |

Cost per ablation at the upstream budget: the same as G1, 3 to 10 GPU hours for 3 Re × 3 seeds. A first pass of A1 and A2 at the reduced budget is affordable locally: 9 runs at 0.35 s per step × 5000 steps, about 4.5 h of MPS each, when the machine is free.

## G4. Frequency of the collapse: PENDING

**Question.** How often does a run collapse to the constant field (F2), and does it depend on Re or on the budget?

Run more seeds at the reduced budget, for example seeds 3 to 9 at Re = 100:

```bash
uv run python scripts/train.py --config local name=local_collapse study.reynolds=[100] study.seeds=[3,4,5,6,7,8,9] device=mps
```

Cost: 7 × 30 min of MPS. Each run also writes a checkpoint, and `audit.py`'s `network_state` reports whether its last hidden layer is saturated. To capture the event itself, save checkpoints every 100 steps around the collapse (not implemented).

## G5. Pressure: NOT IMPLEMENTED

Recover the reference pressure from $\psi$ by solving $\Delta p = 2(u_x v_y - u_y v_x)$ with Neumann data $\partial_n p$ from the momentum equation at the walls, fix the mean, and compare $p - \bar p$ with the network.

Cost: one sparse solve per Reynolds number on CPU, seconds to minutes. The wall Neumann data are singular at the lid corners and need care.

## G6. A second reference at Re = 10 and 400: NOT IMPLEMENTED

At Re = 10 and 400 the reference is checked by grid convergence and by its agreement with benchmarks at neighbouring Re, not by an independent solver. Options:

- a finer grid, 1025 nodes: about $2 \times 10^6$ unknowns; the sparse LU needs several GB and minutes per Newton step, so check the memory first;
- a spectral or finite-element solver.

The network errors are at least 60 times the estimated reference error, so this gap cannot change any conclusion. It matters only if the network errors fall below a few percent.

## G7. Is the trained field a lid layer: PENDING, CPU only

**Question.** Are the trained Re = 100 and 400 networks close to the analytic lid-layer family of F1?

Fit $(\delta, \varepsilon)$ of `diagnostics.LidLayer` to each network's $u$ on the reference grid by least squares and report the residual. The checkpoints exist, and the computation takes minutes on CPU. It quantifies the README's qualitative description, "a thin layer below the lid in which $u$ drops from the lid speed to about zero, above an interior at rest".

## G8. Device

All runs used Apple MPS in float32. G1 adds CUDA. Neither result is bitwise reproducible on the other device. The stored errors recompute on CPU to $2.4 \times 10^{-5}$ relative (F3), which covers evaluation only, not training.

## Order of work

1. G1: removes the budget confound and gives the seed spread at the upstream budget.
2. G7 and G2: decide between an optimisation failure and loss degeneracy.
3. G3 (A1, A2 first): tests the cause, after approval of the methodological change.
4. G4, G5, G6: strengthen secondary points.
