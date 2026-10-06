# Results

Tracked result files. Each study directory is a copy of a run directory without its checkpoints, written by `python scripts/results.py publish <run directory or archive>`.

| Path | Content |
|---|---|
| `reference_check.json` | output of `scripts/verify_reference.py`: grid convergence of the finite-difference reference solver, extrema against published values, differences to the tables of Ghia et al. (1982), centreline profiles |
| `<study>/study_metadata.json` | timestamp, git commit, device, package versions, resolved configuration, record of each reference solution |
| `<study>/summary.csv` | one row per run |
| `<study>/summary.json` | the same rows, and mean, standard deviation, minimum and maximum over seeds per Reynolds number |
| `<study>/fields.npz` | reference and predicted velocity on 129 × 129 nodes for every run, predicted pressure for the first seed |
| `<study>/re<Re>_seed<S>/history.csv` | learning rate, loss, the seven loss terms and the tracked velocity error against the step |
| `<study>/re<Re>_seed<S>/train_metrics.json` | viscosity, parameter count, final loss terms, timings, memory |
| `<study>/re<Re>_seed<S>/eval_metrics.json` | errors against the reference, boundary errors, residuals, inference time |

Studies present:

| Study | Configuration | Device | Status |
|---|---|---|---|
| `local` | `configs/local.yaml`: Re 10, 100, 400, three seeds, 5000 steps, 1000 + 500 points per step | Apple MPS | run |
| `official` | `configs/official.yaml`: the upstream settings, Re 10, one seed, 10 000 steps, 4000 + 2000 points per step | Apple MPS | run |
| `full` | `configs/full.yaml`: Re 10, 100, 400, three seeds at the upstream settings | CUDA | pending |

Units and definitions:

- `velocity_rel_l2`, `u_rel_l2`, `v_rel_l2`: relative $L^2$ errors on all nodes of the reference grid, as fractions. `velocity_rel_l2_core` is restricted to nodes at least a tenth of the cavity width away from every wall.
- `velocity_max_abs_error`, `centreline_*`, `no_slip_rms_error`, `lid_u_rms_error`, `ghia_*`: in units of the lid speed. The `ghia_*` columns are empty for Re = 10, which Ghia et al. do not tabulate.
- `residual_rms_*`: root mean square of the PDE residuals without the distance weight of the training loss, at the centres of a uniform grid of cells, made dimensionless with the cavity width and the lid speed.
- `loss` and the loss terms in `history.csv`: the quantities the optimiser sees, in the units of the upstream example, with its weights.
- Times are wall-clock seconds or milliseconds with the device synchronised. `peak_train_memory_mb` is the CUDA peak allocation and is empty on other devices.

`study_metadata.json` of `local` and `official` carries a `metadata_note`: both studies were launched with `--resume`, which did not write the record at that commit, so it was written from the same clean checkout while the studies were running or after they had finished. `timestamp_utc` is the time of writing. The timings of both studies were taken while another training job was using the same machine and are not benchmarks.
