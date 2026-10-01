# Measurements -> SSM components: conditional VAE

Nothing outside this folder is modified. `SSM_Driver.py` / `measure_limbs.py` are imported and reused.

## Why a conditional VAE

Seven measurements (4 circumferences, 1 length, 2 widths) pin down at most 7 of the 11 SSM coordinates
(10 modes + tibia scale), so "measurements -> limb" is one-to-many. A plain regressor returns the
conditional mean; a CVAE returns a *distribution*:

* encoder `q(z | m, c)` sees the true components during training (this is the "small amount of data from the
  encoder" - `z` has `model.z_dim` numbers, default 4 = the number of leftover degrees of freedom);
* decoder `p(c | m, z)` gets the measurements at its input every time;
* at generation time `z ~ N(0, I)` -> a family of limbs that all (approximately) match `m`.

`model.z_dim=0` turns it into a deterministic MLP, a useful baseline: its residual error is the amount of
variance `z` has to explain.

### Things that make it work for *generation* (not just reconstruction)
1. **Prior-path losses.** A plain CVAE is only trained on `z ~ q(z|m,c)`. Here the decoder is also run on
   `z ~ N(0,I)` (no ground truth available) and penalised for (a) not matching `m` (`w_prior_meas`) and (b) leaving
   the plausible region (`w_plaus`). That is what guarantees sampled limbs honour the input.
2. **Measurement surrogate.** The exact measurement function costs ~5 ms/limb, too slow for a training loop, so
   `train_surrogate.py` fits a small MLP `components -> measurements` (checked against the exact function; see
   `test_metrics` in `data/surrogate_*.pt`). The exact function is used for all *reported* numbers
   (`val_exact/*`, `test_exact/*`).
3. **Exact plausibility test.** The OpenLimbTT generator draws 10 "skin" modes uniformly from a box and maps them
   through a *linear* regression (`LR.pkl`) to the full modes; scale is uniform in [342.8, 439.8]. I extracted the
   regression coefficients (`data/lr_map.npz`), so any component vector can be mapped back to box coordinates:
   inside the box <=> the generator could have produced it. Used as a training penalty and as the metric
   `prior_outside_box`; `generate.py` can rejection-sample against it.

## Workflow

```bash
# 0. one-off: extra dependencies into the existing venv (does not touch pyproject.toml)
uv pip install --python .venv/Scripts/python.exe -r claude/requirements.txt

# 1. surrogate (streams freshly measured limbs, ~few min; writes claude/data/surrogate_scaled.pt)
.venv/Scripts/python.exe claude/train_surrogate.py

# 2. CVAE
.venv/Scripts/python.exe claude/train.py

# 3. generate a family for a set of measurements (mm)
.venv/Scripts/python.exe claude/generate.py ckpt=claude/outputs/single/<run>/checkpoints/best.ckpt     measurements=[300,290,280,270,120,110,115] n_samples=32 save_meshes=true
```
Use `data.scale=false` (steps 1 and 2) for the size-normalised variant.

**No dataset is stored.** Limbs are sampled on the fly (`ComponentSampler`): skin modes ~ U(box), linear
regression, scale ~ U(342.8, 439.8) - the same recipe as `GenerateRandomLimbs.py`, minus docker (the
`LR.pkl` coefficients are in `data/lr_map.npz`, 1 KB). Every epoch sees new limbs. Only small *fixed* validation /
test sets (512 / 1024 limbs, seeded, RAM only) are measured exactly at start-up (~10 s).
CVAE training gets its input measurements from the surrogate (free); the surrogate itself is fitted on
streamed exact measurements, which is the only place the ~4.5 ms/limb exact function is in the loop.

## Results

Full write-up with figures, tables and caveats: [report/WRITEUP.md](report/WRITEUP.md).
Headline (final model, `w_plaus=200`, exact original measurement code): 0.37 mm / 0.17 % mean measurement error,
chamfer 0.95 mm to the true limb, ~5.4 mm shape spread across `z` at fixed measurements, 98 % of samples plausible.

## Sweeps

```bash
# grid (hydra basic sweeper)
python claude/train.py -m model.z_dim=2,3,4,6 model.beta=0.001,0.01,0.1
# Optuna TPE, search space in conf/experiment/optuna.yaml
python claude/train.py -m experiment=optuna
# W&B logging
python claude/train.py logger=wandb
```
Every run/trial gets its own folder under `claude/outputs/multirun/<time>/<n>/` with checkpoints, CSV logs and
`metrics.json`. The sweep objective is `monitor` (default `val/score`).

## Metrics (`val/*` are cheap, surrogate-based; `*_exact/*` use SSM_Driver.measure on a subset)

| metric | meaning |
|---|---|
| `recon_mse` | posterior-path error in z-scored component space |
| `prior_meas_rmse` | z-scored measurement error of samples drawn with `z ~ N(0,I)` - **does the family honour the input** |
| `prior_outside_box` | fraction of prior samples the generator could never produce |
| `diversity_vert_std` | mean per-vertex std (mm) across 8 samples for the same measurements - **how wide is the family** |
| `elbo_nll` | 0.5*D*MSE/sigma^2 + KL with fixed sigma: comparable across `beta` / `z_dim` |
| `vert_rmse` | posterior-path mesh error |
| `score` | `sqrt(recon_mse) + prior_meas_rmse + prior_outside_box` (the default checkpoint/sweep objective) |
| `*_exact/{prior,post}_mae[_<measure>]` | mm error of the exact measurements of the generated limbs |

`score` mixes fit and generative quality; change `monitor=` to optimise something else, e.g.
`monitor=val/prior_meas_rmse`. Note a model with no diversity can still score well - look at
`diversity_vert_std` next to it (`z_dim=0` is the zero-diversity reference).

## Files
* `openlimb_cvae/geometry.py` - `SSMGeometry` (efficient, identical maths to `LegMeasurementDataset.get_verts`, reuses `Measurements` and `get_measures`), `ComponentSampler`, plausibility box.
* `openlimb_cvae/data.py` - `LimbDataModule` (streaming); `lit_cvae.py` - `LimbCVAE`; `lit_surrogate.py`; `networks.py`.
* `openlimb_cvae/fast_measure.py` - `FastMeasure` (same numbers as `SSM_Driver.measure`, 5-8× faster in batches) and `BoxMap` (limb numbers <-> the generator's box coordinates).
* `openlimb_cvae/ga.py` - the notebook GA, its LM refinement fixed, an improved GA, and box-constrained Gauss-Newton; `evaluate_ga.py` compares them with the CVAE (write-up §8.3).
* `conf/` - hydra config groups (`data`, `model`, `trainer`, `logger`, `experiment`).
