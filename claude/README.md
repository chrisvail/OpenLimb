# OpenLimb: measurements → residual-limb shapes (conditional VAE)

Given the seven clinical measurements (4 circumferences, 1 length, 2 widths), generate as many different, plausible
3-D residual limbs as you like, all with those measurements. What the method is, why, and every result:
**[report/WRITEUP.md](report/WRITEUP.md)**. This file is only about running the code.

Nothing outside `claude/` is modified (`SSM_Driver.py`, `measure_limbs.py` in the repository root are imported and reused).

```
claude/
  train_surrogate.py   stage 1: differentiable stand-in for the measurement code
  train.py             stage 2: CVAE (model=cvae; model.meas_dropout=true for missing measurements), MLP baseline (model=mlp)
  common.py            repository paths, hydra resolvers, sys.path set-up
  data/                limb geometry + exact measurement code, limb sampler / DataModule, generate.py;
                       also lr_map.npz and the trained surrogate (surrogate_scaled.pt)
  models/              networks, Lightning modules (CVAE, dropout CVAE, MLP, surrogate), GA / Gauss-Newton baselines
  evaluation/          evaluate*.py, plot_eval.py, make_test_csv.py, annotator.html
  conf/                hydra configs: one yaml per script + groups data/ model/ trainer/ logger/ experiment/
  report/              WRITEUP.md (the write-up), figs/, eval_reports/ (text output of the runs in the write-up)
  outputs/             everything the scripts write (git-ignored)
```

## 0. Setup

All commands are run **from the repository root** (`SSM_Driver` uses paths relative to it; every hydra config sets
`chdir: false`). `python` below means the project's environment (`.venv/Scripts/python.exe` on Windows).

```bash
uv sync                          # dependencies are in pyproject.toml
bash scripts/build_docker.sh     # only for evaluation/evaluate.py: builds the image `openlimbtt-env` (OpenLimbTT's own generator)
```
A trained surrogate is already at `claude/data/surrogate_scaled.pt` and trained models are in `claude/outputs/single/`,
so §1 is enough to *use* the model; §2 retrains everything.

## 1. Generate limbs from measurements

```bash
python claude/data/generate.py ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt \
    measurements=[300,290,280,270,120,110,115] n_samples=32 save_meshes=true
```
Measurements are in mm, in the order **Circ 1, Circ 2, Circ 3, Circ 4, Length 1, Width 1, Width 2**. Writes
`components.npy` (32 × 11), `measurements_check.json` (requested vs exactly re-measured) and `limb_*.obj` to
`claude/outputs/generated/<time>/`. `temperature=0.5` narrows the family; `test_index=3` uses the measurements of held-out limb 3.

With a measurement-dropout model, write `null` for a measurement you do not have:
```bash
python claude/data/generate.py ckpt=claude/outputs/single/fx_dropout/checkpoints/best.ckpt \
    measurements=[300,null,280,null,120,110,null] n_samples=32
```

## 2. Reproduce the results

Each block gives the command(s) and says which part of the write-up it produces. Run them in order. Timings are for a
24-core laptop CPU; the training data is sampled on the fly, so no dataset needs to exist. Every run's folder also
contains its resolved config (`.hydra/`).

### 2.1 Surrogate (write-up §4.4)
```bash
python claude/train_surrogate.py trainer.max_epochs=40          # ~19 min  ->  claude/data/surrogate_scaled.pt
```
Everything below is trained *against* this file; retrain the CVAEs whenever it changes.

### 2.2 Models (§4, §7)
```bash
for w in 1 10 50 200; do      # plausibility-weight sweep; w=200 is the final CVAE        ~13 min each (run in parallel)
  python claude/train.py model.w_plaus=$w hydra.run.dir=claude/outputs/single/fx_w$w
done
python claude/train.py model=mlp monitor=val/mse hydra.run.dir=claude/outputs/single/fx_mlp           # plain-MLP baseline
python claude/train.py model.meas_dropout=true hydra.run.dir=claude/outputs/single/fx_dropout              # measurement dropout (§4.7)
```
Checkpoints are written to `<run dir>/checkpoints/best.ckpt`, `metrics.json` has the validation/test metrics.
Smoke test of the pipeline (seconds): `python claude/train.py trainer=smoke`.

### 2.3 Accuracy, plausibility, shape, diversity of the final CVAE (§7.1–7.9, figures A–E)
```bash
for w in 1 50 200; do         # needs docker; 1024 test limbs from the original generator    ~24 min each (parallel)
  python claude/evaluation/evaluate.py ckpt=claude/outputs/single/fx_w$w/checkpoints/best.ckpt out_dir=claude/outputs/eval/fx_w$w
done
python claude/evaluation/plot_eval.py claude/outputs/eval/fx_w200 --out claude/report/figs \
    --compare "w_plaus=1@claude/outputs/eval/fx_w1" "w_plaus=50@claude/outputs/eval/fx_w50" "w_plaus=200@claude/outputs/eval/fx_w200"
```
Writes `fig_A…fig_E*.png`, `fig_compare_variants.png`, `compare_variants.md` to `report/figs/`. The text of each run is
`claude/outputs/eval/fx_w<w>/report.txt`:
```bash
cp claude/outputs/eval/fx_w1/report.txt   claude/report/eval_reports/base_wplaus1.txt
cp claude/outputs/eval/fx_w50/report.txt  claude/report/eval_reports/wplaus50.txt
cp claude/outputs/eval/fx_w200/report.txt claude/report/eval_reports/wplaus200.txt
```

### 2.4 Against a plain MLP and against optimisation (§7.3)
```bash
python claude/evaluation/evaluate_mlp.py ckpt=claude/outputs/single/fx_mlp/checkpoints/best.ckpt \
    cvae_ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt cvae_eval=claude/outputs/eval/fx_w200 \
    out_dir=claude/outputs/eval/fx_mlp                                      # ~2 min   -> report/figs/fig_compare_mlp.png
python claude/evaluation/evaluate_ga.py cvae_ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt \
    cvae_eval=claude/outputs/eval/fx_w200 mlp_eval=claude/outputs/eval/fx_mlp \
    out_dir=claude/outputs/eval/fx_ga                                       # ~42 min, 12 workers   -> report/figs/fig_compare_ga.png
cp claude/outputs/eval/fx_mlp/report.txt claude/report/eval_reports/mlp.txt
cp claude/outputs/eval/fx_ga/report.txt  claude/report/eval_reports/ga.txt
```
Both must use the same `seed` as `evaluate.py` (default 777) so that they see the same 1024 test limbs.

### 2.5 Missing measurements (§7.11)
```bash
python claude/evaluation/evaluate_dropout.py ckpt=claude/outputs/single/fx_dropout/checkpoints/best.ckpt \
    baseline_ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt out_dir=claude/outputs/eval/fx_dropout
cp claude/outputs/eval/fx_dropout/report.txt claude/report/eval_reports/dropout.txt
```
Does not need docker (test limbs come from this package's sampler with a fixed seed). Also writes `results.json`.

### 2.6 Real-data experiments on synthetic stand-ins (§8)
Measurements table (§8.2):
```bash
python claude/evaluation/make_test_csv.py claude/outputs/test_meas.csv            # 60 rows, faults injected
python claude/evaluation/evaluate_csv.py ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt \
    csv=claude/outputs/test_meas.csv id_col=Patient n_range=3000 out_dir=claude/outputs/csv/final
cp claude/outputs/csv/final/report.txt claude/report/eval_reports/csv_synthetic.txt
```
Meshes with ground truth (§8.1), clean and with 0.3 mm vertex noise:
```bash
python claude/evaluation/evaluate_wild.py mode=make_test_data mesh_dir=claude/outputs/wild_test n_test=20
python claude/evaluation/evaluate_wild.py mode=evaluate ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt \
    mesh_dir=claude/outputs/wild_test out_dir=claude/outputs/wild/final k_samples=16 n_points=8000
python claude/evaluation/evaluate_wild.py mode=make_test_data mesh_dir=claude/outputs/wild_noisy n_test=20 noise_mm=0.3
python claude/evaluation/evaluate_wild.py mode=evaluate ckpt=claude/outputs/single/fx_w200/checkpoints/best.ckpt \
    mesh_dir=claude/outputs/wild_noisy out_dir=claude/outputs/wild/noisy k_samples=16 n_points=8000
cp claude/outputs/wild/final/report.txt claude/report/eval_reports/wild_synthetic.txt
cp claude/outputs/wild/noisy/report.txt claude/report/eval_reports/wild_synthetic_noise0.3mm.txt
```
For **real** data: annotate your own meshes with `evaluate_wild.py mode=annotate mesh_dir=<dir>` (3-D tool in the browser, needs
internet for three.js), then `mode=evaluate` as above (`unit_scale=1000` for metres); for a measurement table use
`evaluate_csv.py csv=<file> [unit_scale=10] [id_col=...]`. The options are documented at the top of each script and in write-up §8.

## 3. Other things you can run

```bash
python claude/train.py -m model.z_dim=2,3,4,6 model.beta=0.001,0.01,0.1     # grid sweep
python claude/train.py -m experiment=optuna                                 # Optuna TPE, search space in conf/experiment/optuna.yaml
python claude/train.py logger=wandb                                         # or logger=tensorboard
python claude/train_surrogate.py data.scale=false                           # size-normalised variant (also train.py data.scale=false)
```
Any hydra key can be overridden on the command line (`conf/*.yaml`, `conf/model/*.yaml`). Metrics logged in training:
`val/*` are cheap (surrogate-based), `val_exact/*` and `test_exact/*` use the exact measurement code on a subset;
`val/score` (= √recon MSE + prior measurement RMSE + fraction outside the plausible box) selects the checkpoint.
For the dropout model, `val/score` uses random masks and `val/full_*` has the same metrics with all seven given.

## 4. Building the write-up as PDF / LaTeX

`report/WRITEUP.md` is plain pandoc-flavoured Markdown (pipe tables, relative image links, no HTML), so pandoc can turn it
into LaTeX or PDF (not run here: pandoc and a TeX distribution are not installed on this machine):
```bash
cd claude/report
pandoc WRITEUP.md -s --toc --number-sections -V geometry:margin=2.5cm -V mainfont="DejaVu Serif" --pdf-engine=xelatex -o WRITEUP.pdf
pandoc WRITEUP.md -s --toc --number-sections -o WRITEUP.tex          # editable LaTeX; figures stay in figs/
```
`--pdf-engine=xelatex` is needed for the Unicode symbols (−, ≈, ², σ …); a font containing them must be set with `mainfont`.
If you want real equations, write them as `$...$` / `$$...$$` in the Markdown and pandoc will emit them as LaTeX math.
