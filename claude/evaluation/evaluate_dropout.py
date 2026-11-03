"""Evaluate a measurement-dropout CVAE over missing-measurement patterns.

    python claude/evaluation/evaluate_dropout.py ckpt=claude/outputs/single/<run>/checkpoints/best.ckpt
    python claude/evaluation/evaluate_dropout.py ckpt=... baseline_ckpt=claude/outputs/single/<run>/checkpoints/best.ckpt

Held-out limbs come from this package's sampler (same recipe as training, fixed seed) with their exact
SSM_Driver measurements; generated limbs are re-measured with the exact code, in mm. For every mask pattern:

  obs MAE / miss MAE   |exact measurement of the generated limb - true measurement|, on observed / missing entries
                       (observed = how well the request is honoured, missing = how well it is inferred)
  outside              fraction of prior samples outside the generator's plausible skin-space box
  vert RMSE            mm, generated vs the true limb (prior sample; it is only expected to shrink as m is added)
  family std           mm, per-vertex std over k_prior samples of the same request (the family's width)

Patterns: all 7 observed; each single measurement dropped; k = 0..6 observed (random subsets).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import register_resolvers, setup_env  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from data.datamodule import LimbDataModule  # noqa: E402
from data.geometry import SSMGeometry  # noqa: E402
from models.lit_cvae import LimbCVAE  # noqa: E402

REPORT = []


def say(s=""):
    print(s, flush=True)
    REPORT.append(s)


def random_mask(n, n_obs, d, gen):
    ranks = torch.rand(n, d, generator=gen).argsort(-1).argsort(-1)
    return (ranks < n_obs).float()


@torch.no_grad()
def run_pattern(model, geo, m, c, mask, k, n_div):
    """Generate k prior samples per limb under `mask`; return a dict of summary metrics."""
    N, d = m.shape
    m_rep, c_rep, mk = m.repeat_interleave(k, 0), c.repeat_interleave(k, 0), mask.repeat_interleave(k, 0)
    z = torch.randn(N * k, model.hparams.z_dim) if model.hparams.z_dim > 0 else None
    cn = model.decode(m_rep, z, mk)
    want = m_rep * model.meas_std + model.meas_mean
    err = (geo.get_measures((cn * model.comp_std + model.comp_mean).double()).float() - want).abs()
    obs, miss = mk > 0, mk == 0
    v, v_true = model.verts(cn), model.verts(c_rep)
    nd = min(n_div, N)
    out = dict(
        obs_mae=err[obs].mean().item() if obs.any() else float("nan"),
        miss_mae=err[miss].mean().item() if miss.any() else float("nan"),
        outside=model.outside_box(cn).float().mean().item(),
        vert_rmse=(v - v_true).pow(2).mean().sqrt().item(),
        family_std=v.reshape(N, k, -1)[:nd].std(1).mean().item() if k > 1 else float("nan"),
        err_per_meas=err.reshape(N, k, d).mean((0, 1)).tolist(),
    )
    return out


def fmt(x, p=3):
    return "-" if x != x else f"{x:.{p}f}"


def table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    say(line(header))
    say("  ".join("-" * x for x in w))
    for r in rows:
        say(line(r))


@hydra.main(config_path="../conf", config_name="evaluate_dropout", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    out = Path(HydraConfig.get().runtime.output_dir)
    model = LimbCVAE.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    hp = model.hparams
    assert hp.meas_dropout, "ckpt was not trained with model.meas_dropout=true"
    d, k = hp.n_measurements, cfg.k_prior
    geo = SSMGeometry(scale=hp.scale, normalise=False)
    names = [x["name"] for x in geo.measurement_details]

    dm = LimbDataModule(scale=hp.scale, n_val=0, n_test=0)
    test = dm._exact_set(cfg.n_limbs, seed=cfg.seed)
    m, c = test["m"], test["c"]
    N = len(m)
    say(f"checkpoint: {cfg.ckpt}\nz_dim={hp.z_dim}  p_full={hp.p_full}  min_keep={hp.min_keep}  w_plaus={hp.w_plaus}")
    say(f"{N} held-out limbs x {k} prior samples per pattern; errors in mm of the exact measurement\n")

    header = ["pattern", "obs MAE", "miss MAE", "outside", "vert RMSE", "family std"]
    row = lambda name, r: [name, fmt(r["obs_mae"]), fmt(r["miss_mae"]), f"{r['outside']:.1%}", fmt(r["vert_rmse"], 2),
                           fmt(r["family_std"], 2)]
    results = {}

    # ------------------------------------------------------------------ all / drop-one
    say("=== A. All measurements, and each measurement dropped in turn ===")
    rows = []
    r = results["all"] = run_pattern(model, geo, m, c, torch.ones_like(m), k, cfg.n_diversity_limbs)
    rows.append(row("all 7", r))
    if cfg.baseline_ckpt:
        base = LimbCVAE.load_from_checkpoint(cfg.baseline_ckpt, map_location="cpu").eval()
        rb = results["baseline_all"] = run_pattern(base, geo, m, c, torch.ones_like(m), k, cfg.n_diversity_limbs)
        rows.append(row("all 7 (baseline CVAE)", rb))
    for i, n in enumerate(names):
        mask = torch.ones_like(m)
        mask[:, i] = 0
        r = results[f"drop_{n}"] = run_pattern(model, geo, m, c, mask, k, cfg.n_diversity_limbs)
        rows.append(row(f"drop {n}", r))
    table(rows, header)

    # ------------------------------------------------------------------ k observed
    say(f"\n=== B. k observed measurements (random subsets, {cfg.n_masks} masks per limb) ===")
    rows = []
    for n_obs in range(d - 1, -1, -1):
        rs = [run_pattern(model, geo, m, c, random_mask(N, n_obs, d, gen), k, cfg.n_diversity_limbs) for _ in range(cfg.n_masks)]
        r = {key: sum(x[key] for x in rs) / len(rs) for key in rs[0] if key != "err_per_meas"}
        r["err_per_meas"] = [sum(x["err_per_meas"][i] for x in rs) / len(rs) for i in range(d)]
        results[f"keep_{n_obs}"] = r
        rows.append(row(f"k={n_obs}", r))
    table(rows, header)

    # ------------------------------------------------------------------ per-measurement error when dropped
    say("\n=== C. MAE (mm) of each measurement: when observed (all 7) vs when it alone is dropped ===")
    rows = [[n, fmt(results["all"]["err_per_meas"][i]), fmt(results[f"drop_{n}"]["err_per_meas"][i])]
            for i, n in enumerate(names)]
    table(rows, ["measure", "observed", "dropped"])

    (out / "report.txt").write_text("\n".join(REPORT))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
