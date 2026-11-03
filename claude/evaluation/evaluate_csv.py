"""Experiment 2: measurement self-consistency of the CVAE on a table of real 1-D measurements.

    python claude/evaluation/evaluate_csv.py ckpt=<best.ckpt> csv=<measurements.csv> [unit_scale=10] [id_col=Patient]

* columns are matched by name (case / spaces / underscores / "(mm)" suffixes ignored; "Circumference 1" == "Circ 1"):
  Circ 1-4, Length 1, Width 1-2.  Other columns are ignored (and `id_col` is carried through to the outputs);
* rows with a missing / null / non-numeric / non-finite value in any of the 7 columns are removed (and counted);
  values <= 0 are physically impossible (entry errors) and are removed too, counted separately;
  nothing else is removed: out-of-range values are real data - they are FLAGGED (per_row.csv, report), included in
  every statistic, and also summarised separately;
* every kept row is decoded (z = 0 and k_samples draws z ~ N(0, T^2 I)); the generated limbs are re-measured with the exact
  measurement code (SSMGeometry.get_measures = SSM_Driver.measure with the edge-on-plane fix) and compared with the requested values.

There is no ground-truth shape here, so this experiment answers "does the family of limbs honour the input numbers?",
not "is the limb right?" (that is evaluate_wild.py).  Rows outside the range the model was trained on are flagged and
reported separately: for them a large discrepancy means "not representable", not "model error".

Outputs (out_dir): per_row.csv, summary.json, report.txt, per_sample.npz.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import register_resolvers, setup_env  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from data.geometry import ComponentSampler, SSMGeometry  # noqa: E402
from models.lit_cvae import LimbCVAE  # noqa: E402

# Model input order (SSM_Driver.get_measurement_details)
MEAS_NAMES = ["Circ 1", "Circ 2", "Circ 3", "Circ 4", "Length 1", "Width 1", "Width 2"]
NULL_STRINGS = {"", "null", "none", "nan", "na", "n/a", "#n/a", "-", "--", "?", "missing"}


# ============================================================================ loading / cleaning
def canon(col):
    c = re.sub(r"\(.*?\)|\[.*?\]", "", str(col)).lower()
    c = re.sub(r"[^a-z0-9]", "", c)
    return c.replace("circumference", "circ")


def load_measurements(path, id_col=None, unit_scale=1.0, drop_nonpositive=True):
    """-> (clean DataFrame with columns MEAS_NAMES (+ 'id', 'row'), counts dict)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    lookup = {canon(c): c for c in raw.columns}
    want = {n: canon(n) for n in MEAS_NAMES}
    missing = [n for n, k in want.items() if k not in lookup]
    if missing:
        raise KeyError(f"columns not found for {missing}; csv has {list(raw.columns)}")
    num = pd.DataFrame({n: pd.to_numeric(raw[lookup[k]].str.strip().where(
        ~raw[lookup[k]].str.strip().str.lower().isin(NULL_STRINGS)), errors="coerce") for n, k in want.items()})
    num["row"] = np.arange(len(raw)) + 2                      # spreadsheet row number (1 = header)
    num["id"] = raw[id_col].values if id_col else num["row"].values
    vals = num[MEAS_NAMES]
    bad_missing = ~np.isfinite(vals.to_numpy(float)).all(1)
    bad_nonpos = ~bad_missing & (vals.to_numpy(float) <= 0).any(1) & drop_nonpositive
    keep = ~bad_missing & ~bad_nonpos
    out = num[keep].reset_index(drop=True)
    out[MEAS_NAMES] = out[MEAS_NAMES] * unit_scale
    counts = dict(rows=len(raw), dropped_missing=int(bad_missing.sum()), dropped_nonpositive=int(bad_nonpos.sum()),
                  kept=int(keep.sum()))
    return out, counts


# ============================================================================ discrepancy statistics
def discrepancy_stats(got, want, tol, scale_std):
    """got, want (N,7). Returns {measure: {stat: value}} plus an 'ALL' entry.  Extend / replace statistics here."""
    err = got - want
    out = {}
    for i, n in enumerate(MEAS_NAMES + ["ALL"]):
        e = err[:, i] if i < 7 else err.reshape(-1)
        w = want[:, i] if i < 7 else want.reshape(-1)
        a = np.abs(e)
        s = dict(n=len(e), mae=a.mean(), medae=np.median(a), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                 loa_low=e.mean() - 1.96 * e.std(), loa_high=e.mean() + 1.96 * e.std(),
                 p95=np.percentile(a, 95), max=a.max(), mape=100 * (a / np.abs(w)).mean(),
                 rmse_pct=100 * np.sqrt(((e / w) ** 2).mean()))
        sd = scale_std[i] if i < 7 else np.tile(scale_std, len(err))
        s["rmse_zscore"] = np.sqrt(((e / sd) ** 2).mean())      # same scale as training-time prior_meas_rmse
        for t in tol:
            s[f"within_{t:g}mm"] = (a <= t).mean()
        if i < 7 and len(e) > 2:
            s["r2"] = 1 - (e ** 2).sum() / ((w - w.mean()) ** 2).sum()
        out[n] = {k: float(v) for k, v in s.items()}
    return out


def fmt_table(stats, keys, title, say):
    say(f"\n{title}")
    hdr = ["measure"] + keys
    rows = [[n] + [f"{s[k]:.3f}" if k in s else "" for k in keys] for n, s in stats.items()]
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    for r in [hdr, ["-" * x for x in w]] + rows:
        say("  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r)))


# ============================================================================ main
@torch.no_grad()
def decode_rows(model, geo, meas_mm, K, T, chunk=64):
    """meas_mm (N,7) -> dict of arrays for z=0 and K prior draws (all re-measured with the exact code)."""
    zd = model.hparams.z_dim
    N = len(meas_mm)
    m = model.norm_meas(torch.as_tensor(meas_mm, dtype=torch.float32))
    z0 = np.zeros((N, 7))
    zs = np.zeros((N, K, 7))
    plaus = np.zeros((N, K), bool)
    z0_plaus = np.zeros(N, bool)
    div = np.zeros(N)
    lw = np.zeros((N, K))
    for s in range(0, N, chunk):
        mb = m[s:s + chunk]
        b = len(mb)
        z = torch.cat([torch.zeros(b, 1, zd), torch.randn(b, K, zd) * T], 1) if zd else None
        mm = mb[:, None].expand(b, K + 1, 7).reshape(-1, 7)
        c = model.decode(mm, z.reshape(-1, zd) if zd else None)
        raw = (c * model.comp_std + model.comp_mean).double()
        got = geo.get_measures(raw, normalise=False).numpy().reshape(b, K + 1, 7)
        out_box = model.outside_box(c).numpy().reshape(b, K + 1)
        v = model.verts(c).reshape(b, K + 1, -1)[:, 1:]
        z0[s:s + b], zs[s:s + b] = got[:, 0], got[:, 1:]
        z0_plaus[s:s + b], plaus[s:s + b] = ~out_box[:, 0], ~out_box[:, 1:]
        div[s:s + b] = v.std(1).mean(-1).numpy()
        if zd and T != 1.0:
            zz = z[:, 1:].numpy()
            lw[s:s + b] = -0.5 * (zz ** 2).sum(-1) + 0.5 * (zz ** 2).sum(-1) / T ** 2 + zd * np.log(T)
    w = np.exp(lw - lw.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True) if (model.hparams.z_dim and T != 1.0) else np.full((N, K), 1.0 / K)
    return dict(z0=z0, zs=zs, plaus=plaus, z0_plaus=z0_plaus, diversity=div, w=w)


def run(cfg):
    torch.manual_seed(cfg.seed)
    out = Path(HydraConfig.get().runtime.output_dir)
    report = []

    def say(s=""):
        print(s, flush=True)
        report.append(s)

    df, counts = load_measurements(cfg.csv, cfg.id_col, cfg.unit_scale, cfg.drop_nonpositive)
    say(f"csv: {cfg.csv}\nrows {counts['rows']}  dropped (missing/null/non-numeric) {counts['dropped_missing']}  "
        f"dropped (<= 0) {counts['dropped_nonpositive']}  kept {counts['kept']}  (unit_scale={cfg.unit_scale})")
    if df.empty:
        raise SystemExit("no complete rows")

    model = LimbCVAE.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    geo = SSMGeometry(scale=model.hparams.scale, normalise=False)
    K, T = cfg.k_samples, cfg.temperature
    meas = df[MEAS_NAMES].to_numpy(float)
    std = model.meas_std.numpy().astype(float)

    # range the model was trained on (measurements of the generator's own limbs, via the exact code)
    rc = ComponentSampler(model.hparams.scale).raw(cfg.n_range, torch.Generator().manual_seed(cfg.seed))
    ref = np.concatenate([geo.get_measures(rc[i:i + 64], normalise=False).numpy() for i in range(0, len(rc), 64)])
    lo, hi = np.percentile(ref, [0.0, 100.0], axis=0)
    in_range = ((meas >= lo) & (meas <= hi)).all(1)
    say(f"inside the generator's measurement range ({cfg.n_range} limbs): {in_range.sum()}/{len(in_range)} rows "
        f"(out-of-range rows are flagged, NOT removed: all statistics below include them)")
    for i, n in enumerate(MEAS_NAMES):
        say(f"  {n:9s} data {meas[:, i].min():7.1f}..{meas[:, i].max():7.1f}   generator {lo[i]:7.1f}..{hi[i]:7.1f}")

    say(f"\ncheckpoint {cfg.ckpt}\nz_dim={model.hparams.z_dim}  K={K}  T={T}  decoding {len(df)} rows ...")
    r = decode_rows(model, geo, meas, K, T)
    zs, z0, w = r["zs"], r["z0"], r["w"]
    tol = list(cfg.tolerances_mm)

    sets = {"z=0 (the family's centre)": (z0, meas),
            f"z~N(0,I): every sample ({len(df)}x{K})": (zs.reshape(-1, 7), np.repeat(meas, K, 0)),
            "z~N(0,I): mean over z per row": ((w[..., None] * zs).sum(1), meas)}
    summary = {"counts": counts, "n_in_range": int(in_range.sum()), "tolerances_mm": tol}
    keys = ["mae", "medae", "rmse", "bias", "p95", "max", "mape", "rmse_pct", "rmse_zscore"] + [f"within_{t:g}mm" for t in tol] + ["r2"]
    for title, (g, wnt) in sets.items():
        st = discrepancy_stats(g, wnt, tol, std)
        summary[title] = st
        fmt_table(st, keys, f"=== {title}: measured(generated) - requested, mm (mape/rmse_pct in %) ===", say)
    if (~in_range).any() and in_range.any():
        for title, mask in (("inside range", in_range), ("OUTSIDE range", ~in_range)):
            st = discrepancy_stats(zs[mask].reshape(-1, 7), np.repeat(meas[mask], K, 0), tol, std)
            summary[f"samples, {title}"] = st
            fmt_table({"ALL": st["ALL"]}, keys, f"--- z-samples, rows {title} ({mask.sum()} rows) ---", say)

    # spread over z and plausibility
    spread = zs.std(1)
    say("\n=== family ===")
    say("std over z of re-measured values (mean over rows, mm): " +
        "  ".join(f"{n}={spread[:, i].mean():.3f}" for i, n in enumerate(MEAS_NAMES)))
    say(f"diversity (mean per-vertex std over z, mm): {r['diversity'].mean():.2f}")
    say(f"inside generator's plausible box: z=0 {r['z0_plaus'].mean():.1%}   samples {r['plaus'].mean():.1%}")
    worst = np.abs(zs - meas[:, None]).max(-1)                     # (N,K) worst measure per sample
    for t in tol:
        say(f"all 7 measurements within {t:g} mm:  z=0 {(np.abs(z0 - meas).max(1) <= t).mean():.1%}   "
            f"samples {(worst <= t).mean():.1%}   at least one of K samples {(worst.min(1) <= t).mean():.1%}")
    summary["family"] = dict(spread_over_z_mm=dict(zip(MEAS_NAMES, spread.mean(0).tolist())),
                             diversity_vert_std_mm=float(r["diversity"].mean()),
                             plausible_z0=float(r["z0_plaus"].mean()), plausible_samples=float(r["plaus"].mean()))

    # per-row table
    res = df.copy()
    res["in_range"] = in_range
    res["out_of_range_measures"] = ["; ".join(n for n, bad in zip(MEAS_NAMES, ~ok) if bad) for ok in ((meas >= lo) & (meas <= hi))]
    for i, n in enumerate(MEAS_NAMES):
        res[f"{n} z0"] = z0[:, i]
        res[f"{n} mean_z"] = (w * zs[:, :, i]).sum(1)
        res[f"{n} std_z"] = spread[:, i]
        res[f"{n} z0_err"] = z0[:, i] - meas[:, i]
    res["worst_abs_err_z0"] = np.abs(z0 - meas).max(1)
    res["worst_abs_err_mean_over_z"] = worst.mean(1)
    res["plausible_z0"] = r["z0_plaus"]
    res.to_csv(out / "per_row.csv", index=False)
    np.savez_compressed(out / "per_sample.npz", requested=meas, z0=z0, samples=zs, weights=w, in_range=in_range)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    say(f"\nsaved to {out}")


CONFIG = dict(
    ckpt="???",                    # trained LimbCVAE checkpoint
    csv="???",                     # table with Circ 1-4, Width 1-2, Length 1 columns
    id_col=None,                   # optional column carried through to per_row.csv
    unit_scale=1.0,                # multiply csv values by this to get mm (cm -> 10)
    drop_nonpositive=True,         # rows with a value <= 0 are impossible and removed (counted in the report)
    k_samples=16,                  # z ~ N(0, T^2 I) draws per row (plus z = 0)
    temperature=1.0,
    tolerances_mm=[1.0, 2.0, 5.0],
    n_range=3000,                  # generator limbs used to establish the trained measurement range
    seed=0,
    out_dir="${claude_dir:}/outputs/csv/${now:%Y-%m-%d_%H-%M-%S}",
    hydra={"job": {"chdir": False}, "run": {"dir": "${out_dir}"}},
)
ConfigStore.instance().store(name="evaluate_csv", node=CONFIG)


@hydra.main(config_path=None, config_name="evaluate_csv", version_base="1.3")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
