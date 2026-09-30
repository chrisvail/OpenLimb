"""Evaluate the plain-MLP regression baseline with the SAME protocol as evaluate.py, and compare with the CVAE.

    python claude/evaluate_mlp.py ckpt=<mlp best.ckpt> cvae_ckpt=<cvae best.ckpt> cvae_eval=<cvae eval dir>

* Test limbs: the 1024 limbs of the original docker generator (same seed as evaluate.py => identical limbs).
* Every mesh / measurement of a generated limb comes from the ORIGINAL SSM_Driver code.
* Statistics use the same definitions for both models (pooled over individual measurements; per-limb worst case).
* The CVAE numbers come from its saved evaluation arrays (8 random codes per request), MLP has 1 answer per request.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlimb_cvae.common import register_resolvers, setup_env  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate import (in_box_u, load_reference, orig_measures, orig_verts,  # noqa: E402
                      original_test_limbs, shape_metrics)
from openlimb_cvae.lit_cvae import LimbCVAE  # noqa: E402
from openlimb_cvae.lit_mlp import LimbMLP  # noqa: E402

REPORT = []


def say(s=""):
    print(s, flush=True)
    REPORT.append(s)


def stats(got, want):
    """got, want (N,7) in mm. Same definitions as the CVAE rows of the write-up (section 8.3)."""
    err = np.abs(got - want)
    pct = 100 * err / np.abs(want)
    return dict(mean_mm=err.mean(), mean_pct=pct.mean(), median_mm=np.median(err), p99_mm=np.percentile(err, 99),
                p99_pct=np.percentile(pct, 99), worst_mm=err.max(), worst_pct=pct.max(),
                frac_any_gt5mm=100 * (err.max(1) > 5).mean(), frac_any_gt1pct=100 * (pct.max(1) > 1).mean(),
                per_measure_mm=err.mean(0), per_measure_pct=pct.mean(0), per_measure_max_mm=err.max(0),
                per_measure_std_mm=err.std(0), n=len(err))


def timeit(fn, reps):
    for _ in range(max(3, reps // 5)):
        fn()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return 1000 * float(np.median(t))


@hydra.main(config_path="conf", config_name="evaluate_mlp", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    out = Path(HydraConfig.get().runtime.output_dir)
    mlp = LimbMLP.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    ref, names = load_reference()
    faces = ref.face2vert

    say(f"MLP checkpoint: {cfg.ckpt}")
    say(f"generating {cfg.n_limbs} test limbs with the original docker generator (seed {cfg.seed}) ...")
    true_c = original_test_limbs(cfg.n_limbs, cfg.seed)
    true_m = orig_measures(ref, true_c)
    cv = np.load(Path(cfg.cvae_eval) / "eval_data.npz", allow_pickle=True)
    same = np.allclose(cv["true_m"], true_m.numpy(), atol=1e-3)
    say(f"test limbs identical to the CVAE evaluation set: {same}")
    assert same, "test limbs differ from the CVAE evaluation - comparison would not be like-for-like"

    # ---------------------------------------------------------------- A: measurement accuracy
    with torch.no_grad():
        raw = (mlp(mlp.norm_meas(true_m.float())) * mlp.comp_std + mlp.comp_mean).double()
    got = orig_measures(ref, raw).numpy()
    want = true_m.numpy()
    S = stats(got, want)
    C = stats(cv["A_prior_got"], cv["A_prior_want"])          # CVAE, prior samples, 8192 limbs
    say("\n=== A. Measurement accuracy (ORIGINAL SSM_Driver measurements of the predicted limb vs the request) ===")
    say(f"[MLP: {S['n']} limbs, 1 answer per request]")
    err = np.abs(got - want)
    pct = 100 * err / want
    rows = [[n, f"{want[:, i].mean():.1f}", f"{err[:, i].mean():.3f}", f"{err[:, i].std():.3f}", f"{err[:, i].max():.2f}",
             f"{(got - want)[:, i].mean():+.3f}", f"{pct[:, i].mean():.3f}", f"{pct[:, i].std():.3f}", f"{pct[:, i].max():.2f}"]
            for i, n in enumerate(names)]
    rows.append(["ALL", "", f"{err.mean():.3f}", f"{err.std():.3f}", f"{err.max():.2f}", f"{(got - want).mean():+.3f}",
                 f"{pct.mean():.3f}", f"{pct.std():.3f}", f"{pct.max():.2f}"])
    hdr = ["measure", "mean mm", "|err| mean", "|err| std", "|err| max", "bias mm", "% mean", "% std", "% max"]
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    for r in [hdr] + rows:
        say("  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r)))

    say("\nHeadline statistics (identical definitions for both models):")
    keys = [("mean_mm", "mean error (mm)"), ("mean_pct", "mean error (%)"), ("median_mm", "median error (mm)"),
            ("p99_mm", "99 % of measurements within (mm)"), ("p99_pct", "99 % of measurements within (%)"),
            ("worst_mm", "worst error (mm)"), ("worst_pct", "worst error (%)"),
            ("frac_any_gt5mm", "limbs with any measurement > 5 mm off (%)"),
            ("frac_any_gt1pct", "limbs with any measurement > 1 % off (%)")]
    for k, lab in keys:
        say(f"  {lab:<48} MLP {S[k]:>9.3f}    CVAE {C[k]:>9.3f}")

    # ---------------------------------------------------------------- B: plausibility
    over = (in_box_u(mlp, raw).abs() - 1).clamp_min(0).max(-1).values
    cv_over = cv["B_over"]
    say("\n=== B. Plausibility (inside the generator's parameter box) ===")
    say(f"  outside box: MLP {(over > 0).float().mean():.1%}  (>5 % of half-range: {(over > 0.05).float().mean():.1%}, "
        f"max exceedance {over.max():.3f})   CVAE {(cv_over > 0).mean():.1%}")

    # ---------------------------------------------------------------- C: shape distance to the true limb
    ns = cfg.n_shape_limbs
    tv, pv = orig_verts(ref, true_c[:ns]), orig_verts(ref, raw[:ns])
    ms = [shape_metrics(pv[i], tv[i], faces) for i in range(ns)]
    ck = list(cv["C_keys"])
    say(f"\n=== C. Shape distance to the TRUE limb ({ns} limbs, mm) ===")
    say(f"  {'':<28}{'chamfer':>10}{'hausdorff':>11}{'pt-to-plane':>13}{'vertex rmse':>13}")
    say(f"  {'MLP (single answer)':<28}" + "".join(f"{np.mean([m[k] for m in ms]):>{w_}.2f}" for k, w_ in
                                                     (("chamfer", 10), ("hausdorff", 11), ("point_to_plane", 13), ("vertex_rmse", 13))))
    cp = cv["C_prior"]
    best = np.take_along_axis(cp, cp[:, :, 0].argmin(1)[:, None, None].repeat(cp.shape[2], 2), 1)[:, 0]
    for lab, arr in (("CVAE encoder z (sees limb)", cv["C_post"]), ("CVAE prior, mean of 4", cp.reshape(-1, cp.shape[2])),
                     ("CVAE prior, best of 4", best)):
        say(f"  {lab:<28}" + "".join(f"{arr[:, ck.index(k)].mean():>{w_}.2f}" for k, w_ in
                                    (("chamfer", 10), ("hausdorff", 11), ("point_to_plane", 13), ("vertex_rmse", 13))))

    # ---------------------------------------------------------------- D: speed, same machine
    cvae = LimbCVAE.load_from_checkpoint(cfg.cvae_ckpt, map_location="cpu").eval()
    say(f"\n=== D. Time per request, this machine ({torch.get_num_threads()} threads, median of repeats, ms) ===")
    T = {}
    with torch.no_grad():
        for n, reps in ((1, 300), (64, 300), (64000, 15)):
            m = mlp.norm_meas(true_m[:1].float()).expand(n, -1).contiguous()
            T[f"mlp_{n}"] = timeit(lambda: mlp(m), reps)
            T[f"cvae_{n}"] = timeit(lambda: cvae.decode(m, torch.randn(n, cvae.hparams.z_dim)), reps)
    for n in (1, 64, 64000):
        say(f"  {n:>6} limbs: MLP {T[f'mlp_{n}']:.3f} ms   CVAE {T[f'cvae_{n}']:.3f} ms")

    # ---------------------------------------------------------------- figure
    nm = [x.replace("Circumference", "Circ").replace("Length", "Len").replace("Width", "Wid") for x in names]
    cw, cg = cv["A_prior_want"], cv["A_prior_got"]
    cerr, cpct = np.abs(cg - cw), 100 * np.abs(cg - cw) / cw
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (e, colour, lab) in enumerate(((pct, "#d62728", "MLP regression"), (cpct, "#1f77b4", "CVAE (prior samples)"))):
        pos = np.arange(7) + (j - 0.5) * 0.36
        bp = ax[0].boxplot([e[:, i] for i in range(7)], positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                           whis=(1, 99), medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set_facecolor(colour); b.set_alpha(0.6)
        ax[0].plot([], [], color=colour, lw=6, alpha=0.6, label=lab)
        ax[1].plot(np.sort(e.ravel()), np.linspace(0, 100, e.size), color=colour, label=lab)
    ax[2].plot(np.sort(err.max(1)), np.linspace(0, 100, len(err)), color="#d62728", label="MLP regression")
    ax[2].plot(np.sort(cerr.max(1)), np.linspace(0, 100, len(cerr)), color="#1f77b4", label="CVAE (prior samples)")
    ax[0].set_xticks(range(7)); ax[0].set_xticklabels(nm, rotation=30); ax[0].set_ylabel("% error"); ax[0].axhline(1, color="r", ls="--", lw=1)
    ax[0].set_title("Percentage error per measurement (whiskers 1-99 %)"); ax[0].legend()
    ax[1].axvline(1, color="r", ls="--", lw=1); ax[1].set_xlim(0, 5); ax[1].set_xlabel("% error (all measurements pooled)")
    ax[1].set_ylabel("cumulative % of measurements"); ax[1].set_title("Error CDF"); ax[1].legend(loc="lower right")
    ax[2].axvline(5, color="r", ls="--", lw=1); ax[2].set_xlim(0, 20); ax[2].set_xlabel("worst of the 7 measurement errors per limb (mm)")
    ax[2].set_ylabel("cumulative % of limbs"); ax[2].set_title("Per-limb worst error"); ax[2].legend(loc="lower right")
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("Plain MLP regression vs CVAE - original measurement code, same 1024 test limbs")
    fig.tight_layout()
    figdir = Path(cfg.fig_dir); figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig_compare_mlp.png", dpi=130); plt.close(fig)

    res = dict(mlp={k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in S.items()},
               cvae={k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in C.items()}, timing_ms=T,
               outside_box=dict(mlp=float((over > 0).float().mean()), cvae=float((cv_over > 0).mean())),
               shape_mlp={k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
    (out / "compare.json").write_text(json.dumps(res, indent=2))
    (out / "report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
