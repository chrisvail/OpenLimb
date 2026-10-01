"""Evaluate the gradient-free (GA) route with the SAME protocol and test limbs as evaluate.py / evaluate_mlp.py.

    python claude/evaluate_ga.py

Methods (openlimb_cvae/ga.py):
* GA (notebook)             - genetic_alg.ipynb's pygad set-up, unchanged: what GA_results*.csv contain
* GA (notebook) + LM fixed  - plus the notebook's Levenberg-Marquardt refinement with its faults fixed
* GA (improved)             - same library, objective and budget; plausible-box genes, shrinking Gaussian mutation
* GA (improved) + GN        - plus projected Gauss-Newton inside the box
* GN (box centre)           - no GA at all: Gauss-Newton from the centre of the box
* GN (random start)         - Gauss-Newton from k random plausible limbs per request -> a family, like the CVAE

The optimisers use FastMeasure (identical numbers, faster); every reported number re-measures the final limbs
with the ORIGINAL SSM_Driver code, exactly as for the CVAE and the MLP.
"""
import json
import multiprocessing as mp
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
import pygad  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate import load_reference, orig_measures, orig_verts, original_test_limbs, shape_metrics  # noqa: E402
from evaluate_mlp import stats  # noqa: E402
from openlimb_cvae import ga  # noqa: E402
from openlimb_cvae.fast_measure import BoxMap  # noqa: E402
from openlimb_cvae.geometry import load_transforms  # noqa: E402
from openlimb_cvae.lit_cvae import LimbCVAE  # noqa: E402

REPORT = []


def say(s=""):
    print(s, flush=True)
    REPORT.append(s)


def outside(raw):
    """Exceedance beyond the generator's box (0 = inside). 1e-9 tolerance: GN limbs sit exactly on the bound."""
    over = (BoxMap().to_u(torch.as_tensor(raw)).abs() - 1).max(-1).values.numpy()
    return np.where(over > 1e-9, over, 0.0)


def family_spread(v):
    """(K, V, 3) -> mean over members of the RMS vertex distance to the family mean (write-up's 'family spread')."""
    return (v - v.mean(0)).pow(2).sum(-1).mean(-1).sqrt().mean().item()


def time_notebook_original(ref, true_c, true_m, seed):
    """genetic_alg.ipynb as written: unbatched fitness through SSM_Driver.MeasurementLoss (measures the true limb too)."""
    from SSM_Driver import MeasurementLoss
    loss = MeasurementLoss(ref)
    comp_tf, _ = load_transforms(True)
    t = time.perf_counter()
    fit = lambda inst, sol, idx: -loss(torch.tensor(sol).double()[None] * comp_tf[1] + comp_tf[0], true_c[None]).item()
    g = pygad.GA(num_generations=50, num_parents_mating=4, fitness_func=fit, sol_per_pop=32, num_genes=11,
                 gene_space={"low": -3, "high": 3}, parent_selection_type="sss", keep_parents=2,
                 mutation_percent_genes=20, crossover_type="single_point", mutation_type="random",
                 random_seed=seed, suppress_warnings=True)
    g.run()
    g.best_solution()
    return time.perf_counter() - t


@hydra.main(config_path="conf", config_name="evaluate_ga", version_base="1.3")
def main(cfg: DictConfig):
    out = Path(HydraConfig.get().runtime.output_dir)
    ref, names = load_reference()
    faces = ref.face2vert
    say(f"generating {cfg.n_limbs} test limbs with the original docker generator (seed {cfg.seed}) ...")
    true_c = original_test_limbs(cfg.n_limbs, cfg.seed)
    true_m = orig_measures(ref, true_c)
    cv = np.load(Path(cfg.cvae_eval) / "eval_data.npz", allow_pickle=True)
    same = np.allclose(cv["true_m"], true_m.numpy(), atol=1e-3)
    say(f"test limbs identical to the CVAE / MLP evaluation set: {same}")
    assert same, "test limbs differ from the CVAE evaluation - comparison would not be like-for-like"
    if cfg.n_requests:                                   # smoke tests only; the CVAE rows still use all 1024
        true_c, true_m = true_c[:cfg.n_requests], true_m[:cfg.n_requests]
    N, K = len(true_m), cfg.k_random
    targets = true_m.numpy()
    kw_imp = OmegaConf.to_container(cfg.ga_improved)
    say(f"improved GA settings: {kw_imp}")

    # ------------------------------------------------------------------ optimise every request
    R = {}
    seeds = cfg.seed + np.arange(N)
    starts = (torch.rand(N * K, 11, generator=torch.Generator().manual_seed(cfg.seed), dtype=torch.float64) * 2 - 1).numpy()
    with mp.Pool(cfg.workers, initializer=ga._init_worker) as pool:
        for variant, tg, sd, st, kw in (("notebook", targets, seeds, None, {}),
                                        ("improved", targets, seeds, None, kw_imp),
                                        ("gn", targets, seeds, None, {}),
                                        ("gn_random", np.repeat(targets, K, 0), np.repeat(seeds, K), starts, {})):
            t = time.perf_counter()
            R.update(ga.run(pool, variant, tg, sd, starts=st, **kw))
            say(f"  {variant}: {time.perf_counter() - t:.0f} s wall")

    # ------------------------------------------------------------------ A/B: accuracy + plausibility, original code
    S, O = {}, {}
    for stage, r in R.items():
        want = targets if len(r["raw"]) == N else np.repeat(targets, K, 0)
        r["got"] = orig_measures(ref, torch.tensor(r["raw"])).numpy()
        S[stage], O[stage] = stats(r["got"], want), outside(r["raw"])
    C = stats(cv["A_prior_got"], cv["A_prior_want"])
    mlp = json.loads((Path(cfg.mlp_eval) / "compare.json").read_text())

    say("\n=== A/B. Measurement accuracy (ORIGINAL SSM_Driver measurements vs the request) and plausibility ===")
    keys = [("mean_mm", "mean mm"), ("mean_pct", "mean %"), ("median_mm", "median mm"), ("p99_mm", "p99 mm"),
            ("p99_pct", "p99 %"), ("worst_mm", "worst mm"), ("worst_pct", "worst %"), ("frac_any_gt5mm", ">5mm %"),
            ("frac_any_gt1pct", ">1% %")]
    rows = [(st, S[st], 100 * (O[st] > 0).mean(), R[st]["evals"].mean()) for st in R]
    rows += [("CVAE (8 random codes)", C, 100 * (cv["B_over"] > 0).mean(), float("nan")),
             ("plain MLP", mlp["mlp"], 100 * mlp["outside_box"]["mlp"], float("nan"))]
    say(f"{'method':<28}" + "".join(f"{lab:>11}" for _, lab in keys) + f"{'outside %':>11}{'evals':>8}{'n':>7}")
    for st, s, ob, ev in rows:
        say(f"{st:<28}" + "".join(f"{s[k]:>11.4g}" for k, _ in keys) + f"{ob:>11.1f}{ev:>8.0f}{s['n']:>7.0f}")
    say("\nper-measurement mean |error| (mm):  " + "  ".join(n.replace("Circumference", "Circ") for n in names))
    for st, s, _, _ in rows:
        say(f"  {st:<28}" + "  ".join(f"{x:.3g}" for x in s["per_measure_mm"]))
    say("\nexceedance beyond the box of the implausible limbs (fraction of half-width): " +
        "; ".join(f"{st}: median {np.median(O[st][O[st] > 0]):.2f}, max {O[st].max():.2f}" for st in R if (O[st] > 0).any()))

    # ------------------------------------------------------------------ C: distance to the true limb
    ns = cfg.n_shape_limbs
    tv = orig_verts(ref, true_c[:ns])
    SH = {}
    say(f"\n=== C. Shape distance to the TRUE limb ({ns} limbs, mm) ===")
    say(f"  {'':<28}{'chamfer':>10}{'hausdorff':>11}{'pt-to-plane':>13}{'vertex rmse':>13}")
    for st in [s for s in R if len(R[s]["raw"]) == N]:
        pv = orig_verts(ref, torch.tensor(R[st]["raw"][:ns]))
        ms = [shape_metrics(pv[i], tv[i], faces) for i in range(ns)]
        SH[st] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        say(f"  {st:<28}" + "".join(f"{SH[st][k]:>{w}.2f}" for k, w in
                                    (("chamfer", 10), ("hausdorff", 11), ("point_to_plane", 13), ("vertex_rmse", 13))))

    # ------------------------------------------------------------------ D: families (GN random start vs CVAE), K each
    cvae = LimbCVAE.load_from_checkpoint(cfg.cvae_ckpt, map_location="cpu").eval()
    nf = min(500, N)
    g_raw = R["GN (random start)"]["raw"].reshape(N, K, 11)
    torch.manual_seed(cfg.seed)
    F = {k: [] for k in ("gn_spread", "cvae_spread", "gn_true", "cvae_true", "gn_to_cvae", "cvae_to_gn")}
    for i in range(nf):
        with torch.no_grad():
            m = cvae.norm_meas(true_m[i:i + 1].float()).expand(K, -1)
            c_raw = (cvae.decode(m, torch.randn(K, cvae.hparams.z_dim)) * cvae.comp_std + cvae.comp_mean).double()
        gv, cvv, ti = orig_verts(ref, torch.tensor(g_raw[i])), orig_verts(ref, c_raw), orig_verts(ref, true_c[i:i + 1])
        F["gn_spread"].append(family_spread(gv))
        F["cvae_spread"].append(family_spread(cvv))
        rm = lambda a, b: (torch.cdist(a.flatten(1), b.flatten(1)) / a.shape[1] ** 0.5)
        F["gn_true"].append(rm(ti, gv).min().item())
        F["cvae_true"].append(rm(ti, cvv).min().item())
        F["gn_to_cvae"].append(rm(gv, cvv).min(1).values.mean().item())
        F["cvae_to_gn"].append(rm(cvv, gv).min(1).values.mean().item())
    F = {k: np.array(v) for k, v in F.items()}
    say(f"\n=== D. Families: {K} limbs per request, {nf} requests (vertex RMSE, mm) ===")
    say(f"  family spread:               GN random start {F['gn_spread'].mean():.2f}   CVAE {F['cvae_spread'].mean():.2f}")
    say(f"  true limb -> nearest member: GN random start {F['gn_true'].mean():.2f}   CVAE {F['cvae_true'].mean():.2f}")
    say(f"  GN member -> nearest CVAE member {F['gn_to_cvae'].mean():.2f};  CVAE member -> nearest GN member {F['cvae_to_gn'].mean():.2f}")

    # ------------------------------------------------------------------ timing, main process, all threads
    say(f"\n=== Time per request, this machine ({torch.get_num_threads()} threads, median over {cfg.n_timing} requests) ===")
    prob = ga.Problem()
    T = {}
    for variant, kw in (("notebook", {}), ("improved", kw_imp), ("gn", {})):
        per = {}
        for i in range(cfg.n_timing):
            prob.n_evals = 0
            for st, (_, ev, sec) in ga.solve(prob, variant, true_m[i], int(seeds[i]), **kw).items():
                per.setdefault(st, []).append(sec)
        T.update({st: float(np.median(v)) for st, v in per.items()})
    T["GA (notebook), as written: original code, unbatched"] = float(np.median(
        [time_notebook_original(ref, true_c[i], true_m[i], int(seeds[i])) for i in range(cfg.n_timing_original)]))
    for st, v in T.items():
        say(f"  {st:<52} {v:8.3f} s")
    say(f"  CVAE: {mlp['timing_ms']['cvae_1']:.2f} ms for 1 limb, {mlp['timing_ms']['cvae_64']:.2f} ms for 64 (evaluate_mlp.py)")

    # ------------------------------------------------------------------ figure
    figure(R, S, O, cv, targets, K, Path(cfg.fig_dir))
    np.savez_compressed(out / "eval_data.npz", targets=targets, true_c=true_c.numpy(),
                        **{f"{k}|{st}": v for st, r in R.items() for k, v in r.items()}, **{f"F_{k}": v for k, v in F.items()})
    js = lambda d: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}
    (out / "ga.json").write_text(json.dumps(dict(stats={k: js(v) for k, v in S.items()}, cvae=js(C),
                                                 outside_pct={k: float(100 * (v > 0).mean()) for k, v in O.items()},
                                                 evals={k: float(r["evals"].mean()) for k, r in R.items()},
                                                 shape=SH, families={k: float(v.mean()) for k, v in F.items()},
                                                 timing_s=T, improved_settings=kw_imp), indent=2))
    (out / "report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nsaved to {out}")


def figure(R, S, O, cv, targets, K, figdir):
    # categorical slots in fixed order (validated palette), line style as a second cue
    series = [("CVAE (random codes)", "#2a78d6", "-"), ("GA (notebook)", "#eb6834", "-"),
              ("GA (notebook) + LM (fixed)", "#1baf7a", "--"), ("GA (improved)", "#eda100", "-"),
              ("GA (improved) + GN", "#e87ba4", "--"), ("GN (box centre)", "#4a3aa7", ":")]
    err = {st: np.abs(R[st]["got"] - targets) for st in R if len(R[st]["raw"]) == len(targets)}
    err["CVAE (random codes)"] = np.abs(cv["A_prior_got"] - cv["A_prior_want"])
    ink, muted = "#0b0b0b", "#898781"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    for st, col, ls in series:
        e = np.clip(err[st], 1e-8, None)
        ax[0].plot(np.sort(e.ravel()), np.linspace(0, 100, e.size), color=col, ls=ls, lw=2, label=st)
        ax[1].plot(np.sort(e.max(1)), np.linspace(0, 100, len(e)), color=col, ls=ls, lw=2, label=st)
    for a, x, lab in ((ax[0], None, "|error| of each measurement (mm, log; < 1e-8 drawn at 1e-8)"),
                      (ax[1], 5, "worst of the 7 errors per limb (mm, log; < 1e-8 drawn at 1e-8)")):
        a.set_xscale("log"); a.set_xlim(5e-9, 30); a.set_xlabel(lab); a.grid(alpha=0.25, color=muted)
        if x:
            a.axvline(x, color=muted, ls="--", lw=1)
    ax[0].set_ylabel("cumulative % of measurements"); ax[0].set_title("Measurement error, all measurements pooled")
    ax[1].set_ylabel("cumulative % of limbs"); ax[1].set_title("Per-limb worst error (dashed: 5 mm)")
    labs = [s for s, _, _ in series]
    vals = [100 * (cv["B_over"] > 0).mean() if s.startswith("CVAE") else 100 * (O[s] > 0).mean() for s in labs]
    y = np.arange(len(labs))[::-1]
    ax[2].barh(y, vals, color=[c for _, c, _ in series], height=0.6)
    for yi, v in zip(y, vals):
        ax[2].text(v + 1.5, yi, f"{v:.1f} %", va="center", color=ink, fontsize=9)
    ax[2].set_yticks(y); ax[2].set_yticklabels(labs, fontsize=9, color=ink); ax[2].set_xlim(0, 115)
    ax[2].set_xlabel("% of returned limbs outside the plausible box"); ax[2].set_title("Plausibility")
    ax[2].grid(axis="x", alpha=0.25, color=muted)
    fig.suptitle("Optimisation (GA, GN) vs CVAE - original measurement code, same 1024 test limbs", color=ink)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig_compare_ga.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
