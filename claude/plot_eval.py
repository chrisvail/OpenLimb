"""Figures for the evaluation produced by evaluate.py.

    python claude/plot_eval.py <eval_dir> --out claude/report/figs
    python claude/plot_eval.py <eval_dir> --out claude/report/figs --compare "w_plaus=1@<dir>" "w_plaus=10@<dir>"
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C_PRIOR, C_POST, C_REF, C_TRUE = "#1f77b4", "#ff7f0e", "#2ca02c", "#111111"


def load(d):
    return np.load(Path(d) / "eval_data.npz", allow_pickle=True)


def short(names):
    return [n.replace("Circumference", "Circ").replace("Length", "Len").replace("Width", "Wid") for n in names]


def fig_a(D, out):
    names = short(D["names"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (key, colour, lab) in enumerate((("prior", C_PRIOR, "random z ~ N(0,I)"), ("post", C_POST, "encoded z = mu(m*, c)"))):
        err = np.abs(D[f"A_{key}_got"] - D[f"A_{key}_want"])
        pct = 100 * err / D[f"A_{key}_want"]
        pos = np.arange(7) + (j - 0.5) * 0.36
        for a, data in ((ax[0], err), (ax[1], pct)):
            bp = a.boxplot([data[:, i] for i in range(7)], positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                           whis=(1, 99), medianprops=dict(color="k"))
            for b in bp["boxes"]:
                b.set_facecolor(colour); b.set_alpha(0.6)
        ax[2].plot(np.sort(pct.ravel()), np.linspace(0, 100, pct.size), color=colour, label=f"{lab} (n={len(err)})")
    for a, t, yl in ((ax[0], "Absolute error |M(c_hat) - m*|", "mm"), (ax[1], "Relative error 100|M(c_hat) - m*| / m*", "%")):
        a.set_xticks(range(7)); a.set_xticklabels(names, rotation=30); a.set_title(t); a.set_ylabel(yl)
    ax[0].plot([], [], color=C_PRIOR, lw=6, alpha=0.6, label="random z"); ax[0].plot([], [], color=C_POST, lw=6, alpha=0.6, label="encoded z")
    ax[0].legend(); ax[1].axhline(1, color="r", ls="--", lw=1); ax[1].text(6.4, 1.03, "1 % target", color="r", ha="right")
    ax[2].axvline(1, color="r", ls="--", lw=1); ax[2].set_xlim(0, 3); ax[2].set_xlabel("% error (all 7 measurements pooled)")
    ax[2].set_ylabel("cumulative % of limbs"); ax[2].set_title("Error CDF"); ax[2].legend(loc="lower right")
    fig.suptitle("A. Measurement error of generated limbs: exact measurements M(c_hat) vs requested m* (boxes 25-75 %, whiskers 1-99 %)")
    fig.tight_layout(); fig.savefig(out / "fig_A_measurement_accuracy.png"); plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    for i, a in enumerate(ax.ravel()[:7]):
        w, g = D["A_prior_want"][:, i], D["A_prior_got"][:, i]
        a.hexbin(w, g - w, gridsize=40, cmap="Blues", mincnt=1, bins="log")
        a.axhline(0, color="k", lw=0.8); a.set_title(short(D["names"])[i]); a.set_xlabel("requested m* (mm)"); a.set_ylabel("signed error M(c_hat) - m* (mm)")
    ax.ravel()[7].axis("off")
    fig.suptitle("A2. Signed error vs requested value (random z; darker = more limbs)"); fig.tight_layout()
    fig.savefig(out / "fig_A2_error_vs_value.png"); plt.close(fig)


def fig_b(D, out):
    over = D["B_over"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(over[over > 0], bins=60, color=C_PRIOR)
    ax[0].set_xlabel("exceedance e = max_j |u_j| - 1 (fraction of box half-width)"); ax[0].set_ylabel("limbs (only those outside the box)")
    ax[0].set_title(f"outside box: {(over > 0).mean():.1%}   >5 %: {(over > 0.05).mean():.1%}   (n={len(over)})")
    ax[1].plot(np.sort(over), np.linspace(0, 100, len(over)), color=C_PRIOR); ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel("exceedance e (0 = inside box)"); ax[1].set_ylabel("cumulative % of limbs"); ax[1].set_title("CDF")
    fig.suptitle("B. Plausibility of random-z limbs: distance outside the generator's skin-mode box"); fig.tight_layout()
    fig.savefig(out / "fig_B_plausibility.png"); plt.close(fig)


def fig_c(D, out):
    keys = list(D["C_keys"]); post = D["C_post"]; prior = D["C_prior"]
    best = np.take_along_axis(prior, prior[:, :, 0].argmin(1)[:, None, None].repeat(prior.shape[2], 2), 1)[:, 0]
    sets = [("encoded z", post, C_POST), ("random z, every sample", prior.reshape(-1, prior.shape[2]), C_PRIOR),
            ("random z, best of %d" % prior.shape[1], best, C_REF)]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, k, t in zip(ax, ("chamfer", "hausdorff", "point_to_plane", "vertex_rmse"),
                       ("Chamfer", "Hausdorff", "Point-to-plane (mean)", "Vertex-correspondence RMSE")):
        j = keys.index(k)
        hi = np.percentile(np.concatenate([s[1][:, j] for s in sets]), 99.5)
        for lab, s, c in sets:
            a.hist(s[:, j], bins=np.linspace(0, hi, 60), alpha=0.55, color=c, density=True, label=f"{lab}  mean {s[:, j].mean():.2f}")
        a.set_title(t); a.set_xlabel("mm"); a.legend(fontsize=7)
    ax[0].axvline(float(np.mean(D["C_ctx"])), color="k", ls="--"); ax[0].text(float(np.mean(D["C_ctx"])), 0, " unrelated limbs", rotation=90, va="bottom")
    fig.suptitle(f"C. Distance from generated limb to the true limb ({len(post)} limbs). Encoded z has seen the true limb; random-z limbs are "
                 "other limbs with the same measurements"); fig.tight_layout(); fig.savefig(out / "fig_C_shape_distance.png"); plt.close(fig)


def fig_d(D, out):
    names = short(D["names"])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    a = ax[0, 0]
    a.hist(D["D_rms_dev"], bins=40, alpha=0.6, color=C_PRIOR, label=f"spread S (RMS dist. to family mean)  ({D['D_rms_dev'].mean():.2f} mm)")
    a.hist(D["D_pair_mean"], bins=40, alpha=0.6, color=C_POST, label=f"RMSE between two members ({D['D_pair_mean'].mean():.2f} mm)")
    a.hist(D["D_chamfer"], bins=40, alpha=0.6, color=C_REF, label=f"chamfer between two members ({D['D_chamfer'].mean():.2f} mm)")
    a.set_xlabel("mm"); a.set_ylabel(f"measurement sets (n={len(D['D_rms_dev'])})"); a.legend(fontsize=7); a.set_title("D2. Shape differences within a family")
    a = ax[0, 1]
    a.boxplot([D["D_meas_std"][:, i] for i in range(7)], whis=(1, 99), showfliers=False, patch_artist=True, boxprops=dict(facecolor=C_PRIOR, alpha=0.6),
              medianprops=dict(color="k"))
    a.set_xticklabels(names, rotation=30); a.set_ylabel("std of M over the 64 family members (mm)"); a.set_title("D1. Measurement scatter within a family")
    a = ax[0, 2]
    f = D["D_pca_frac"]; x = np.arange(1, f.shape[1] + 1)
    a.bar(x, f.mean(0), color=C_PRIOR, alpha=0.7); a.errorbar(x, f.mean(0), yerr=[f.mean(0) - np.percentile(f, 5, 0), np.percentile(f, 95, 0) - f.mean(0)],
                                                            fmt="none", ecolor="k", capsize=2)
    a.set_xlabel("principal direction of within-family shape variation"); a.set_ylabel("variance fraction (mean, 5-95 % range)")
    a.set_title(f"D3. How many independent shape directions? (PR {D['D_pr'].mean():.2f})")
    a = ax[1, 0]
    a.hist(D["D_null_frac"], bins=40, color=C_PRIOR); a.set_xlabel("null-space fraction rho (1 = z never changes measurements)")
    a.set_title(f"D4. Is variation measurement-preserving? (mean {D['D_null_frac'].mean():.3f})"); a.set_ylabel("measurement sets")
    a = ax[1, 1]
    gs, rs = D["R_gen_spread"], D["R_ref_spread"]
    a.scatter(rs, gs, s=14, color=C_REF); lim = [0, max(gs.max(), rs.max()) * 1.1]; a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("reference family spread S_ref (mm)"); a.set_ylabel("generated family spread S (mm)")
    a.set_title(f"D5. Spread: generated vs reference family (n={len(gs)} sets; dashed = equal)")
    a = ax[1, 2]
    vals = [D["R_baseline"].mean(), D["R_coverage"].mean(), D["R_precision"].mean(), D["D_true_to_gen"].mean()]
    errs = [D["R_baseline"].std(), D["R_coverage"].std(), D["R_precision"].std(), D["D_true_to_gen"].std()]
    a.bar(range(4), vals, yerr=errs, color=[C_TRUE, C_PRIOR, C_POST, C_REF], alpha=0.7, capsize=3)
    a.set_xticks(range(4)); a.set_xticklabels(["baseline\nref -> nearest\nother ref", "coverage\nref -> nearest\ngenerated", "precision\ngenerated ->\nnearest ref",
                                               "true limb ->\nnearest generated"], fontsize=8)
    a.set_ylabel("vertex RMSE (mm), mean ± std over sets"); a.set_title("D5. Nearest-neighbour distances (lower = closer)")
    fig.suptitle(f"D. Families: {len(D['D_rms_dev'])} measurement sets m* x 64 random z each ({len(D['D_rms_dev']) * 64} limbs; exact meshes and measurements)")
    fig.tight_layout(); fig.savefig(out / "fig_D_family_diversity.png"); plt.close(fig)


def _outline(v, h, band=2.5):
    """Closed cross-section outline of mesh vertices v (V,3) in a thin z-band around height h."""
    p = v[np.abs(v[:, 2] - h) < band][:, :2]
    if len(p) < 8:
        return None
    ang = np.arctan2(p[:, 1] - p[:, 1].mean(), p[:, 0] - p[:, 0].mean())
    p = p[np.argsort(ang)]
    return np.vstack([p, p[:1]])


def fig_silhouettes(D, out):
    sets = [k[:-4] for k in D.files if k.startswith("set") and k.endswith("_gen")]
    if not sets:
        return
    fig, ax = plt.subplots(len(sets), 4, figsize=(16, 4.3 * len(sets)), squeeze=False)
    cm = plt.cm.viridis(np.linspace(0, 1, 8))
    for r, s in enumerate(sets):
        gen, tru = D[f"{s}_gen"], D[f"{s}_true"]
        ref = D[f"{s}_ref"] if f"{s}_ref" in D.files else None
        m = D["true_m"][int(s[3:])]
        z0, z1 = np.percentile(tru[:, 2], [5, 95])
        for c, q in enumerate((0.2, 0.5, 0.8)):
            a, h = ax[r, c], z0 + q * (z1 - z0)
            if ref is not None:
                for v in ref[:8]:
                    o = _outline(v, h)
                    if o is not None:
                        a.plot(o[:, 0], o[:, 1], color="0.55", lw=0.8, ls="--", alpha=0.8)
            for k in range(len(gen)):
                o = _outline(gen[k], h)
                if o is not None:
                    a.plot(o[:, 0], o[:, 1], color=cm[k], lw=1.1, alpha=0.9)
            o = _outline(tru, h)
            if o is not None:
                a.plot(o[:, 0], o[:, 1], color="k", lw=1.6)
            a.set_aspect("equal"); a.set_title(f"set {s[3:]}: cross-section at height h = {h:.0f} mm"); a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
        a = ax[r, 3]
        dev = np.linalg.norm(gen - gen.mean(0), axis=-1)
        lines = ["black = true limb", "coloured = 8 generated limbs (different z)", "grey dashed = 8 reference limbs",
                 "  (projected SSM limbs, not patients)", "",
                 "shared measurements m* (mm):"] + [f"  {n}: {v:.1f}" for n, v in zip(short(D["names"]), m)]
        lines += ["", "distance of each vertex from family mean", f"  mean {dev.mean():.1f} mm, max {dev.max():.1f} mm"]
        txt = chr(10).join(lines)
        a.axis("off"); a.text(0, 0.95, txt, va="top", family="monospace", fontsize=8)
    fig.suptitle("D6. Example families: cross-sections of limbs that share the same measurements"); fig.tight_layout()
    fig.savefig(out / "fig_D6_example_families.png"); plt.close(fig)


def fig_e(D, out):
    zd = int(D["zd"]); sh, dr = D["E_shape"], D["E_drift"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].boxplot(list(sh), whis=(1, 99), showfliers=False, patch_artist=True, boxprops=dict(facecolor=C_PRIOR, alpha=0.6), medianprops=dict(color="k"))
    ax[0].set_xticklabels([f"z{d + 1}" for d in range(zd)]); ax[0].set_ylabel("RMS vertex distance, z_k = 0 vs z_k = ±2 (mm)"); ax[0].set_title("Shape change when one latent z_k moves 0 -> ±2")
    ax[1].boxplot(list(dr), whis=(1, 99), showfliers=False, patch_artist=True, boxprops=dict(facecolor=C_POST, alpha=0.6), medianprops=dict(color="k"))
    ax[1].set_xticklabels([f"z{d + 1}" for d in range(zd)]); ax[1].set_ylabel("largest change in any measurement (mm)"); ax[1].set_title("Measurement change for the same move")
    ax[2].bar(np.arange(zd) - 0.2, D["mu"].var(0), 0.4, color=C_PRIOR, label="Var of encoder mean mu_k"); ax[2].bar(np.arange(zd) + 0.2, D["kl"], 0.4, color=C_REF, label="KL (nats)")
    ax[2].set_xticks(range(zd)); ax[2].set_xticklabels([f"z{d + 1}" for d in range(zd)]); ax[2].legend(); ax[2].set_title("Is each latent used? (0 = unused)")
    fig.suptitle(f"E. Latent traversal ({sh.shape[1] // 2} measurement sets; other latents held at 0)"); fig.tight_layout(); fig.savefig(out / "fig_E_latent_traversal.png"); plt.close(fig)


def summary(D):
    err = np.abs(D["A_prior_got"] - D["A_prior_want"]); pct = 100 * err / D["A_prior_want"]
    prior = D["C_prior"]; keys = list(D["C_keys"])
    return {"outside box (%)": 100 * (D["B_over"] > 0).mean(), "outside box by >5 % (%)": 100 * (D["B_over"] > 0.05).mean(),
            "mean |err| (mm)": err.mean(), "mean |err| (%)": pct.mean(), "99th pct err (%)": np.percentile(pct, 99),
            "family spread S (mm)": D["D_rms_dev"].mean(), "null-space fraction rho": D["D_null_frac"].mean(),
            "encoded-z chamfer (mm)": D["C_post"][:, keys.index("chamfer")].mean(),
            "random-z chamfer (mm)": prior[:, :, keys.index("chamfer")].mean()}


def fig_compare(items, out):
    S = {lab: summary(load(d)) for lab, d in items}
    metrics = list(next(iter(S.values())).keys())
    fig, ax = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.6))
    for a, m in zip(ax, metrics):
        a.bar(range(len(S)), [S[l][m] for l in S], color=plt.cm.tab10(np.arange(len(S))), alpha=0.8)
        a.set_xticks(range(len(S))); a.set_xticklabels(list(S), rotation=45, ha="right", fontsize=8); a.set_title(m, fontsize=8)
        for i, l in enumerate(S):
            a.text(i, S[l][m], f"{S[l][m]:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Model variants (same held-out data)"); fig.tight_layout(); fig.savefig(out / "fig_compare_variants.png"); plt.close(fig)
    with open(out / "compare_variants.md", "w") as f:
        f.write("| metric | " + " | ".join(S) + " |\n|---|" + "---|" * len(S) + "\n")
        for m in metrics:
            f.write(f"| {m} | " + " | ".join(f"{S[l][m]:.3f}" for l in S) + " |\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir")
    p.add_argument("--out", default="claude/report/figs")
    p.add_argument("--compare", nargs="*", default=[], help='label@dir pairs')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = load(a.eval_dir)
    for f in (fig_a, fig_b, fig_c, fig_d, fig_silhouettes, fig_e):
        f(D, out)
    if a.compare:
        fig_compare([tuple(c.split("@", 1)) for c in a.compare], out)
    print("figures ->", out)
