"""Independent evaluation of a trained CVAE against the ORIGINAL limb generator / measurement system.

    python claude/evaluate.py ckpt=claude/outputs/single/<run>/checkpoints/best.ckpt

* Test limbs come from GenerateRandomLimbs.py (docker), not from this package's sampler.
* Meshes and measurements of every limb (true and generated) go through the original
  LegMeasurementDataset.get_verts / get_measures (SSM_Driver.py), in mm.
* Sections: A measurement accuracy, B plausibility, C shape distances, D family diversity, E z-traversal.
* Raw per-sample arrays are saved to eval_data.npz; `python claude/plot_eval.py <eval dir>` draws the figures.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlimb_cvae.common import register_resolvers, setup_env  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from openlimb_cvae.geometry import ROOT, ComponentSampler, _in_root  # noqa: E402
from openlimb_cvae.lit_cvae import LimbCVAE  # noqa: E402

REPORT = []


def say(s=""):
    print(s, flush=True)
    REPORT.append(s)


# ----------------------------------------------------------------------------- original system
def load_reference():
    """LegMeasurementDataset instance WITHOUT its __init__ (which shells out to docker/WSL and deletes
    ./stls/*.npy). All tensors are loaded exactly as __init__ does, so get_verts/get_measures are the originals."""
    with _in_root():
        import SSM_Driver
        from SSM_Driver import LegMeasurementDataset

        d = ROOT / "data_components"
        ref = object.__new__(LegMeasurementDataset)
        ref.dtype, ref.device, ref.scale = torch.float64, "cpu", 1
        ref.measure = SSM_Driver.measure
        ref.raw_components = torch.load(d / "vert_components.pt").double()
        ref.mean_verts = torch.load(d / "mean_verts.pt").double()
        ref.face2vert = torch.load(d / "face2vert.pt")
        ref.vert_mapping = torch.load(d / "vert_mapping.pt")
        names = [x["name"] for x in SSM_Driver.measurement_details]
    return ref, names


@torch.no_grad()
def orig_verts(ref, comps, chunk=16):
    out = [ref.get_verts(comps[i:i + chunk]).reshape(-1, ref.vert_mapping.shape[0], 3).float()
           for i in range(0, len(comps), chunk)]
    return torch.cat(out)


@torch.no_grad()
def orig_measures(ref, comps, chunk=16):
    return torch.cat([ref.get_measures(components=comps[i:i + chunk], normalise=False).reshape(-1, 7)
                      for i in range(0, len(comps), chunk)])


def original_test_limbs(n, seed):
    tmp = ROOT / "claude" / "data" / f"eval_tmp_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for i, start in enumerate(range(0, n, 256)):
            cmd = ["docker", "run", "--rm", "-v", f"{ROOT}:/opt/app/working", "-w", "/opt/app/working", "openlimbtt-env",
                   "python", "/opt/app/working/GenerateRandomLimbs.py", "--num_limbs", str(min(256, n - start)),
                   "--path", f"./claude/data/{tmp.name}/", "--start", str(start), "--save_mesh", "0",
                   "--scale", "1", "--seed", str(seed + i)]
            subprocess.run(cmd, check=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        comps = np.concatenate([np.load(f) for f in sorted(tmp.glob("components_*.npy"))])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # nothing kept on disk
    return torch.tensor(comps).double()


# ----------------------------------------------------------------------------- metrics
def vertex_normals(v, faces):
    fn = torch.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]], dim=-1)
    vn = torch.zeros_like(v)
    for k in range(3):
        vn.index_add_(0, faces[:, k], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def shape_metrics(g, t, faces):
    """g, t: (V,3) generated / true vertices in mm. Vertex-sampled symmetric distances."""
    g, t = g.float(), t.float()
    nt, ng = vertex_normals(t, faces), vertex_normals(g, faces)
    D = torch.cdist(g, t)
    d_gt, j = D.min(1)
    d_tg, i = D.min(0)
    p_gt = ((g - t[j]) * nt[j]).sum(-1).abs()
    p_tg = ((t - g[i]) * ng[i]).sum(-1).abs()
    return dict(chamfer=0.5 * (d_gt.mean() + d_tg.mean()).item(), hausdorff=max(d_gt.max(), d_tg.max()).item(),
                point_to_plane=0.5 * (p_gt.mean() + p_tg.mean()).item(), point_to_plane_max=max(p_gt.max(), p_tg.max()).item(),
                vertex_rmse=(g - t).pow(2).sum(-1).mean().sqrt().item())


def table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    say(line(header))
    say("  ".join("-" * x for x in w))
    for r in rows:
        say(line(r))


def pairwise_rmse(a, b):
    """a (N,V,3), b (M,V,3) -> (N,M) correspondence RMSE in mm."""
    return torch.cdist(a.flatten(1).float(), b.flatten(1).float()) / a.shape[1] ** 0.5


@torch.no_grad()
def decode_raw(model, m_norm, z):
    return (model.decode(m_norm, z) * model.comp_std + model.comp_mean).double()


def in_box_u(model, raw):
    return model.box_coords((raw.float() - model.comp_mean) / model.comp_std)


@hydra.main(config_path="conf", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    out = Path(HydraConfig.get().runtime.output_dir)
    model = LimbCVAE.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    zd = model.hparams.z_dim
    ref, names = load_reference()
    faces = ref.face2vert

    say(f"checkpoint: {cfg.ckpt}\nz_dim={zd}  scale={model.hparams.scale}  w_plaus={model.hparams.w_plaus}")
    say(f"generating {cfg.n_limbs} test limbs with the original docker generator ...")
    true_c = original_test_limbs(cfg.n_limbs, cfg.seed)                 # (N, 11) raw
    true_m = orig_measures(ref, true_c)                                  # (N, 7) mm, original code
    m_norm = model.norm_meas(true_m.float())
    c_norm = (true_c.float() - model.comp_mean) / model.comp_std
    N, K = len(true_c), cfg.k_prior

    # sanity: does my training-time sampler match the original generator's statistics?
    s = ComponentSampler(True).raw(20000, torch.Generator().manual_seed(0))
    say(f"sampler vs docker generator (max |component mean diff|/std, max |std ratio-1|): "
        f"{((s.mean(0) - true_c.mean(0)) / true_c.std(0)).abs().max():.2f} / "
        f"{((s.std(0) / true_c.std(0)) - 1).abs().max():.2f}\n")

    # ------------------------------------------------------------------ A: measurement accuracy
    with torch.no_grad():
        mus, logv = model.encode(m_norm, c_norm) if zd else (torch.zeros(N, 0), torch.zeros(N, 0))
    post_raw = decode_raw(model, m_norm, mus if zd else None)
    z = torch.randn(N * K, zd) if zd else None
    prior_raw = decode_raw(model, m_norm.repeat_interleave(K, 0), z)
    say("=== A. Measurement accuracy (ORIGINAL SSM_Driver measurements of decoded limbs vs requested) ===")
    A_arr = {}
    for key, tag, raw, rep in (("prior", "prior z~N(0,I)", prior_raw, K), ("post", "posterior z=mu(m,c)", post_raw, 1)):
        got = orig_measures(ref, raw)
        want = true_m.repeat_interleave(rep, 0)
        A_arr[f"A_{key}_got"], A_arr[f"A_{key}_want"] = got.numpy(), want.numpy()
        err = got - want
        rel = 100 * err.abs() / want.abs()
        say(f"\n[{tag}]  {len(raw)} limbs")
        rows = []
        for i, n in enumerate(names):
            rows.append([n, f"{want[:, i].mean():.1f}", f"{err[:, i].abs().mean():.3f}", f"{err[:, i].abs().std():.3f}",
                         f"{err[:, i].abs().max():.2f}", f"{err[:, i].mean():+.3f}", f"{rel[:, i].mean():.3f}",
                         f"{rel[:, i].std():.3f}", f"{rel[:, i].max():.2f}"])
        rows.append(["ALL", "", f"{err.abs().mean():.3f}", f"{err.abs().std():.3f}", f"{err.abs().max():.2f}",
                     f"{err.mean():+.3f}", f"{rel.mean():.3f}", f"{rel.std():.3f}", f"{rel.max():.2f}"])
        table(rows, ["measure", "mean mm", "|err| mean", "|err| std", "|err| max", "bias mm", "% mean", "% std", "% max"])

    # ------------------------------------------------------------------ B: plausibility
    over = (in_box_u(model, prior_raw).abs() - 1).clamp_min(0).max(-1).values
    say("\n=== B. Plausibility of prior samples (inside the generator's skin-mode box) ===")
    say(f"outside box: {(over > 0).float().mean():.1%}   outside by >5% of box half-range: {(over > 0.05).float().mean():.1%}   "
        f"max exceedance {over.max():.3f}   ({len(over)} samples)")

    # ------------------------------------------------------------------ C: shape distances
    ns, ks = cfg.n_shape_limbs, cfg.k_shape
    say(f"\n=== C. Shape distance to the TRUE limb ({ns} limbs, mm, vertex-sampled; original get_verts) ===")
    tv = orig_verts(ref, true_c[:ns])
    pv = orig_verts(ref, post_raw[:ns])
    qv = orig_verts(ref, prior_raw.reshape(N, K, -1)[:ns, :ks].reshape(ns * ks, -1))
    keys = ["chamfer", "hausdorff", "point_to_plane", "point_to_plane_max", "vertex_rmse"]
    post_m = [shape_metrics(pv[i], tv[i], faces) for i in range(ns)]
    prior_m = [[shape_metrics(qv[i * ks + k], tv[i], faces) for k in range(ks)] for i in range(ns)]
    rows = []
    for tag, ms in (("posterior (z=mu)", post_m), ("prior, mean over samples", [x for r in prior_m for x in r]),
                    ("prior, best of %d" % ks, [min(r, key=lambda m: m["chamfer"]) for r in prior_m])):
        rows.append([tag] + [f"{np.mean([m[k] for m in ms]):.2f} ± {np.std([m[k] for m in ms]):.2f}" for k in keys])
    table(rows, ["vs true limb", "chamfer", "hausdorff", "pt-to-plane", "pt-to-plane max", "vertex rmse"])
    ctx = [shape_metrics(tv[i], tv[(i + 1) % ns], faces)["chamfer"] for i in range(min(ns, 200))]
    say("(mean ± std over limbs; 'max' column = worst vertex per limb. Prior samples are *different* limbs with the "
        "same measurements, so they are not expected to reproduce the true shape exactly.)")
    say(f"context: chamfer between two unrelated limbs = {np.mean(ctx):.2f} mm; "
        f"true-limb vertex spread (RMS from mean) = {(tv[:200] - tv[:200].mean(0)).pow(2).sum(-1).mean().sqrt():.2f} mm")

    # ------------------------------------------------------------------ D: family with fixed measurements
    nf, kf = cfg.n_family_limbs, cfg.k_family
    say(f"\n=== D. Same measurements, different z: {nf} measurement sets x {kf} samples (= {nf * kf} limbs) ===")
    samp = ComponentSampler(True)
    D = {k: [] for k in ("meas_std", "meas_absdev", "rms_dev", "pair_mean", "pair_max", "chamfer", "hausdorff", "pca_frac",
                         "pr", "null_frac", "true_to_gen")}
    R = {k: [] for k in ("coverage", "precision", "baseline", "ref_spread", "gen_spread", "n_ref")}
    examples = {}
    for i in range(nf):
        raw = decode_raw(model, m_norm[i].expand(kf, -1), torch.randn(kf, zd) if zd else None)
        fm = orig_measures(ref, raw)                                              # (kf, 7)
        fv = orig_verts(ref, raw)                                                 # (kf, V, 3)
        fc = (raw.float() - model.comp_mean) / model.comp_std
        D["meas_std"].append(fm.std(0))
        D["meas_absdev"].append((fm - true_m[i]).abs().mean(0))
        dev = (fv - fv.mean(0)).pow(2).sum(-1).mean(-1).sqrt()                     # (kf,)
        D["rms_dev"].append(dev.mean())
        pw = pairwise_rmse(fv, fv)
        D["pair_mean"].append(pw.sum() / (kf * (kf - 1)))
        D["pair_max"].append(pw.max())
        sm = shape_metrics(fv[0], fv[1], faces)
        D["chamfer"].append(sm["chamfer"])
        D["hausdorff"].append(sm["hausdorff"])
        ev = torch.linalg.svdvals(fv.flatten(1) - fv.flatten(1).mean(0)).pow(2)
        D["pca_frac"].append((ev / ev.sum())[:8])
        D["pr"].append(ev.sum() ** 2 / ev.pow(2).sum())
        cbar = fc.mean(0).requires_grad_(True)
        J = torch.autograd.functional.jacobian(model.surrogate, cbar)              # (7, 11)
        Nsp = torch.linalg.svd(J)[2][7:].T                                          # (11, 4) measurement-preserving directions
        cov = torch.cov((fc - fc.mean(0)).T)
        D["null_frac"].append((Nsp.T @ cov @ Nsp).trace() / cov.trace())
        tvi = orig_verts(ref, true_c[i:i + 1])
        D["true_to_gen"].append(pairwise_rmse(tvi, fv).min())

        if i < cfg.n_reference_sets:
            # reference family: random in-box limbs projected (Gauss-Newton on the surrogate) onto {measurements = target}
            c = samp.normalised(cfg.n_reference, torch.Generator().manual_seed(1000 + i))
            for _ in range(25):
                r = model.surrogate(c) - m_norm[i]
                Jb = torch.func.vmap(torch.func.jacrev(model.surrogate))(c)
                c = (c - (torch.linalg.pinv(Jb) @ r[..., None]).squeeze(-1)).detach()
            rr = (c * model.comp_std + model.comp_mean).double()
            rr = rr[in_box_u(model, rr).abs().max(-1).values <= 1]
            if len(rr):
                rr = rr[(orig_measures(ref, rr) - true_m[i]).abs().max(-1).values < 1.5]   # verified with the original system
            if len(rr) >= 20:
                rv = orig_verts(ref, rr)
                d_rg = pairwise_rmse(rv, fv)
                d_rr = pairwise_rmse(rv, rv)
                d_rr.fill_diagonal_(float("inf"))
                R["coverage"].append(d_rg.min(1).values.mean())
                R["precision"].append(d_rg.min(0).values.mean())
                R["baseline"].append(d_rr.min(1).values.mean())
                R["ref_spread"].append((rv - rv.mean(0)).pow(2).sum(-1).mean(-1).sqrt().mean())
                R["gen_spread"].append(dev.mean())
                R["n_ref"].append(float(len(rr)))
                if i < 3:
                    examples[f"set{i}_ref"] = rv[:8].numpy()
        if i < 3:
            examples[f"set{i}_gen"] = fv[:8].numpy()
            examples[f"set{i}_true"] = tvi[0].numpy()
        if (i + 1) % 25 == 0:
            say(f"  ... {i + 1}/{nf} sets")
    D = {k: torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in v]).numpy() for k, v in D.items()}
    R = {k: np.array([float(x) for x in v]) for k, v in R.items()}
    ci = lambda x: f"{np.mean(x):.3f} (95% range over sets {np.percentile(x, 2.5):.3f}-{np.percentile(x, 97.5):.3f})"
    tm = true_m[:nf].numpy()

    say("\nD1. measurements stay put while z varies (exact, original code) - mean over sets of the per-set std over z:")
    rows = [[n, f"{D['meas_std'][:, i].mean():.3f}", f"{100 * (D['meas_std'][:, i] / tm[:, i]).mean():.3f}",
             f"{D['meas_absdev'][:, i].mean():.3f}", f"{100 * (D['meas_absdev'][:, i] / tm[:, i]).mean():.3f}"]
            for i, n in enumerate(names)]
    table(rows, ["measure", "std over z mm", "std over z %", "|err| mm", "|err| %"])
    say("\nD2. shapes differ while z varies:")
    say(f"  RMS vertex deviation from family mean (mm): {ci(D['rms_dev'])}")
    say(f"  mean pairwise vertex RMSE between two samples (mm): {ci(D['pair_mean'])};  largest pair: {ci(D['pair_max'])}")
    say(f"  chamfer between two samples of the same set (mm): {ci(D['chamfer'])};  hausdorff: {ci(D['hausdorff'])}")
    say("\nD3. dimensionality of the family (PCA of vertices across z):")
    frac = D["pca_frac"].mean(0)
    say("  variance fraction per principal direction: " + "  ".join(f"{f:.3f}" for f in frac) +
        f"\n  participation ratio (effective # of dims): {D['pr'].mean():.2f}   dims for 95% var: {int((np.cumsum(frac) < 0.95).sum()) + 1}")
    say("\nD4. fraction of family variance inside the 4-D nullspace of the measurement Jacobian "
        f"(1.0 = z moves the limb only along measurement-preserving directions): {ci(D['null_frac'])}")
    say(f"\nD5. vs a reference family of real limbs sharing the same measurements ({len(R['n_ref'])} sets, {R['n_ref'].mean():.0f} ref limbs each)")
    say(f"  spread (RMS vertex deviation from family mean): generated {R['gen_spread'].mean():.2f} mm   reference {R['ref_spread'].mean():.2f} mm")
    say(f"  coverage  (ref limb -> nearest generated sample): {R['coverage'].mean():.2f} mm   [ref->nearest other ref, density baseline: {R['baseline'].mean():.2f} mm]")
    say(f"  precision (generated -> nearest ref limb):         {R['precision'].mean():.2f} mm")
    say(f"  true limb -> nearest of {kf} generated samples: {ci(D['true_to_gen'])} mm (vertex RMSE)")

    # ------------------------------------------------------------------ E: z traversal
    nt = cfg.n_traversal_limbs
    say(f"\n=== E. Latent traversal: z = ±2 along one axis, others 0 ({nt} measurement sets) ===")
    with torch.no_grad():
        kls = (-0.5 * (1 + logv - mus ** 2 - logv.exp())).mean(0)
    E = {"shape": np.zeros((zd, 2 * nt)), "drift": np.zeros((zd, 2 * nt))}
    rows = []
    for d in range(zd):
        sc, mc = [], []
        for sign in (-2.0, 2.0):
            z0, z1 = torch.zeros(nt, zd), torch.zeros(nt, zd)
            z1[:, d] = sign
            r0, r1 = decode_raw(model, m_norm[:nt], z0), decode_raw(model, m_norm[:nt], z1)
            sc.append((orig_verts(ref, r1) - orig_verts(ref, r0)).pow(2).sum(-1).mean(-1).sqrt())
            mc.append((orig_measures(ref, r1) - orig_measures(ref, r0)).abs().max(-1).values)
        E["shape"][d], E["drift"][d] = torch.cat(sc).numpy(), torch.cat(mc).numpy()
        rows.append([f"z{d}", f"{mus[:, d].var():.3f}", f"{kls[d]:.3f}", f"{E['shape'][d].mean():.2f}",
                     f"{E['drift'][d].mean():.2f}", f"{E['drift'][d].max():.2f}"])
    table(rows, ["axis", "var(mu)", "KL nats", "shape change mm (RMS vert)", "max meas drift mm", "worst drift mm"])

    # ------------------------------------------------------------------ save everything needed for plots
    np.savez_compressed(
        out / "eval_data.npz", names=np.array(names), zd=zd, B_over=over.numpy(),
        C_post=np.array([[m[k] for k in keys] for m in post_m]),
        C_prior=np.array([[[m[k] for k in keys] for m in r] for r in prior_m]),
        C_keys=np.array(keys), C_ctx=np.array(ctx), true_m=true_m.numpy(), mu=mus.numpy(), kl=kls.numpy(),
        **A_arr, **{f"D_{k}": v for k, v in D.items()}, **{f"R_{k}": v for k, v in R.items()},
        E_shape=E["shape"], E_drift=E["drift"], **examples)
    (out / "report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
