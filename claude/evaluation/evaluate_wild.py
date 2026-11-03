"""Experiment 1: CVAE vs. ground-truth 3D limb meshes ("in the wild" scans).

    python claude/evaluation/evaluate_wild.py mode=make_test_data mesh_dir=<dir> n_test=20       # synthetic stand-in data
    python claude/evaluation/evaluate_wild.py mode=annotate       mesh_dir=<dir>                 # interactive, one json per mesh
    python claude/evaluation/evaluate_wild.py mode=evaluate ckpt=<best.ckpt> mesh_dir=<dir>      # the experiment

Pipeline per mesh
  1. read the mesh + its annotation (<mesh stem>.json next to the mesh, or in ann_dir);
  2. measure it with the OpenLimb definitions (SSM_Driver.measure re-implemented for arbitrary triangle meshes:
     planar circumferences, extents of planar slices, length along the limb axis) -> the 7 CVAE inputs;
  3. decode k_samples limbs z ~ N(0, I)  (+ the z = 0 limb, + the mean limb as a "no information" baseline);
  4. rigidly align the ground-truth mesh to every generated limb (annotated landmarks -> Kabsch, then trimmed ICP);
  5. shape error (symmetric point-to-surface / point-to-plane: mean, RMSE, Hausdorff, HD95) and
     measurement error of every generated limb, then `distribution_metrics` over the z-family.

Annotation file (`<mesh stem>.json`, coordinates in mm in the frame of the mesh after `unit_scale`):
    {"version": 2,
     "planes": {"circ1": [p, p, p], ..., "circ4": [p, p, p]},   # 3 points on each circumference plane (any 3 around the limb)
     "widths": {"width1": [p, p], "width2": [p, p]},            # the two extremes (medial / lateral) of the slice
     "length1": [p, p],                                         # the two end points of Length 1
     "reference_measurements": [7 numbers, mm]}                 # optional (e.g. tape measure / known truth), only reported
The limb axis is the mean (sign-aligned) normal of the four circumference planes. Circ i = perimeter of the mesh cut by
plane i (loop nearest the points); Width j = extent of the cut at the pair's mid-point (plane normal to the axis) along
the pair's direction; Length 1 = |(a - b) . axis|.  `mode=annotate` serves a 3D viewer (claude/evaluation/annotator.html) with a
translucent mesh, draggable points, live slices and the SSM mean limb with its landmarks as a guide.

Measurement convention: the cut rule is the corrected one of measure_fixed.py (a vertex is on the "+" side
iff d >= 0), identical to what the model is trained on (SSMGeometry / FastMeasure).

Everything that is likely to change while the statistics are being decided lives in the two registries
SHAPE_METRICS and DISTRIBUTION_METRICS (and in `shape_distances`).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import register_resolvers, setup_env  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import igl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from data.geometry import ComponentSampler, SSMGeometry, load_shape_model  # noqa: E402

FACES = None   # SSM mesh faces, set in evaluate()
from models.lit_cvae import LimbCVAE  # noqa: E402

# Model input order (SSM_Driver.get_measurement_details): Circ 1-4, Length 1, Width 1, Width 2.
MEAS_NAMES = ["Circ 1", "Circ 2", "Circ 3", "Circ 4", "Length 1", "Width 1", "Width 2"]
PLANES = ["circ1", "circ2", "circ3", "circ4"]
WIDTHS = ["width1", "width2"]
LOOP_MODES = ("nearest", "largest", "all")  # "all" = what SSM_Driver does (sums every intersected triangle)


# ============================================================================ mesh + annotation I/O
def load_mesh(path, unit_scale=1.0):
    V, F = igl.read_triangle_mesh(str(path))
    return np.asarray(V, np.float64) * unit_scale, np.asarray(F, np.int64)


def unit(v):
    return v / np.linalg.norm(v)


def load_annotation(path):
    """v2 annotation -> planes [(centre, normal)] x4, widths [(mid, dir)] x2, length (a, b), limb axis."""
    j = json.loads(Path(path).read_text())
    if j.get("version") != 2:
        raise ValueError(f"{path}: unsupported annotation version (expected 2, produced by mode=annotate)")
    P = {k: np.asarray(j["planes"][k], np.float64) for k in PLANES}
    planes = [(p.mean(0), unit(np.cross(p[1] - p[0], p[2] - p[0]))) for p in P.values()]
    axis = planes[0][1].copy()
    for _, n in planes[1:]:
        axis += n if n @ planes[0][1] >= 0 else -n
    axis = unit(axis)
    widths = []
    for k in WIDTHS:
        a, b = np.asarray(j["widths"][k], np.float64)
        d = (a - b) - ((a - b) @ axis) * axis
        widths.append(((a + b) / 2, unit(d)))
    ref = j.get("reference_measurements")
    return dict(planes=planes, widths=widths, length=np.asarray(j["length1"], np.float64), axis=axis,
                reference=None if ref is None else np.asarray(ref, float))


# ============================================================================ measuring a mesh (OpenLimb definitions)
def slice_segments(V, F, point, normal):
    """Intersect the triangle mesh with the plane (point, normal). Returns segment end points (m,2,3) and a
    connected-component label per segment (segments sharing a cut edge belong to the same loop).
    A vertex is on the "+" side iff d >= 0, so vertices ON the plane and coplanar edges / faces are handled
    consistently (every face has 0 or 2 cut edges)."""
    d = (V - point) @ normal
    side = (d >= 0)[F]                                               # (nf,3)
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    cross = side[:, pairs[:, 0]] != side[:, pairs[:, 1]]
    keep = np.flatnonzero(cross.sum(1) == 2)
    if len(keep) == 0:
        return np.zeros((0, 2, 3)), np.zeros(0, int)
    e = np.argsort(~cross[keep], axis=1, kind="stable")[:, :2]        # the two crossed edges of each face
    fk = F[keep]
    va, vb = np.take_along_axis(fk, pairs[e, 0], 1), np.take_along_axis(fk, pairs[e, 1], 1)
    t = d[va] / (d[va] - d[vb])
    P = V[va] + t[..., None] * (V[vb] - V[va])
    code = np.minimum(va, vb) * len(V) + np.maximum(va, vb)           # undirected edge id
    uniq, inv = np.unique(code, return_inverse=True)
    inv = inv.reshape(-1, 2)
    g = coo_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(len(uniq), len(uniq)))
    _, lab = connected_components(g, directed=False)
    return P, lab[inv[:, 0]]


def select_loops(P, lab, point, mode):
    if mode == "all" or len(P) == 0:
        return np.ones(len(P), bool)
    seg_len = np.linalg.norm(P[:, 0] - P[:, 1], axis=-1)
    if mode == "largest":
        return lab == np.bincount(lab, weights=seg_len).argmax()
    dist = np.linalg.norm(P.mean(1) - point, axis=-1)
    best = np.full(lab.max() + 1, np.inf)
    np.minimum.at(best, lab, dist)
    return lab == best.argmin()


def cut_loop(V, F, point, normal, loop_mode="nearest"):
    P, lab = slice_segments(V, F, point, normal)
    return P[select_loops(P, lab, point, loop_mode)], (0 if len(lab) == 0 else lab.max() + 1)


def measure_mesh(V, F, ann, loop_mode="nearest"):
    """The 7 OpenLimb measurements (mm), in MEAS_NAMES order, + number of loops found per slice."""
    out, nloops = np.full(7, np.nan), np.zeros(7, int)
    for i, (c, n) in enumerate(ann["planes"]):
        P, nloops[i] = cut_loop(V, F, c, n, loop_mode)
        if len(P):
            out[i] = np.linalg.norm(P[:, 0] - P[:, 1], axis=-1).sum()
    out[4] = abs((ann["length"][0] - ann["length"][1]) @ ann["axis"])
    for i, (mid, d) in zip((5, 6), ann["widths"]):
        P, nloops[i] = cut_loop(V, F, mid, ann["axis"], loop_mode)
        if len(P):
            c = P.reshape(-1, 3) @ d
            out[i] = c.max() - c.min()
    return out, nloops


def slice_centroid(V, F, point, normal, loop_mode="nearest"):
    P, _ = cut_loop(V, F, point, normal, loop_mode)
    w = np.linalg.norm(P[:, 0] - P[:, 1], axis=-1)
    return (P.mean(1) * w[:, None]).sum(0) / w.sum()


def landmark_annotation(V, F, ids):
    """Exact v2 annotation of an SSM limb from its landmark vertices (ids: circ1-4, length a/b, width1/2)."""
    z = np.array([0.0, 0.0, 1.0])
    out = {"version": 2, "planes": {}, "widths": {}}
    for k, i in zip(PLANES, ids[:4]):
        pts = cut_loop(V, F, V[i], z)[0].reshape(-1, 3)
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        out["planes"][k] = [pts[np.argmin(np.abs(np.angle(np.exp(1j * (ang - a0)))))].tolist()
                            for a0 in (0.0, 2 * np.pi / 3, -2 * np.pi / 3)]
    for k, i in zip(WIDTHS, ids[6:8]):
        pts = cut_loop(V, F, V[i], z)[0].reshape(-1, 3)
        a = pts[np.argmin(pts[:, 0])]                       # synthetic pair: one extreme + the x-extent along x,
        out["widths"][k] = [a.tolist(), (a + [np.ptp(pts[:, 0]), 0.0, 0.0]).tolist()]   # so d = x exactly as in the SSM
    out["length1"] = [V[ids[4]].tolist(), V[ids[5]].tolist()]
    return out


# ============================================================================ alignment
def kabsch(src, dst):
    """Rigid (R, t) minimising |R src + t - dst|."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def trimmed_icp(src, dst, R, t, iters=50, trim=0.9, tol=1e-5):
    tree = cKDTree(dst)
    prev = np.inf
    for _ in range(iters):
        p = src @ R.T + t
        d, j = tree.query(p)
        keep = d <= np.quantile(d, trim)
        dR, dt = kabsch(p[keep], dst[j[keep]])
        R, t = dR @ R, dR @ t + dt
        err = d[keep].mean()
        if prev - err < tol:
            break
        prev = err
    return R, t, err


def frame_candidates(V, F, ann, gen_V, gen_ids, loop_mode):
    """Rigid poses (GT -> SSM frame) from the annotation alone, independent of the scan's pose.
    Frame = (limb axis pointing circ1 -> circ4, width direction, their cross product); the sign of the width
    direction is not recoverable from the annotation so both are returned (resolved by the fit)."""
    z, x = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    cg = np.stack([slice_centroid(V, F, c, n, loop_mode) for c, n in ann["planes"]])
    cs = np.stack([slice_centroid(gen_V, FACES, gen_V[i], z, loop_mode) for i in gen_ids[:4]])
    a = ann["axis"] if (cg[3] - cg[0]) @ ann["axis"] > 0 else -ann["axis"]
    az = z if (cs[3] - cs[0]) @ z > 0 else -z
    w = ann["widths"][0][1]
    out = []
    for s in (1.0, -1.0):
        Fa = np.stack([a, w, np.cross(a, w)], 1)
        Fg = np.stack([az, s * x, np.cross(az, s * x)], 1)
        R = Fg @ Fa.T
        out.append((R, cs.mean(0) - R @ cg.mean(0)))
    return out


def align(mode, V, F, gt_pts, ann, gen_V, gen_ids, loop_mode, cache):
    """Rigid transform taking the ground truth into the generated limb's frame.
    none: identity; frame: from the annotation only; icp: frame + trimmed ICP (the better of the two sign choices)."""
    if mode == "none":
        return np.eye(3), np.zeros(3)
    cands = frame_candidates(V, F, ann, gen_V, gen_ids, loop_mode)
    tree = cKDTree(gen_V)
    sub = gt_pts[::max(1, len(gt_pts) // 3000)]
    if mode == "frame":
        errs = [np.quantile(tree.query(sub @ R.T + t)[0], 0.9) for R, t in cands]
        return cands[int(np.argmin(errs))]
    if "choice" in cache:                              # decided once per mesh (on the z = 0 limb)
        return trimmed_icp(sub, gen_V, *cands[cache["choice"]])[:2]
    fits = [trimmed_icp(sub, gen_V, R, t) for R, t in cands]
    cache["choice"] = int(np.argmin([f[2] for f in fits]))
    return fits[cache["choice"]][:2]


# ============================================================================ shape distances
def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)


def one_way(P, V, F, N):
    """Distances from points P to the surface (V,F): euclidean, and to the plane of the closest triangle."""
    sq, idx, C = igl.point_mesh_squared_distance(np.ascontiguousarray(P), np.ascontiguousarray(V), F)
    return np.sqrt(sq), np.abs(((P - C) * N[idx]).sum(1))


def shape_distances(A_pts, A_mesh, B_pts, B_mesh):
    """Symmetric surface distances between two shapes given as (sampled points, (V, F)).
    Returns the raw per-point arrays so any statistic can be built on them (see SHAPE_METRICS)."""
    (VA, FA), (VB, FB) = A_mesh, B_mesh
    ab, ab_pl = one_way(A_pts, VB, FB, face_normals(VB, FB))
    ba, ba_pl = one_way(B_pts, VA, FA, face_normals(VA, FA))
    return dict(euc=(ab, ba), plane=(ab_pl, ba_pl))


def _mean(x): return 0.5 * (x[0].mean() + x[1].mean())
def _rmse(x): return float(np.sqrt(0.5 * ((x[0] ** 2).mean() + (x[1] ** 2).mean())))
def _max(x): return max(x[0].max(), x[1].max())
def _p95(x): return max(np.percentile(x[0], 95), np.percentile(x[1], 95))


# name -> f(dist dict) -> float.  Add / swap statistics here.
SHAPE_METRICS = {
    "chamfer": lambda d: _mean(d["euc"]),            # mean symmetric point-to-surface distance (ASSD)
    "rmse": lambda d: _rmse(d["euc"]),               # symmetric RMS point-to-surface distance
    "hausdorff": lambda d: _max(d["euc"]),
    "hd95": lambda d: _p95(d["euc"]),
    "p2plane_mean": lambda d: _mean(d["plane"]),     # symmetric point-to-(tangent)-plane distance
    "p2plane_rmse": lambda d: _rmse(d["plane"]),
    "p2plane_max": lambda d: _max(d["plane"]),
}


def sample_surface(V, F, n, rng):
    """Area-weighted uniform surface samples: barycentric coords (n,3), face ids (n,), points (n,3)."""
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    B = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    return B, fi, np.einsum("nk,nkd->nd", B, V[F[fi]])


def cloud_chamfer(A, B):
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


# ============================================================================ distribution of the generated family
def distribution_metrics(e, e_pc, pair, meas_err, w, plausible, base):
    """Statistics of the generated family for ONE subject.

    e         (K,)   primary shape error of every z-sample w.r.t. the ground truth
    e_pc      (K,)   point-cloud chamfer to ground truth (same distance as `pair`, for the energy score)
    pair      (K,K)  point-cloud chamfer between samples
    meas_err  (K,7)  |re-measured generated limb - requested measurement|  (mm)
    w         (K,)   self-normalised importance weights (uniform when temperature == 1)
    plausible (K,)   bool: inside the generator's box
    base      float  primary error of the mean limb (no-information baseline)

    expected_error is the Monte-Carlo estimate of  int p(z) d(G(m, z), x_true) dz.  Replace / extend freely.
    """
    K = len(e)
    off = pair[~np.eye(K, dtype=bool)]
    out = {
        "expected_error": float((w * e).sum()),                # E_z[d]  (importance weighted if T != 1)
        "expected_error_plausible": float(e[plausible].mean()) if plausible.any() else np.nan,
        "error_std_over_z": float(e.std()),
        "error_p05": float(np.percentile(e, 5)), "error_p95": float(np.percentile(e, 95)),
        "best_of_k": float(e.min()),                           # oracle: closest sample to truth
        "family_spread": float(off.mean()),                    # E d(s, s')
        "energy_score": float((w * e_pc).sum() - 0.5 * off.mean()),   # proper scoring rule, lower = better
        "calibration_ratio": float(e_pc.mean() / off.mean()),  # ~1 if truth looks like a draw from the family
        "skill_vs_mean_limb": float(1 - (w * e).sum() / base),  # 1 = perfect, 0 = no better than the mean limb
        "expected_meas_err_mm": float((w[:, None] * meas_err).sum(0).mean()),   # E_z of mean |meas error|
        "frac_plausible": float(plausible.mean()),
    }
    return out


# ============================================================================ synthetic data / annotation tooling
def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def landmark_ids(geo):
    d = geo.measurement_details
    return [int(d[i]["plane_point"]) for i in range(4)] + [int(d[4]["v1"]), int(d[4]["v2"]),
                                                          int(d[5]["plane_point"]), int(d[6]["plane_point"])]


def make_test_data(cfg):
    """OpenLimb limbs -> randomly posed (+ re-tessellated, noisy) meshes with exact annotations.
    A stand-in for scans: the true measurements are stored as `reference_measurements` (corrected convention)."""
    out = Path(cfg.mesh_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    geo = SSMGeometry(scale=True, normalise=False)
    faces = load_shape_model()[2].numpy()
    ids = landmark_ids(geo)
    comps = ComponentSampler(True).raw(cfg.n_test, torch.Generator().manual_seed(cfg.seed))
    verts = geo.get_verts(comps).numpy()
    meas = geo.get_measures(comps, normalise=False).numpy()
    for i in range(cfg.n_test):
        V, F = verts[i], faces
        ann = landmark_annotation(V, F, ids)
        if cfg.retessellate:            # different topology than the SSM; same surface
            Vu, Fu = igl.upsample(V, F, 1)
            V, F = np.asarray(Vu), np.asarray(Fu, np.int64)
        R, t = random_rotation(rng), rng.normal(size=3) * 100
        Vw = (V + rng.normal(size=V.shape) * cfg.noise_mm) @ R.T + t
        tf = lambda p: (np.asarray(p) @ R.T + t).tolist()
        stem = f"limb_{i:04d}"
        igl.write_triangle_mesh(str(out / f"{stem}.obj"), Vw, F)
        (out / f"{stem}.json").write_text(json.dumps({
            "version": 2, "planes": {k: [tf(p) for p in v] for k, v in ann["planes"].items()},
            "widths": {k: [tf(p) for p in v] for k, v in ann["widths"].items()}, "length1": [tf(p) for p in ann["length1"]],
            "reference_measurements": meas[i].tolist(), "true_components": comps[i].tolist()}, indent=1))
    print(f"wrote {cfg.n_test} meshes + annotations to {out}")


def annotate(cfg):
    """Serve the 3D annotation tool (claude/evaluation/annotator.html) and open it in the browser. Ctrl+C to stop."""
    import base64
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    meshes = {p.name: p for e in cfg.mesh_ext for p in sorted(Path(cfg.mesh_dir).glob(f"*.{e}"))}
    ann_dir = Path(cfg.ann_dir or cfg.mesh_dir)
    page_path = Path(__file__).parent / "annotator.html"
    cache = {}

    def payload(V, F):
        V = np.asarray(V, np.float32)
        return {"v": base64.b64encode(V.tobytes()).decode(), "f": base64.b64encode(np.asarray(F, np.uint32).tobytes()).decode(),
                "center": ((V.max(0) + V.min(0)) / 2).tolist()}

    def reference():
        geo = SSMGeometry(scale=True, normalise=False)
        c = torch.zeros(1, 11, dtype=torch.float64)
        c[0, -1] = 390.0
        V, F = geo.get_verts(c)[0].numpy(), load_shape_model()[2].numpy()
        return payload(V, F) | {"ann": landmark_annotation(V, F, landmark_ids(geo))}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _name(self):
            return parse_qs(urlparse(self.path).query).get("name", [""])[0]

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                self._send(page_path.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/list":
                self._send([{"name": n, "annotated": (ann_dir / f"{p.stem}.json").exists()} for n, p in meshes.items()])
            elif route == "/api/ref":
                cache.setdefault("ref", reference())
                self._send(cache["ref"])
            elif route == "/api/mesh" and self._name() in meshes:
                V, F = load_mesh(meshes[self._name()], cfg.unit_scale)
                self._send(payload(V, F))
            elif route == "/api/ann" and self._name() in meshes:
                jp = ann_dir / f"{meshes[self._name()].stem}.json"
                self._send(json.loads(jp.read_text()) if jp.exists() else {})
            else:
                self._send({"error": "not found"}, code=404)

        def do_POST(self):
            if urlparse(self.path).path == "/api/ann" and self._name() in meshes:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                jp = ann_dir / f"{meshes[self._name()].stem}.json"
                old = json.loads(jp.read_text()) if jp.exists() else {}
                new = json.loads(body)
                if "reference_measurements" in old:
                    new["reference_measurements"] = old["reference_measurements"]
                jp.write_text(json.dumps(new, indent=1))
                self._send({"ok": True})
            else:
                self._send({"error": "not found"}, code=404)

    if not meshes:
        raise SystemExit(f"no meshes ({cfg.mesh_ext}) in {cfg.mesh_dir}")
    class Server(ThreadingHTTPServer):
        allow_reuse_address = False         # on Windows SO_REUSEADDR would silently allow a second listener

    srv = Server(("127.0.0.1", cfg.port), Handler)
    url = f"http://127.0.0.1:{cfg.port}/"
    print(f"annotation tool for {len(meshes)} meshes at {url}  (annotations -> {ann_dir}; Ctrl+C to stop)", flush=True)
    if cfg.open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ============================================================================ the experiment
def table(rows, header, say):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header, ["-" * x for x in w]] + rows:
        say("  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r)))


def stat_rows(table_dict):
    keys = next(iter(table_dict.values())).keys()
    rows = []
    for k in keys:
        x = np.array([r[k] for r in table_dict.values()], float)
        x = x[np.isfinite(x)]
        rows.append([k, len(x), f"{x.mean():.3f}", f"{np.median(x):.3f}", f"{x.std():.3f}",
                     f"{np.percentile(x, 95):.3f}", f"{x.max():.3f}"])
    return rows


def evaluate(cfg):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(HydraConfig.get().runtime.output_dir)
    report = []

    def say(s=""):
        print(s, flush=True)
        report.append(s)

    model = LimbCVAE.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    zd, K, T = model.hparams.z_dim, cfg.k_samples, cfg.temperature
    geo = SSMGeometry(scale=model.hparams.scale, normalise=False)
    ids = landmark_ids(geo)
    faces = load_shape_model()[2].numpy()
    global FACES
    FACES = faces
    ref = geo.get_measures(ComponentSampler(model.hparams.scale).raw(
        cfg.n_range, torch.Generator().manual_seed(cfg.seed)), normalise=False).numpy()
    lo, hi = ref.min(0), ref.max(0)
    mean_limb_c = model.comp_mean[None]
    primary = cfg.primary

    meshes = sorted(p for e in cfg.mesh_ext for p in Path(cfg.mesh_dir).glob(f"*.{e}"))
    if cfg.limit:
        meshes = meshes[:cfg.limit]
    say(f"checkpoint {cfg.ckpt}\n{len(meshes)} meshes in {cfg.mesh_dir}; K={K} samples, T={T}, align={cfg.align}, "
        f"loop_mode={cfg.loop_mode}, primary metric={primary}\n")

    subj, subj_meas, subj_dist, subj_shape, samples = {}, {}, {}, {}, {}
    for path in meshes:
        jp = Path(cfg.ann_dir or cfg.mesh_dir) / f"{path.stem}.json"
        if not jp.exists():
            say(f"[skip] {path.name}: no annotation {jp.name}")
            continue
        V, F = load_mesh(path, cfg.unit_scale)
        ann = load_annotation(jp)
        meas, nloops = measure_mesh(V, F, ann, cfg.loop_mode)
        if not np.isfinite(meas).all():
            say(f"[skip] {path.name}: a measurement plane misses the mesh ({meas})")
            continue
        oor = [n for n, v, a, b in zip(MEAS_NAMES, meas, lo, hi) if not a <= v <= b]
        if oor:   # flagged, never removed: real data is allowed to be unusual
            say(f"[flag] {path.name}: outside the generator's measurement range: {oor}")
        if (nloops[:4] > 1).any() or (nloops[5:] > 1).any():
            say(f"[warn] {path.name}: slices with several loops {nloops.tolist()} (loop_mode={cfg.loop_mode})")

        # ---- generate: row 0 = z=0, rows 1..K = z ~ N(0, T^2 I)
        m_norm = model.norm_meas(torch.tensor(meas, dtype=torch.float32))[None]
        with torch.no_grad():
            z = torch.cat([torch.zeros(1, zd), torch.randn(K, zd) * T]) if zd else None
            n_rows = K + 1
            cn = model.decode(m_norm.expand(n_rows, -1), z)
            raw = (cn * model.comp_std + model.comp_mean)
            gen_V = model.verts(cn).reshape(n_rows, -1, 3).double().numpy()
            plausible = ~model.outside_box(cn).numpy()
            base_V = model.verts((mean_limb_c - model.comp_mean) / model.comp_std).reshape(-1, 3).double().numpy()
        regen = geo.get_measures(raw.double(), normalise=False).numpy()          # exact, original code
        meas_err = np.abs(regen - meas)
        if zd and T != 1.0:   # importance weights N(z;0,I) / N(z;0,T^2 I)
            zz = z[1:].numpy()
            lw = -0.5 * (zz ** 2).sum(1) + 0.5 * (zz ** 2).sum(1) / T ** 2 + zd * np.log(T)
            w = np.exp(lw - lw.max())
            w /= w.sum()
        else:
            w = np.full(K, 1.0 / K)

        # ---- compare with the ground-truth surface
        _, _, gt_pts = sample_surface(V, F, cfg.n_points, rng)
        Bg, fg, _ = sample_surface(base_V, faces, cfg.n_points, rng)     # same barycentric points on every sample
        bary = lambda Vg: np.einsum("nk,nkd->nd", Bg, Vg[faces[fg]])
        cache = {}

        def compare(Vg):
            R, t = align(cfg.align, V, F, gt_pts, ann, Vg, ids, cfg.loop_mode, cache)
            Vt, Pt = V @ R.T + t, gt_pts @ R.T + t
            dist = shape_distances(Pt, (Vt, F), bary(Vg), (Vg, faces))
            return {k: float(f(dist)) for k, f in SHAPE_METRICS.items()}, cloud_chamfer(Pt, bary(Vg)), (Vt, R, t)

        res = [compare(gen_V[i]) for i in range(n_rows)]
        base_res = compare(base_V)
        shape = {k: np.array([r[0][k] for r in res]) for k in SHAPE_METRICS}
        e, e_pc = shape[primary][1:], np.array([r[1] for r in res[1:]])
        clouds = np.stack([bary(g) for g in gen_V[1:]])
        pair = np.array([[cloud_chamfer(a, b) if i < j else 0.0 for j, b in enumerate(clouds)]
                         for i, a in enumerate(clouds)])
        pair = pair + pair.T

        row = distribution_metrics(e, e_pc, pair, meas_err[1:], w, plausible[1:], base_res[0][primary])
        row["z0_error"] = float(shape[primary][0])
        row["mean_limb_error"] = float(base_res[0][primary])
        row["n_measures_out_of_range"] = float(len(oor))
        subj[path.stem] = row
        subj_shape[path.stem] = {f"z0_{k}": float(v[0]) for k, v in shape.items()} | \
                                {f"mean_{k}": float(v.mean()) for k, v in shape.items()} | \
                                {f"max_{k}": float(v.max()) for k, v in shape.items()}
        subj_meas[path.stem] = {f"{n} (z0 err mm)": float(meas_err[0, i]) for i, n in enumerate(MEAS_NAMES)} | \
                               {f"{n} (mean err mm)": float(meas_err[1:, i].mean()) for i, n in enumerate(MEAS_NAMES)}
        if ann["reference"] is not None:
            subj_dist[path.stem] = {f"annotation-vs-reference {n} (mm)": abs(meas[i] - ann["reference"][i])
                                    for i, n in enumerate(MEAS_NAMES)}
        samples[path.stem] = dict(meas=meas, e=e, w=w, gen_V=gen_V if cfg.save_meshes else None, res=res)
        say(f"{path.name}: E_z[{primary}]={row['expected_error']:.2f} mm  best-of-{K}={row['best_of_k']:.2f}  "
            f"z0={row['z0_error']:.2f}  mean-limb={row['mean_limb_error']:.2f}  calib={row['calibration_ratio']:.2f}")
        if cfg.save_meshes:
            for i in range(min(cfg.save_meshes, n_rows)):
                Vt = res[i][2][0]
                igl.write_triangle_mesh(str(out / f"{path.stem}_gen{i:02d}.obj"), gen_V[i], faces)
                if i == 0:
                    igl.write_triangle_mesh(str(out / f"{path.stem}_gt_aligned.obj"), Vt, F)

    if not subj:
        say("nothing evaluated")
        return
    say(f"\n=== Shape error vs ground truth, mm ({len(subj)} meshes; stats over meshes) ===")
    hdr = ["quantity", "n", "mean", "median", "std", "p95", "max"]
    table(stat_rows(subj_shape), hdr, say)
    say("\n=== Distribution of the generated family (per-mesh statistics, see distribution_metrics) ===")
    table(stat_rows(subj), hdr, say)
    say("\n=== Measurement fidelity of generated limbs (exact corrected measurement vs requested), mm ===")
    table(stat_rows(subj_meas), hdr, say)
    if subj_dist:
        say("\n=== Mesh-derived measurements vs reference_measurements in the annotation, mm ===")
        table(stat_rows(subj_dist), hdr, say)

    allrows = {k: subj[k] | subj_shape[k] | subj_meas[k] | subj_dist.get(k, {}) for k in subj}
    cols = list(next(iter(allrows.values())).keys())
    with open(out / "subjects.csv", "w") as f:
        f.write("mesh," + ",".join(f"{c}" for c in cols) + "," + ",".join(f"input {n} (mm)" for n in MEAS_NAMES) + "\n")
        for k, r in allrows.items():
            f.write(f"{k}," + ",".join(f"{r[c]:.5f}" for c in cols) + "," +
                    ",".join(f"{x:.3f}" for x in samples[k]["meas"]) + "\n")
    np.savez_compressed(out / "per_sample.npz", names=np.array(list(samples)),
                        **{f"{k}_e": v["e"] for k, v in samples.items()})
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    say(f"\nsaved to {out}")


# ============================================================================ entry point
CONFIG = dict(
    mode="evaluate",              # evaluate | annotate | make_test_data
    ckpt="???",                   # trained LimbCVAE checkpoint (evaluate)
    mesh_dir="???",               # directory of meshes (obj/stl/ply/off)
    ann_dir=None,                 # annotation jsons (default: next to the meshes)
    mesh_ext=["obj", "stl", "ply", "off"],
    unit_scale=1.0,               # multiply mesh / annotation coordinates by this to get mm (e.g. 1000 for metres)
    loop_mode="nearest",          # which slice loop(s) to measure: nearest the points | largest | all
    align="icp",                  # none | frame | icp   (rigid alignment of ground truth to each generated limb)
    k_samples=16,                 # z samples per mesh (plus z=0)
    temperature=1.0,              # z ~ N(0, T^2 I); metrics are importance-weighted back to N(0, I)
    n_points=10000,               # surface samples for distances
    primary="chamfer",            # key of SHAPE_METRICS used for the distribution statistics
    save_meshes=0,                # write this many generated limbs (+ aligned ground truth) per mesh
    limit=None,
    n_range=1000,                 # generator limbs used to flag out-of-range measurements
    port=8765, open_browser=True, # annotate mode
    seed=0,
    n_test=20, noise_mm=0.0, retessellate=True,       # make_test_data only
    out_dir="${claude_dir:}/outputs/wild/${now:%Y-%m-%d_%H-%M-%S}",
    hydra={"job": {"chdir": False}, "run": {"dir": "${out_dir}"}},
)
ConfigStore.instance().store(name="evaluate_wild", node=CONFIG)


@hydra.main(config_path=None, config_name="evaluate_wild", version_base="1.3")
def main(cfg: DictConfig):
    assert cfg.loop_mode in LOOP_MODES and cfg.align in ("none", "frame", "icp") and cfg.primary in SHAPE_METRICS
    {"evaluate": evaluate, "annotate": annotate, "make_test_data": make_test_data}[cfg.mode](cfg)


if __name__ == "__main__":
    main()
