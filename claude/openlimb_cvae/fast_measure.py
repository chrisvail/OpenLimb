"""Fast, exact evaluation of the seven measurements for search loops (GA fitness, finite differences).

SSM_Driver.measure intersects every one of the ~50k mesh edges with each of the six horizontal measurement
planes, although a plane only ever cuts a thin band of them. Here the same formulas are applied to that band only:

* at start-up, probe limbs decide per plane which vertices are *always* above / *always* below it;
  an edge joining two always-above (or two always-below) vertices can never be cut and is dropped;
* at run time the "always" assumption is verified for every limb (a cheap comparison on the z coordinates);
  limbs that violate it are sent through the original SSM_Driver.measure instead.

So the result equals the original code for every limb (up to float summation order, ~1e-12 mm) - the band only
decides how much work is done, never the answer.
"""
import torch

from .geometry import SCALE_MAX, SCALE_MIN, SKIN_MAX, SKIN_MIN, ComponentSampler, SSMGeometry


class BoxMap:
    """u in [-1, 1]^11  <->  raw components. u = box coordinates of OpenLimbTT's own generator
    (10 skin-mode scores + tibia length), so |u| <= 1 is exactly "the generator could have produced this limb"."""

    def __init__(self):
        s = ComponentSampler(True)
        self.A, self.b = s.A, s.b
        self.Ainv = torch.linalg.inv(s.A.T)
        self.mid = torch.cat([(s.lo + s.hi) / 2, torch.tensor([(SCALE_MIN + SCALE_MAX) / 2]).double()])
        self.half = torch.cat([(s.hi - s.lo) / 2, torch.tensor([(SCALE_MAX - SCALE_MIN) / 2]).double()])

    def to_raw(self, u):
        x = u.double() * self.half + self.mid
        return torch.cat([x[:, :10] @ self.A.T + self.b, x[:, 10:]], -1)

    def to_u(self, raw):
        raw = raw.double()
        x = torch.cat([(raw[:, :10] - self.b) @ self.Ainv, raw[:, 10:]], -1)
        return (x - self.mid) / self.half


class FastMeasure:
    def __init__(self, geo: SSMGeometry = None, n_probe=4096, margin=1.5, seed=0):
        self.geo = geo or SSMGeometry(scale=True, normalise=False)
        import SSM_Driver

        e2v, f2e = SSM_Driver.edge2vert.long(), SSM_Driver.face2edge.long()
        # probe limbs: the generator's box blown up by `margin`, so limbs a little outside it stay on the fast path
        g = torch.Generator().manual_seed(seed)
        u = (torch.rand(n_probe, 11, generator=g, dtype=torch.float64) * 2 - 1) * margin
        u[:, 10].clamp_(-1, 1)                                   # sign of (z - z_plane) does not depend on scale
        z = self.geo.get_verts(BoxMap().to_raw(u))[..., 2]        # (n_probe, V)

        self.planes = []
        for d in self.geo.measurement_details:
            if d["type"] == "length":
                assert d["direction"].flatten().tolist() == [0, 0, 1]
                self.planes.append(dict(type="length", v1=int(d["v1"]), v2=int(d["v2"])))
                continue
            assert d["plane_normal"].flatten().tolist() == [0, 0, 1], "fast path assumes horizontal planes"
            p = int(d["plane_point"])
            above = z > z[:, p:p + 1]
            always_above, always_below = above.all(0), (~above).all(0)
            static = always_above[e2v] | always_below[e2v]        # (E, 2): endpoint never changes side
            same_side = (always_above[e2v].all(1)) | (always_below[e2v].all(1))
            keep = ~(static.all(1) & same_side)                   # edges that can be cut
            eidx = keep.nonzero().squeeze(1)
            plane = dict(type=d["type"], p=p, e0=e2v[eidx, 0], e1=e2v[eidx, 1],
                         expect=always_above, band=~(always_above | always_below))
            if d["type"] == "circumference":
                local = torch.full((e2v.shape[0],), -1, dtype=torch.long)
                local[eidx] = torch.arange(len(eidx))
                fe = local[f2e]                                   # (F, 3) local edge ids, -1 = cannot be cut
                fe = fe[(fe >= 0).sum(1) >= 2]
                plane["fe"], plane["fe_ok"] = fe.clamp_min(0), fe >= 0
            else:
                assert d["plane_direction"].flatten().tolist() == [1, 0, 0]
            self.planes.append(plane)

    @torch.no_grad()
    def __call__(self, raw):
        """raw (B, 11) components -> (B, 7) measurements in mm, identical to SSM_Driver.measure."""
        verts = self.geo.get_verts(raw)
        z = verts[..., 2]
        out = torch.empty(len(raw), len(self.planes), dtype=verts.dtype)
        ok = torch.ones(len(raw), dtype=torch.bool)
        for k, pl in enumerate(self.planes):
            if pl["type"] == "length":
                out[:, k] = (z[:, pl["v1"]] - z[:, pl["v2"]]).abs()
                continue
            zp = z[:, pl["p"]:pl["p"] + 1]
            ok &= (((z > zp) == pl["expect"]) | pl["band"]).all(1)
            a, b = verts[:, pl["e0"]], verts[:, pl["e1"]]        # (B, Es, 3)
            t = (zp - a[..., 2]) / (b[..., 2] - a[..., 2])        # same expression as plane_edge_intersection
            cut = (0 < t) & (t < 1)
            if pl["type"] == "width":
                x = a[..., 0] + t * (b[..., 0] - a[..., 0])
                inf = torch.full_like(x, float("inf"))
                lo = torch.where(cut, x, inf).min(1).values.clamp_max(0)      # original takes the min together with 0
                hi = torch.where(cut, x, -inf).max(1).values
                out[:, k] = (hi - lo).clamp_min(0)
            else:
                pts = torch.nan_to_num(a + t[..., None] * (b - a), nan=float("inf"))
                fp = pts[:, pl["fe"]]                              # (B, Fs, 3, 3)
                fc = cut[:, pl["fe"]] & pl["fe_ok"]               # (B, Fs, 3)
                total = 0
                for i, j in ((0, 1), (2, 1), (0, 2)):
                    seg = torch.linalg.norm(fp[:, :, i] - fp[:, :, j], dim=-1)
                    total = total + torch.where(fc[:, :, i] & fc[:, :, j], seg, torch.zeros_like(seg)).sum(1)
                out[:, k] = total
        if not ok.all():                                           # outside the probed band: original code
            out[~ok] = self.geo.get_measures(raw[~ok], normalise=False)
        self.last_fallbacks = int((~ok).sum())
        return out
