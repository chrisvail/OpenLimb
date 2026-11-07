"""Corrected planar measurements (drop-in for SSM_Driver.measure).

measure_limbs.py classifies an edge as cut iff 0 < t < 1.  Because every OpenLimb measurement plane passes exactly
through a mesh vertex (and a face may even have an edge lying in the plane), that test silently drops

  * every segment of the faces touching the vertex on the plane           (edges with t == 0 or t == 1),
  * every edge lying IN the plane                                         (0 / 0 -> nan -> not cut),

so circumferences come out short (mean ~1.1 %, up to ~15 mm).  Widths additionally take the minimum together with
x = 0 (`values *= t_mask` before the max-min), so a slice that does not straddle x = 0 is measured from the origin.

Here a vertex belongs to the "+" side iff its signed distance d >= 0 (a symbolic nudge of the plane by +eps) and an
edge is cut iff its end points are on different sides.  Every triangle then has 0 or 2 cut edges whatever the
degeneracy, so the polyline is closed, continuous in the vertices and equals the exact planar perimeter.
"""
import torch
from torch import nn


def _sides(verts, plane_point, plane_normal):
    """verts (B,V,3) -> signed distance (B,V)."""
    p = verts[:, plane_point].unsqueeze(1) if torch.is_tensor(
        plane_point) or isinstance(plane_point, int) else plane_point
    return torch.einsum("bvi,i->bv", verts - p, plane_normal.reshape(-1).to(verts.dtype))


def _cut_points(verts, edge2vert, d):
    a, b = edge2vert[:, 0], edge2vert[:, 1]
    da, db = d[:, a], d[:, b]
    cut = (da >= 0) != (db >= 0)
    t = torch.where(cut, da / (da - db).where(cut,
                    torch.ones_like(da)), torch.zeros_like(da))
    pts = verts[:, a] + t.unsqueeze(-1) * (verts[:, b] - verts[:, a])
    return cut, pts


def circumference(verts, edge2vert, face2edge, plane_point, plane_normal):
    d = _sides(verts, plane_point, plane_normal)
    cut, pts = _cut_points(verts, edge2vert, d)
    # (B,F,3), (B,F,3,3)
    fc, fp = cut[:, face2edge], pts[:, face2edge]
    total = 0
    for i, j in ((0, 1), (2, 1), (0, 2)):
        seg = torch.linalg.norm(fp[:, :, i] - fp[:, :, j], dim=-1)
        total = total + (seg * (fc[:, :, i] & fc[:, :, j])).sum(1)
    return total


def width(verts, edge2vert, plane_point, plane_normal, plane_direction):
    d = _sides(verts, plane_point, plane_normal)
    cut, pts = _cut_points(verts, edge2vert, d)
    v = torch.einsum("bei,i->be", pts,
                     plane_direction.reshape(-1).to(verts.dtype))
    inf = torch.full_like(v, float("inf"))
    hi, lo = torch.where(
        cut, v, -inf).max(1).values, torch.where(cut, v, inf).min(1).values
    return torch.where(cut.any(1), hi - lo, torch.zeros_like(hi))


def measure_length(verts, v1, v2, direction):
    # print(f"Length:        {verts.shape=}    {v1}    {v2}    {direction.shape=}")

    if len(verts.shape) == 2:
        verts = verts[None]

    if isinstance(direction, (tuple, list)):
        direction = verts[direction[0]] - verts[direction[1]]
        direction /= torch.linalg.norm(direction)

    return torch.abs(torch.einsum("ij, ij -> i", verts[:, v1] - verts[:, v2], direction))


class Measure(nn.Module):
    """Same interface / details as SSM_Driver.Measurements."""

    def __init__(self, edge2vert, face2edge, details):
        super().__init__()

        self.details = details
        self.measures = []
        for dt in details:
            if dt["type"] == "circumference":
                self.measures.append(lambda v, dt=dt: circumference(
                    v, edge2vert, face2edge, dt["plane_point"], dt["plane_normal"]))
            elif dt["type"] == "width":
                self.measures.append(lambda v, dt=dt: width(
                    v, edge2vert, dt["plane_point"], dt["plane_normal"], dt["plane_direction"]))
            else:
                self.measures.append(lambda v, dt=dt: measure_length(
                    v, dt["v1"], dt["v2"], dt["direction"]))

    def forward(self, x, verbose=False):
        if x.dim() == 2:                       # a single limb (LegMeasurementDataset.get_verts squeezes the batch)
            x = x[None]
        return torch.stack([m(x) for m in self.measures], dim=-1)
