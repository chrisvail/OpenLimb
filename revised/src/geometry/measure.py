from functools import partial

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


def _edges_cut_in_batch(edge2vert, distances):
    """(E,) edges cut in at least one limb of the batch."""
    first_above = distances[:, edge2vert[:, 0]] >= 0
    second_above = distances[:, edge2vert[:, 1]] >= 0
    return (first_above != second_above).any(0)


def circumference(verts, edge2vert, face2edge, plane_point, plane_normal):
    distances = _sides(verts, plane_point, plane_normal)
    cut_edges = _edges_cut_in_batch(edge2vert, distances)
    if not cut_edges.any():
        return torch.zeros(len(verts), dtype=verts.dtype, device=verts.device)

    # Edge ids in the reduced edge list, -1 for dropped edges
    reduced_id = torch.full((len(edge2vert),), -1, device=edge2vert.device)
    reduced_id[cut_edges] = torch.arange(
        int(cut_edges.sum()), device=edge2vert.device)
    face_edges = reduced_id[face2edge]
    face_edges = face_edges[(face_edges >= 0).sum(1) >= 2]
    kept = face_edges >= 0
    face_edges = face_edges.clamp_min(0)

    is_cut, points = _cut_points(verts, edge2vert[cut_edges], distances)
    # (B,F,3), (B,F,3,3)
    face_cut, face_points = is_cut[:, face_edges] & kept, points[:, face_edges]
    perimeter = 0
    for first, second in ((0, 1), (2, 1), (0, 2)):
        length = torch.linalg.norm(
            face_points[:, :, first] - face_points[:, :, second], dim=-1)
        perimeter = perimeter + \
            (length * (face_cut[:, :, first] & face_cut[:, :, second])).sum(1)
    return perimeter


def width(verts, edge2vert, plane_point, plane_normal, plane_direction):
    distances = _sides(verts, plane_point, plane_normal)
    cut_edges = _edges_cut_in_batch(edge2vert, distances)
    if not cut_edges.any():
        return torch.zeros(len(verts), dtype=verts.dtype, device=verts.device)

    is_cut, points = _cut_points(verts, edge2vert[cut_edges], distances)
    positions = torch.einsum("bei,i->be", points,
                             plane_direction.reshape(-1).to(verts.dtype))
    inf = torch.full_like(positions, float("inf"))
    upper = torch.where(is_cut, positions, -inf).max(1).values
    lower = torch.where(is_cut, positions, inf).min(1).values
    return torch.where(is_cut.any(1), upper - lower, torch.zeros_like(upper))


def measure_length(verts, v1, v2, direction):
    if len(verts.shape) == 2:
        verts = verts[None]

    if isinstance(direction, (tuple, list)):
        direction = verts[direction[0]] - verts[direction[1]]
        direction /= torch.linalg.norm(direction)

    return torch.abs(torch.einsum("ij, ij -> i", verts[:, v1] - verts[:, v2], direction))


class Measure(nn.Module):
    """
    Measurement of a model give measurement details.

    Measurement details can comprise of circumfrential measurements, widths and lengths.
    """

    def __init__(self, edge2vert, face2edge, details, data_dir):
        super().__init__()

        self.details = details
        self.measures = []
        for dt in details:
            if dt["type"] == "circumference":
                self.measures.append(partial(circumference, edge2vert=edge2vert, face2edge=face2edge,
                                     plane_point=dt["plane_point"], plane_normal=dt["plane_normal"]))
            elif dt["type"] == "width":
                self.measures.append(partial(
                    width,
                    edge2vert=edge2vert,
                    plane_point=dt["plane_point"],
                    plane_normal=dt["plane_normal"],
                    plane_direction=dt["plane_direction"]))
            else:
                self.measures.append(
                    partial(measure_length, v1=dt["v1"], v2=dt["v2"], direction=dt["direction"]))

        scaled_transform = torch.load(
            data_dir / "scaled_measurement_transforms.pt")
        self.scaled_mean, self.scaled_std = scaled_transform["mean"], scaled_transform["std"]

        unscaled_transform = torch.load(
            data_dir / "unscaled_measurement_transforms.pt")
        self.unscaled_mean, self.unscaled_std = unscaled_transform[
            "mean"], unscaled_transform["std"]

    def __call__(self, x, normalise=True, scaled=True):
        if x.dim() == 2:
            x = x[None]

        meas = torch.stack([m(x) for m in self.measures], dim=-1)   # (B, 7)
        if not normalise:
            return meas

        if scaled:
            return (meas - self.scaled_mean) / self.scaled_std
        else:
            return (meas - self.unscaled_mean) / self.unscaled_std

    def get_transforms(self, scaled=True):
        if scaled:
            return {"mean": self.scaled_mean, "std": self.scaled_std}
        else:
            return {"mean": self.unscaled_mean, "std": self.unscaled_std}
