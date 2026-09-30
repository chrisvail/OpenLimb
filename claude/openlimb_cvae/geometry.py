"""Thin, differentiable wrapper around SSM_Driver's measurement machinery.

LegMeasurementDataset.__init__ deletes files and shells out to docker/WSL, so we
never instantiate it. Instead we build a bare instance (object.__new__), fill in the
same tensors its __init__ would load, and borrow its get_measures/normalise_measures
methods unchanged. Only get_verts is re-implemented (mathematically identical) because
the original materialises a (B, 10, 298854) tensor which is ~GBs for realistic batches.
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _in_root():
    # SSM_Driver uses "./data_components/..." relative paths at import time.
    old = os.getcwd()
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        yield
    finally:
        os.chdir(old)


class SSMGeometry(nn.Module):
    """components (+scale) -> vertices -> (normalised) measurements."""

    def __init__(self, scale: bool = True, normalise: bool = True, dtype=torch.float64):
        super().__init__()
        with _in_root():
            import SSM_Driver
            from SSM_Driver import LegMeasurementDataset

            d = ROOT / "data_components"
            self.scale = bool(scale)
            self.normalise = normalise
            self.dtype = dtype
            self.measure = SSM_Driver.measure
            self.measurement_details = SSM_Driver.measurement_details

            raw = torch.load(d / "vert_components.pt").to(dtype)          # (10, 3V)
            mean = torch.load(d / "mean_verts.pt").to(dtype)              # (V, 3)
            vmap = torch.load(d / "vert_mapping.pt")                      # (Vm,)
            V = mean.shape[0]
            # Pre-select the mapped vertices so only (10, Vm*3) is ever multiplied.
            self.register_buffer("comp_basis", raw.reshape(raw.shape[0], V, 3)[:, vmap].reshape(raw.shape[0], -1))
            self.register_buffer("mean_verts", mean[vmap])
            self.n_modes = raw.shape[0]

            prefix = "" if self.scale else "un"
            self.register_buffer("comp_tf", torch.load(d / f"{prefix}scaled_component_transforms.pt").to(dtype))
            self.register_buffer("meas_tf", torch.load(d / f"{prefix}scaled_measurement_transforms.pt").to(dtype))

        # Borrow the reference implementation for measurement + normalisation.
        ref = object.__new__(LegMeasurementDataset)
        ref.measure = self.measure
        ref.measurement_transforms = self.meas_tf
        ref.scale = int(self.scale)
        self._ref = ref

    @property
    def n_components(self):
        return self.n_modes + int(self.scale)

    @property
    def n_measurements(self):
        return len(self.measurement_details)

    def get_verts(self, components):
        components = components.to(self.dtype)
        if self.scale:
            c, s = components[:, :-1], components[:, -1:]
        else:
            c, s = components, torch.ones_like(components[:, :1])
        verts = self.mean_verts[None] + (c @ self.comp_basis).reshape(-1, *self.mean_verts.shape)
        return verts * s[..., None]

    def get_measures(self, components, normalise=None):
        normalise = self.normalise if normalise is None else normalise
        m = self.measure.forward(self.get_verts(components))
        return self._ref.normalise_measures(m) if normalise else m

    # -- normalisation of component space (per-dimension z-score) ------------------
    def norm_components(self, c):
        return (c - self.comp_tf[0]) / self.comp_tf[1]

    def denorm_components(self, z):
        return z * self.comp_tf[1] + self.comp_tf[0]


# ---------------------------------------------------------------------------------
# Light-weight helpers that do NOT import SSM_Driver (no igl / measurement setup)
# ---------------------------------------------------------------------------------
DATA_DIR = ROOT / "data_components"
CLAUDE_DATA = ROOT / "claude" / "data"

# Ranges hard-coded in GenerateRandomLimbs.py for the 10 "skin-only" modes, plus the tibia
# length range for the scale factor. Every training limb is c = A @ s + b with s drawn
# uniformly from this box, so "is this a plausible limb" == "does s lie inside the box".
SKIN_MIN = [-14.80692194113721, -5.37869110537926, -3.996990835549319, -4.05537984190567,
            -2.403525754650053, -1.854646894835898, -2.13215613028021, -1.197319576810893,
            -0.798154129842514, -0.7353096205329]
SKIN_MAX = [25.4537448635005, 10.06971435469401, 4.12168595787859, 4.24308399616669,
            4.73074159745632, 2.390247129446665, 2.19420856255569, 1.221847275562656,
            0.802807003549, 0.95283973472981]
SCALE_MIN, SCALE_MAX = 342.8, 439.8


def load_transforms(scale: bool):
    """(comp_tf, meas_tf), each (2, D): row 0 = mean, row 1 = std. Mirrors the dataset."""
    n = 10 + int(scale)
    prefix = "" if scale else "un"
    comp = torch.load(DATA_DIR / "scaled_component_transforms.pt")[:, :n].double()
    meas = torch.load(DATA_DIR / f"{prefix}scaled_measurement_transforms.pt").double()
    return comp, meas


def load_shape_model():
    """(mean_verts (Vm,3), basis (10, Vm*3), faces) restricted to the mapped vertices."""
    raw = torch.load(DATA_DIR / "vert_components.pt").double()
    mean = torch.load(DATA_DIR / "mean_verts.pt").double()
    vmap = torch.load(DATA_DIR / "vert_mapping.pt")
    V = mean.shape[0]
    basis = raw.reshape(raw.shape[0], V, 3)[:, vmap].reshape(raw.shape[0], -1)
    return mean[vmap], basis, torch.load(DATA_DIR / "face2vert.pt")


def plausibility_affine(scale: bool):
    """Affine map (W, w0) taking z-scored components to box coordinates u.

    u = cn @ W + w0 is in [-1, 1] for every training sample; |u| > 1 flags a limb the
    generator could never have produced. Uses the LR.pkl coefficients extracted to
    claude/data/lr_map.npz (see README).
    """
    import numpy as np

    lr = np.load(CLAUDE_DATA / "lr_map.npz")
    A, b = torch.tensor(lr["coef"]).double(), torch.tensor(lr["intercept"]).double()
    comp_tf, _ = load_transforms(scale)
    lo, hi = torch.tensor(SKIN_MIN).double(), torch.tensor(SKIN_MAX).double()
    n = comp_tf.shape[1]
    # c = cn*std + mean ; s = (c[:10]-b) @ inv(A^T) ; u = (s-lo)/(hi-lo)*2-1
    inv = torch.linalg.inv(A.T)
    half = (hi - lo) / 2
    W = torch.zeros(n, n, dtype=torch.float64)
    w0 = torch.zeros(n, dtype=torch.float64)
    W[:10, :10] = torch.diag(comp_tf[1, :10]) @ inv @ torch.diag(1 / half)
    w0[:10] = ((comp_tf[0, :10] - b) @ inv - (lo + hi) / 2) / half
    if scale:
        h = (SCALE_MAX - SCALE_MIN) / 2
        W[10, 10] = comp_tf[1, 10] / h
        w0[10] = (comp_tf[0, 10] - (SCALE_MIN + SCALE_MAX) / 2) / h
    return W, w0


class ComponentSampler:
    """On-the-fly replacement for GenerateRandomLimbs.py (no docker, no disk).

    Identical recipe: skin modes ~ U(box), components = LR(skin) = A s + b, scale ~ U(342.8, 439.8).
    """

    def __init__(self, scale: bool):
        import numpy as np

        lr = np.load(CLAUDE_DATA / "lr_map.npz")
        self.A = torch.tensor(lr["coef"]).double()
        self.b = torch.tensor(lr["intercept"]).double()
        self.lo, self.hi = torch.tensor(SKIN_MIN).double(), torch.tensor(SKIN_MAX).double()
        self.scale = bool(scale)
        self.comp_tf, _ = load_transforms(scale)

    def raw(self, n, generator=None):
        s = self.lo + torch.rand(n, 10, generator=generator, dtype=torch.float64) * (self.hi - self.lo)
        c = s @ self.A.T + self.b
        if self.scale:
            sc = SCALE_MIN + torch.rand(n, 1, generator=generator, dtype=torch.float64) * (SCALE_MAX - SCALE_MIN)
            c = torch.cat([c, sc], -1)
        return c

    def normalised(self, n, generator=None):
        return ((self.raw(n, generator) - self.comp_tf[0]) / self.comp_tf[1]).float()
