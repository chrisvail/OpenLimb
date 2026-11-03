import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class MLP(nn.Module):
    """Linear -> depth x residual block -> Linear."""

    def __init__(self, d_in, d_out, width=256, depth=4, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, d_out))

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class Encoder(nn.Module):
    """q(z | measurements, components) -> (mu, logvar)."""

    def __init__(self, d_meas, d_comp, z_dim, **kw):
        super().__init__()
        self.net = MLP(d_meas + d_comp, 2 * z_dim, **kw)

    def forward(self, m, c):
        mu, logvar = self.net(torch.cat([m, c], -1)).chunk(2, -1)
        return mu, logvar.clamp(-12, 6)


class Decoder(nn.Module):
    """p(components | measurements, z). Measurements are re-injected at the input."""

    def __init__(self, d_meas, z_dim, d_comp, **kw):
        super().__init__()
        self.net = MLP(d_meas + z_dim, d_comp, **kw)

    def forward(self, m, z=None):
        return self.net(m if z is None else torch.cat([m, z], -1))


class MeasurementSurrogate(nn.Module):
    """Fast differentiable stand-in for SSM_Driver.measure (z-scored in, z-scored out).

    The exact measurement (plane/edge intersections over the full mesh) costs ~5ms per
    limb, far too slow to sit inside a training step, so the CVAE's measurement-consistency
    loss goes through this network and the exact function is only used for evaluation.
    """

    def __init__(self, d_comp, d_meas, width=256, depth=3):
        super().__init__()
        self.net = MLP(d_comp, d_meas, width=width, depth=depth)

    def forward(self, c):
        return self.net(c)
