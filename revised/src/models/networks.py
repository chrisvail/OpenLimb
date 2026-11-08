import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, width, dropout, activation=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), activation(), nn.Linear(
                width, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class MLP(nn.Module):
    """Linear -> depth x residual block -> Linear."""

    def __init__(self, in_dim, out_dim, width=256, depth=4, dropout=0.0, activation=nn.GELU):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(
            *[ResBlock(width, dropout, activation) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(
            width), activation(), nn.Linear(width, out_dim))

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class Encoder(nn.Module):
    """q(z | measurements, components) -> (mu, logvar)."""

    def __init__(self, n_measurements, z_dim, n_components, **kw):
        super().__init__()
        self.net = MLP(n_measurements + n_components, 2 * z_dim, **kw)

    def forward(self, m, c):
        mu, logvar = self.net(torch.cat([m, c], -1)).chunk(2, -1)
        return mu, logvar.clamp(-12, 6)


class Decoder(nn.Module):
    """p(components | measurements, z). Measurements are re-injected at the input."""

    def __init__(self, n_measurements, z_dim, n_components, **kw):
        super().__init__()
        self.net = MLP(n_measurements + z_dim, n_components, **kw)

    def forward(self, m, z=None):
        return self.net(m if z is None else torch.cat([m, z], -1))


class MeasurementSurrogate(nn.Module):
    """Fast differentiable stand-in for SSM_Driver.measure (z-scored in, z-scored out).

    The exact measurement (plane/edge intersections over the full mesh) costs ~5ms per
    limb, far too slow to sit inside a training step, so the CVAE's measurement-consistency
    loss goes through this network and the exact function is only used for evaluation.
    """

    def __init__(self, n_components, n_measurements, width=256, depth=3, activation=nn.GELU):
        super().__init__()
        self.net = MLP(n_components, n_measurements, width=width,
                       depth=depth, activation=activation)

    def forward(self, c):
        return self.net(c)
