import math

import lightning as L
import torch
import torch.nn.functional as F

from .geometry import load_transforms, plausibility_affine
from .networks import MLP, MeasurementSurrogate


class LimbMLP(L.LightningModule):
    """Plain regression baseline: measurements -> SSM components, MSE only.

    Deliberately identical to the CVAE's decoder with z_dim=0 (same residual MLP, width/depth, optimiser,
    schedule, data stream, step budget) but WITHOUT any of the extra losses (no KL, no measurement
    consistency, no plausibility). Training measurements come from the frozen surrogate exactly as for the
    CVAE, so the only difference between the two experiments is the objective / the missing latent.
    """

    def __init__(self, n_components=11, n_measurements=7, scale=True, width=256, depth=4, dropout=0.0,
                 lr=1e-3, weight_decay=1e-4, warmup_epochs=3, meas_noise=0.0, surrogate_arch=None):
        super().__init__()
        self.save_hyperparameters()
        self.net = MLP(n_measurements, n_components, width=width, depth=depth, dropout=dropout)
        self.surrogate = MeasurementSurrogate(n_components, n_measurements, **(surrogate_arch or {}))
        self.surrogate.requires_grad_(False)
        comp_tf, meas_tf = load_transforms(scale)
        W, w0 = plausibility_affine(scale)
        self.register_buffer("comp_mean", comp_tf[0].float())
        self.register_buffer("comp_std", comp_tf[1].float())
        self.register_buffer("meas_mean", meas_tf[0].float())
        self.register_buffer("meas_std", meas_tf[1].float())
        self.register_buffer("pl_W", W.float())
        self.register_buffer("pl_b", w0.float())

    def train(self, mode=True):
        super().train(mode)
        self.surrogate.eval()
        return self

    def forward(self, m):
        return self.net(m)

    def norm_meas(self, meas):
        return (meas.to(self.meas_mean) - self.meas_mean) / self.meas_std

    def box_coords(self, cn):
        return cn @ self.pl_W + self.pl_b

    def training_step(self, batch, _):
        c = batch["c"]
        with torch.no_grad():
            m = batch["m"] if "m" in batch else self.surrogate(c)
            if self.hparams.meas_noise > 0:
                m = m + self.hparams.meas_noise * torch.randn_like(m)
        loss = F.mse_loss(self(m), c)
        self.log("train/mse", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(c))
        return loss

    def _eval_step(self, batch, name):
        m, c = batch["m"], batch["c"]
        c_hat = self(m)
        self.log_dict({
            f"{name}/mse": F.mse_loss(c_hat, c),
            f"{name}/meas_rmse": F.mse_loss(self.surrogate(c_hat), m).sqrt(),
            f"{name}/outside_box": (self.box_coords(c_hat).abs() > 1).any(-1).float().mean(),
        }, on_epoch=True, on_step=False, prog_bar=(name == "val"), batch_size=len(m))

    def validation_step(self, batch, _):
        self._eval_step(batch, "val")

    def test_step(self, batch, _):
        self._eval_step(batch, "test")

    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        total = self.trainer.estimated_stepping_batches
        warm = int(total * hp.warmup_epochs / max(1, self.trainer.max_epochs))
        lam = lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lam), "interval": "step"}}
