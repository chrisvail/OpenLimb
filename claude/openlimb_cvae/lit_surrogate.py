import lightning as L
import torch
import torch.nn.functional as F

from .geometry import load_transforms
from .networks import MeasurementSurrogate


class LitSurrogate(L.LightningModule):
    """Regress z-scored measurements from z-scored components."""

    def __init__(self, n_components, n_measurements, scale=True, width=256, depth=3, lr=2e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.net = MeasurementSurrogate(n_components, n_measurements, width, depth)
        self.register_buffer("meas_std", load_transforms(scale)[1][1].float())

    def forward(self, c):
        return self.net(c)

    def _step(self, batch, name):
        m, c = batch["m"], batch["c"]
        pred = self(c)
        loss = F.mse_loss(pred, m)
        self.log(f"{name}/mse", loss, prog_bar=True, batch_size=len(m))
        # mean absolute error in physical units (mm, or size-normalised units if scale=False)
        mae = ((pred - m).abs() * self.meas_std).mean(0)
        self.log(f"{name}/mae", mae.mean(), batch_size=len(m))
        if name != "train":
            for i, v in enumerate(mae):
                self.log(f"{name}/mae_{i}", v, batch_size=len(m))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
