import lightning as L
import torch
from torch import nn

from models.networks import MeasurementSurrogate


class SurrogateMeasurementModel(L.LightningModule):
    def __init__(self, n_components, n_measurements, width, depth, activation, measurement_std, lr=5E-3, weight_decay=1E-5):
        super().__init__()
        self.save_hyperparameters()

        activation = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Tanh": nn.Tanh,
        }[activation]
        self.model = MeasurementSurrogate(
            n_components, n_measurements, width, depth, activation)
        self.register_buffer(
            "measurement_std", torch.as_tensor(measurement_std).float())

    def forward(self, components):
        return self.model(components)

    def _step(self, batch, name):
        components, measurements = batch
        batch_size = components.shape[0]
        preds = self.model(components)
        loss = nn.functional.mse_loss(preds, measurements)

        self.log(f"{name}/MSE", loss,
                 batch_size=batch_size, prog_bar=True)
        with torch.no_grad():
            diff = (preds.detach() - measurements).abs() * self.measurement_std
            mae = diff.mean()
            worst = diff.max()

            self.log(f"{name}/MAE", mae, batch_size)
            self.log(f"{name}/Worst", worst, batch_size, reduce_fx="max")

            if not self.training:
                for i, d in enumerate(diff.T):
                    self.log(f"{name}/MAE_{i}", d.mean(),
                             batch_size=batch_size)
                    self.log(f"{name}/Worst_{i}", d.max(),
                             batch_size=batch_size, reduce_fx="max")

        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def val_step(self, batch):
        return self._step(batch, "val")

    def test_step(self, batch):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
