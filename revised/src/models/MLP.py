import lightning as L  # noqa: N999
import torch
from torch import nn

from models.networks import MLP
from models.surrogate import SurrogateMeasurementModel


class MLPModel(L.LightningModule):
    def __init__(self,
                 n_components,
                 n_measurements,
                 width,
                 depth,
                 mlp_dropout,
                 activation,
                 surrogate_details,
                 plausibility,
                 measurement_std,
                 measurement_dropout=None,
                 w_measure=1,
                 w_components=1,
                 w_plausibility=200,
                 lr=5E-3,
                 weight_decay=1E-5):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "plausibility"
        ])

        activation = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Tanh": nn.Tanh,
        }[activation]

        self.measurement_dropout = measurement_dropout

        if self.measurement_dropout is not None:
            self.model = MLP(
                2*n_measurements, n_components, width, depth, mlp_dropout, activation)
        else:
            self.model = MLP(
                n_measurements, n_components, width, depth, mlp_dropout, activation)

        plausibility_mat, plausibility_vec = plausibility
        self.register_buffer("plausibility_mat", plausibility_mat.float())
        self.register_buffer("plausibility_vec", plausibility_vec.float())

        self.surrogate = SurrogateMeasurementModel(
            n_components, n_measurements, measurement_std=measurement_std, **surrogate_details)
        self.surrogate.requires_grad_(False)

        self.w_measure = w_measure
        self.w_components = w_components
        self.w_plausibility = w_plausibility

        self.register_buffer(
            "measurement_std", torch.as_tensor(measurement_std).float())

    def load_surrogate(self, checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.surrogate.load_state_dict(
            ckpt["state_dict"], strict=True)
        self.surrogate.requires_grad_(False)
        self.surrogate.eval()

    def _step(self, batch, name):
        components, measurements = batch
        batch_size = components.shape[0]

        if self.measurement_dropout is not None:
            pred_components = self.drop_measurements(measurements)
        else:
            pred_components = self.model(measurements)

        pred_measures = self.surrogate(pred_components)
        component_loss = nn.functional.mse_loss(pred_components, components)
        measurement_loss = nn.functional.mse_loss(pred_measures, measurements)
        plausibility = self._plausibility(pred_components)
        plausibility_loss = nn.functional.relu(plausibility - 1).pow(2).mean()

        loss = self.w_components*component_loss \
            + self.w_measure*measurement_loss \
            + self.w_plausibility*plausibility_loss

        self.log(f"{name}/Loss", loss,
                 batch_size=batch_size, prog_bar=True)
        with torch.no_grad():
            diff = (pred_measures.detach() - measurements).abs() * \
                self.measurement_std
            mae = diff.mean()
            worst = diff.max()

            self.log_dict({
                f"{name}/MAE": mae,
                f"{name}/Component": component_loss,
                f"{name}/Measurement": measurement_loss,
                f"{name}/Plausibility": plausibility_loss,
                f"{name}/Plausibile Percentage": (plausibility <= 1).all(-1).float().mean(),
            }, batch_size=batch_size)
            self.log(f"{name}/Worst", worst,
                     batch_size=batch_size, reduce_fx="max")

            if not self.training:
                for i, d in enumerate(diff.T):
                    self.log(f"{name}/mae_{i}", d.mean(),
                             batch_size=batch_size)
                    self.log(f"{name}/Worst_{i}", d.max(),
                             batch_size=batch_size, reduce_fx="max")

        return loss

    def drop_measurements(self, measurements):
        batch_size = measurements.shape[0]
        if self.training:
            dropout = torch.rand(
                batch_size, device=measurements.device) < self.measurement_dropout["prob"]
            k_dropped = torch.randint(
                1, self.measurement_dropout["max_drop"] + 1, (batch_size, ), device=measurements.device)
            rank = torch.rand_like(measurements).argsort(-1)
            mask = torch.logical_and(
                rank < k_dropped[:, None], dropout[:, None])
            masked_measurements = measurements.clone()
            masked_measurements[mask] = 0
        else:
            masked_measurements = measurements
            mask = torch.zeros_like(measurements)

        pred_components = self.model(torch.concat(
            (masked_measurements, mask.float()), dim=-1))
        return pred_components

    def _plausibility(self, components):
        return (components @ self.plausibility_mat + self.plausibility_vec).abs()

    def train(self, mode=True):
        super().train(mode)
        # Force surrogate into eval mode even during training
        self.surrogate.eval()
        return self

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")

    def test_step(self, batch):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
