import lightning as L  # noqa: N999
import torch
from torch import nn

from models.networks import Decoder, Encoder
from models.surrogate import SurrogateMeasurementModel


class CVAEModel(L.LightningModule):
    def __init__(self,
                 n_components,
                 n_measurements,
                 z_dim,
                 encoder_width,
                 decoder_width,
                 encoder_depth,
                 decoder_depth,
                 encoder_dropout,
                 decoder_dropout,
                 activation,
                 surrogate_details,
                 plausibility,
                 measurement_std,
                 measurement_dropout=None,
                 w_measure=1,
                 w_components=1,
                 w_plausibility=200,
                 w_kl_div=0.1,
                 kl_warmup=10,
                 w_prior=1,
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
        self.z_dim = z_dim

        if self.measurement_dropout is not None:
            self.encoder = Encoder(2*n_measurements, z_dim, n_components,
                                   width=encoder_width, depth=encoder_depth, activation=activation, dropout=encoder_dropout)
            self.decoder = Decoder(2*n_measurements, z_dim, n_components,
                                   width=decoder_width, depth=decoder_depth, activation=activation, dropout=decoder_dropout)
        else:
            self.encoder = Encoder(n_measurements, z_dim, n_components,
                                   width=encoder_width, depth=encoder_depth, activation=activation, dropout=encoder_dropout)
            self.decoder = Decoder(n_measurements, z_dim, n_components,
                                   width=decoder_width, depth=decoder_depth, activation=activation, dropout=decoder_dropout)

        plausibility_mat, plausibility_vec = plausibility
        self.register_buffer("plausibility_mat", plausibility_mat)
        self.register_buffer("plausibility_vec", plausibility_vec)

        self.surrogate = SurrogateMeasurementModel(**surrogate_details)
        self.surrogate.requires_grad_(False)

        self.w_measure = w_measure
        self.w_components = w_components
        self.w_plausibility = w_plausibility
        self.w_kl_div = w_kl_div
        self.kl_warmup = kl_warmup
        self.w_prior = w_prior

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
            mask = self.drop_measurements(measurements)
        else:
            mask = None

        posterior_loss = self.posterior_path(
            components, measurements, name, mask)
        prior_loss = self.prior_path(measurements, name, mask)

        loss = posterior_loss + self.w_prior*prior_loss

        self.log(f"{name}/Loss", loss,
                 batch_size=batch_size, prog_bar=True)

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
        else:
            mask = torch.zeros_like(measurements, dtype=torch.bool)

        return mask

    def posterior_path(self, components, measurements, name, mask=None):
        if mask is not None:
            measurement_inputs = measurements.clone()
            measurement_inputs[mask] = 0
            measurement_inputs = torch.concat(
                (measurement_inputs, mask.float()), dim=-1)
        else:
            measurement_inputs = measurements

        mu, log_var = self.encoder(measurement_inputs, components)
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()

        pred_components = self.decoder(measurement_inputs, z)

        pred_measures = self.surrogate(pred_components)
        component_loss = nn.functional.mse_loss(pred_components, components)
        measurement_loss = nn.functional.mse_loss(pred_measures, measurements)
        plausibility = self._plausibility(pred_components)
        plausibility_loss = nn.functional.relu(plausibility - 1).pow(2).mean()
        kl_div_loss = (-0.5 * (1 + log_var - mu**2 -
                       log_var.exp())).sum(-1).mean()

        if self.training:
            kl_warmup = min(1.0, (self.current_epoch + 1) / self.kl_warmup)
            kl_weight = self.w_kl_div * kl_warmup
        else:
            kl_weight = self.w_kl_div

        with torch.no_grad():
            batch_size = measurements.shape[0]
            diff = (pred_measures.detach() - measurements).abs() * \
                self.measurement_std
            mae = diff.mean()
            worst = diff.max()

            self.log_dict({
                f"{name}/posterior/MAE": mae,
                f"{name}/posterior/Component": component_loss,
                f"{name}/posterior/Measurement": measurement_loss,
                f"{name}/posterior/Plausibility": plausibility_loss,
                f"{name}/posterior/Plausibile Percentage": (plausibility <= 1).all(-1).float().mean(),
            }, batch_size=batch_size)
            self.log(f"{name}/posterior/Worst", worst,
                     batch_size=batch_size, reduce_fx="max")

            if not self.training:
                self.log_per_measurement_error(
                    name, "posterior", batch_size, diff)

        return self.w_components*component_loss \
            + self.w_measure*measurement_loss \
            + self.w_plausibility*plausibility_loss \
            + kl_weight*kl_div_loss

    def log_per_measurement_error(self, name, tag, batch_size, diff):
        for i, d in enumerate(diff.T):
            self.log(f"{name}/{tag}/mae_{i}", d.mean(),
                     batch_size=batch_size)
            self.log(f"{name}/{tag}/Worst_{i}", d.max(),
                     batch_size=batch_size, reduce_fx="max")

    def prior_path(self, measurements, name, mask=None):
        batch_size = measurements.shape[0]
        z = torch.randn((batch_size, self.z_dim),
                        device=measurements.device, dtype=measurements.dtype)

        if mask is not None:
            masked_measurements = measurements.clone()
            masked_measurements[mask] = 0

            pred_components = self.decoder(torch.concat(
                (masked_measurements, mask.float()), dim=-1), z)
        else:
            pred_components = self.decoder(measurements, z)

        pred_measures = self.surrogate(pred_components)
        if mask is not None:
            measurement_loss = self._masked_mse(
                pred_measures, measurements, mask)
        else:
            measurement_loss = nn.functional.mse_loss(
                pred_measures, measurements)
        plausibility = self._plausibility(pred_components)
        plausibility_loss = nn.functional.relu(plausibility - 1).pow(2).mean()

        with torch.no_grad():
            batch_size = measurements.shape[0]
            diff = (pred_measures.detach() - measurements).abs() * \
                self.measurement_std
            mae = diff.mean()
            worst = diff.max()

            self.log_dict({
                f"{name}/prior/MAE": mae,
                f"{name}/prior/Measurement": measurement_loss,
                f"{name}/prior/Plausibility": plausibility_loss,
                f"{name}/prior/Plausibile Percentage": (plausibility <= 1).all(-1).float().mean(),
            }, batch_size=batch_size)
            self.log(f"{name}/prior/Worst", worst,
                     batch_size=batch_size, reduce_fx="max")

            if not self.training:
                self.log_per_measurement_error(name, "prior", batch_size, diff)

        return self.w_measure*measurement_loss \
            + self.w_plausibility*plausibility_loss

    @staticmethod
    def _masked_mse(pred, ground_truth, mask):
        return (((pred - ground_truth) * ~mask)**2).sum() / (~mask).sum()

    def _plausibility(self, components):
        return (components @ self.plausibility_mat + self.plausibility_vec).abs()

    def train(self, mode=True):
        super().train(mode)
        # Force surrogate into eval mode even during training
        self.surrogate.eval()
        return self

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._check_spread(batch[1], n=8)
        return self._step(batch, "val")

    def test_step(self, batch):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def sample(self, measurements, n=8, temperature=1.0):
        was_training = self.training
        self.eval()

        if measurements.dim() == 1:
            measurements = measurements[None]

        batch_size = measurements.shape[0]

        mask = torch.isnan(measurements)
        if self.measurement_dropout is not None:
            measurements = measurements.masked_fill(mask, 0)
            mask = torch.repeat_interleave(mask, n, dim=0)
            measurements = torch.repeat_interleave(measurements, n, dim=0)
            inputs = torch.concat((measurements, mask.float()), dim=-1)
        else:
            if mask.any():
                raise ValueError(
                    "Measurements dropped on full measurement model")
            inputs = torch.repeat_interleave(measurements, n, dim=0)

        z = temperature * torch.randn((batch_size*n, self.z_dim),
                                      device=measurements.device, dtype=measurements.dtype)

        pred_components = self.decoder(inputs, z)

        self.train(was_training)
        return pred_components.reshape((batch_size, n, -1))

    @torch.no_grad()
    def _check_spread(self, measurements, n=8):
        components = self.sample(measurements, n)
        std = components.std(dim=1)
        self.log_dict({
            f"val/Component {i} spread": x for i, x in enumerate(std.mean(0))
        } | {"val/Spread mean": std.mean()}, batch_size=measurements.shape[0])
