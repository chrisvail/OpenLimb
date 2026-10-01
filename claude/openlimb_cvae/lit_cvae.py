import math

import lightning as L
import torch
import torch.nn.functional as F

from .geometry import load_shape_model, load_transforms, plausibility_affine
from .networks import Decoder, Encoder, MeasurementSurrogate


class LimbCVAE(L.LightningModule):
    """Conditional VAE: measurements (+ small latent z) -> SSM components (+ scale).

    Training:   q(z | m, c) encoder sees the true components; decoder p(c | m, z).
    Generation: z ~ N(0, I) -> a *family* of limbs, all (approximately) satisfying m.
                The 7 measurements under-determine the 11 SSM coordinates, so z only has to
                carry the ~4 leftover degrees of freedom; z_dim=0 gives a deterministic regressor.

    Losses (all in z-scored component space unless noted):
      recon      MSE(c_hat, c)                                  [posterior path]
      kl         KL(q(z|m,c) || N(0,I)), optional free bits
      vert       mesh-vertex error / vertex variance            [posterior path]
      meas       MSE(surrogate(c_hat), m)                       [posterior path]
      prior_meas MSE(surrogate(dec(m, z~N(0,I))), m)            [generative path - no ground truth]
      plaus      hinge on the generator's skin-space box        [generative path]
    """

    def __init__(self, n_components=11, n_measurements=7, scale=True, z_dim=4,
                 width=256, depth=4, dropout=0.0, surrogate_arch=None,
                 lr=1e-3, weight_decay=1e-4, warmup_epochs=3,
                 beta=0.01, kl_warmup_epochs=20, free_bits=0.0,
                 w_recon=1.0, w_vert=0.0, w_meas=1.0, w_prior_meas=1.0, w_plaus=1.0,
                 meas_noise=0.0, eval_sigma=0.1, n_diversity=8, exact_eval_n=256, exact_eval_every=10,
                 test_exact_n=1024):
        super().__init__()
        self.save_hyperparameters()
        kw = {"width": width, "depth": depth, "dropout": dropout}
        self.encoder = Encoder(n_measurements, n_components, z_dim, **kw) if z_dim > 0 else None
        self.decoder = Decoder(n_measurements, z_dim, n_components, **kw)
        self.surrogate = MeasurementSurrogate(n_components, n_measurements, **(surrogate_arch or {}))
        self.surrogate.requires_grad_(False)

        comp_tf, meas_tf = load_transforms(scale)
        mean_v, basis, _ = load_shape_model()
        W, w0 = plausibility_affine(scale)
        f = lambda t: t.float()
        self.register_buffer("comp_mean", f(comp_tf[0]))
        self.register_buffer("comp_std", f(comp_tf[1]))
        self.register_buffer("meas_mean", f(meas_tf[0]))
        self.register_buffer("meas_std", f(meas_tf[1]))
        self.register_buffer("mean_verts", f(mean_v.reshape(-1)))
        self.register_buffer("basis", f(basis))
        self.register_buffer("pl_W", f(W))
        self.register_buffer("pl_b", f(w0))
        self.register_buffer("vert_var", torch.ones(()))
        self._geo = []  # exact geometry (CPU, float64), built lazily; kept out of the module tree

    # ------------------------------------------------------------------ helpers -----
    def train(self, mode=True):
        super().train(mode)
        self.surrogate.eval()  # frozen: no dropout / norm updates
        return self

    def on_fit_start(self):
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "sample_c"):
            v = self.verts(dm.sample_c(2000).to(self.device))
            self.vert_var.copy_(v.var(0).mean())

    def verts(self, cn):
        """z-scored components -> flattened vertices (B, Vm*3), same maths as SSM_Driver.get_verts."""
        c = cn * self.comp_std + self.comp_mean
        if self.hparams.scale:
            modes, s = c[:, :-1], c[:, -1:]
        else:
            modes, s = c, 1.0
        return (self.mean_verts + modes @ self.basis) * s

    def box_coords(self, cn):
        return cn @ self.pl_W + self.pl_b

    def outside_box(self, cn, tol=0.0):
        return (self.box_coords(cn).abs() > 1 + tol).any(-1)

    def encode(self, m, c):
        return self.encoder(m, c)

    def decode(self, m, z):
        return self.decoder(m, z if self.hparams.z_dim > 0 else None)

    def norm_meas(self, meas):
        return (meas.to(self.meas_mean) - self.meas_mean) / self.meas_std

    # ------------------------------------------------------------------ generation ---
    @torch.no_grad()
    def sample(self, measurements, n=16, temperature=1.0, reject_implausible=False, max_tries=20):
        """measurements: (B,7) or (7,) in physical units (mm for scale=True).
        Returns denormalised components (B, n, n_components).

        reject_implausible: resample until n limbs lie inside the generator's skin-space box
        (falls back to filling the remainder with unfiltered samples after max_tries)."""
        was_training = self.training
        self.eval()
        meas = torch.as_tensor(measurements).reshape(-1, self.hparams.n_measurements)
        m = self.norm_meas(meas.to(self.device))
        zd = self.hparams.z_dim
        out = torch.empty(len(m), n, self.hparams.n_components, device=self.device)
        for b in range(len(m)):
            kept, total, tries = [], 0, 0
            while total < n and tries < (max_tries if reject_implausible else 1):
                z = torch.randn(n, zd, device=self.device) * temperature
                c = self.decode(m[b].expand(n, -1), z)
                if reject_implausible:
                    c = c[~self.outside_box(c)]
                kept.append(c)
                total += len(c)
                tries += 1
            if total < n:  # could not find enough plausible ones: top up
                kept.append(self.decode(m[b].expand(n, -1), torch.randn(n, zd, device=self.device) * temperature))
            out[b] = torch.cat(kept)[:n]
        self.train(was_training)
        return out * self.comp_std + self.comp_mean

    # ------------------------------------------------------------------ training -----
    def _kl(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z)
        per_dim = kl.mean(0)
        if self.hparams.free_bits > 0:
            per_dim = per_dim.clamp(min=self.hparams.free_bits)
        return per_dim.sum(), kl.sum(-1).mean()

    def _losses(self, m, c, deterministic=False):
        hp = self.hparams
        m_in = m + hp.meas_noise * torch.randn_like(m) if (hp.meas_noise > 0 and self.training) else m
        if hp.z_dim > 0:
            mu, logvar = self.encode(m_in, c)
            z = mu if deterministic else mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            kl_loss, kl = self._kl(mu, logvar)
        else:
            z, kl_loss, kl = None, m.new_zeros(()), m.new_zeros(())
        c_hat = self.decode(m_in, z)
        out = dict(kl=kl, kl_loss=kl_loss, recon=F.mse_loss(c_hat, c),
                   meas=F.mse_loss(self.surrogate(c_hat), m_in))
        out["vert"] = F.mse_loss(self.verts(c_hat), self.verts(c)) / self.vert_var if hp.w_vert > 0 else m.new_zeros(())
        # generative path: z from the prior, no ground truth, only "honour m and stay plausible"
        z_p = torch.randn(len(m), hp.z_dim, device=m.device) if hp.z_dim > 0 else None
        c_p = self.decode(m_in, z_p)
        out["prior_meas"] = F.mse_loss(self.surrogate(c_p), m_in)
        out["plaus"] = F.relu(self.box_coords(c_p).abs() - 1).pow(2).mean()
        out["_c_p"] = c_p.detach()
        out["_c_hat"] = c_hat.detach()
        return out

    def training_step(self, batch, _):
        c = batch["c"]
        m = batch.get("m")
        if m is None:  # limbs are sampled on the fly; measurements come from the surrogate
            with torch.no_grad():
                m = self.surrogate(c)
        hp = self.hparams
        o = self._losses(m, c)
        beta = hp.beta * min(1.0, (self.current_epoch + 1) / max(1, hp.kl_warmup_epochs))
        loss = (hp.w_recon * o["recon"] + beta * o["kl_loss"] + hp.w_vert * o["vert"] + hp.w_meas * o["meas"]
                + hp.w_prior_meas * o["prior_meas"] + hp.w_plaus * o["plaus"])
        self.log_dict({"train/recon": o["recon"], "train/kl": o["kl"], "train/meas": o["meas"],
                       "train/prior_meas": o["prior_meas"], "train/plaus": o["plaus"], "train/beta": beta},
                      on_step=False, on_epoch=True, batch_size=len(m))
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(m))
        return loss

    # ------------------------------------------------------------------ evaluation ---
    def _eval_step(self, batch, name):
        m, c = batch["m"], batch["c"]
        o = self._losses(m, c, deterministic=True)
        hp = self.hparams
        elbo_nll = 0.5 * c.shape[1] * o["recon"] / hp.eval_sigma ** 2 + o["kl"]  # fixed sigma => comparable across sweeps
        outside = self.outside_box(o["_c_p"]).float().mean()
        # spread of the family: per-vertex std over n_diversity prior samples of the same measurement
        k, nb = hp.n_diversity, min(len(m), 32)
        m_rep = m[:nb].repeat_interleave(k, 0)
        z = torch.randn(len(m_rep), hp.z_dim, device=m.device) if hp.z_dim > 0 else None
        diversity = self.verts(self.decode(m_rep, z)).reshape(nb, k, -1).std(1).mean()
        vert_rmse = (self.verts(o["_c_hat"]) - self.verts(c)).pow(2).mean().sqrt()
        prior_rmse = o["prior_meas"].sqrt()
        score = o["recon"].sqrt() + prior_rmse + outside
        self.log_dict({
            f"{name}/recon_mse": o["recon"], f"{name}/kl": o["kl"], f"{name}/elbo_nll": elbo_nll,
            f"{name}/vert_rmse": vert_rmse, f"{name}/post_meas_rmse": o["meas"].sqrt(),
            f"{name}/prior_meas_rmse": prior_rmse, f"{name}/prior_outside_box": outside,
            f"{name}/diversity_vert_std": diversity, f"{name}/score": score,
        }, on_epoch=True, on_step=False, prog_bar=(name == "val"), batch_size=len(m))

    def validation_step(self, batch, _):
        self._eval_step(batch, "val")

    def test_step(self, batch, _):
        self._eval_step(batch, "test")

    # -- exact (SSM_Driver.measure) evaluation: slow, so run on a subset ---------------
    def _geometry(self):
        if not self._geo:
            from .geometry import SSMGeometry
            self._geo.append(SSMGeometry(scale=self.hparams.scale, normalise=False))
        return self._geo[0]

    @torch.no_grad()
    def exact_eval(self, m, c, prefix):
        geo = self._geometry()
        names = [d["name"] for d in geo.measurement_details]
        meas = (m * self.meas_std + self.meas_mean).double().cpu()
        zd = self.hparams.z_dim
        z_prior = torch.randn(len(m), zd, device=m.device) if zd > 0 else None
        z_post = self.encode(m, c)[0] if zd > 0 else None
        res = {}
        for tag, cc in (("prior", self.decode(m, z_prior)), ("post", self.decode(m, z_post))):
            err = (geo.get_measures((cc * self.comp_std + self.comp_mean).double().cpu()) - meas).abs()
            res[f"{prefix}_exact/{tag}_mae"] = err.mean().item()
            res[f"{prefix}_exact/{tag}_rel_err"] = (err / meas.abs()).mean().item()
            for i, n in enumerate(names):
                res[f"{prefix}_exact/{tag}_mae_{n.replace(' ', '_')}"] = err[:, i].mean().item()
        return res

    def _run_exact(self, ds_name, n, prefix):
        fixed = self.trainer.datamodule.fixed[ds_name]
        m, c = fixed["m"][:n].to(self.device), fixed["c"][:n].to(self.device)
        self.log_dict(self.exact_eval(m, c, prefix), on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        hp = self.hparams
        if hp.exact_eval_n and hp.exact_eval_every > 0 and (self.current_epoch + 1) % hp.exact_eval_every == 0 \
                and not self.trainer.sanity_checking:
            self._run_exact("val", hp.exact_eval_n, "val")

    def on_test_epoch_end(self):
        if self.hparams.test_exact_n:
            self._run_exact("test", self.hparams.test_exact_n, "test")

    # ------------------------------------------------------------------ optimiser ----
    def configure_optimizers(self):
        hp = self.hparams
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
        total = self.trainer.estimated_stepping_batches
        warm = int(total * hp.warmup_epochs / max(1, self.trainer.max_epochs))
        lam = lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lam), "interval": "step"}}
