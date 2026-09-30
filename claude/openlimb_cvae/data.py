import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from .geometry import ComponentSampler, load_transforms


class _Stream(IterableDataset):
    """Endless stream of freshly sampled, already-batched limbs (nothing is stored).

    Yields {"c": z-scored components} and, when ``exact`` is set, {"m": z-scored exact measurements}
    as well (slow: ~4.5 ms/limb). Without ``exact`` the consumer derives m from the surrogate.
    RNG / geometry state is kept on the object, so with persistent workers every epoch continues
    the stream instead of replaying it.
    """

    def __init__(self, scale, batch_size, steps, seed, exact):
        self.scale, self.batch_size, self.steps, self.seed, self.exact = scale, batch_size, steps, seed, exact
        self._state = None

    def __len__(self):
        return self.steps

    def _init_worker(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        geo = None
        if self.exact:
            from .geometry import SSMGeometry  # heavy import (igl); only in the process that needs it
            geo = SSMGeometry(scale=self.scale, normalise=True)
        self._state = dict(
            sampler=ComponentSampler(self.scale), geo=geo, wid=wid, nw=nw,
            gen=torch.Generator().manual_seed(self.seed * 1_000_003 + wid))

    def __iter__(self):
        if self._state is None:
            self._init_worker()
        st = self._state
        n = self.steps // st["nw"] + (st["wid"] < self.steps % st["nw"])
        sampler = st["sampler"]
        for _ in range(n):
            raw = sampler.raw(self.batch_size, st["gen"])
            batch = {"c": ((raw - sampler.comp_tf[0]) / sampler.comp_tf[1]).float()}
            if st["geo"] is not None:
                with torch.no_grad():
                    batch["m"] = st["geo"].get_measures(raw).float()
            yield batch


class _Fixed(Dataset):
    """A handful of pre-built batches (validation / test)."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


class LimbDataModule(L.LightningDataModule):
    """Limbs are sampled on the fly with the same recipe as GenerateRandomLimbs.py (skin modes
    ~ U(box) -> linear regression -> components, scale ~ U(342.8, 439.8)) - no docker, no files.

    * train: endless stream, ``steps_per_epoch`` batches per "epoch". Measurements come from the
      consumer's surrogate unless ``exact_train`` (used to fit the surrogate itself).
    * val / test: small fixed sets (seeded, kept in RAM) with *exact* SSM_Driver measurements.
    """

    def __init__(self, scale=True, batch_size=512, steps_per_epoch=100, n_val=512, n_test=1024,
                 seed=0, num_workers=0, exact_train=False):
        super().__init__()
        self.save_hyperparameters()
        self.scale = bool(scale)
        self.sampler = ComponentSampler(scale)
        self.d_comp = 11 if scale else 10
        self.d_meas = 7
        self.fixed = {}

    def sample_c(self, n, seed=123):
        return self.sampler.normalised(n, torch.Generator().manual_seed(seed))

    def _exact_set(self, n, seed):
        from .geometry import SSMGeometry
        geo = SSMGeometry(scale=self.scale, normalise=True)
        gen = torch.Generator().manual_seed(seed)
        ms, cs = [], []
        for i in range(0, n, 256):
            raw = self.sampler.raw(min(256, n - i), gen)
            with torch.no_grad():
                ms.append(geo.get_measures(raw).float())
            cs.append(((raw - self.sampler.comp_tf[0]) / self.sampler.comp_tf[1]).float())
        m, c = torch.cat(ms), torch.cat(cs)
        bs = self.hparams.batch_size
        return {"m": m, "c": c, "batches": _Fixed([{"m": m[i:i + bs], "c": c[i:i + bs]} for i in range(0, n, bs)])}

    def setup(self, stage=None):
        if self.fixed:
            return
        # Distinct fixed seeds => val/test are identical across runs & sweep trials, and disjoint from
        # the training stream in practice (continuous 10-D sampling).
        self.fixed["val"] = self._exact_set(self.hparams.n_val, seed=10_001)
        if stage in ("test", None) or self.hparams.n_test:
            self.fixed["test"] = self._exact_set(self.hparams.n_test, seed=10_002)

    def train_dataloader(self):
        h = self.hparams
        ds = _Stream(self.scale, h.batch_size, h.steps_per_epoch, h.seed, h.exact_train)
        return DataLoader(ds, batch_size=None, num_workers=h.num_workers, persistent_workers=h.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.fixed["val"]["batches"], batch_size=None)

    def test_dataloader(self):
        return DataLoader(self.fixed["test"]["batches"], batch_size=None)
