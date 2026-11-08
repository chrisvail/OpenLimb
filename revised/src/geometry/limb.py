from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from geometry.measure import Measure
from geometry.sampler import OpenLimbSampler


class LimbDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path, scale: bool = True, batch_size: int = 512, train_batches: int = 512, val_batches=4, test_batches: int = 16,
                 seed: int = 2441, val_seed=9001, test_seed: int = 7393, num_workers: int = 0, normalise_components: bool = True, normalise_measurements: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.test_seed = test_seed
        self.train_batches = train_batches
        self.test_batches = test_batches
        self.normalise_components, self.normalise_measurements = normalise_components, normalise_measurements
        self.scale = scale
        self.num_workers = num_workers

        self.train_sampler = OpenLimbSampler(
            batch_size, seed, self.data_dir, scale)

        self.measure = Measure(self.train_sampler.edge2vert,
                               self.train_sampler.face2edge, get_measurement_details(self.data_dir, torch.float64), self.data_dir)

    def train_dataloader(self):
        dataset = _StreamingDataset(
            self.train_sampler, self.measure, self.train_batches, self.normalise_components, self.normalise_measurements)
        return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        val_sampler = OpenLimbSampler(
            self.batch_size, self.val_seed, self.data_dir, self.scale)
        dataset = _FixedDataset(
            val_sampler, self.measure, self.val_batches, self.normalise_components, self.normalise_measurements)
        return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    def test_dataloader(self):
        test_sampler = OpenLimbSampler(
            self.batch_size, self.test_seed, self.data_dir, self.scale)
        dataset = _FixedDataset(
            test_sampler, self.measure, self.test_batches, self.normalise_components, self.normalise_measurements)
        return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)


class _StreamingDataset(IterableDataset):
    def __init__(self, sampler: OpenLimbSampler, measure: Measure, steps: int, normalise_components: bool, normalise_measurements: bool):
        super().__init__()
        self.sampler = sampler
        self.measure = measure
        self.steps = steps
        self.normalise_components = normalise_components
        self.normalise_measurements = normalise_measurements
        self._seeded = False

    def __len__(self):
        return self.steps

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)

        if not self._seeded:
            # + 1 so it doesnt collide with the test dataset
            self.sampler.generator.manual_seed(
                self.sampler.seed * 3581 + wid + 1)
            self._seeded = True

        for _ in range(wid, self.steps, nw):
            components = self.sampler.sample()
            with torch.no_grad():
                measurements = self.measure(self.sampler.to_mesh(
                    components, False)[0], self.normalise_measurements, self.sampler.scale)

                if self.normalise_components:
                    components = self.sampler.normalise_components(components)

            yield components.to(torch.float32), measurements.to(torch.float32)


class _FixedDataset(Dataset):
    def __init__(self, sampler: OpenLimbSampler, measure: Measure, batches, normalise_components: bool, normalise_measurements: bool):
        super().__init__()
        self.batches = []
        for _ in range(batches):
            components = sampler.sample()
            with torch.no_grad():
                measurements = measure(sampler.to_mesh(
                    components, False)[0], normalise_measurements, sampler.scale)

                if normalise_components:
                    components = sampler.normalise_components(components)

                self.batches.append(
                    (components.to(torch.float32), measurements.to(torch.float32)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def get_measurement_details(data_dir, dtype):
    """Helper function that defines the set of measurements"""
    # Order
    # Mid patella tendon
    # Distal tibia
    # Knee widest
    # Knee above? This one feels off
    # Over fib head
    # Fib head
    # Circ 3
    # Circ 4

    vert_idxs = torch.load(Path(data_dir) / "mesh.pt")["selected_verts"]
    return (
        {
            # Circ one
            "type": "circumference",
            "plane_point": vert_idxs[4],
            "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
            "name": "Circumference 1",
        },
        {
            "type": "circumference",
            "plane_point": vert_idxs[5],
            "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
            "name": "Circumference 2",
        },
        {
            "type": "circumference",
            "plane_point": vert_idxs[6],
            "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
            "name": "Circumference 3",
        },
        {
            "type": "circumference",
            "plane_point": vert_idxs[7],
            "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
            "name": "Circumference 4",
        },
        {
            "type": "length",
            "v1": vert_idxs[0],
            "v2": vert_idxs[1],
            "direction": torch.tensor([[0, 0, 1]], dtype=dtype),
            "name": "Length 1",
        },
        {
            "type": "width",
            "plane_point": vert_idxs[2],
            "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
            "plane_direction": torch.tensor([[1, 0, 0]], dtype=dtype),
            "name": "Width 1",
        },
        {
            "type": "width",
            "plane_point": vert_idxs[3],
            "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
            "plane_direction": torch.tensor([[1, 0, 0]], dtype=dtype),
            "name": "Width 2",
        },
    )


if __name__ == "__main__":
    # cd revised/src && uv run python -m geometry.limb
    data_dir = Path("../../data_components")
    dm = LimbDataModule(data_dir, num_workers=2,
                        train_batches=8, test_batches=2)
    # expect (512, 11) and (512, 7), float32
    c, m = next(iter(dm.train_dataloader()))
    t = list(dm.test_dataloader())
    assert all(torch.equal(a[0], b[0]) for a, b in zip(
        t, dm.test_dataloader()))   # same test data each call
