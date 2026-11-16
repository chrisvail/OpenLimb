from pathlib import Path

import lightning as L
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data.limb import get_measurement_details
from data.measure import Measure
from data.sampler import OpenLimbSampler


class MeasurementDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        measure_data_dir: Path,
        batch_size: int,
        n_nulls: int,
        normalise: bool = True,
        scale: bool = True,
    ):
        super().__init__()
        self.n_nulls = n_nulls
        self.batch_size = batch_size
        self.normalise = normalise
        data_path = Path(data_path)
        measure_data_dir = Path(measure_data_dir)

        self.train_sampler = OpenLimbSampler(
            batch_size, 0, measure_data_dir, scale=scale
        )

        self.n_components = self.train_sampler.n_components

        measurement_details = get_measurement_details(measure_data_dir, torch.float64)
        self.n_measurements = len(measurement_details)
        self.measure = Measure(
            self.train_sampler.edge2vert,
            self.train_sampler.face2edge,
            measurement_details,
            measure_data_dir,
        )

        self.cols = ["C1", "C2", "C3", "C4", "L1", "W1", "W2"]
        self.df = (
            pl.read_csv(data_path, null_values=["NA", "26 or 25"])
            .fill_nan(None)
            .with_row_index("idx")
            .select(["idx"] + self.cols)
            .filter(
                (pl.col("C1") > 15).fill_null(True),
                (pl.col("C2") > 15).fill_null(True),
                (pl.col("C3") > 15).fill_null(True),
                (pl.col("C4") > 15).fill_null(True),
                (pl.col("L1") > 2).fill_null(True),
                (pl.col("W1") > 4).fill_null(True),
                (pl.col("W2") < 15).fill_null(True),
            )
            .with_columns(n_nulls=pl.sum_horizontal(pl.col(self.cols).is_null()))
            .filter(pl.col("n_nulls") <= self.n_nulls)
        )

        if self.normalise:
            self.chunks = [
                self.measure.normalise(
                    chunk.select(self.cols).to_torch().double(), scaled=True
                ).float()
                for chunk in self.df.iter_slices(n_rows=self.batch_size)
            ]
        else:
            self.chunks = [
                chunk.select(self.cols).to_torch().float()
                for chunk in self.df.iter_slices(n_rows=self.batch_size)
            ]


    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        
        return None, self.chunks[idx]

class MeasurementDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        measure_data_dir: Path,
        batch_size: int,
        n_nulls: int,
        normalise: bool = True,
        scale: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.params = {
            "data_path": data_path,
            "batch_size": batch_size,
            "n_nulls": n_nulls,
            "measure_data_dir": measure_data_dir,
            "normalise": normalise,
            "scale": scale,
        }

        self.dataset = MeasurementDataset(**self.params)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=None, shuffle=False, num_workers=0)

    def predict_dataloader(self):
        return self.test_dataloader()

    def __getattr__(self, attr: str):
        if attr == "dataset":
            raise AttributeError(attr)
        return self.dataset.__getattribute__(attr)
