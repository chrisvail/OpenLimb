from pathlib import Path

import lightning as L
import polars as pl
from torch.utils.data import DataLoader, IterableDataset

from ..geometry.measure import Measure


class MeasurementDataset(IterableDataset):
    def __init__(self, data_path: Path, batch_size: int, n_nulls: int, measure: Measure, normalise: bool=True):
        super().__init__()
        self.n_nulls = n_nulls
        self.batch_size = batch_size
        self.measure = measure
        self.normalise = normalise
        

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
            .with_columns(
                n_nulls=pl.sum_horizontal(pl.col(self.cols).is_null())
            )
            .filter(pl.col("n_nulls") <= self.n_nulls)
        )

    def len(self):
        return (len(self.df) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for chunk in self.df.iter_slices(n_rows=self.batch_size):
            x  = chunk.select(self.cols).to_torch().double()

            if self.normalise:
                x = self.measure.normalise(x, scaled=True)

            # Components (Unavailable obviously), measurements
            yield None, x



class MeasurementDataModule(L.LightningDataModule):
    def __init__(self, data_path: Path, batch_size: int, n_nulls: int, measure: Measure, normalise: bool=True):
        super().__init__()
        self.save_hyperparameters()

        self.params = {
            "data_path": data_path,
            "batch_size": batch_size,
            "n_nulls": n_nulls,
            "measure": measure,
            "normalise": normalise
        }

    def test_dataloader(self):
        return DataLoader(
            MeasurementDataset(**self.params),
            batch_size=None,
            shuffle=False,
            num_workers=0
        )

    def predict_dataloader(self):
        return self.test_dataloader()