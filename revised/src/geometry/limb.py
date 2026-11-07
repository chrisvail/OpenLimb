from pathlib import Path

import torch
import lightning as L
from .sampler import OpenLimbSampler
from .measure import ExactMeasure


class LimbDataModule(L.LightningDataModule):
    def __init__(self, data_dir, scale=True, batch_size=512, val_batches=32, test_batches=1024,
                 seed=0, num_workers=0):
        super(L.LightningDataModule, self).__init__()
        self.save_hyperparameters()

        self.scale = scale
        self.sampler = OpenLimbSampler(batch_size, seed=seed)
        self.measure = ExactMeasure(self.sampler.edge2vert,
                                    self.sampler.face2edge, {})


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
