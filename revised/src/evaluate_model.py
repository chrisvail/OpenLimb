"""Main runner."""
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import hydra
import lightning as L
import torch
from geometry.measure import Measure
from geometry.sampler import OpenLimbSampler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Suppress the PyTorch / Lightning deprecation and performance warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="../configs", config_name="eval_model_default", version_base="1.3")
@torch.no_grad()
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    L.seed_everything(cfg.seed, workers=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = hydra.utils.instantiate(cfg.data)
    n_components = data.n_components
    n_measurements = data.n_measurements
    plausibility = data.train_sampler.get_plausibility_transform(True)
    measurement_transforms = data.measure.get_transforms(True)
    sampler: OpenLimbSampler = data.train_sampler
    measure: Measure = data.measure

    model = hydra.utils.instantiate(cfg.model)
    test_loader = data.test_dataloader()

    if not ("gauss_newton" in cfg.model._target_.lower()):
        logger = hydra.utils.instantiate(cfg.logger)(
            name=f"{cfg.experiment}_{cfg.model._target_.split(".")[-1]}_s{cfg.seed}_{time_stamp}",
            tags=[cfg.model._target_.split(".")[-1], time_stamp, "v1", "train"]
            )
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            allow_val_change=True,
        )
        # Model is neural nework
        model = model(n_components=n_components, n_measurements=n_measurements,
                      plausibility=plausibility, measurement_std=measurement_transforms["std"])

        ckpt = torch.load(cfg.model_checkpoint,
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        trainer: L.Trainer = hydra.utils.instantiate(
            cfg.trainer, logger=logger)

        pred_components = trainer.predict(model, data)
        save_results(cfg, time_stamp, sampler, measure, pred_components)

        if model.measurement_dropout is not None:
            # Model has dropout so lets analyse that
            datasets = {
                "random_drop_1": {"mode": "random", "drop_num": 1},
                "random_drop_2": {"mode": "random", "drop_num": 2},
                **{f"index_drop_1({i})": {"mode": "index", "idxs": (i,)}
                   for i in range(n_measurements)},
                **{f"index_drop_2({x[0]}_{x[1]})": {"mode": "index", "idxs": x} for x in combinations(range(n_measurements), 2)}
            }
            dropout_res = {}
            for name, details in datasets.items():
                ds = DataLoader(EvaluationDataLoader(
                                test_loader.dataset, **details), batch_size=None, shuffle=False)
                pred_components = trainer.predict(model, ds)
                pred_measures = torch.concat([measure(sampler.to_mesh(
                    comp, True)[0], normalise=False).float() for comp in pred_components], dim=0)

                dropout_res[name +
                            "_components"] = torch.concat(pred_components, dim=0).detach().cpu()
                dropout_res[name +
                            "_measurements"] = pred_measures.detach().cpu()

            drop_path = get_file_path(cfg, time_stamp, "dropout")
            torch.save(dropout_res, drop_path)

    else:
        model = model(sampler=sampler, measure=measure)
        dataloader = test_loader
        pred_components = [model(measurements).float()
                           for _, measurements in tqdm(dataloader)]
        save_results(cfg, time_stamp, sampler, measure, pred_components)

        # Dropout
        datasets = {
            "random_drop_1": {"mode": "random", "drop_num": 1},
            "random_drop_2": {"mode": "random", "drop_num": 2},
            **{f"index_drop_1({i})": {"mode": "index", "idxs": (i,)}
                for i in range(n_measurements)},
            **{f"index_drop_2({x[0]}_{x[1]})": {"mode": "index", "idxs": x} for x in combinations(range(n_measurements), 2)}
        }
        dropout_res = {}
        for name, details in tqdm(datasets.items()):
            ds = DataLoader(EvaluationDataLoader(
                test_loader.dataset, **details), batch_size=None, shuffle=False)
            pred_components = [model(measurements).float()
                               for _, measurements in tqdm(ds)]
            pred_measures = torch.concat([measure(sampler.to_mesh(
                comp, True)[0], normalise=False).float() for comp in pred_components], dim=0)

            dropout_res[name +
                        "_components"] = torch.concat(pred_components, dim=0).detach().cpu()
            dropout_res[name +
                        "_measurements"] = pred_measures.detach().cpu()

        drop_path = get_file_path(cfg, time_stamp, "dropout")
        torch.save(dropout_res, drop_path)


class EvaluationDataLoader(Dataset):
    def __init__(self, dataset, mode, idxs=None, drop_num=0):
        self.dataset = dataset
        batch_size, n_measurements = self.dataset[0][1].shape

        if mode == "random":
            gen = torch.Generator()
            gen.manual_seed(5657)
            self.masks = torch.rand(
                (len(dataset), batch_size, n_measurements), generator=gen).argsort(-1) < drop_num
        elif mode == "index":
            self.masks = torch.zeros(
                (len(dataset), batch_size, n_measurements), dtype=torch.bool)
            self.masks[:, :, idxs] = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        components, measurements = self.dataset[idx]
        return components, measurements.masked_fill(self.masks[idx], torch.nan)


def save_results(cfg, time_stamp, sampler, measure, pred_components):
    pred_measures = torch.concat([measure(sampler.to_mesh(
        comp, True)[0], normalise=False).float() for comp in pred_components], dim=0)

    file_path = get_file_path(cfg, time_stamp)
    torch.save({
        "pred_components": torch.concat(pred_components, dim=0).detach().cpu(),
        "pred_measures": pred_measures.detach().cpu(),
        "cfg": OmegaConf.to_container(cfg, resolve=True)
    }, file_path)


def get_file_path(cfg, time_stamp, suffix=""):
    model_name = cfg.model._target_.split(".")[-1]
    eval_path = Path(__file__).parent.parent / "evaluations"
    out_dir = eval_path / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time_stamp}_seed{cfg.seed}_{suffix}.pt"
    return path


if __name__ == "__main__":
    main()
