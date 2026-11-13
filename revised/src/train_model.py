"""Main runner."""
import warnings
from datetime import datetime

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

# Suppress the PyTorch / Lightning deprecation and performance warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="../configs", config_name="train_model_default", version_base="1.3")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    L.seed_everything(cfg.seed, workers=True)

    data = hydra.utils.instantiate(cfg.data)
    n_components = data.n_components
    n_measurements = data.n_measurements
    plausibility = data.train_sampler.get_plausibility_transform(True)
    measurement_transforms = data.measure.get_transforms(True)

    model = hydra.utils.instantiate(cfg.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not ("surrogate" in cfg.model._target_.lower()):
        model = model(n_components=n_components, n_measurements=n_measurements,
                      plausibility=plausibility, measurement_std=measurement_transforms["std"])

        if cfg.surrogate_checkpoint is None:
            raise ValueError("Set surrogate_checkpoint to a trained surrogate")
        model.load_surrogate(cfg.surrogate_checkpoint)
    else:
        model = model(n_components=n_components, n_measurements=n_measurements,
                      measurement_std=measurement_transforms["std"])

    logger = hydra.utils.instantiate(cfg.logger)(
        name=f"{cfg.experiment}_{cfg.model._target_.split(".")[-1]}_s{cfg.seed}_{timestamp}",
        tags=[cfg.model._target_.split(".")[-1], timestamp, "v1", "train"]
        )
    logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        allow_val_change=True,
    )

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    try:
        trainer.fit(model, data)
    finally:
        logger.finish() 


if __name__ == "__main__":
    main()
    # Run from inside revised
    # Quotes around measurement_dropout is important.
    # python src/train_model.py -m model=mlp,cvae seed=range(5) surrogate_checkpoint=/path/to/surrogate.ckpt 'model.measurement_dropout={prob:0.3,max_drop:2}'
