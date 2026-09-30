"""Stage 2: train the conditional VAE (hydra + lightning).

    python claude/train.py                                   # single run
    python claude/train.py -m model.z_dim=2,3,4,6 model.beta=0.001,0.01,0.1   # grid sweep
    python claude/train.py -m experiment=optuna              # Optuna (TPE) sweep
    python claude/train.py trainer=smoke data.n_max=2048      # quick smoke test

Returns the best value of `monitor` (default val/score, lower is better) so sweepers can optimise it.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlimb_cvae.common import register_resolvers, setup_env, surrogate_path  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import lightning as L  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from openlimb_cvae.data import LimbDataModule  # noqa: E402


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    out = Path(HydraConfig.get().runtime.output_dir)

    dm = LimbDataModule(**cfg.data)
    dm.setup()

    sur_file = surrogate_path(cfg.data.scale, cfg.surrogate_path)
    if not sur_file.exists():
        raise FileNotFoundError(f"{sur_file} missing - run `python claude/train_surrogate.py data.scale={cfg.data.scale}` first")
    sur = torch.load(sur_file)

    model = hydra.utils.instantiate(cfg.model, _convert_="all", n_components=dm.d_comp, n_measurements=dm.d_meas,
                                    scale=cfg.data.scale, surrogate_arch=sur["arch"])
    model.surrogate.load_state_dict(sur["state_dict"])

    ckpt = ModelCheckpoint(dirpath=out / "checkpoints", monitor=cfg.monitor, mode="min", filename="best")
    callbacks = [ckpt, LearningRateMonitor("epoch")]
    if cfg.early_stopping_patience:
        callbacks.append(EarlyStopping(cfg.monitor, patience=cfg.early_stopping_patience, mode="min"))
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule=dm)
    best = float(ckpt.best_model_score) if ckpt.best_model_score is not None else math.inf
    metrics = {"best_" + cfg.monitor: best, "best_ckpt": ckpt.best_model_path}
    if cfg.test_after_fit and ckpt.best_model_path:
        metrics |= {k: float(v) for k, v in trainer.test(model, datamodule=dm, ckpt_path="best")[0].items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return best if math.isfinite(best) else 1e6


if __name__ == "__main__":
    main()
