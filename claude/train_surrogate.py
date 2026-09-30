"""Stage 1: fit the differentiable measurement surrogate used inside the CVAE loss.

    python claude/train_surrogate.py                 # scale=true  -> claude/data/surrogate_scaled.pt
    python claude/train_surrogate.py data.scale=false
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlimb_cvae.common import register_resolvers, setup_env, surrogate_path  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import lightning as L  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from openlimb_cvae.data import LimbDataModule  # noqa: E402
from openlimb_cvae.lit_surrogate import LitSurrogate  # noqa: E402


@hydra.main(config_path="conf", config_name="surrogate", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    dm = LimbDataModule(**cfg.data)
    dm.setup()
    model = LitSurrogate(dm.d_comp, dm.d_meas, scale=cfg.data.scale, **cfg.model)
    out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt = ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/mse", mode="min")
    trainer = L.Trainer(**cfg.trainer, logger=hydra.utils.instantiate(cfg.logger), callbacks=[ckpt])
    trainer.fit(model, datamodule=dm)
    best = LitSurrogate.load_from_checkpoint(ckpt.best_model_path)
    metrics = trainer.test(best, datamodule=dm)[0]
    path = surrogate_path(cfg.data.scale)
    torch.save({"state_dict": best.net.state_dict(),
                "arch": {"width": cfg.model.width, "depth": cfg.model.depth},
                "test_metrics": metrics}, path)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved surrogate -> {path}\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
