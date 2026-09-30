"""Generate a family of plausible limbs from measurements using a trained CVAE.

    python claude/generate.py ckpt=<path/to/best.ckpt> measurements=[300,290,280,270,120,110,115] n_samples=32
    python claude/generate.py ckpt=<...> test_index=3 save_meshes=true

Writes components.npy (n, 11) [10 SSM components + scale], measurements_check.json comparing the
requested measurements with the *exact* SSM_Driver measurements of the generated limbs, and
optionally one .obj per limb.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlimb_cvae.common import register_resolvers, setup_env  # noqa: E402

setup_env()
register_resolvers()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from openlimb_cvae.data import LimbDataModule  # noqa: E402
from openlimb_cvae.geometry import load_shape_model  # noqa: E402
from openlimb_cvae.lit_cvae import LimbCVAE  # noqa: E402


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        f.writelines(f"v {x} {y} {z}\n" for x, y, z in verts)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)


@hydra.main(config_path="conf", config_name="generate", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    model = LimbCVAE.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    scale = model.hparams.scale

    if cfg.test_index is not None:
        dm = LimbDataModule(scale=scale, n_val=1, n_test=cfg.test_index + 1)
        dm.setup()
        m_norm = dm.fixed["test"]["m"][cfg.test_index]
        target = (m_norm * model.meas_std + model.meas_mean)
    elif cfg.measurements is not None:
        target = torch.tensor(list(cfg.measurements), dtype=torch.float32)
    else:
        raise ValueError("provide `measurements=[...]` or `test_index=N`")

    comps = model.sample(target, n=cfg.n_samples, temperature=cfg.temperature,
                         reject_implausible=cfg.reject_implausible)[0]           # (n, n_comp)
    out = Path(HydraConfig.get().runtime.output_dir)
    np.save(out / "components.npy", comps.numpy())

    # exact check of how well the family honours the requested measurements
    from openlimb_cvae.geometry import SSMGeometry
    achieved = SSMGeometry(scale=scale, normalise=False).get_measures(comps.double()).float()
    names = [d["name"] for d in SSMGeometry(scale=scale).measurement_details]
    report = {n: {"target": target[i].item(), "mean": achieved[:, i].mean().item(),
                  "std": achieved[:, i].std().item(), "max_abs_err": (achieved[:, i] - target[i]).abs().max().item()}
              for i, n in enumerate(names)}
    (out / "measurements_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    inside = (~model.outside_box((comps - model.comp_mean) / model.comp_std)).float().mean().item()
    print(f"inside plausible box: {inside:.0%}")

    if cfg.save_meshes:
        _, _, faces = load_shape_model()
        verts = model.verts((comps - model.comp_mean) / model.comp_std).reshape(len(comps), -1, 3).numpy()
        for i, v in enumerate(verts):
            write_obj(out / f"limb_{i:03d}.obj", v, faces.numpy())
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
