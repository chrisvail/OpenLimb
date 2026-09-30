import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

CLAUDE_DIR = Path(__file__).resolve().parents[1]


def register_resolvers():
    if not OmegaConf.has_resolver("claude_dir"):
        OmegaConf.register_new_resolver("claude_dir", lambda: CLAUDE_DIR.as_posix())


def surrogate_path(scale, override=None):
    return Path(override) if override else CLAUDE_DIR / "data" / f"surrogate_{'scaled' if scale else 'unscaled'}.pt"


def setup_env():
    """Make `openlimb_cvae` and the repo root (SSM_Driver, measure_limbs) importable."""
    for p in (CLAUDE_DIR, CLAUDE_DIR.parent):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    torch.set_float32_matmul_precision("high")
