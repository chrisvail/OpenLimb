def load_model(ckpt):
    """Load a LimbCVAE or LimbMLP checkpoint (recognised by its `z_dim` hyper-parameter) on the CPU."""
    import torch

    hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    if "z_dim" in hp:
        from .lit_cvae import LimbCVAE as cls
    else:
        from .lit_mlp import LimbMLP as cls
    return cls.load_from_checkpoint(ckpt, map_location="cpu").eval()
