"""Regenerate everything in ./data_components without docker.

    python populate_data_components.py [--version 2023-06] [--openlimbtt PATH] [--out ./data_components]

--version picks the OpenLimbTT release (2022-10, 2023-06, 2025-07). 2023-06 is what GenerateRandomLimbs.py /
the docker image use, and the only one the rest of this repository was built for (10 modes + scale).

Each release maps uniformly sampled skin-only mode scores to full-model scores with a linear regression stored in
LR.pkl. The coefficients are read from an .npz (keys `coef` (n_out, n_in), `intercept` (n_out,)):
    2023-06: claude/data/lr_map.npz (already in the repository)
    others:  claude/data/lr_map_<version>.npz, or --lr_map PATH
--trust_pickle extracts them from that release's LR.pkl instead and caches them to claude/data/lr_map_<version>.npz.
Unpickling runs code from the file, so only pass it if you trust the OpenLimbTT repository.

Files written (and the notebook each one originally came from):
    mesh.pt                             dict     the skin-only mesh, everything indexed by the same Vs skin verts:
                                                 mean_verts (Vs, 3)        OpenLimbTT mean limb shape
                                                 vert_components (M, 3Vs)  OpenLimbTT modes
                                                   (mean and modes were dumped from GenerateRandomLimbs.py inside
                                                    the docker image, then restricted to the skin verts)
                                                 faces (Fs, 3)             skin faces (measure_limbs.ipynb)
                                                 selected_verts (8,)       nearest vert to each point in verts.txt
                                                                           (SSM_Driver.ipynb)
    component_range.pt                  dict     skin_min, skin_max (S,): sampling box of the skin-only modes;
                                                 scale_min, scale_max (): tibia-length range of the scale factor
    lr_map.pt                           dict     coef (M, S), intercept (M,): skin -> full-model linear regression,
                                                 full = skin @ coef.T + intercept
    scaled_measurement_transforms.pt    (2, 7)   component_exploration.ipynb  - [mean; std] of their measurements
    unscaled_measurement_transforms.pt  (2, 7)
    ssm_version.txt                              which release the files above came from

The OpenLimbTT repository (the one the dockerfile clones) is cloned to --openlimbtt if it is not there.
The boneless mesh measure_limbs.ipynb used (meshes/limb_00000000_boneless.stl) no longer exists; the release's
mean skin mesh is used instead - its vertices are exactly the skin shell of the mean limb mesh.
verts.txt was picked in 2023-06 coordinates; other releases are aligned/cut differently, so check the landmarks.
"""
import argparse
import subprocess
from functools import partial
from pathlib import Path

import igl
import numpy as np
import torch
from torch import nn

import measure_limbs

ROOT = Path(__file__).resolve().parent
OPENLIMBTT_URL = "https://github.com/abel-research/OpenLimbTT.git"
CLAUDE_DATA = ROOT / "claude" / "data"

# Files and constants per release. Skin-mode ranges and tibia-length (scale) ranges are copied from each release's
# "Generate Random Synthetic Model.py" (2023-06 from GenerateRandomLimbs.py, which uses the same values).
VERSIONS = {
    "2022-10": dict(
        limb="Mean_Limb_Shape.stl", skin="Mean_Skin_Shape.stl", modes="Modes.npy", modes_transposed=True,
        minima=[-0.645328929043159, -0.31931947079786, -0.531048773913778, -0.46741046041567, -0.507896845719971,
                -0.549559768779508, -0.638809195801619, -0.3986621998533641, -0.389913947175072, -0.587790496589690],
        maxima=[0.4273670099841691, 0.72298577328671, 0.56827592399203, 0.490646382465809, 0.588489283542763,
                0.534364309153956, 0.527485323330151, 0.41065896977363, 0.599926073881045, 0.622085661222956],
        scale=(342.8, 439.8),
    ),
    "2023-06": dict(
        limb="Mean_Limb_Shape.stl", skin="Mean_Skin_Shape.stl", modes="Components.npy", modes_transposed=False,
        minima=[-14.80692194113721, -5.37869110537926, -3.996990835549319, -4.05537984190567, -2.403525754650053,
                -1.854646894835898, -2.13215613028021, -1.197319576810893, -0.798154129842514, -0.7353096205329],
        maxima=[25.4537448635005, 10.06971435469401, 4.12168595787859, 4.24308399616669, 4.73074159745632,
                2.390247129446665, 2.19420856255569, 1.221847275562656, 0.802807003549, 0.95283973472981],
        scale=(342.8, 439.8),
        lr_map=CLAUDE_DATA / "lr_map.npz",
    ),
    "2025-07": dict(
        limb="Mean-Limb.stl", skin="Mean-Skin.stl", modes="Components16.npy", modes_transposed=False,
        # Mode 6's minimum is written as -4/3 in the release script (possibly meant -4.3); kept as published.
        minima=[-17.04, -10.38, -8.41, -6.42, -4.73, -4 / 3, -4.01, -2.55, -2.26, -1.97, -1.54, -0.96, -1.46, -0.96,
                -1.12, -0.95, -0.9, -0.7, -0.66, -0.49, -0.62, -0.53, -0.61, -0.45, -0.37, -0.46, -0.39, -0.32,
                -0.28, -0.32, -0.22, -0.22],
        maxima=[24.25, 16.42, 10.04, 11.66, 5.21, 4.38, 4.01, 2.68, 2.5, 2.49, 2.2, 2.48, 2.49, 1.66, 1.65, 1.14,
                0.97, 0.83, 1.23, 0.96, 0.73, 0.69, 0.75, 0.54, 0.5, 0.5, 0.46, 0.42, 0.35, 0.32, 0.26, 0.37],
        scale=(313.05, 466.34),
    ),
}


# ---------------------------------------------------------------------------------
# SSM (what GenerateRandomLimbs.py loads inside docker)
# ---------------------------------------------------------------------------------
def ensure_openlimbtt(path, version):
    ssm = path / f"version-{version}"
    if not path.exists():
        print(f"Cloning {OPENLIMBTT_URL} -> {path}")
        subprocess.run(["git", "clone", "--depth", "1", OPENLIMBTT_URL, str(path)], check=True)
    if not ssm.exists():
        raise FileNotFoundError(f"{ssm} not found - is {path} an OpenLimbTT clone?")
    return ssm


def load_lr_map(version, ssm, lr_map=None, trust_pickle=False):
    """(coef (n_out, n_in), intercept (n_out,)) of the release's skin -> full-model regression."""
    path = lr_map or VERSIONS[version].get("lr_map", CLAUDE_DATA / f"lr_map_{version}.npz")
    if not Path(path).exists():
        if not trust_pickle:
            raise FileNotFoundError(
                f"No regression coefficients at {path}. Pass --lr_map PATH, or --trust_pickle to extract them "
                f"from {ssm / 'LR.pkl'} (unpickling executes code from that file).")
        import pickle

        with open(ssm / "LR.pkl", "rb") as f:
            model = pickle.load(f)
        np.savez(path, coef=np.asarray(model.coef_), intercept=np.asarray(model.intercept_))
        print(f"cached LR.pkl coefficients to {path}")
    lr = np.load(path)
    return lr["coef"], lr["intercept"]


def generate_components(n_samples, scale, version, coef, intercept, seed=42):
    """GenerateRandomLimbs.py without docker/ampscan: skin modes ~ U(box), full modes = LR(skin), scale ~ U(range)."""
    v = VERSIONS[version]
    minima, maxima = np.array(v["minima"]), np.array(v["maxima"])
    scale_min, scale_max = v["scale"]
    np.random.seed(seed)
    component = np.random.rand(n_samples, len(minima)) * (maxima - minima) + minima
    newcomponent = component @ coef.T + intercept
    if scale:
        scale_factor = np.random.rand(n_samples, 1) * (scale_max - scale_min) + scale_min
    else:
        scale_factor = np.ones((n_samples, 1))
    return torch.tensor(np.concatenate((newcomponent, scale_factor), axis=-1))


# ---------------------------------------------------------------------------------
# measure_limbs.ipynb: "Removing Bones from file"
# ---------------------------------------------------------------------------------
def boneless_mapping(verts_bones, verts_boneless, face2vert_boneless):
    verts_bones, verts_boneless = np.round(verts_bones, decimals=8), np.round(verts_boneless, decimals=8)

    _, boneless_idxs = np.unique(verts_boneless, return_index=True, axis=0)
    _, bones_idxs = np.unique(verts_bones, return_index=True, axis=0)

    boneless_set = {tuple(x) for x in verts_boneless[boneless_idxs]}
    mapping = []
    face_mapping = {}
    for idx in bones_idxs:
        if tuple(verts_bones[idx]) in boneless_set:
            face_mapping[tuple(verts_bones[idx])] = len(mapping)
            mapping.append(idx)

    face2vert = np.array([[face_mapping[tuple(verts_boneless[v])] for v in face] for face in face2vert_boneless])
    return torch.tensor(np.array(mapping)), torch.tensor(face2vert)


# ---------------------------------------------------------------------------------
# SSM_Driver.ipynb: selected verts
# ---------------------------------------------------------------------------------
def select_verts(mapped_verts, path=ROOT / "verts.txt"):
    selected = []
    with open(path) as f:
        for line in f:
            if line.strip() and line[0] != "#":
                selected.append([float(x) for x in line.strip().split(", ")])
    selected = torch.tensor(selected, dtype=mapped_verts.dtype)
    return torch.argmin(torch.linalg.norm(mapped_verts[None] - selected[:, None], dim=-1), dim=-1)


# ---------------------------------------------------------------------------------
# component_exploration.ipynb / SSM_Driver.ipynb: measurements
# ---------------------------------------------------------------------------------
class Measurements(nn.Module):
    def __init__(self, edge2vert, face2edge, details):
        super().__init__()
        self.details = details
        self.measures = []
        for detail in details:
            match detail["type"]:
                case "width":
                    measure = partial(measure_limbs.measure_width, edge2vert=edge2vert,
                                      plane_point=detail["plane_point"], plane_normal=detail["plane_normal"],
                                      plane_direction=detail["plane_direction"])
                case "length":
                    measure = partial(measure_limbs.measure_length,
                                      v1=detail["v1"], v2=detail["v2"], direction=detail["direction"])
                case "circumference":
                    measure = partial(measure_limbs.measure_planar_circumference, edge2vert=edge2vert,
                                      face2edge=face2edge, plane_point=detail["plane_point"],
                                      plane_normal=detail["plane_normal"])
            self.measures.append(measure)

    def forward(self, x):
        return torch.stack([measure(x) for measure in self.measures], dim=-1)


def get_measurement_details(vert_idxs, dtype):
    z = torch.tensor([[0, 0, 1]], dtype=dtype)
    x = torch.tensor([[1, 0, 0]], dtype=dtype)
    return (
        {"type": "circumference", "plane_point": vert_idxs[4], "plane_normal": z, "name": "Circumference 1"},
        {"type": "circumference", "plane_point": vert_idxs[5], "plane_normal": z, "name": "Circumference 2"},
        {"type": "circumference", "plane_point": vert_idxs[6], "plane_normal": z, "name": "Circumference 3"},
        {"type": "circumference", "plane_point": vert_idxs[7], "plane_normal": z, "name": "Circumference 4"},
        {"type": "length", "v1": vert_idxs[0], "v2": vert_idxs[1], "direction": z, "name": "Length 1"},
        {"type": "width", "plane_point": vert_idxs[2], "plane_normal": z, "plane_direction": x, "name": "Width 1"},
        {"type": "width", "plane_point": vert_idxs[3], "plane_normal": z, "plane_direction": x, "name": "Width 2"},
    )


def get_verts(components, mean_verts, raw_components):
    """LegMeasurementDataset.get_verts as a matmul (the original broadcast needs ~25 GB for 1024 limbs)."""
    components, scale = components[:, :-1], components[:, -1:]
    verts = mean_verts[None] + (components @ raw_components).reshape(-1, *mean_verts.shape)
    return verts * scale[..., None]


def measure_all(components, measure, mean_verts, raw_components, chunk=128):
    return torch.cat([measure(get_verts(c, mean_verts, raw_components)) for c in components.split(chunk)])


def mean_std(x):
    return torch.vstack([torch.mean(x, dim=0, keepdim=True), torch.std(x, dim=0, keepdim=True)])


def summarize_tensor(tensor, name):
    print(f"Summary statistics for {name}:")
    for i, (m, s, lo, hi) in enumerate(zip(tensor.mean(0), tensor.std(0), tensor.min(0).values, tensor.max(0).values)):
        print(f"Component {i}: mean={m:.4f}, std={s:.4f}, min={lo:.4f}, max={hi:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", choices=sorted(VERSIONS), default="2023-06", help="OpenLimbTT release")
    parser.add_argument("--openlimbtt", type=Path, default=ROOT / "OpenLimbTT",
                        help="Path to a clone of OpenLimbTT (cloned there if missing)")
    parser.add_argument("--out", type=Path, default=ROOT / "data_components")
    parser.add_argument("--lr_map", type=Path, default=None,
                        help="npz with the release's regression `coef`/`intercept` (see module docstring)")
    parser.add_argument("--trust_pickle", action="store_true",
                        help="Extract the regression from the release's LR.pkl (executes code from the file)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Limbs used for the transforms")
    parser.add_argument("--seed", type=int, default=42, help="GenerateRandomLimbs.py's default seed")
    args = parser.parse_args()

    dtype = torch.float64
    version = VERSIONS[args.version]
    ssm = ensure_openlimbtt(args.openlimbtt, args.version)
    coef, intercept = load_lr_map(args.version, ssm, args.lr_map, args.trust_pickle)
    if coef.shape[1] != len(version["minima"]):
        raise ValueError(f"regression takes {coef.shape[1]} skin modes, {args.version} samples {len(version['minima'])}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.version != "2023-06":
        print(f"WARNING: {args.version} is not the release the rest of this repository was built for (2023-06)")

    def save(obj, name):
        torch.save(obj, args.out / name)
        print(f"saved {name:38s} {tuple(obj.shape)}  {obj.dtype}")

    def save_dict(obj, name):
        torch.save(obj, args.out / name)
        print(f"saved {name:38s} " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in obj.items()))

    # SSM: igl (like ampscan with unify=False) keeps the raw STL order the modes are defined in
    verts_bones, _ = igl.read_triangle_mesh(str(ssm / version["limb"]))
    verts_boneless, face2vert_boneless = igl.read_triangle_mesh(str(ssm / version["skin"]))
    raw_mean_verts = torch.tensor(verts_bones, dtype=dtype)
    raw_components = torch.tensor(np.load(ssm / version["modes"]), dtype=dtype)
    if version["modes_transposed"]:
        raw_components = raw_components.T.contiguous()
    assert raw_components.shape[1] == raw_mean_verts.numel()
    if coef.shape[0] != raw_components.shape[0]:
        raise ValueError(f"regression predicts {coef.shape[0]} modes, {version['modes']} has {raw_components.shape[0]}")
    (args.out / "ssm_version.txt").write_text(f"{args.version}\n")

    # Generator sampling box and skin -> full-model linear regression (full = skin @ coef.T + intercept)
    save_dict({
        "skin_min": torch.tensor(version["minima"], dtype=dtype),
        "skin_max": torch.tensor(version["maxima"], dtype=dtype),
        "scale_min": torch.tensor(version["scale"][0], dtype=dtype),
        "scale_max": torch.tensor(version["scale"][1], dtype=dtype),
    }, "component_range.pt")
    save_dict({"coef": torch.tensor(coef, dtype=dtype), "intercept": torch.tensor(intercept, dtype=dtype)},
              "lr_map.pt")

    # measure_limbs.ipynb: restrict the mean and modes to the skin verts once, here, so nothing downstream maps
    vert_mapping, faces = boneless_mapping(verts_bones, verts_boneless, face2vert_boneless)
    mean_verts = raw_mean_verts[vert_mapping]
    raw_components = raw_components.reshape(-1, len(raw_mean_verts), 3)[:, vert_mapping].reshape(
        raw_components.shape[0], -1)

    # SSM_Driver.ipynb (originally snapped onto meshes/limb_00000.stl; the mean shape is used here)
    vert_idxs = select_verts(mean_verts)
    save_dict({"mean_verts": mean_verts, "vert_components": raw_components, "faces": faces,
               "selected_verts": vert_idxs}, "mesh.pt")

    # component_exploration.ipynb: both runs used the generator's default seed, so share shape components
    unscaled = generate_components(args.batch_size, False, args.version, coef, intercept, seed=args.seed)
    scaled = generate_components(args.batch_size, True, args.version, coef, intercept, seed=args.seed)

    edge2vert, face2edge, _ = igl.edge_topology(mean_verts.numpy(), faces.numpy())
    measure = Measurements(torch.from_numpy(edge2vert), torch.from_numpy(face2edge),
                           get_measurement_details(vert_idxs, dtype))
    with torch.no_grad():
        scaled_measures = measure_all(scaled, measure, mean_verts, raw_components)
        unscaled_measures = measure_all(unscaled, measure, mean_verts, raw_components)
    summarize_tensor(unscaled_measures, "Unscaled Measures")
    summarize_tensor(scaled_measures, "Scaled Measures")
    save(mean_std(scaled_measures), "scaled_measurement_transforms.pt")
    save(mean_std(unscaled_measures), "unscaled_measurement_transforms.pt")


if __name__ == "__main__":
    main()
