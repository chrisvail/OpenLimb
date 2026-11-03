"""Synthetic stand-in for a table of clinical measurements, for evaluate_csv.py.

    python claude/evaluation/make_test_csv.py claude/outputs/test_meas.csv [n=60] [seed=0]

Rows are the exact measurements (mm) of OpenLimbTT-generator limbs, under aliased headers ("Circumference 1" ...) and a
`Patient` id column. Faults are injected so the cleaning rules can be seen working: 4 rows with an empty / null / NA /
non-numeric value, 1 negative value, and 1 row whose Circ 1 is 1.4x too large (outside the model's range, kept and flagged).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup_env  # noqa: E402

setup_env()

import torch  # noqa: E402

from data.geometry import ComponentSampler, SSMGeometry  # noqa: E402

HEADER = ["Patient", "Circumference 1", "Circumference 2", "Circumference 3", "Circumference 4",
          "Length 1", "Width 1", "Width 2"]


def main(path, n=60, seed=0):
    raw = ComponentSampler(True).raw(n, torch.Generator().manual_seed(seed))
    meas = SSMGeometry(scale=True, normalise=False).get_measures(raw).numpy()
    rows = [[f"P{i:03d}"] + [f"{v:.1f}" for v in m] for i, m in enumerate(meas)]
    for r, c, bad in ((5, 2, ""), (17, 4, "null"), (29, 6, "NA"), (41, 3, "n/a?"), (23, 5, "-12.0"), (50, 1, f"{1.4 * meas[50, 0]:.1f}")):
        rows[r][c] = bad
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([HEADER] + rows)
    print(f"wrote {n} rows to {path}")


if __name__ == "__main__":
    args = dict(a.split("=") for a in sys.argv[2:])
    main(sys.argv[1], int(args.get("n", 60)), int(args.get("seed", 0)))
