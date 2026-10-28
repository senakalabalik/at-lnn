# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
Regenerate the datasets of the paper (configurations only; accelerations are computed from the analytic potential
during training, see lnn/dataset_creation.py).

    python data_generation/generate_datasets.py --check          # regenerate in memory and compare with data/
    python data_generation/generate_datasets.py --out data_new   # write fresh copies

Every file is a list [states] with states a float64 tensor of shape (N, 12):
(x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3). All runs use a box of [-10, 10]^2, seed 2003 for the
training set and 2006 for the validation set, and the dense/sparse log-uniform mixture of Appendix A.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generator_v1_at as G1  # noqa: E402
import generator_v2 as G2     # noqa: E402

BOX = (-10.0, 10.0)
# file stem -> (generator, r_min, r_dense, r_max, p_dense)
DATASETS = {
    "threebody_{n}_maxr_5_minr06_d90_rd15":   (G1, 0.6, 1.5, 5.0, 0.9),   # pure AT, dense/sparse (Table II)
    "threebody_{n}_maxr_5_minr06_loguniform": (G1, 0.6, 0.6, 5.0, 0.0),   # pure AT, plain log-uniform
    "argonlike_liquid_{n}_maxr_3_minr09":     (G2, 0.9, 1.5, 3.0, 0.5),   # LJ and LJ+AT (Table II)
    "argonlike_liquid_{n}_maxr_3_minr07":     (G2, 0.7, 1.5, 3.0, 0.5),   # LJ, r_min = 0.7 (Sec. V B 3)
    "newtonian_01_5":                         (G2, 0.1, 1.5, 5.0, 0.8),   # Newtonian control
}
SPLITS = {"train": (120000, 2003), "val": (24000, 2006)}


def filename(stem, split):
    if stem.startswith("newtonian"):
        return f"{stem}_dataset.pt" if split == "train" else f"{stem}_validation_dataset.pt"
    n = "120k" if split == "train" else "24k"
    return stem.format(n=n) + ("_dataset.pt" if split == "train" else "_validation_dataset.pt")


def generate(stem, split):
    G, r_min, r_dense, r_max, p_dense = DATASETS[stem]
    n, seed = SPLITS[split]
    kw = dict(n_samples=n, bounds=BOX, r_min=r_min, r_dense=r_dense, r_max=r_max, p_dense=p_dense, seed=seed,
              verbose=False)
    if G is G2:
        kw["T_star"] = 1.0
    states, _ = G.generate_dataset_r_based_positions_maxwell_vel(**kw)
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare regenerated files with data/")
    ap.add_argument("--out", default=None, help="directory to write regenerated files to")
    a = ap.parse_args()
    data = HERE.parent / "data"
    ok = True
    for stem, spec in DATASETS.items():
        for split in SPLITS:
            f = filename(stem, split)
            states = generate(stem, split)
            if a.out:
                Path(a.out).mkdir(parents=True, exist_ok=True)
                torch.save([torch.tensor(states, dtype=torch.float64)], Path(a.out) / f)
            if a.check:
                ref = torch.load(data / f, weights_only=False)[0].numpy()
                dpos = float(np.abs(states[:, :6] - ref[:, :6]).max())
                dvel = float(np.abs(states[:, 6:] - ref[:, 6:]).max())
                exact_vel = spec[0] is G2
                good = states.shape == ref.shape and dpos == 0.0 and (dvel == 0.0 or not exact_vel)
                ok &= good
                note = "" if exact_vel else "  (velocities unseeded in generator v1: not compared)"
                print(f"{'OK  ' if good else 'FAIL'} {f:62s} positions max|diff| = {dpos:.1e}"
                      + (f", velocities max|diff| = {dvel:.1e}" if exact_vel else note))
    if a.check:
        print("ALL DATASETS REPRODUCED" if ok else "SOME DATASETS DIFFER")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
