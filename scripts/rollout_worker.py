# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
Learned-model trajectory rollout with the adaptive torchode solver (lnn/ode_solve.py), run as a separate process
so that notebooks importing PySR (Julia) and torch-compiled solvers do not interfere.

    python scripts/rollout_worker.py <input.npz> <output.npz>
input.npz:  run_dir (str, a folder in weights/), x0 (float32 [B,12]), t (float32 [T]), chunk (int)
output.npz: traj (float64 [B,T,12])
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["MPLBACKEND"] = "Agg"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lnn"))
from release_io import load_model, load_system  # noqa: E402
from ode_solve import solve_with_lnn_torchode    # noqa: E402


def main(in_path, out_path):
    d = np.load(in_path, allow_pickle=True)
    name = Path(str(d["run_dir"])).name
    x0 = torch.tensor(d["x0"], dtype=torch.float32)
    t = torch.tensor(d["t"], dtype=torch.float32)
    chunk = int(d["chunk"])
    device, dtype = torch.device("cpu"), torch.float32
    model = load_model(name, device, dtype)
    particle = load_system(name)
    particle.eps_LJ = 0.0     # as in the original worker: pure-AT rollouts
    traj = solve_with_lnn_torchode(particle, model, x0, (float(t[0]), float(t[-1])), t,
                                   dtype=torch.float64, device=device, max_batch_size=chunk)
    np.savez(out_path, traj=np.asarray(traj))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
