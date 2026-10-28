# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
End-to-end consistency check of the release (uses only the code in this repository):

  1. every model rebuilt from weights/<name>/model_state.pt reproduces the ORIGINAL checkpoint's outputs
     (potential, forces, and the full Euler-Lagrange forward) stored in reference_outputs.npz;
  2. every model reproduces its logged best validation loss on its validation set, which also pins down
     the dataset each model was trained with;
  3. (--data) every dataset regenerates from its seed (see data_generation/generate_datasets.py).

    python scripts/verify_release.py [--data]
"""
import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
print = functools.partial(print, flush=True)   # keep order with the dataset subprocess
sys.path.insert(0, str(REPO / "lnn"))
from release_io import MODELS, WEIGHTS, load_model, load_system, load_states  # noqa: E402


def potential_and_forces(model, states):
    q = states[:, :6].clone().requires_grad_(True)
    V = -model.network(q)                    # q-only input: the kinetic term vanishes, L = -V
    a = -torch.autograd.grad(V.sum(), q)[0]
    return V.detach().numpy(), a.detach().numpy()


def logged_val_loss(model, system, states, batch=128):
    """The quantity train.py logs as validation loss: mean over batches of mean|pred - target| on (qdot, qddot),
    plus the eigenvalue penalty (identically zero for the fixed kinetic term)."""
    q = states[:, :6].clone().requires_grad_(True)
    qd = states[:, 6:].clone().requires_grad_(True)
    target = torch.cat([states[:, 6:], system.solve_acceleration(q, qd).detach()], 1)
    losses = [(model(states[i:i + batch]).detach() - target[i:i + batch]).abs().mean().item()
              for i in range(0, len(states), batch)]
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="also regenerate and compare all datasets (~40 s)")
    a = ap.parse_args()
    ok = True
    print("1) rebuilt models vs. original checkpoints (max |difference|)")
    for name in MODELS:
        model = load_model(name)
        ref = np.load(WEIGHTS / name / "reference_outputs.npz")
        states = torch.as_tensor(ref["states"])
        V, acc = potential_and_forces(model, states)
        dV, dA = np.abs(V - ref["V"]).max(), np.abs(acc - ref["accel"]).max()
        dEL = np.nan if np.isnan(ref["el"]).all() else np.abs(model(states).detach().numpy() - ref["el"]).max()
        good = dV <= 1e-5 * max(1.0, np.abs(ref["V"]).max()) and dA <= 1e-5 * max(1.0, np.abs(ref["accel"]).max()) \
            and (np.isnan(dEL) or dEL <= 1e-5 * max(1.0, np.abs(ref["el"]).max()))
        ok &= bool(good)
        print(f"   {'OK  ' if good else 'FAIL'} {name:34s} V {dV:.1e}  forces {dA:.1e}  EL forward "
              + ("n/a (distances-only model)" if np.isnan(dEL) else f"{dEL:.1e}"))

    print("2) logged best validation loss reproduced on the model's validation set")
    for name, meta in MODELS.items():
        model, system = load_model(name), load_system(name)
        logged = json.load(open(WEIGHTS / name / "history_metadata.json"))["best_loss"]
        L = logged_val_loss(model, system, load_states(meta["val"]).float())
        good = abs(L - logged) <= 1e-4 * logged
        ok &= good
        print(f"   {'OK  ' if good else 'FAIL'} {name:34s} recomputed {L:.6f}   logged {logged:.6f}   ({meta['val']})")

    if a.data:
        print("3) datasets regenerated from their seeds", flush=True)
        r = subprocess.run([sys.executable, str(REPO / "data_generation" / "generate_datasets.py"), "--check"])
        ok &= r.returncode == 0

    print("\nRELEASE VERIFIED" if ok else "\nVERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
