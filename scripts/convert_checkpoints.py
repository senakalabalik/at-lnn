# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
One-time conversion of the original pickled checkpoints into release weights (run by the authors; included for
transparency). It must run against the ORIGINAL source tree, because the pickles re-bind to it:

    python scripts/convert_checkpoints.py --orig-src <path>/lnn-advanced/src --orig-runs <path with the model folders>

For every model in lnn.release_io.MODELS it writes weights/<name>/:
    model_state.pt         state_dict (plain tensors)
    config.json            architecture needed to rebuild it
    particle.json, history.csv, history_metadata.json   copied from the original run folder
    reference_outputs.npz  outputs of the ORIGINAL model on 256 validation states, used by verify_release.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-src", required=True)
    ap.add_argument("--orig-runs", required=True)
    ap.add_argument("--n-ref", type=int, default=256)
    a = ap.parse_args()

    sys.path.insert(0, a.orig_src)        # the ORIGINAL code (pickles re-bind to it)
    import torch
    from fixedkinetic_LNN import LNN
    torch.serialization.add_safe_globals([LNN])
    from file_exports import load_model

    sys.path.insert(0, str(REPO / "lnn"))
    from release_io import MODELS  # only the registry (plain dict); model classes come from the original tree

    for name, meta in MODELS.items():
        src = Path(a.orig_runs) / name
        out = REPO / "weights" / name
        out.mkdir(parents=True, exist_ok=True)
        model = load_model(str(src / "model.pt"), torch.device("cpu"), torch.float32)
        model.eval()
        mlp = model.network.mlp
        pj = json.load(open(src / "particle.json"))

        cfg = dict(input_size=pj["dof"], num_layers=pj["model_params"]["num_layers"],
                   hidden_dim=pj["model_params"]["hidden_dim"], activation="GELU",
                   features=meta["features"], mlp_in_features=mlp[0].in_features)
        assert cfg["mlp_in_features"] == (3 if meta["features"] == "r" else 6), name
        torch.save(model.state_dict(), out / "model_state.pt")
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        for f in ("particle.json", "history.csv", "history_metadata.json"):
            shutil.copy(src / f, out / f)

        # reference outputs of the ORIGINAL model
        val = torch.load(Path(a.orig_src) / meta["val"], weights_only=False)
        states = (val[0] if isinstance(val, (list, tuple)) else val.tensors[0]).float()[: a.n_ref]
        q = states[:, :6].clone().requires_grad_(True)
        if meta["features"] == "r":     # the original forward() builds 6 features; this run used the distances-only branch
            p = q.view(-1, 3, 2)
            feats = torch.cat([torch.norm(p[:, 0] - p[:, 1], dim=1, keepdim=True),
                               torch.norm(p[:, 1] - p[:, 2], dim=1, keepdim=True),
                               torch.norm(p[:, 2] - p[:, 0], dim=1, keepdim=True)], 1)
            V = mlp(feats).squeeze(-1)
            el = np.full((len(states), 12), np.nan)
        else:
            V = -model.network(q)
            el = model(states).detach().numpy()          # full Euler-Lagrange forward, (qdot, qddot)
        acc = -torch.autograd.grad(V.sum(), q)[0]
        np.savez(out / "reference_outputs.npz", states=states.numpy(), V=V.detach().numpy(),
                 accel=acc.detach().numpy(), el=el)
        print(f"{name:34s} features={meta['features']:4s} -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
