# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
Retrain any model of the paper with the original training code (lnn/train.py: train_from_scratch), using the
hyperparameters stored in weights/<name>/particle.json and the datasets in data/.

    python scripts/train_model.py --model default_pure_AT_rd15 --out runs/rd15 [--epochs 200] [--device cuda]

This is the code path the paper's runs used (main.py --mode train_from_scratch --model_name fixedkinetic
--lambda_val 100 with normalize=False). Training was done on GPUs (Kaggle); results are reproducible in
distribution, not bit-for-bit (different hardware and random streams).
Output: <out>/{model_state.pt, config.json, particle.json, history.csv, history_metadata.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lnn"))
import fixedkinetic_LNN                      # noqa: E402
from release_io import MODELS, DATA, WEIGHTS  # noqa: E402
from physical_systems import initialize_particle_system  # noqa: E402
from file_exports import load_particle, save_history     # noqa: E402
from train import train_from_scratch                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override total_epochs (paper: 200)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--subset", type=int, default=None,
                    help="TEST ONLY: train/validate on the first N configurations (quick smoke test)")
    a = ap.parse_args()
    meta = MODELS[a.model]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.epochs is not None and a.epochs < 10:
        ap.error("--epochs must be >= 10: train.py uses a cosine schedule with period total_epochs // 10")

    pj = WEIGHTS / a.model / "particle.json"
    cls = json.load(open(pj))["class_name"]
    particle = load_particle(initialize_particle_system(cls), str(pj))
    if a.epochs is not None:
        particle.train_hyperparams["total_epochs"] = a.epochs

    train_path, val_path = DATA / meta["train"], DATA / meta["val"]
    if a.subset:
        for src, name in ((train_path, "train_subset.pt"), (val_path, "val_subset.pt")):
            d = torch.load(src, weights_only=False)
            torch.save([d[0][: a.subset]], out / name)
        train_path, val_path = out / "train_subset.pt", out / "val_subset.pt"

    # train.py loads the network class from the module "<model_name>_LNN". The paper models use
    # "fixedkinetic" (lnn/fixedkinetic_LNN.py). For the distances-only control we register a module
    # "fixedkinetic_r_LNN" whose LNN builds the same network with features="r".
    model_name = "fixedkinetic"
    if meta["features"] == "r":
        import types
        mod = types.ModuleType("fixedkinetic_r_LNN")
        mod.LNN = lambda *args, **kw: fixedkinetic_LNN.LNN(*args, features="r", **kw)
        sys.modules["fixedkinetic_r_LNN"] = mod
        model_name = "fixedkinetic_r"

    device = torch.device(a.device)
    model, history = train_from_scratch(
        particle, str(out), 100.0, torch.float32, device,
        training_data_path=str(train_path), validation_data_path=str(val_path),
        normalize=False, experiment_name=model_name)

    torch.save(model.state_dict(), out / "model_state.pt")
    cfg = dict(input_size=particle.dof, num_layers=particle.model_params["num_layers"],
               hidden_dim=particle.model_params["hidden_dim"], activation="GELU", features=meta["features"])
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    json.dump(json.load(open(pj)), open(out / "particle.json", "w"), indent=2)
    save_history(history, str(out), overwrite=True)
    print("saved to", out)


if __name__ == "__main__":
    main()
