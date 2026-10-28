# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
Registry of the trained models used in the paper, and loaders that rebuild them from plain weights.

The original checkpoints were pickled whole objects (torch.save(model)), which re-bind to whatever the model
source file contains at load time. Here every model is stored as a state_dict + config.json and rebuilt from
lnn/fixedkinetic_LNN.py. scripts/verify_release.py checks that the rebuilt models reproduce the original
checkpoints' outputs (stored in weights/<name>/reference_outputs.npz) exactly.
"""
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
LNN_DIR = REPO / "lnn"
if str(LNN_DIR) not in sys.path:
    sys.path.insert(0, str(LNN_DIR))

from fixedkinetic_LNN import LNN                       # noqa: E402
from physical_systems import initialize_particle_system  # noqa: E402
from file_exports import load_particle                  # noqa: E402

# name -> what it is, where it is used, and the (verified) datasets it was trained / validated on
MODELS = {
    "default_pure_AT_rd15": dict(
        system="three_body", features="rcos", role="pure AT, dense/sparse sampler (reference model)",
        paper="Figs. 3, 4, 5, 8, 9, 10; Sec. VI; Table IV",
        train="threebody_120k_maxr_5_minr06_d90_rd15_dataset.pt",
        val="threebody_24k_maxr_5_minr06_d90_rd15_validation_dataset.pt"),
    "default_pure_AT_loguniform": dict(
        system="three_body", features="rcos", role="pure AT, plain log-uniform sampler",
        paper="Fig. 3",
        train="threebody_120k_maxr_5_minr06_loguniform_dataset.pt",
        val="threebody_24k_maxr_5_minr06_loguniform_validation_dataset.pt"),
    "argonlike-liquid_pure_LJ": dict(
        system="three_body", features="rcos", role="pure LJ (pairwise model / LJ control), r_min = 0.9",
        paper="Figs. 6, 11; Table V; Sec. V B",
        train="argonlike_liquid_120k_maxr_3_minr09_dataset.pt",
        val="argonlike_liquid_24k_maxr_3_minr09_validation_dataset.pt"),
    "argonlike-liquid_second_version": dict(
        system="three_body", features="rcos", role="LJ + AT, physical coupling nu = 0.1",
        paper="Fig. 11; Sec. V B",
        train="argonlike_liquid_120k_maxr_3_minr09_dataset.pt",
        val="argonlike_liquid_24k_maxr_3_minr09_validation_dataset.pt"),
    "argonlike-amplified_version": dict(
        system="three_body", features="rcos", role="LJ + AT, amplified coupling nu = 1.0",
        paper="Fig. 11; Sec. V B",
        train="argonlike_liquid_120k_maxr_3_minr09_dataset.pt",
        val="argonlike_liquid_24k_maxr_3_minr09_validation_dataset.pt"),
    "pure_LJ_07_ground": dict(
        system="three_body", features="rcos", role="pure LJ, r_min = 0.7 (cutoff sensitivity)",
        paper="Sec. V B 3",
        train="argonlike_liquid_120k_maxr_3_minr07_dataset.pt",
        val="argonlike_liquid_24k_maxr_3_minr07_validation_dataset.pt"),
    "newtonian_threebody_r01_5": dict(
        system="newt_three_body", features="rcos", role="Newtonian gravity control, (r, cos) features",
        paper="Fig. 7",
        train="newtonian_01_5_dataset.pt", val="newtonian_01_5_validation_dataset.pt"),
    "newtonian_threebody_r01_5_3DOF": dict(
        system="newt_three_body", features="r", role="Newtonian gravity control, distances only",
        paper="Fig. 7",
        train="newtonian_01_5_dataset.pt", val="newtonian_01_5_validation_dataset.pt"),
}

WEIGHTS = REPO / "weights"
DATA = REPO / "data"


def model_config(name):
    return json.load(open(WEIGHTS / name / "config.json"))


def load_model(name, device="cpu", dtype=torch.float32):
    """Rebuild a paper model from weights/<name>/{config.json, model_state.pt}."""
    cfg = model_config(name)
    act = {"GELU": torch.nn.GELU()}[cfg["activation"]]
    model = LNN(cfg["input_size"], cfg["num_layers"], act, cfg["hidden_dim"], torch.device(device), dtype,
                features=cfg["features"])
    state = torch.load(WEIGHTS / name / "model_state.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_model_path(path, device="cpu", dtype=torch.float32):
    """Drop-in for the original file_exports.load_model(f"<run_dir>/model.pt", device, dtype) calls in the
    notebooks: <run_dir> is a folder in weights/, and the model is rebuilt from its plain weights."""
    if isinstance(device, str):
        device = torch.device(device)
    return load_model(Path(path).parent.name, device, dtype)


def load_system(name):
    """The analytic system (with the physical parameters) the model was trained on."""
    pj = WEIGHTS / name / "particle.json"
    cls = json.load(open(pj))["class_name"]
    return load_particle(initialize_particle_system(cls), str(pj))


def load_states(filename):
    """A dataset file: list [states] with states of shape (N, 12) = (x1,y1,x2,y2,x3,y3, vx1,...,vy3)."""
    d = torch.load(DATA / filename, weights_only=False)
    return d[0] if isinstance(d, (list, tuple)) else d.tensors[0]
