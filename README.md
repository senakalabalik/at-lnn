# at-lnn: Recovering irreducible three-body interactions with Lagrangian neural networks

[![arXiv](https://img.shields.io/badge/arXiv-2610.13292-b31b1b.svg)](https://arxiv.org/abs/2610.13292)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23997752.svg)](https://doi.org/10.5281/zenodo.23997752)

Code, data and trained models for

> S. Kalabalık, Ş. Çağlar, A. U. Hamzaoğulları, A. Özakın,
> *Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks*, arXiv:2610.13292 (2026).

A fixed-kinetic Lagrangian neural network, $L_\theta = T(\dot q) - V_\theta(r_{12}, r_{23}, r_{31}, \cos\theta_1, \cos\theta_2, \cos\theta_3)$,
learns the Axilrod–Teller three-body potential from position–acceleration data. The network and training loop build on
[hessian_regularized_lnn](https://github.com/abdullahumuth/hessian_regularized_lnn) (Hamzaoğulları & Özakın); see `NOTICE`
for what was taken over and what was changed.

## Quick start

```bash
conda env create -f environment.yml && conda activate at-lnn
python scripts/verify_release.py --data        # check models, losses and datasets (~1 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/paper_figures.ipynb   # all figures -> figures/ (~4 min)
```

## Contents

| Folder | What is in it |
|---|---|
| `lnn/` | Model (`fixedkinetic_LNN.py`), physical systems (`physical_systems.py`: LJ + Axilrod–Teller, Newtonian), training loop and utilities; `release_io.py` registers the paper's models and loads them |
| `weights/` | The 8 trained models of the paper (plain `state_dict` + `config.json`), with their training configuration (`particle.json`) and training history |
| `data/` | The 5 training/validation dataset pairs (configurations; accelerations are computed from the analytic potential during training), `SHA256SUMS`, and the 4 initial states of Fig. 10 |
| `data_generation/` | The dataset generators and the exact parameters and seeds of every file in `data/` |
| `notebooks/` | `paper_figures.ipynb` (all data figures), `symbolic_regression.ipynb` (Sec. VI and Appendix D) |
| `scripts/` | `verify_release.py`, `train_model.py`, `evaluate_sr_fronts.py`, `rollout_worker.py`, `convert_checkpoints.py` |
| `results/` | Precomputed results the figures and Sec. VI read: N = 100 rollout-ensemble statistics (Fig. 9b) and the PySR Pareto fronts |

### Models

| `weights/` folder | System | Used in |
|---|---|---|
| `default_pure_AT_rd15` | pure Axilrod–Teller, dense/sparse sampler (reference model) | Figs. 3, 4, 5, 8, 9, 10; Sec. VI; Table IV |
| `default_pure_AT_loguniform` | pure AT, plain log-uniform sampler | Fig. 3 |
| `argonlike-liquid_pure_LJ` | pure Lennard-Jones, $r_{\min} = 0.9$ | Figs. 6, 11; Table V; Sec. V B |
| `argonlike-liquid_second_version` | LJ + AT, $\tilde\nu = 0.1$ | Fig. 11; Sec. V B |
| `argonlike-amplified_version` | LJ + AT, $\tilde\nu = 1.0$ | Fig. 11; Sec. V B |
| `pure_LJ_07_ground` | pure LJ, $r_{\min} = 0.7$ | Sec. V B 3 |
| `newtonian_threebody_r01_5` | Newtonian gravity, $(r, \cos\theta)$ features | Fig. 7 |
| `newtonian_threebody_r01_5_3DOF` | Newtonian gravity, distances only | Fig. 7 |

The datasets each model was trained and validated on are listed in `lnn/release_io.py` (`MODELS`).

## Installation

```bash
conda env create -f environment.yml
conda activate at-lnn
```
PySR (symbolic regression only) installs its Julia backend on first import; this needs internet access once.

## Reproducing the results

**1. Check the release** (≈1 min; add `--data` to also regenerate all datasets, ≈1 more min):
```bash
python scripts/verify_release.py --data
```
This checks that every model rebuilt from `weights/` reproduces the original trained checkpoint exactly, that every
model reproduces its logged validation loss on its validation set, and that every dataset regenerates from its seed.

**2. Figures** (≈5 min on a laptop CPU): run `notebooks/paper_figures.ipynb`. The figures are written to `figures/`
with the file names used in the manuscript, and the cells print the numbers quoted in the text.

**3. Symbolic regression (Sec. VI, Appendix D).**
```bash
python scripts/evaluate_sr_fronts.py        # seconds: recomputes the Sec. VI / App. D numbers from the saved fronts
```
`notebooks/symbolic_regression.ipynb` reruns the searches themselves (tens of minutes; needs PySR).

**4. Retraining a model** (the paper's runs used GPUs; a CPU works but is slow):
```bash
python scripts/train_model.py --model default_pure_AT_rd15 --out runs/rd15 --device cuda
```
This calls the original training loop (`lnn/train.py: train_from_scratch`) with the stored hyperparameters
(200 epochs, RAdam, learning rate $10^{-3}$, cosine schedule, batch 128, 3 × 256 GELU MLP; Appendix B of the paper).

**5. Datasets:** `python data_generation/generate_datasets.py --out <dir>` writes fresh copies.

## Reproducibility notes

* **Datasets.** All configurations (positions) regenerate bit-for-bit from the seeds. Velocities of the LJ, LJ + AT and
  Newtonian datasets do too. The two pure-AT datasets were made with a generator whose velocity draw was not seeded, so
  their velocities are not reproducible; velocities do not enter the learned potential, and the shipped files are the
  ones the models were trained on.
* **Trained models.** The original checkpoints were pickled model objects; they were converted to plain weights with
  `scripts/convert_checkpoints.py`, and `verify_release.py` confirms the conversion is exact. Retraining reproduces the
  models in distribution, not bit-for-bit (GPU nondeterminism, random streams).
* **Logged losses.** The training loop computes the mean absolute error over the full state derivative
  $(\dot q, \ddot q)$, whose velocity half is reproduced exactly, plus an eigenvalue penalty that is identically zero for
  a fixed kinetic term; logged losses are therefore half the mean absolute acceleration error.
* **Symbolic regression.** The Appendix D operator-set controls on the learned potential were run seeded and serially
  and reproduce exactly. The other searches were not seeded; a rerun returns an equivalent Pareto front. The fronts used
  in the paper are in `results/sr_outputs/`.
* **Fig. 9b** reads precomputed statistics of the N = 100 rollout ensemble (`results/fig9b_ensemble_summary.npz`,
  including the one trajectory shown in Fig. 9a(ii)).

## Data archive

Each tagged release of this repository, including the datasets and model weights, is archived at Zenodo:
[10.5281/zenodo.23997752](https://doi.org/10.5281/zenodo.23997752) (all versions). The version used in the paper is
v1.0.0, [10.5281/zenodo.23997753](https://doi.org/10.5281/zenodo.23997753).

## Citation

If you use this code or data, please cite:

```bibtex
@article{kalabalik2026recovering,
  title   = {Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks},
  author  = {Kalabal{\i}k, Sena and {\c{C}}a{\u{g}}lar, {\c{S}}{\"u}kr{\"u} and Hamzao{\u{g}}ullar{\i}, Abdullah Umut and {\"O}zak{\i}n, Arkada{\c{s}}},
  journal = {arXiv preprint arXiv:2610.13292},
  year    = {2026}
}
```

If you use the network or training code, please also cite the underlying stabilized LNN:

```bibtex
@article{hamzaogullari2026learning,
  title   = {Learning Relativistic Geodesics and Chaotic Dynamics via Stabilized Lagrangian Neural Networks},
  author  = {Hamzao{\u{g}}ullar{\i}, Abdullah Umut and {\"O}zak{\i}n, Arkada{\c{s}}},
  journal = {arXiv preprint arXiv:2601.12519},
  year    = {2026}
}
```

A machine-readable version is in `CITATION.cff`.

## License

Apache License 2.0 (`LICENSE`); third-party attribution in `NOTICE`.
