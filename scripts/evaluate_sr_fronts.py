# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
"""
Recompute the symbolic-regression numbers of Sec. VI and Appendix D from the saved PySR Pareto fronts
(results/sr_outputs/*.csv), without running PySR. Selection follows PySR's model_selection="best": the highest
score among expressions whose loss is within 1.5x of the lowest loss. Evaluation sets are those of the original
notebook (notebooks/symbolic_regression.ipynb).

    python scripts/evaluate_sr_fronts.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lnn"))
from release_io import load_model, load_system, load_states  # noqa: E402

FRONTS = REPO / "results" / "sr_outputs"
SYMS = sp.symbols("r12 r23 r31 c1 c2 c3")
LOCALS = {"square": lambda x: x**2, "cube": lambda x: x**3, "inv": lambda x: 1 / x,
          **{str(s): s for s in SYMS}}

model = load_model("default_pure_AT_rd15")
particle = load_system("default_pure_AT_rd15")


def features_from_q6(q6):            # as in the notebook (mirrors CombinedNetwork.forward, incl. the 1e-6)
    p = np.asarray(q6, float).reshape(3, 2)
    r12 = np.linalg.norm(p[0] - p[1]); r23 = np.linalg.norm(p[1] - p[2]); r31 = np.linalg.norm(p[2] - p[0])
    v12 = p[1] - p[0]; v13 = p[2] - p[0]; v23 = p[2] - p[1]
    return np.array([r12, r23, r31, (v12 @ v13) / (r12 * r31 + 1e-6), (-v12 @ v23) / (r12 * r23 + 1e-6),
                     ((-v13) @ (-v23)) / (r31 * r23 + 1e-6)])


def learned_V(Q):
    with torch.no_grad():
        return -model.lagrangian(torch.as_tensor(Q, dtype=torch.float32)).numpy().astype(float)


def true_V(Q):
    return np.array([float(particle.potential(torch.as_tensor(q, dtype=torch.float32), torch.zeros(6)))
                     for q in Q])


def core_set(n, seed, box=1.4):      # notebook cell "_corel3": uniform triangles, all sides in [0.6, 2.0]
    rng = np.random.default_rng(seed); Q = []
    while len(Q) < n:
        p = rng.uniform(-box, box, size=(3, 2))
        r = [np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[2] - p[0])]
        if min(r) < 0.6 or max(r) > 2.0:
            continue
        Q.append((p - p.mean(0)).reshape(-1).astype(np.float32))
    return np.array(Q)


def heldout_valset():                # notebook cell 18: Dq[perm[6000:7500]], seed 0, rd15 validation file
    Dq = load_states("threebody_24k_maxr_5_minr06_d90_rd15_validation_dataset.pt").numpy()[:, :6]
    return Dq[np.random.default_rng(0).permutation(len(Dq))[6000:7500]].astype(np.float32)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate(expr_str, F):
    e = sp.sympify(expr_str, locals=LOCALS)
    f = sp.lambdify(SYMS, e, "numpy")
    return np.broadcast_to(f(*F.T), (len(F),)).astype(float), e


def pysr_best(df):
    ok = df[df["loss"] <= 1.5 * df["loss"].min()]
    return ok.loc[ok["score"].idxmax()]


def report(title, csv, Q, rows=None):
    df = pd.read_csv(FRONTS / csv)
    F = np.array([features_from_q6(q) for q in Q])
    VL, VT = learned_V(Q), true_V(Q)
    picks = [("best", pysr_best(df))] + [(f"complexity {c}", df[df.complexity == c].iloc[0]) for c in (rows or [])]
    print(f"\n== {title}  [{csv}]")
    for lab, row in picks:
        p, e = evaluate(row["sympy_format"], F)
        print(f"   {lab:14s} (complexity {int(row['complexity'])}): R2 vs learned V = {r2(VL, p):.6f}   "
              f"R2 vs analytic AT = {r2(VT, p):.4f}")
        print(f"   {'':14s} expanded: {sp.expand(e).evalf(5)}")
    return df, F, VL, VT


if __name__ == "__main__":
    # positive control: SR on the analytic potential (evaluated against AT on the core set)
    df = pd.read_csv(FRONTS / "eqs_true.csv"); Qc = core_set(1200, 7)
    Fc = np.array([features_from_q6(q) for q in Qc]); row = pysr_best(df)
    p, e = evaluate(row["sympy_format"], Fc)
    print(f"== positive control (analytic AT)  [eqs_true.csv]\n   best: R2 vs analytic AT = {r2(true_V(Qc), p):.6f}"
          f"\n   expanded: {sp.expand(e).evalf(6)}   (exact: 1/(r12 r23 r31)^3 + 3 c1 c2 c3/(r12 r23 r31)^3)")

    # main recovery (Sec. VI, Eq. 11): the symmetric expression (complexity 18) and the one with the delta term (27)
    df, F, VL, VT = report("learned potential, S3-augmented search (Sec. VI)", "eqs_learned_recovered.csv",
                           heldout_valset(), rows=(18, 27))
    delta = 0.006805059 * (F[:, 1] + F[:, 2])
    print(f"   delta term 0.0068(r23+r31): range {delta.min():.4f}-{delta.max():.4f}; |delta|/|V_theta| median "
          f"{np.median(delta / np.abs(VL)):.4f}, max {np.max(delta / np.abs(VL)):.2f}")
    report("learned potential, search without augmentation", "eqs_learned_noaug.csv", heldout_valset())

    # Appendix D: operator-set controls on the learned potential (seeded, serial), evaluated on core_set(1200, 7)
    Qo = core_set(1200, 7)
    report("App. D, no cube operator (square + inv)", "eqs_learned_ops_nocube_seeded.csv", Qo)
    report("App. D, binary operators only", "eqs_learned_ops_binaryonly_seeded.csv", Qo)
    for seed in (1, 2, 3):   # the no-cube search rerun with new seeds
        report(f"App. D, no cube operator, random_state={seed}", f"eqs_learned_ops_nocube_seed{seed}.csv", Qo)
