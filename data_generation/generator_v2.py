# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
# Generator VERSION 2, used for the LJ, LJ+AT and Newtonian datasets (Maxwell velocities at T* = 1, seeded).
# Copied verbatim from the authors' notebook generate_threebody_training_data_x10.ipynb (cell 31). Fully reproducible.

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PHYSICAL ASSUMPTION (IMPORTANT)
# ---------------------------------------------------------
# All coordinates are in reduced Lennard-Jones units:
#   r* = r / sigma
#   v* = v * sqrt(m / epsilon)
#   E* = E / epsilon
# so sigma, epsilon, mass are implicit in scaling.
# =========================================================

# -----------------------
# r-shaped proposal
# -----------------------

def sample_r_mixture(r_min, r_dense, r_max, p_dense, rng):
    """
    Log-uniform mixture:
    - with prob p_dense: sample from [r_min, r_dense]
    - else: sample from [r_dense, r_max]
    """
    if rng.random() < p_dense:
        lo, hi = np.log(r_min), np.log(r_dense)
    else:
        lo, hi = np.log(r_dense), np.log(r_max)
    return float(np.exp(rng.uniform(lo, hi)))


def pairwise_distances(pos6):
    x1, y1, x2, y2, x3, y3 = pos6
    r12 = np.hypot(x1 - x2, y1 - y2)
    r23 = np.hypot(x2 - x3, y2 - y3)
    r31 = np.hypot(x3 - x1, y3 - y1)
    return r12, r23, r31


def canonicalize_triangle(pos6):
    """
    Body1 at origin, body2 on +x axis (geometry normalization only).
    """
    x1, y1, x2, y2, x3, y3 = pos6

    x2 -= x1; y2 -= y1
    x3 -= x1; y3 -= y1

    ang = np.arctan2(y2, x2)
    c, s = np.cos(-ang), np.sin(-ang)

    x2r, y2r = c*x2 - s*y2, s*x2 + c*y2
    x3r, y3r = c*x3 - s*y3, s*x3 + c*y3

    if x2r < 0:
        x2r, y2r = -x2r, -y2r
        x3r, y3r = -x3r, -y3r

    return x2r, y2r, x3r, y3r


def random_rotate_translate(pos6, bounds=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x1, y1, x2, y2, x3, y3 = pos6

    ang = rng.uniform(0, 2*np.pi)
    c, s = np.cos(ang), np.sin(ang)

    def rot(x, y):
        return c*x - s*y, s*x + c*y

    x1, y1 = rot(x1, y1)
    x2, y2 = rot(x2, y2)
    x3, y3 = rot(x3, y3)

    if bounds is not None:
        low, high = bounds
        tx = rng.uniform(low, high)
        ty = rng.uniform(low, high)
        x1 += tx; y1 += ty
        x2 += tx; y2 += ty
        x3 += tx; y3 += ty

    return np.array([x1, y1, x2, y2, x3, y3], dtype=np.float64)


# -----------------------
# velocity prior (REDUCED UNITS)
# -----------------------

def sample_vel_reduced(dim=6, T_star=1.0, rng=None):
    """
    Maxwell-like velocities in reduced LJ units.
    m = 1 => std ~ sqrt(T*)
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(scale=np.sqrt(T_star), size=dim).astype(np.float64)


# -----------------------
# main generator
# -----------------------

def generate_dataset_r_based_positions_maxwell_vel(
    n_samples=60000,
    bounds=(-1.5, 1.5),
    r_min=0.75,
    r_dense=1.4,
    r_max=3.0,
    p_dense=0.5,
    rotate=True,
    translate=True,
    T_star=1.0,
    max_attempts=5_000_000,
    seed=1234,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    low, high = bounds

    states = []
    attempts = 0

    while len(states) < n_samples and attempts < max_attempts:
        attempts += 1

        # -----------------------
        # sample reference particle
        # -----------------------
        x1, y1 = rng.uniform(low, high, size=2)

        # -----------------------
        # sample 2nd and 3rd particles
        # -----------------------
        r12 = sample_r_mixture(r_min, r_dense, r_max, p_dense, rng)
        a12 = rng.uniform(0, 2*np.pi)
        x2 = x1 + r12*np.cos(a12)
        y2 = y1 + r12*np.sin(a12)

        r31 = sample_r_mixture(r_min, r_dense, r_max, p_dense, rng)
        a31 = rng.uniform(0, 2*np.pi)
        x3 = x1 + r31*np.cos(a31)
        y3 = y1 + r31*np.sin(a31)

        # -----------------------
        # reject out of box
        # -----------------------
        if not (low <= x2 <= high and low <= y2 <= high and
                low <= x3 <= high and low <= y3 <= high):
            continue

        pos6 = np.array([x1, y1, x2, y2, x3, y3], dtype=np.float64)

        r12c, r23c, r31c = pairwise_distances(pos6)

        # -----------------------
        # enforce physical regime
        # -----------------------
        if not (r_min <= r12c <= r_max and
                r_min <= r23c <= r_max and
                r_min <= r31c <= r_max):
            continue

        # -----------------------
        # symmetry augmentation
        # -----------------------
        if rotate or translate:
            pos6 = random_rotate_translate(
                pos6,
                bounds=bounds if translate else None,
                rng=rng
            )

            x1,y1,x2,y2,x3,y3 = pos6

            if not (low <= x1 <= high and low <= y1 <= high and
                    low <= x2 <= high and low <= y2 <= high and
                    low <= x3 <= high and low <= y3 <= high):
                continue

        # -----------------------
        # velocity sampling (IMPORTANT FIX)
        # -----------------------
        vel6 = sample_vel_reduced(dim=6, T_star=T_star, rng=rng)

        state12 = np.concatenate([pos6, vel6])
        states.append(state12)

        if verbose and len(states) % 5000 == 0:
            print(f"[DATA] {len(states)}/{n_samples} (attempts={attempts})")

    states = np.array(states, dtype=np.float64)

    # --- random permute particle labels inside each sample (fix "body1 is special") ---
    rngp = np.random.default_rng(seed + 999)

    # reshape (N, 12) -> (N, 3, 4) where each body has [x,y,vx,vy]
    S = np.zeros((states.shape[0], 3, 4), dtype=states.dtype)

    S[:, 0, :] = states[:, [0,1, 6,7]]   # body1: x1,y1,vx1,vy1
    S[:, 1, :] = states[:, [2,3, 8,9]]   # body2: x2,y2,vx2,vy2
    S[:, 2, :] = states[:, [4,5,10,11]]  # body3: x3,y3,vx3,vy3

    perms = np.array([
        [0,1,2],
        [0,2,1],
        [1,0,2],
        [1,2,0],
        [2,0,1],
        [2,1,0],
    ], dtype=np.int64)

    perm_idx = rngp.integers(0, 6, size=S.shape[0])
    P = perms[perm_idx]                      # (N, 3)

    S = S[np.arange(S.shape[0])[:, None], P] # permute body axis per-sample
    states2 = np.empty_like(states)
    states2[:, 0:2]  = S[:, 0, 0:2]
    states2[:, 2:4]  = S[:, 1, 0:2]
    states2[:, 4:6]  = S[:, 2, 0:2]
    states2[:, 6:8]  = S[:, 0, 2:4]
    states2[:, 8:10] = S[:, 1, 2:4]
    states2[:,10:12] = S[:, 2, 2:4]
    states = states2

    # -------------------------------------------------------------------------------

    rng2 = np.random.default_rng(seed + 1)
    states = states[rng2.permutation(states.shape[0])]


    info = {
        "attempts": attempts,
        "vel_norm": np.linalg.norm(states[:, 6:], axis=1),
        "r_range": (r_min, r_dense, r_max),
    }

    return states, info
