# Part of the code release for "Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks".
# Apache-2.0.
# Generator VERSION 1, used for the pure Axilrod-Teller datasets (rd15 dense/sparse and log-uniform).
# Copied verbatim from the authors' notebook generate_threebody_training_data_x10.ipynb (cell 9); plotting helpers removed.
# NOTE: velocities are drawn by sample_vel_realistic() with a FRESH, UNSEEDED generator, so positions are exactly
# reproducible from the seed but velocities are not. Velocities do not enter the learned potential (fixed kinetic term).


### a new one ###
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# r-shaped proposal
# -----------------------

def sample_r_mixture(r_min, r_dense, r_max, p_dense, rng):
    """Log-uniform mixture: dense in [r_min, r_dense] with prob p_dense, else in [r_dense, r_max]."""
    if rng.random() < p_dense:
        lo, hi = np.log(r_min), np.log(r_dense)
    else:
        lo, hi = np.log(r_dense), np.log(r_max)
    return float(np.exp(rng.uniform(lo, hi)))

def pairwise_distances(pos6):
    x1,y1,x2,y2,x3,y3 = pos6
    r12 = np.hypot(x1-x2, y1-y2)
    r23 = np.hypot(x2-x3, y2-y3)
    r31 = np.hypot(x3-x1, y3-y1)
    return r12, r23, r31

def canonicalize_triangle(pos6):
    """Translate so body1 at origin; rotate so body2 lies on +x. Returns (x2,y2,x3,y3)."""
    x1,y1,x2,y2,x3,y3 = pos6
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
    """Optional augmentation: rotate + translate (translation keeps you inside a box if bounds provided)."""
    if rng is None:
        rng = np.random.default_rng()
    x1,y1,x2,y2,x3,y3 = pos6

    ang = rng.uniform(0, 2*np.pi)
    c,s = np.cos(ang), np.sin(ang)
    def rot(x,y): return (c*x - s*y, s*x + c*y)
    x1,y1 = rot(x1,y1); x2,y2 = rot(x2,y2); x3,y3 = rot(x3,y3)

    if bounds is not None:
        low, high = bounds
        tx = rng.uniform(low, high)
        ty = rng.uniform(low, high)
        x1+=tx; y1+=ty; x2+=tx; y2+=ty; x3+=tx; y3+=ty

    return np.array([x1,y1,x2,y2,x3,y3], dtype=np.float64)

# -----------------------
# velocity prior (Maxwell-ish)
# -----------------------

def sample_vel_maxwell(dim=6, std=1.0, v_clip=2.8*np.sqrt(3), rng=None):
    """Gaussian per-component + optional norm clip."""
    if rng is None:
        rng = np.random.default_rng()
    v = rng.normal(scale=std, size=dim)
    n = np.linalg.norm(v)
    if v_clip is not None and n > v_clip:
        v *= (v_clip / (n + 1e-12))
    return v.astype(np.float64)

def sample_vel_realistic(dim=6, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # from my real data
    #std_small = 0.5        # matches per-component std
    #std_big   = 1.0        # allows rare fast events. not too much tho.
    #p_big     = 0.02       # ~2% fast samples (tune 5–10% ?)
    #v_clip    = 30.0       # slightly above observed max (27)

    # tuned for r=(0.7,1.2), masses=1 -> v_body ~(0.25,2.8) by energy comparison.
    std_small = 0.80   # typical per-component scale
    std_big   = 1.25   # rare faster mode
    p_big     = 0.02   # 2% fast samples
    v_clip    = 2.8*np.sqrt(3)    # the total speed limit roughly, so that we don't introduce any stricter limits on per-body per-component speeds

    std = std_big if rng.random() < p_big else std_small
    v = rng.normal(scale=std, size=dim)

    n = np.linalg.norm(v)
    if n > v_clip:
        v *= (v_clip / (n + 1e-12))

    return v.astype(np.float64)


# -----------------------
# main generator
# -----------------------

def generate_dataset_r_based_positions_maxwell_vel(
    n_samples=60000,
    bounds=(-1.5, 1.5),          # box for all three bodies (optional but convenient) (-1.5, 1.5), 
    r_min=1e-2,
    r_dense=3.0,
    r_max=20.0,
    p_dense=0.9,                # probability of sampling in [r_min, r_dense]
    rotate=True,
    translate=True,             # if True, uses bounds for translation (keeps points in box after augmentation)
    vel_std=1.0,
    v_clip=20.0,
    max_attempts=5_000_000,
    seed=1234,
    verbose=True,
):
    """
    Returns:
      states: (N, 12) array [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
      info: dict with r statistics and canonical shape samples for visualization
    """
    rng = np.random.default_rng(seed)
    low, high = bounds

    states = []
    r12s, r23s, r31s, rmins = [], [], [], []
    x2s, y2s, x3s, y3s = [], [], [], []
    attempts = 0

    # how we place points:
    # 1) pick body1 in box
    # 2) pick body2 around body1 at radius r12 ~ mixture, angle uniform
    # 3) pick body3 around body1 at radius r31 ~ mixture, angle uniform
    # 4) accept if all pairwise distances in [r_min, r_max] and points in bounds
    while len(states) < n_samples and attempts < max_attempts:
        attempts += 1

        x1, y1 = rng.uniform(low, high, size=2)

        r12 = sample_r_mixture(r_min, r_dense, r_max, p_dense, rng)
        a12 = rng.uniform(0, 2*np.pi)
        x2 = x1 + r12*np.cos(a12)
        y2 = y1 + r12*np.sin(a12)

        r31 = sample_r_mixture(r_min, r_dense, r_max, p_dense, rng)
        a31 = rng.uniform(0, 2*np.pi)
        x3 = x1 + r31*np.cos(a31)
        y3 = y1 + r31*np.sin(a31)

        # stay in box
        if not (low <= x2 <= high and low <= y2 <= high and low <= x3 <= high and low <= y3 <= high):
            continue

        pos6 = np.array([x1,y1,x2,y2,x3,y3], dtype=np.float64)
        r12c, r23c, r31c = pairwise_distances(pos6)

        if not (r_min <= r12c <= r_max and r_min <= r23c <= r_max and r_min <= r31c <= r_max):
            continue


        # optional augmentation (rotation + translation)
        if rotate or translate:
            pos6 = random_rotate_translate(
                pos6,
                bounds=bounds if translate else None,
                rng=rng
            )
            # if translate=True, we might exit box only if bounds very tight; re-check
            x1,y1,x2,y2,x3,y3 = pos6
            if not (low <= x1 <= high and low <= y1 <= high and
                    low <= x2 <= high and low <= y2 <= high and
                    low <= x3 <= high and low <= y3 <= high):
                continue
            r12c, r23c, r31c = pairwise_distances(pos6)
            if not (r_min <= r12c <= r_max and r_min <= r23c <= r_max and r_min <= r31c <= r_max):
                continue

       # vel6 = sample_vel_maxwell(dim=6, std=vel_std, v_clip=v_clip, rng=rng)
        #vel6 = sample_vel_heavy_tail(dim=6, std_small=0.5, std_big=3.0, p_big=0.1, v_clip=50.0, rng=rng)
        r_max_edge = max(r12c, r23c, r31c)
        scale = 1.0 if r_max_edge <= r_dense else (r_dense / r_max_edge)**0.5
        vel6 = sample_vel_realistic() * scale
        #vel6=sample_vel_realistic()

        state12 = np.concatenate([pos6, vel6])

        states.append(state12)
        r12s.append(r12c); r23s.append(r23c); r31s.append(r31c)
        rmins.append(min(r12c, r23c, r31c))

        x2r, y2r, x3r, y3r = canonicalize_triangle(pos6)
        x2s.append(x2r); y2s.append(y2r); x3s.append(x3r); y3s.append(y3r)

        if verbose and (len(states) % 5000 == 0):
            print(f"[DATA] collected {len(states)}/{n_samples} (attempts={attempts})")
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
        "r12": np.array(r12s),
        "r23": np.array(r23s),
        "r31": np.array(r31s),
        "rmin": np.array(rmins),
        "canon": (np.array(x2s), np.array(y2s), np.array(x3s), np.array(y3s)),
        "vel_norm": np.linalg.norm(states[:, 6:], axis=1),
    }
    return states, info

