"""
Empirical Rademacher complexity of the trajectory class H_T versus the
theoretical spectral-norm-product upper bound used to derive C_1 \approx 30.

Goal: quantify the calibration gap. The paper currently asserts
G(H_T) << G(H), explaining why calibrated \widehat C_1 \approx 0.01 works while
the theoretical worst-case constant is C_1 \approx 30. This experiment turns
that assertion into a measurement.

Procedure (per width):
1. Train a CP-MMD discriminator for T steps on real (X, Y); save kernels h^{(t)}.
2. For each Rademacher draw \sigma \in {-1,+1}^{2N}:
     - Relabel pooled Z into (X^\sigma, Y^\sigma).
     - For each saved h^{(t)}: compute MMD_u^2(h^{(t)}, X^\sigma, Y^\sigma).
     - Take max over t -> sup_{h \in H_T} of MMD_u^2 on relabelled data.
3. Average max-over-trajectory across \sigma draws -> R_emp(H_T).
4. Compute the spectral-product bound \widetilde G_max
   = max_t (\prod_j ||W_j^{(t)}||_2) \cdot ||Z||/\sqrt{2N} / \sigma_t.
5. Report ratio \widetilde G_max / R_emp; this is the slack between the
   theoretical penalty constant and what the trajectory actually realises.

Setting: d=50, n=100, delta=0.5; widths {200, 1000}; 50 Rademacher draws.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _DEVICE
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'empirical_rademacher')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES = 100
DIM = 50
DELTA = 0.5
FEATURE_DIM = 10
N_STEPS = 100
LR = 0.005
SNAPSHOT_EVERY = 5
N_RADEMACHER = 50
N_REPS = 5
WIDTHS = [200, 1000]
C1 = 0.01


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def gen(n, d, delta, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32) + delta
    return X, Y


def train_with_snapshots(X, Y, d, width, C1, seed):
    """Train CP-MMD; return list of (state_dict, sigma, spec_product) snapshots."""
    torch.manual_seed(seed)
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
    D = DeepDiscriminator(d, FEATURE_DIM, width).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    opt = torch.optim.Adam(params, lr=LR)

    snapshots = []
    for step in range(N_STEPS):
        opt.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        spec = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec = spec * torch.linalg.matrix_norm(p, ord=2)
        G = spec * data_norm / sigma
        loss = -(mmd2 - C1 * G)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % SNAPSHOT_EVERY == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                snap_state = {k: v.detach().clone() for k, v in D.state_dict().items()}
                snap_sigma = log_sigma.exp().item()
                snap_spec = 1.0
                for name, p in D.named_parameters():
                    if 'weight' in name:
                        snap_spec *= torch.linalg.matrix_norm(p, ord=2).item()
                snapshots.append((snap_state, snap_sigma, snap_spec))
    return snapshots, data_norm


def features_at_snapshot(D, snap_state, Z_t):
    D.load_state_dict(snap_state)
    D.eval()
    with torch.no_grad():
        return D(Z_t).cpu().numpy()


def rademacher_estimate(snapshots, X, Y, d, width):
    """For each Rademacher relabelling of Z, compute sup_t MMD_u^2(h^(t), .)."""
    N = len(X)
    Z = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(Z).to(_DEVICE)
    D_eval = DeepDiscriminator(d, FEATURE_DIM, width).to(_DEVICE)

    # Cache features for each snapshot
    feats = []
    sigmas = []
    for state, sig, _ in snapshots:
        feats.append(features_at_snapshot(D_eval, state, Z_t))
        sigmas.append(sig)

    rng = np.random.default_rng(12345)
    sup_vals = []
    raw_means_per_t = np.zeros(len(snapshots))
    for r in range(N_RADEMACHER):
        sgn = rng.choice([-1, 1], size=2 * N)
        idx_pos = np.where(sgn == 1)[0]
        idx_neg = np.where(sgn == -1)[0]
        if len(idx_pos) < 2 or len(idx_neg) < 2:
            continue
        per_t = []
        for f, sig in zip(feats, sigmas):
            mmd2 = compute_mmd_u_statistic(f[idx_pos], f[idx_neg], 'gaussian', sig)
            per_t.append(abs(mmd2))
        per_t = np.asarray(per_t)
        sup_vals.append(per_t.max())
        raw_means_per_t += per_t / N_RADEMACHER

    return float(np.mean(sup_vals)), float(np.std(sup_vals)), raw_means_per_t


def theoretical_bound(snapshots, data_norm):
    """\widetilde G_max along the trajectory: max_t spec_t * data_norm / sigma_t."""
    vals = []
    for _, sig, spec in snapshots:
        vals.append(spec * data_norm / max(sig, 1e-8))
    return float(max(vals)), float(np.mean(vals))


def run_width(width, n_reps=N_REPS):
    log(f"--- width={width} ---")
    rows = []
    for rep in range(n_reps):
        seed = width * 13 + rep * 7
        X, Y = gen(N_SAMPLES, DIM, DELTA, seed=seed)
        snapshots, data_norm = train_with_snapshots(X, Y, DIM, width, C1, seed)
        R_emp_mean, R_emp_std, _ = rademacher_estimate(snapshots, X, Y, DIM, width)
        G_max, G_mean = theoretical_bound(snapshots, data_norm)
        ratio = G_max / max(R_emp_mean, 1e-12)
        log(f"  rep {rep}: R_emp={R_emp_mean:.4e} (sd {R_emp_std:.1e})  "
            f"G_tilde_max={G_max:.4e}  ratio={ratio:.2f}")
        rows.append({
            'width': width, 'rep': rep,
            'R_emp_mean': R_emp_mean, 'R_emp_std': R_emp_std,
            'G_tilde_max': G_max, 'G_tilde_mean': G_mean,
            'ratio': ratio,
            'data_norm': data_norm,
        })
    return rows


def main():
    log("=" * 70)
    log(f"Empirical Rademacher of trajectory class: widths={WIDTHS}, C1={C1}")
    log(f"d={DIM}, n={N_SAMPLES}, delta={DELTA}, n_rad={N_RADEMACHER}, n_reps={N_REPS}")
    log(f"Device: {_DEVICE}")
    log("=" * 70)

    rows = []
    for w in WIDTHS:
        t0 = time.time()
        rows.extend(run_width(w))
        log(f"  width {w}: {time.time()-t0:.1f}s")

    out = os.path.join(RESULTS_DIR, 'empirical_rademacher.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"Saved {out}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
