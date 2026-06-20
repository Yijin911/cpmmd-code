"""Calibration-ablation row: deep HDGM at d=20, n=200, Delta=0.5.

This is the deep-kernel headline row of the §5.2 ablation table. Reports
shared-G_ref Ĉ_1 and CP-MMD test power on HDGM (high-dimensional Gaussian
mean shift) using the journal's shared-G_ref calibration.

Reference: conference companion arXiv:2605.06883 Table 2, cell (d=20, n=200, Δ=0.5).
Conference reports power 1.00.

Headline configuration only — no sweep over (d, n) cells, no Liu/MMDAgg
baselines.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn

# Use python -u for unbuffered output instead of fdopen-replacing stdout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    DeepDiscriminator, mmd_loss_torch, _to_tensor,
    _sq_euclidean_distances, _DEVICE,
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Headline configuration (conference companion arXiv:2605.06883 Table 2 cell d=20,n=200,Δ=0.5).
# DIM is overridable via --dim CLI arg; the (d=2, n=200, Δ=0.5) cell from
# Table 2 has CP-MMD power 0.83 (non-saturated), useful for ablation.
DIM = 20
N_SAMPLES = 200
DELTA = 0.5
N_REPS = 200
N_PERM = 200
ALPHA = 0.05

# Deep kernel hyperparameters (match run_higgs.py / paper §5.1 setup)
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005

LOG_FILE = os.path.join(RESULTS_DIR, 'hdgm_log.txt')
CSV_FILE = os.path.join(RESULTS_DIR, 'hdgm.csv')
log_fh = None


def _apply_cli_args():
    """Override DIM, LOG_FILE, CSV_FILE, C1_OVERRIDE from CLI."""
    global DIM, LOG_FILE, CSV_FILE, C1_OVERRIDE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=DIM)
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for log/csv filenames (e.g. _d2)')
    parser.add_argument('--c1-override', type=float, default=None,
                        help='Skip calibration; use this C_1 directly')
    args, _ = parser.parse_known_args()
    DIM = args.dim
    C1_OVERRIDE = args.c1_override
    if args.suffix:
        LOG_FILE = os.path.join(RESULTS_DIR, f'hdgm{args.suffix}_log.txt')
        CSV_FILE = os.path.join(RESULTS_DIR, f'hdgm{args.suffix}.csv')


C1_OVERRIDE = None


def log(msg):
    print(msg)
    if log_fh is not None:
        log_fh.write(msg + '\n')
        log_fh.flush()


# ----------------------------------------------------------------------
# Data: P = N(0, I_d), Q = N(Δ·1, I_d)
# ----------------------------------------------------------------------
def sample_hdgm(n, d, delta, rng):
    P = rng.standard_normal((n, d)).astype(np.float32)
    Q = (delta + rng.standard_normal((n, d))).astype(np.float32)
    return P, Q


# ----------------------------------------------------------------------
# Spectral-norm complexity proxy G̃_T = prod ||W_i||_2 * ||Z||_F / N
# (Bartlett-Foster-Telgarsky 2017)
# ----------------------------------------------------------------------
def compute_G_spectral(model, X, Y, sigma):
    spectral_product = 1.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            spectral_product *= torch.linalg.matrix_norm(param, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    data_geom = np.linalg.norm(Z) / N
    G = spectral_product * data_geom
    if sigma is not None and sigma > 0:
        G = G / sigma
    return G


# ----------------------------------------------------------------------
# Train plain-MMD deep kernel (no penalty), returns trained discriminator + sigma + MMD
# ----------------------------------------------------------------------
def train_plain(X, Y, dim):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        mmd2 = mmd_loss_torch(D(X_t), D(Y_t), sigma)
        (-mmd2).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        mmd_val = mmd_loss_torch(D(X_t), D(Y_t), sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


# ----------------------------------------------------------------------
# Train CP-MMD deep kernel with calibrated C_1
# ----------------------------------------------------------------------
def train_cpmmd(X, Y, dim, C1):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z_t = torch.cat([X_t, Y_t], dim=0)
    N = X_t.shape[0] + Y_t.shape[0]
    data_norm = torch.linalg.matrix_norm(Z_t).item() / N
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        mmd2 = mmd_loss_torch(D(X_t), D(Y_t), sigma)
        # CP-MMD penalty: C_1 * (prod ||W_i||_2) * ||Z||_F / (N * sigma)
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, param in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(param, ord=2)
        G = spec_prod * data_norm / sigma.clamp(min=1e-6)
        J = mmd2 - C1 * G
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        mmd_val = mmd_loss_torch(D(X_t), D(Y_t), sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


# ----------------------------------------------------------------------
# Shared-G_ref Ĉ_1 calibration: 95th-percentile of mmd_perm / G_ref
# ----------------------------------------------------------------------
def calibrate_C1_shared(X_null, Y_null, dim, n_cal=10, seed=42):
    rng = np.random.default_rng(seed)
    m, n = len(X_null), len(Y_null)
    N = m + n
    pooled = np.concatenate([X_null, Y_null], axis=0)
    # G_ref: train once on un-permuted null data
    res_ref = train_plain(X_null, Y_null, dim)
    G_ref = compute_G_spectral(res_ref['model'], X_null, Y_null, res_ref['sigma'])
    G_ref = max(abs(G_ref), 1e-12)
    ratios = []
    for _ in range(n_cal):
        perm = rng.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        res = train_plain(X_p, Y_p, dim)
        ratios.append(res['mmd'] / G_ref)
    return float(np.percentile(ratios, (1 - ALPHA) * 100)), G_ref


# ----------------------------------------------------------------------
# PermSplit_2 test using CP-MMD trained kernel
# ----------------------------------------------------------------------
def permsplit_test(X, Y, dim, C1, n_perm=N_PERM):
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_tr, Y_tr = X[:m_tr], Y[:n_tr]
    X_te, Y_te = X[m_tr:], Y[n_tr:]
    res = train_cpmmd(X_tr, Y_tr, dim, C1)
    model, sigma = res['model'], res['sigma']
    te_all = np.concatenate([X_te, Y_te], axis=0)
    te_t = _to_tensor(te_all).to(_DEVICE)
    with torch.no_grad():
        f_all = model(te_t)
    m_te, n_te = len(X_te), len(Y_te)
    test_stat = mmd_loss_torch(f_all[:m_te], f_all[m_te:], sigma).item()
    null_mmds = np.empty(n_perm)
    rng = np.random.default_rng()
    for i in range(n_perm):
        perm = rng.permutation(m_te + n_te)
        null_mmds[i] = mmd_loss_torch(f_all[perm[:m_te]], f_all[perm[m_te:]], sigma).item()
    threshold = float(np.percentile(null_mmds, (1 - ALPHA) * 100))
    return int(test_stat > threshold)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global log_fh
    _apply_cli_args()
    log_fh = open(LOG_FILE, 'w', buffering=1)
    log(f"Ablation HDGM run: d={DIM}, n={N_SAMPLES}, Δ={DELTA}, n_reps={N_REPS}")
    log(f"Device: {_DEVICE}")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calibrate Ĉ_1 once on a dedicated null sample (P vs P), or use override.
    if C1_OVERRIDE is not None:
        C1, G_ref = float(C1_OVERRIDE), float('nan')
        log(f"\n=== Calibration (override) ===")
        log(f"  Ĉ_1 (override) = {C1:.6f}  (skipping calibration)")
    else:
        log("\n=== Calibration (shared G_ref) ===")
        rng = np.random.default_rng(42)
        X_cal, _ = sample_hdgm(2 * N_SAMPLES, DIM, 0.0, rng)
        X_cal_a = X_cal[:N_SAMPLES]
        X_cal_b = X_cal[N_SAMPLES:]
        t0 = time.time()
        C1, G_ref = calibrate_C1_shared(X_cal_a, X_cal_b, DIM, n_cal=10, seed=42)
        log(f"  Ĉ_1 (shared) = {C1:.6f}  (G_ref = {G_ref:.6f}, {time.time()-t0:.1f}s)")

    # H1: power at Δ=0.5
    log(f"\n=== H1: Δ={DELTA} (power) ===")
    h1_reject = 0
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 7919
        rng = np.random.default_rng(seed)
        X, Y = sample_hdgm(N_SAMPLES, DIM, DELTA, rng)
        h1_reject += permsplit_test(X, Y, DIM, C1)
        if (rep + 1) % 20 == 0:
            log(f"  rep {rep+1}/{N_REPS}: power so far {h1_reject/(rep+1):.3f}")
    power = h1_reject / N_REPS
    log(f"  POWER = {power:.3f} ({h1_reject}/{N_REPS})  ({time.time()-t0:.1f}s)")

    # H0: Type-I at Δ=0
    log(f"\n=== H0: Δ=0 (Type-I) ===")
    h0_reject = 0
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 7919 + 1
        rng = np.random.default_rng(seed)
        X, Y = sample_hdgm(N_SAMPLES, DIM, 0.0, rng)
        h0_reject += permsplit_test(X, Y, DIM, C1)
        if (rep + 1) % 20 == 0:
            log(f"  rep {rep+1}/{N_REPS}: Type-I so far {h0_reject/(rep+1):.3f}")
    type1 = h0_reject / N_REPS
    log(f"  TYPE_I = {type1:.3f} ({h0_reject}/{N_REPS})  ({time.time()-t0:.1f}s)")

    # Save CSV
    with open(CSV_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'd', 'n', 'delta', 'n_reps', 'C1_shared', 'G_ref', 'power', 'type_I'])
        w.writerow([f'hdgm_deep_d{DIM}', DIM, N_SAMPLES, DELTA, N_REPS, C1, G_ref, power, type1])

    log(f"\nSaved {CSV_FILE}")
    log(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
