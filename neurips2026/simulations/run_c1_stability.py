"""
Per-architecture Ĉ_1 stability experiment for App. I X.1.

Hypothesis: For fixed data distribution and fixed kernel family, varying only
the architecture (MLP width) yields approximately stable Ĉ_1 across widths
within an order of magnitude. This provides direct controlled-sweep evidence
for the local-Rademacher-complexity stability claim made in App. I X.1.

Setup matches run_srm_width.py:
- d=50, n=100, delta=0.5, Gaussian mean shift, P=N(0,I), Q=N(delta·1, I)
- Same DeepDiscriminator template (FEATURE_DIM=10)
- Plain-max calibration: train discriminator on -gamma_k_u^2 (no penalty)
- N_CAL=10 null permutations per width
- 3 random seeds per width to estimate within-setup variance

Output: results/c1_stability/c1_stability_results.csv with columns
  width, seed, c1_estimate, c1_n_valid_ratios, calib_seconds
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    DeepDiscriminator, mmd_loss_torch, _to_tensor,
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'c1_stability')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Device selection: prefer CUDA, then MPS, fall back to CPU
# ----------------------------------------------------------------------
# MPS lacks aten::_linalg_svd.U which torch.linalg.matrix_norm(ord=2) requires;
# fall back to CPU on Apple silicon since the SVD dominates and would CPU-fallback anyway.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Hyperparameters mirror run_srm_width.py
N_SAMPLES = 100
DIM = 50
DELTA = 0.5
FEATURE_DIM = 10
N_STEPS = 100
LR = 0.005
N_CAL = 10
WIDTHS = [50, 100, 200, 500, 1000]
SEEDS = [0, 1, 2]


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def gen(n, d, delta, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32) + delta
    return X, Y


def train_plain_max(X, Y, d, width, seed):
    """Plain-max trajectory: ascend gamma_k_u^2(h) with no penalty.

    Mirrors the calibration optimizer used in run_srm_width.py:135 with
    criterion='plain' (loss = -mmd2). Returns final sigma and spectral product.
    """
    torch.manual_seed(seed)
    X_t = _to_tensor(X).to(DEVICE)
    Y_t = _to_tensor(Y).to(DEVICE)
    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
    D = DeepDiscriminator(d, FEATURE_DIM, width).to(DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=DEVICE))
    params = list(D.parameters()) + [log_sigma]
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        opt.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        loss = -mmd2
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        prod = 1.0
        for name, p in D.named_parameters():
            if 'weight' in name:
                prod *= torch.linalg.matrix_norm(p, ord=2).item()
        f_x, f_y = D(X_t), D(Y_t)
        final_mmd2 = mmd_loss_torch(f_x, f_y, torch.tensor(sigma_val, device=DEVICE)).item()
    return sigma_val, prod, final_mmd2, data_norm


def calibrate_c1_at_width(X, Y, d, width, seed):
    """Plain-max calibration of Ĉ_1 at a given width.

    Mirrors run_srm_width.py:126 calibrate_c1 exactly: pool, permute,
    plain-max train, record gamma^2 / G̃. Take 95th percentile of valid ratios.
    """
    rng = np.random.default_rng(seed + 999)
    Z = np.concatenate([X, Y], axis=0)
    m = len(X)
    ratios = []
    for i in range(N_CAL):
        perm = rng.permutation(2 * m)
        Xp, Yp = Z[perm[:m]], Z[perm[m:]]
        try:
            sigma, prod, mmd2, data_norm = train_plain_max(
                Xp, Yp, d, width, seed=seed + 100 + i
            )
            G = prod * data_norm / max(sigma, 1e-8)
            if G > 1e-10:
                ratios.append(abs(mmd2) / G)
        except Exception as e:
            log(f"    rep {i} failed: {e}")
    if not ratios:
        return 0.01, 0
    return float(np.percentile(ratios, 95)), len(ratios)


DELTAS = [0.3, 0.5, 0.7, 1.0]
DELTA_FIXED_WIDTH = 200  # Within the well-behaved regime per architecture sweep


def sweep_delta_fixed_width():
    """Data-similarity sweep: vary delta at fixed architecture (width=200).

    Tests whether Ĉ_1 is stable across mild data-shift perturbations holding
    kernel + architecture + sample-size fixed.
    """
    log("\n" + "=" * 70)
    log(f"Data-similarity sweep: width={DELTA_FIXED_WIDTH} fixed, varying delta")
    log(f"Deltas: {DELTAS}")
    log("=" * 70)
    rows = []
    for seed in SEEDS:
        log(f"\n--- seed={seed} ---")
        for delta in DELTAS:
            X, Y = gen(N_SAMPLES * 2, DIM, delta, seed=seed)
            X_tr, Y_tr = X[:N_SAMPLES], Y[:N_SAMPLES]
            t0 = time.time()
            c1, n_valid = calibrate_c1_at_width(X_tr, Y_tr, DIM, DELTA_FIXED_WIDTH, seed)
            dt = time.time() - t0
            log(f"  delta={delta:.1f}  Ĉ_1={c1:.5f}  n_valid={n_valid}/{N_CAL}  ({dt:.1f}s)")
            rows.append({
                'delta': delta,
                'seed': seed,
                'width': DELTA_FIXED_WIDTH,
                'c1_estimate': c1,
                'c1_n_valid_ratios': n_valid,
                'calib_seconds': round(dt, 2),
            })
    csv_path = os.path.join(RESULTS_DIR, 'c1_stability_delta_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"\nWrote {csv_path}")

    log("\n=== Per-delta summary ===")
    log(f"  {'delta':>6}  {'mean Ĉ_1':>10}  {'std':>8}  {'min':>8}  {'max':>8}  {'CV (%)':>7}")
    for delta in DELTAS:
        c1s = [r['c1_estimate'] for r in rows if r['delta'] == delta]
        if c1s:
            mean = np.mean(c1s)
            std = np.std(c1s)
            cv = 100 * std / mean if mean > 0 else float('nan')
            log(f"  {delta:>6.1f}  {mean:>10.5f}  {std:>8.5f}  {min(c1s):>8.5f}  {max(c1s):>8.5f}  {cv:>7.1f}")
    all_c1 = [r['c1_estimate'] for r in rows]
    log(f"\nCross-delta range: min={min(all_c1):.5f}  max={max(all_c1):.5f}  ratio={max(all_c1)/max(min(all_c1),1e-8):.2f}x")


def main():
    log("=" * 70)
    log(f"Ĉ_1 stability sweep: d={DIM}, n={N_SAMPLES}, delta={DELTA}")
    log(f"Widths: {WIDTHS}")
    log(f"Seeds:  {SEEDS}")
    log(f"N_CAL:  {N_CAL}")
    log(f"Device: {DEVICE}")
    log("=" * 70)

    rows = []
    for seed in SEEDS:
        log(f"\n--- seed={seed} ---")
        X, Y = gen(N_SAMPLES * 2, DIM, DELTA, seed=seed)
        X_tr, Y_tr = X[:N_SAMPLES], Y[:N_SAMPLES]
        for width in WIDTHS:
            t0 = time.time()
            c1, n_valid = calibrate_c1_at_width(X_tr, Y_tr, DIM, width, seed)
            dt = time.time() - t0
            log(f"  width={width:5d}  Ĉ_1={c1:.5f}  n_valid={n_valid}/{N_CAL}  ({dt:.1f}s)")
            rows.append({
                'width': width,
                'seed': seed,
                'c1_estimate': c1,
                'c1_n_valid_ratios': n_valid,
                'calib_seconds': round(dt, 2),
            })

    csv_path = os.path.join(RESULTS_DIR, 'c1_stability_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"\nWrote {csv_path}")

    # Quick analysis
    log("\n=== Per-width summary across seeds ===")
    log(f"  {'width':>6}  {'mean Ĉ_1':>10}  {'std':>8}  {'min':>8}  {'max':>8}  {'CV (%)':>7}")
    for width in WIDTHS:
        c1s = [r['c1_estimate'] for r in rows if r['width'] == width]
        if c1s:
            mean = np.mean(c1s)
            std = np.std(c1s)
            cv = 100 * std / mean if mean > 0 else float('nan')
            log(f"  {width:>6}  {mean:>10.5f}  {std:>8.5f}  {min(c1s):>8.5f}  {max(c1s):>8.5f}  {cv:>7.1f}")

    log("\n=== Cross-width range ===")
    all_c1 = [r['c1_estimate'] for r in rows]
    log(f"  global min: {min(all_c1):.5f}  global max: {max(all_c1):.5f}  ratio: {max(all_c1)/min(all_c1):.2f}x")

    # Now run the data-similarity sweep
    sweep_delta_fixed_width()


if __name__ == '__main__':
    main()
