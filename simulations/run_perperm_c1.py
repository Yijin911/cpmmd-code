"""Per-permutation Ĉ_1 calibration for the 3 missing ablation experiments.

Companion to run_ablation_{linear,polynomial,hdgm}.py. Those scripts use
the journal's shared-G_ref procedure; this one computes the conference
companion's per-permutation procedure on the same null data with the
same seed.

For each of (linear, polynomial, hdgm):
  - Use the same null data and seed as run_ablation_<exp>.py.
  - Run the same optimizer (bandwidth grid / polynomial bandwidth grid /
    deep-MLP training) on each of the n_cal=10 permutations.
  - Compute G on the PERMUTED sample (not the un-permuted reference).
  - Report the 95th-percentile mmd_p / G_p as the per-perm Ĉ_1.

Writes results to results/ablation/perperm_c1.csv.

This script does NOT run power / Type-I — only calibrates Ĉ_1. Fast.
"""

import os, sys, time, csv, warnings
from itertools import combinations_with_replacement
import numpy as np

# Use python -u for unbuffered output.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_FILE = os.path.join(RESULTS_DIR, 'perperm_c1.csv')

ALPHA = 0.05
N_CAL = 10
SEED = 42

# ======================================================================
# Linear (multi-scale 2D Gaussian)
# ======================================================================
def calibrate_linear():
    from run_ablation_linear import (
        sample_multiscale, cpmmd_select, N_SAMPLES, DIM
    )
    print("[linear] calibrating...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    X_cal, _ = sample_multiscale(2 * N_SAMPLES, 0.0, rng)
    X_a, X_b = X_cal[:N_SAMPLES], X_cal[N_SAMPLES:]
    pooled = np.concatenate([X_a, X_b], axis=0)
    N = len(pooled)
    rng_perm = np.random.default_rng(SEED)
    ratios = []
    for _ in range(N_CAL):
        perm = rng_perm.permutation(N)
        Xp, Yp = pooled[perm[:len(X_a)]], pooled[perm[len(X_a):]]
        sigma_p, mmd_p = cpmmd_select(Xp, Yp)
        pooled_p = np.concatenate([Xp, Yp], axis=0)
        G_p = (1.0 / max(sigma_p, 1e-6)) * np.linalg.norm(pooled_p) / N
        ratios.append(mmd_p / max(abs(G_p), 1e-12))
    C1_perperm = float(np.percentile(ratios, (1 - ALPHA) * 100))
    print(f"[linear] Ĉ_1 (per-perm) = {C1_perperm:.6f} ({time.time()-t0:.1f}s)")
    return C1_perperm


# ======================================================================
# Polynomial (kurtosis shift, d=10)
# ======================================================================
def calibrate_polynomial():
    from run_ablation_polynomial import (
        sample_kurtosis, cpmmd_poly_select, poly_features,
        N_SAMPLES, DIM, POLY_DEGREE
    )
    print("[polynomial] calibrating...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    X_cal, _ = sample_kurtosis(2 * N_SAMPLES, DIM, None, rng)
    X_a, X_b = X_cal[:N_SAMPLES], X_cal[N_SAMPLES:]
    pooled = np.concatenate([X_a, X_b], axis=0)
    N = len(pooled)
    rng_perm = np.random.default_rng(SEED)
    ratios = []
    for _ in range(N_CAL):
        perm = rng_perm.permutation(N)
        Xp, Yp = pooled[perm[:len(X_a)]], pooled[perm[len(X_a):]]
        sigma_p, mmd_p, _, _ = cpmmd_poly_select(Xp, Yp)
        pooled_p = np.concatenate([Xp, Yp], axis=0)
        pooled_feat = poly_features(pooled_p, POLY_DEGREE)
        G_p = (1.0 / max(sigma_p, 1e-6)) * np.linalg.norm(pooled_feat) / N
        ratios.append(mmd_p / max(abs(G_p), 1e-12))
    C1_perperm = float(np.percentile(ratios, (1 - ALPHA) * 100))
    print(f"[polynomial] Ĉ_1 (per-perm) = {C1_perperm:.6f} ({time.time()-t0:.1f}s)")
    return C1_perperm


# ======================================================================
# HDGM (deep d=20)
# ======================================================================
def calibrate_hdgm(dim_override=None, suffix=''):
    import run_ablation_hdgm as m
    from run_ablation_hdgm import (
        sample_hdgm, train_plain, compute_G_spectral,
    )
    DIM = dim_override if dim_override is not None else m.DIM
    N_SAMPLES = m.N_SAMPLES
    print(f"[hdgm{suffix}] calibrating (DIM={DIM})...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    X_cal, _ = sample_hdgm(2 * N_SAMPLES, DIM, 0.0, rng)
    X_a, X_b = X_cal[:N_SAMPLES], X_cal[N_SAMPLES:]
    pooled = np.concatenate([X_a, X_b], axis=0)
    N = len(pooled)
    rng_perm = np.random.default_rng(SEED)
    ratios = []
    for i in range(N_CAL):
        perm = rng_perm.permutation(N)
        Xp, Yp = pooled[perm[:len(X_a)]], pooled[perm[len(X_a):]]
        res = train_plain(Xp, Yp, DIM)
        G_p = compute_G_spectral(res['model'], Xp, Yp, res['sigma'])
        G_p = max(abs(G_p), 1e-12)
        ratios.append(res['mmd'] / G_p)
        print(f"  [hdgm{suffix}] rep {i+1}/{N_CAL}: mmd={res['mmd']:.6f} G_p={G_p:.6f}")
    C1_perperm = float(np.percentile(ratios, (1 - ALPHA) * 100))
    print(f"[hdgm{suffix}] Ĉ_1 (per-perm) = {C1_perperm:.6f} ({time.time()-t0:.1f}s)")
    return C1_perperm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['linear', 'polynomial', 'hdgm', 'all'],
                        default='all')
    parser.add_argument('--dim', type=int, default=None,
                        help='Override HDGM dim (e.g. --dim 2 for harder cell)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Label suffix for output rows (e.g. _d2)')
    args = parser.parse_args()

    results = {}
    if args.exp in ('linear', 'all'):
        results[f'linear{args.suffix}'] = calibrate_linear()
    if args.exp in ('polynomial', 'all'):
        results[f'polynomial{args.suffix}'] = calibrate_polynomial()
    if args.exp in ('hdgm', 'all'):
        results[f'hdgm{args.suffix}'] = calibrate_hdgm(dim_override=args.dim,
                                                       suffix=args.suffix)

    # Append to CSV (preserves prior rows if running individually)
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['experiment', 'C1_perperm'])
        for exp, c1 in results.items():
            w.writerow([exp, c1])
    print(f"\nSaved {CSV_FILE}")


if __name__ == '__main__':
    main()
