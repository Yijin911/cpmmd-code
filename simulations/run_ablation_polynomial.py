"""Calibration ablation row: polynomial composite kernel at df=20, n=500.

§5.2 polynomial cell, conference-comparable setup.

Matches the conference companion's `run_polynomial_200reps.py`:
  - polynomial degree p = 4 is fixed (the experimenter's choice for the
    4th-moment kurtosis alternative)
  - search class is sigma in [0.01, 100] only
The only differences from the conference are:
  (i)  the per-h proxy is active in sigma-selection:
         J_CP(sigma) = MMD^2(sigma) - C1 * ||Psi_4(D)||_F / (sigma * N)
  (ii) two calibration procedures are compared (shared vs per-perm G_ref)

Reference: conference Fig 2(b), power 0.90 at df=20, n=500.
"""

import os, sys, time, csv, warnings
from itertools import combinations_with_replacement
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import _DEVICE
from fast_ablation import (
    optimize_bandwidth_jcp, optimize_bandwidth_plain, permutation_pvalue,
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)

DIM = 10
DF_H1 = 20
N_SAMPLES = 500
N_REPS = 200
N_PERM = 200
N_CAL = 10
ALPHA = 0.05
POLY_DEGREE = 4
BW_LOW, BW_HIGH = 0.01, 100.0
KERNEL = 'laplacian'

LOG_FILE = os.path.join(RESULTS_DIR, 'polynomial_log.txt')
CSV_FILE = os.path.join(RESULTS_DIR, 'polynomial.csv')
log_fh = None


def log(msg):
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + '\n')
        log_fh.flush()


def sample_kurtosis(n, d, df, rng):
    X = rng.standard_normal((n, d)).astype(np.float32)
    if df is None:
        Y = rng.standard_normal((n, d)).astype(np.float32)
    else:
        W = rng.standard_normal((n, d))
        chi2 = rng.chisquare(df, size=(n, 1))
        Z = W / np.sqrt(chi2 / df)
        Y = (Z * np.sqrt((df - 2.0) / df)).astype(np.float32)
    return X, Y


def poly_features(X, degree):
    n, d = X.shape
    out = [X]
    for deg in range(2, degree + 1):
        tuples = list(combinations_with_replacement(range(d), deg))
        feat = np.empty((n, len(tuples)), dtype=X.dtype)
        for k, tup in enumerate(tuples):
            v = np.ones(n, dtype=X.dtype)
            for i in tup:
                v = v * X[:, i]
            feat[:, k] = v
        out.append(feat)
    return np.hstack(out)


def g_ref_feat(pooled_feat, sigma):
    N = pooled_feat.shape[0]
    return (1.0 / max(sigma, 1e-6)) * float(np.linalg.norm(pooled_feat)) / N


def calibrate(Xp_null, Yp_null, share_g_ref, n_cal=N_CAL, seed=42):
    """Calibrate on feature-mapped data at the fixed degree p.

    Shared: worst-case L^* = 1/sigma_l on the reference data.
    Per-perm: at each permutation's empirical argmax sigma_pi.
    """
    rng = np.random.default_rng(seed)
    m, n = len(Xp_null), len(Yp_null)
    N = m + n
    pooled = np.concatenate([Xp_null, Yp_null], axis=0)

    if share_g_ref:
        G_ref_fixed = (1.0 / BW_LOW) * float(np.linalg.norm(pooled)) / N
    else:
        G_ref_fixed = None

    ratios = []
    for _ in range(n_cal):
        perm = rng.permutation(N)
        Xp_p, Yp_p = pooled[perm[:m]], pooled[perm[m:]]
        opt = optimize_bandwidth_plain(Xp_p, Yp_p, KERNEL, (BW_LOW, BW_HIGH))
        mmd_p = opt['max_mmd']
        if share_g_ref:
            G = G_ref_fixed
        else:
            G = g_ref_feat(np.concatenate([Xp_p, Yp_p], axis=0),
                           opt['optimal_sigma'])
        ratios.append(mmd_p / max(abs(G), 1e-12))
    C1 = float(np.percentile(ratios, (1 - ALPHA) * 100))
    return C1, G_ref_fixed


def permsplit_test(X, Y, C1, seed):
    """Fixed p = POLY_DEGREE; sigma selected via J_CP on the training half."""
    n_half = len(X) // 2
    Xp_tr = poly_features(X[:n_half], POLY_DEGREE)
    Yp_tr = poly_features(Y[:n_half], POLY_DEGREE)
    Xp_te = poly_features(X[n_half:], POLY_DEGREE)
    Yp_te = poly_features(Y[n_half:], POLY_DEGREE)
    pooled_tr = np.concatenate([Xp_tr, Yp_tr], axis=0)
    norm_tr = float(np.linalg.norm(pooled_tr))
    opt = optimize_bandwidth_jcp(Xp_tr, Yp_tr, C1, KERNEL,
                                  (BW_LOW, BW_HIGH),
                                  data_norm_F=norm_tr)
    sigma = opt['optimal_sigma']
    rng = np.random.default_rng(seed)
    rej = permutation_pvalue(Xp_te, Yp_te, sigma, KERNEL,
                             n_perm=N_PERM, alpha=ALPHA, rng=rng)
    return rej, sigma


def run_power_typeI(C1, label):
    h1 = 0; sigs_h1 = []
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 7919
        rng = np.random.default_rng(seed)
        X, Y = sample_kurtosis(2 * N_SAMPLES, DIM, DF_H1, rng)
        rej, sigma = permsplit_test(X, Y, C1, seed)
        h1 += rej
        sigs_h1.append(sigma)
        if (rep + 1) % 50 == 0:
            log(f"  [{label}] rep {rep+1}/{N_REPS}: power so far {h1/(rep+1):.3f}  ({time.time()-t0:.1f}s)")
    power = h1 / N_REPS
    log(f"  [{label}] POWER = {power:.3f} ({h1}/{N_REPS})  median sigma = {np.median(sigs_h1):.4f}  ({time.time()-t0:.1f}s)")

    h0 = 0
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 7919 + 1
        rng = np.random.default_rng(seed)
        X, Y = sample_kurtosis(2 * N_SAMPLES, DIM, None, rng)
        rej, _ = permsplit_test(X, Y, C1, seed)
        h0 += rej
    type1 = h0 / N_REPS
    log(f"  [{label}] TYPE-I = {type1:.3f} ({h0}/{N_REPS})  ({time.time()-t0:.1f}s)")
    return power, type1, float(np.median(sigs_h1))


def main():
    global log_fh
    log_fh = open(LOG_FILE, 'w', buffering=1)
    log(f"Ablation polynomial fixed p={POLY_DEGREE} (kurtosis, df={DF_H1}, d={DIM}, n={N_SAMPLES}) [per-h proxy on sigma]")
    log(f"Conference-comparable: same fixed-p experimental setup; per-h proxy added in sigma-selection")
    log(f"Device: {_DEVICE}")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_global = time.time()

    rng = np.random.default_rng(42)
    X_cal, _ = sample_kurtosis(2 * N_SAMPLES, DIM, None, rng)
    Xp_cal_a = poly_features(X_cal[:N_SAMPLES], POLY_DEGREE)
    Xp_cal_b = poly_features(X_cal[N_SAMPLES:], POLY_DEGREE)

    log("\n=== Calibration ===")
    t0 = time.time()
    C1_shared, G_shared = calibrate(Xp_cal_a, Xp_cal_b, share_g_ref=True)
    log(f"  C1_shared    = {C1_shared:.6f}  (G_ref = {G_shared:.6f}, worst-case L*=1/sigma_l={1.0/BW_LOW:.2f}, {time.time()-t0:.1f}s)")
    t0 = time.time()
    C1_perperm, _ = calibrate(Xp_cal_a, Xp_cal_b, share_g_ref=False)
    log(f"  C1_perperm   = {C1_perperm:.6f}  ({time.time()-t0:.1f}s)")
    delta_pct = 100.0 * (C1_shared - C1_perperm) / max(abs(C1_perperm), 1e-12)
    log(f"  Delta%       = {delta_pct:+.1f}%")

    log("\n=== Deployment with C1_shared ===")
    pow_s, t1_s, sigma_s = run_power_typeI(C1_shared, "shared")
    log("\n=== Deployment with C1_perperm ===")
    pow_p, t1_p, sigma_p = run_power_typeI(C1_perperm, "perperm")

    log(f"\nTotal wall time: {time.time()-t_global:.1f}s")
    with open(CSV_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'd', 'n', 'df', 'poly_degree', 'n_reps',
                    'C1_shared', 'C1_perperm', 'pct_delta_C1',
                    'sigma_med_shared', 'sigma_med_perperm',
                    'power_shared', 'power_perperm',
                    'typeI_shared', 'typeI_perperm'])
        w.writerow(['polynomial_p4_d10', DIM, N_SAMPLES, DF_H1, POLY_DEGREE, N_REPS,
                    C1_shared, C1_perperm, delta_pct,
                    sigma_s, sigma_p,
                    pow_s, pow_p, t1_s, t1_p])
    log(f"Saved {CSV_FILE}")
    log(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
