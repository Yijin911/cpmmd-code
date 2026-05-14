"""Calibration-ablation row: linear bandwidth selection at Delta=0.30, n=200.

§5.2 head-line row, journal version with per-h proxy.

The CP-MMD selection now uses the per-h penalty
    J_CP(sigma) = MMD^2(sigma) - C1 * ||D||_F / (sigma * N),
so the calibrated C1 actively moves the selected sigma. We compare two
procedures:
  shared:   single G_ref computed once on the original data
  per_perm: G_ref recomputed on every calibration permutation
Both calibrate the same C1 functional (1-alpha quantile of plain-MMD
argmax / G_ref ratios); they differ only in whether G_ref is shared.

Reports: (C1_shared, C1_perperm, median sigma at each procedure, power
at each procedure, type-I at each procedure).
"""

import os, sys, time, csv, warnings
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

DIM = 2
N_SAMPLES = 200
DELTA_H1 = 0.30
N_REPS = 100
N_PERM = 200
N_CAL = 10
ALPHA = 0.05
BW_LOW, BW_HIGH = 0.05, 5.0
KERNEL = 'gaussian'

LOG_FILE = os.path.join(RESULTS_DIR, 'linear_log.txt')
CSV_FILE = os.path.join(RESULTS_DIR, 'linear.csv')
log_fh = None


def log(msg):
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + '\n')
        log_fh.flush()


def sample_multiscale(n, delta, rng):
    cluster_std = 0.1
    cluster_sep = 3.0

    def draw(offset):
        labels = rng.integers(0, 2, size=n)
        centers = np.zeros((n, DIM))
        centers[labels == 1, 0] = cluster_sep
        return centers + cluster_std * rng.standard_normal((n, DIM)) + offset

    P = draw(np.zeros(DIM))
    shift = np.zeros(DIM); shift[0] = delta
    Q = draw(shift)
    return P.astype(np.float32), Q.astype(np.float32)


def g_ref(pooled, sigma):
    """G_T(h_sigma) = (1/sigma) * ||D||_F / N for the linear/bandwidth case."""
    N = pooled.shape[0]
    return (1.0 / max(sigma, 1e-6)) * float(np.linalg.norm(pooled)) / N


def calibrate(X_null, Y_null, share_g_ref, n_cal=N_CAL, seed=42):
    """C1 = (1-alpha)-quantile of mmd^2 / G_ref under null permutations.

    Shared: G_ref evaluated at the worst-case Lipschitz envelope
            L^* = 1/sigma_l (one scalar, reused across all permutations).
    Per-perm: G_ref evaluated at each permutation's empirical argmax sigma_pi.
    """
    rng = np.random.default_rng(seed)
    m, n = len(X_null), len(Y_null)
    N = m + n
    pooled = np.concatenate([X_null, Y_null], axis=0)

    if share_g_ref:
        # Worst-case envelope: L^* = 1/sigma_l
        G_ref_fixed = (1.0 / BW_LOW) * float(np.linalg.norm(pooled)) / N
    else:
        G_ref_fixed = None

    ratios = []
    for _ in range(n_cal):
        perm = rng.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        opt = optimize_bandwidth_plain(X_p, Y_p, KERNEL, (BW_LOW, BW_HIGH))
        mmd_p = opt['max_mmd']
        if share_g_ref:
            G = G_ref_fixed
        else:
            G = g_ref(np.concatenate([X_p, Y_p], axis=0), opt['optimal_sigma'])
        ratios.append(mmd_p / max(abs(G), 1e-12))
    C1 = float(np.percentile(ratios, (1 - ALPHA) * 100))
    return C1, G_ref_fixed


def permsplit_test(X, Y, C1, seed):
    n_half = len(X) // 2
    X_tr, Y_tr = X[:n_half], Y[:n_half]
    X_te, Y_te = X[n_half:], Y[n_half:]
    pooled_tr = np.concatenate([X_tr, Y_tr], axis=0)
    norm_tr = float(np.linalg.norm(pooled_tr))
    opt = optimize_bandwidth_jcp(X_tr, Y_tr, C1, KERNEL,
                                  (BW_LOW, BW_HIGH),
                                  data_norm_F=norm_tr)
    sigma = opt['optimal_sigma']
    rng = np.random.default_rng(seed)
    rej = permutation_pvalue(X_te, Y_te, sigma, KERNEL,
                             n_perm=N_PERM, alpha=ALPHA, rng=rng)
    return rej, sigma


def run_power_typeI(C1, label, t_global):
    rng = np.random.default_rng(42)
    X_cal, _ = sample_multiscale(2 * N_SAMPLES, 0.0, rng)
    # H1
    h1, sigs_h1 = 0, []
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 7919
        rng = np.random.default_rng(seed)
        X, Y = sample_multiscale(2 * N_SAMPLES, DELTA_H1, rng)
        rej, sigma = permsplit_test(X, Y, C1, seed)
        h1 += rej
        sigs_h1.append(sigma)
    power = h1 / N_REPS
    log(f"  [{label}] POWER = {power:.3f} ({h1}/{N_REPS})  median sigma = {np.median(sigs_h1):.4f}  ({time.time()-t0:.1f}s)")

    # Type-I
    h0 = 0
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 7919 + 1
        rng = np.random.default_rng(seed)
        X, Y = sample_multiscale(2 * N_SAMPLES, 0.0, rng)
        rej, _ = permsplit_test(X, Y, C1, seed)
        h0 += rej
    type1 = h0 / N_REPS
    log(f"  [{label}] TYPE-I = {type1:.3f} ({h0}/{N_REPS})  ({time.time()-t0:.1f}s)")
    return power, type1, float(np.median(sigs_h1))


def main():
    global log_fh
    log_fh = open(LOG_FILE, 'w', buffering=1)
    log(f"Ablation linear (multi-scale 2D, Delta={DELTA_H1}, n={N_SAMPLES}) [per-h proxy]")
    log(f"Device: {_DEVICE}")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_global = time.time()

    rng = np.random.default_rng(42)
    X_cal, _ = sample_multiscale(2 * N_SAMPLES, 0.0, rng)
    X_cal_a = X_cal[:N_SAMPLES]
    X_cal_b = X_cal[N_SAMPLES:]

    log("\n=== Calibration ===")
    t0 = time.time()
    C1_shared, G_shared = calibrate(X_cal_a, X_cal_b, share_g_ref=True)
    log(f"  C1_shared    = {C1_shared:.6f}  (G_ref = {G_shared:.6f}, worst-case L*=1/sigma_l={1.0/BW_LOW:.2f}, {time.time()-t0:.1f}s)")
    t0 = time.time()
    C1_perperm, _ = calibrate(X_cal_a, X_cal_b, share_g_ref=False)
    log(f"  C1_perperm   = {C1_perperm:.6f}  ({time.time()-t0:.1f}s)")
    delta_pct = 100.0 * (C1_shared - C1_perperm) / max(abs(C1_perperm), 1e-12)
    log(f"  Delta%       = {delta_pct:+.1f}%")

    log("\n=== Deployment with C1_shared ===")
    pow_s, t1_s, sigma_s = run_power_typeI(C1_shared, "shared", t_global)
    log("\n=== Deployment with C1_perperm ===")
    pow_p, t1_p, sigma_p = run_power_typeI(C1_perperm, "perperm", t_global)

    log(f"\nTotal wall time: {time.time()-t_global:.1f}s")
    with open(CSV_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'd', 'n', 'delta', 'n_reps',
                    'C1_shared', 'C1_perperm', 'pct_delta_C1',
                    'sigma_med_shared', 'sigma_med_perperm',
                    'power_shared', 'power_perperm',
                    'typeI_shared', 'typeI_perperm'])
        w.writerow(['linear_multiscale_d2', DIM, N_SAMPLES, DELTA_H1, N_REPS,
                    C1_shared, C1_perperm, delta_pct,
                    sigma_s, sigma_p,
                    pow_s, pow_p,
                    t1_s, t1_p])

    log(f"Saved {CSV_FILE}")
    log(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
