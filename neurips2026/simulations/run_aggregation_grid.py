"""
MMDAgg-cannot-grid experiment: as the bandwidth grid coarsens, MMDAgg's power
degrades because the optimal bandwidth falls between grid points; CP-MMD with
continuous bandwidth selection holds power throughout.

Setting: 1D multi-scale mixture
  P = 0.5 * N(0, sigma_a^2) + 0.5 * N(0, sigma_b^2)
  Q = 0.5 * N(mu, sigma_a^2) + 0.5 * N(-mu, sigma_b^2)
with sigma_a = 0.5, sigma_b = 5.0, mu = 0.6, n=200, alpha=0.05, 200 reps.

The optimal Gaussian bandwidth on this task is around sigma_* \approx 1.4 --
deliberately irrational w.r.t. MMDAgg's geometric grid {2^{k/2} : k=...}.

Methods:
  - MMDAgg with B \in {3, 5, 10, 20} bandwidths in [0.5, 5.0]
  - CP-MMD with continuous bandwidth selection in [0.1, 10.0]
  - Plain (no penalty) baseline at the same continuous range

Hypothesis:
  - MMDAgg power monotone increases in B but plateaus at the per-bandwidth limit.
  - CP-MMD power matches MMDAgg-large-B and exceeds MMDAgg-small-B.
  - The gap (CP-MMD - MMDAgg) at small B quantifies the cost of grid restriction.
"""

import os, sys, time, csv, warnings
import numpy as np
from datetime import datetime
from scipy.optimize import minimize_scalar

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import compute_mmd_u_statistic

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'aggregation_grid')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 200
N_PERM = 200
N = 200
SIGMA_A = 0.5
SIGMA_B = 5.0
MU = 0.6
GRID_SIZES = [3, 5, 10, 20]
SIGMA_RANGE = (0.5, 5.0)
SIGMA_RANGE_CONT = (0.1, 10.0)
N_CAL = 5


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def gen(n, mu, seed):
    rng = np.random.default_rng(seed)
    # P: mixture at zero with two scales
    comp = rng.integers(0, 2, size=n)
    sigmas_p = np.where(comp == 0, SIGMA_A, SIGMA_B)
    X = rng.standard_normal(n) * sigmas_p
    # Q: same scales but opposite-sign means tied to scale
    comp = rng.integers(0, 2, size=n)
    sigmas_q = np.where(comp == 0, SIGMA_A, SIGMA_B)
    means = np.where(comp == 0, mu, -mu)
    Y = rng.standard_normal(n) * sigmas_q + means
    return X.reshape(-1, 1).astype(np.float32), Y.reshape(-1, 1).astype(np.float32)


def geom_grid(lo, hi, B):
    return np.geomspace(lo, hi, B)


def perm_test_at_sigma(X, Y, sigma, seed):
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    m = len(X)
    obs = compute_mmd_u_statistic(X, Y, 'gaussian', sigma)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = rng.permutation(2 * m)
        null[i] = compute_mmd_u_statistic(Z[perm[:m]], Z[perm[m:]], 'gaussian', sigma)
    thr = float(np.percentile(null, (1 - ALPHA) * 100))
    return obs, null, thr


def mmdagg_test(X, Y, B, seed):
    """Bonferroni-style aggregation over a B-point geometric grid."""
    sigmas = geom_grid(SIGMA_RANGE[0], SIGMA_RANGE[1], B)
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    m = len(X)
    obs_per_b = np.empty(B)
    null_per_b = np.empty((B, N_PERM))
    for j, sig in enumerate(sigmas):
        obs_per_b[j] = compute_mmd_u_statistic(X, Y, 'gaussian', sig)
        for i in range(N_PERM):
            perm = rng.permutation(2 * m)
            null_per_b[j, i] = compute_mmd_u_statistic(
                Z[perm[:m]], Z[perm[m:]], 'gaussian', sig)
    # Per-kernel rejection at level alpha/B (Bonferroni)
    thrs = np.percentile(null_per_b, (1 - ALPHA / B) * 100, axis=1)
    return int(np.any(obs_per_b > thrs))


def cpmmd_test(X, Y, C1, seed):
    """Continuous bandwidth selection: maximise MMD - C1 * G_tilde over sigma in [0.1,10],
       then permutation-test at the selected sigma using a held-out half."""
    n_half = len(X) // 2
    Xtr, Xte = X[:n_half], X[n_half:]
    Ytr, Yte = Y[:n_half], Y[n_half:]
    Z_tr = np.concatenate([Xtr, Ytr], axis=0)
    data_norm_tr = np.linalg.norm(Z_tr) / np.sqrt(len(Z_tr))

    def objective(sig):
        m2 = compute_mmd_u_statistic(Xtr, Ytr, 'gaussian', sig)
        # G_tilde for a fixed-bandwidth Gaussian: data_norm / sigma
        G = data_norm_tr / max(sig, 1e-6)
        return -(m2 - C1 * G)

    res = minimize_scalar(objective, bounds=SIGMA_RANGE_CONT, method='bounded',
                          options={'xatol': 1e-3})
    sig_star = res.x
    obs, null, thr = perm_test_at_sigma(Xte, Yte, sig_star, seed)
    return int(obs > thr), sig_star


def plain_test(X, Y, seed):
    n_half = len(X) // 2
    Xtr, Xte = X[:n_half], X[n_half:]
    Ytr, Yte = Y[:n_half], Y[n_half:]

    def objective(sig):
        return -compute_mmd_u_statistic(Xtr, Ytr, 'gaussian', sig)

    res = minimize_scalar(objective, bounds=SIGMA_RANGE_CONT, method='bounded',
                          options={'xatol': 1e-3})
    sig_star = res.x
    obs, null, thr = perm_test_at_sigma(Xte, Yte, sig_star, seed)
    return int(obs > thr), sig_star


def calibrate_c1(seed):
    """Calibrate \widehat C_1 via null permutations on a single (P,P) split."""
    rng = np.random.default_rng(seed + 99999)
    ratios = []
    for i in range(N_CAL):
        X, _ = gen(2 * N, MU, seed=seed + 100 + i)
        Z = X.copy()
        # Treat as a single-population sample, split arbitrarily.
        rng.shuffle(Z)
        Xa, Xb = Z[:N], Z[N:2 * N]
        n_half = N // 2
        Xa_tr, Xa_te = Xa[:n_half], Xa[n_half:]
        Xb_tr, Xb_te = Xb[:n_half], Xb[n_half:]
        data_norm = np.linalg.norm(np.concatenate([Xa_tr, Xb_tr])) / np.sqrt(2 * n_half)

        def neg_mmd(sig):
            return -compute_mmd_u_statistic(Xa_tr, Xb_tr, 'gaussian', sig)
        res = minimize_scalar(neg_mmd, bounds=SIGMA_RANGE_CONT, method='bounded',
                              options={'xatol': 1e-3})
        sig_star = res.x
        m2 = compute_mmd_u_statistic(Xa_te, Xb_te, 'gaussian', sig_star)
        G = data_norm / max(sig_star, 1e-8)
        if G > 1e-10:
            ratios.append(abs(m2) / G)
    if not ratios:
        return 0.01
    return float(np.percentile(ratios, 95))


def run_one_rep(rep, C1):
    seed = rep * 17
    X, Y = gen(N, MU, seed=seed)
    out = {'rep': rep}
    for B in GRID_SIZES:
        out[f'mmdagg_B{B}'] = mmdagg_test(X, Y, B, seed=seed + 1)
    rej_cp, sig_cp = cpmmd_test(X, Y, C1, seed=seed + 2)
    rej_pl, sig_pl = plain_test(X, Y, seed=seed + 3)
    out['cpmmd'] = rej_cp
    out['cpmmd_sigma'] = sig_cp
    out['plain'] = rej_pl
    out['plain_sigma'] = sig_pl
    return out


def main():
    log("=" * 70)
    log(f"MMDAgg-cannot-grid: n={N}, mu={MU}, sigma_a={SIGMA_A}, sigma_b={SIGMA_B}")
    log(f"Grids: {GRID_SIZES}, n_reps={N_REPS}")
    log("=" * 70)

    log("Calibrating C1 via null permutations...")
    C1 = calibrate_c1(seed=0)
    if C1 < 1e-5:
        C1 = 0.01
    log(f"\widehat C_1 = {C1:.5f}")

    rows = []
    t0 = time.time()
    for rep in range(N_REPS):
        rows.append(run_one_rep(rep, C1))
        if (rep + 1) % 20 == 0:
            log(f"  rep {rep+1}/{N_REPS} ({time.time()-t0:.0f}s)")

    out = os.path.join(RESULTS_DIR, 'aggregation_grid_results.csv')
    keys = list(rows[0].keys())
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"Saved {out}")

    # Summary
    summary = {'C1': C1, 'n_reps': N_REPS}
    for B in GRID_SIZES:
        summary[f'power_mmdagg_B{B}'] = float(np.mean([r[f'mmdagg_B{B}'] for r in rows]))
    summary['power_cpmmd'] = float(np.mean([r['cpmmd'] for r in rows]))
    summary['power_plain'] = float(np.mean([r['plain'] for r in rows]))
    summary['cpmmd_sigma_med'] = float(np.median([r['cpmmd_sigma'] for r in rows]))

    out2 = os.path.join(RESULTS_DIR, 'aggregation_grid_summary.csv')
    with open(out2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    log(f"Saved {out2}")
    log("\nPower summary:")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
