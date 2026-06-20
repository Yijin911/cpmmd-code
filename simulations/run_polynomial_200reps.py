"""
Polynomial pillar at 200 reps: kurtosis shift.

Setting where polynomial features are ESSENTIAL:
  P = N(0, I_d)
  Q = symmetric heavy-tailed (scaled multivariate t_v, matching mean and covariance)

Mean and covariance are matched, so any Gaussian-bandwidth MMD has zero population MMD
for ANY sigma. The difference is in the kurtosis (4th moment). Detecting it requires
degree-4 polynomial features (or equivalent).

This is the cleanest "polynomial > aggregation" benchmark:
  - MMDAgg with raw Gaussian/Laplacian kernels = ~zero power
  - CP-MMD with degree-4 polynomial features = should detect

Methods:
  - Lap+poly2: degree-2 polynomials (insufficient for kurtosis)
  - Lap+poly4: degree-4 polynomials (sufficient)
  - Median heuristic (Gaussian kernel, no polynomial features)
  - MMDAgg (Laplacian + Gaussian aggregation, no polynomial features)

Sweep: df in {5, 8, 12, 20} (lower df = heavier tail = easier to detect)
"""

import os, sys, time, csv, warnings
import numpy as np
from datetime import datetime
from itertools import combinations_with_replacement

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import compute_mmd_u_statistic, optimize_bandwidth, _DEVICE

try:
    from mmdagg import mmdagg as mmdagg_fn
    HAS_AGG = True
except ImportError:
    HAS_AGG = False

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'polynomial_200reps')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 200
N_PERM = 200
N_SAMPLES = 500
D = 10  # dimension


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def generate_kurtosis_data(n, d, df=None, seed=None):
    """
    P = N(0, I_d)
    Q = scaled multivariate t_df (mean=0, Cov=I_d).
        Scaled so Cov matches: if Z ~ t_df(0, I_d), then Cov(Z) = df/(df-2)*I_d.
        We scale: Q = Z * sqrt((df-2)/df) so Cov(Q) = I_d.
        If df is None (H0), Q = N(0, I_d) also.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    if df is None:
        Y = rng.standard_normal((n, d)).astype(np.float32)
    else:
        # Multivariate t: Z = W / sqrt(chi2/df), W ~ N(0, I_d)
        W = rng.standard_normal((n, d))
        chi2 = rng.chisquare(df, size=(n, 1))
        Z = W / np.sqrt(chi2 / df)
        # Scale to match covariance
        scale = np.sqrt((df - 2.0) / df)
        Y = (Z * scale).astype(np.float32)
    return X, Y


def poly_features(X, degree):
    """Compute polynomial features up to given degree."""
    n, d = X.shape
    if degree == 2:
        pairs = list(combinations_with_replacement(range(d), 2))
        out = np.empty((n, len(pairs)), dtype=X.dtype)
        for k, (i, j) in enumerate(pairs):
            out[:, k] = X[:, i] * X[:, j]
        return np.hstack([X, out])
    elif degree == 4:
        # degree 2
        pairs2 = list(combinations_with_replacement(range(d), 2))
        feat2 = np.empty((n, len(pairs2)), dtype=X.dtype)
        for k, (i, j) in enumerate(pairs2):
            feat2[:, k] = X[:, i] * X[:, j]
        # degree 3
        trips = list(combinations_with_replacement(range(d), 3))
        feat3 = np.empty((n, len(trips)), dtype=X.dtype)
        for k, tup in enumerate(trips):
            v = np.ones(n, dtype=X.dtype)
            for i in tup:
                v = v * X[:, i]
            feat3[:, k] = v
        # degree 4
        quads = list(combinations_with_replacement(range(d), 4))
        feat4 = np.empty((n, len(quads)), dtype=X.dtype)
        for k, tup in enumerate(quads):
            v = np.ones(n, dtype=X.dtype)
            for i in tup:
                v = v * X[:, i]
            feat4[:, k] = v
        return np.hstack([X, feat2, feat3, feat4])
    else:
        return X


def median_pairwise(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    n = len(Z)
    subset = Z[np.random.choice(n, min(n, 500), replace=False)]
    diffs = subset[:, None, :] - subset[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(len(subset), k=1)
    return float(np.median(dists[iu]))


def perm_test(X, Y, sigma, kernel='gaussian', n_perm=N_PERM, seed=None):
    rng = np.random.default_rng(seed)
    obs = float(compute_mmd_u_statistic(X, Y, kernel_type=kernel, sigma=sigma))
    pooled = np.concatenate([X, Y], axis=0)
    m, n = len(X), len(Y)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(m + n)
        null[k] = compute_mmd_u_statistic(pooled[perm[:m]], pooled[perm[m:]],
                                           kernel_type=kernel, sigma=sigma)
    thr = float(np.percentile(null, (1 - ALPHA) * 100))
    return int(obs > thr)


def run_median(X, Y, seed=None):
    sigma = max(median_pairwise(X, Y), 1e-6)
    return perm_test(X, Y, sigma, seed=seed)


def run_bw_select(X, Y, kernel='gaussian', seed=None):
    n = len(X) // 2
    Xtr, Ytr = X[:n], Y[:n]
    Xte, Yte = X[n:], Y[n:]
    res = optimize_bandwidth(Xtr, Ytr, kernel_type=kernel, search_space=(0.01, 50.0))
    return perm_test(Xte, Yte, res['optimal_sigma'], kernel=kernel, seed=seed)


def run_poly_select(X, Y, degree, kernel='laplacian', seed=None):
    Xp = poly_features(X, degree)
    Yp = poly_features(Y, degree)
    n = len(Xp) // 2
    Xtr, Ytr = Xp[:n], Yp[:n]
    Xte, Yte = Xp[n:], Yp[n:]
    res = optimize_bandwidth(Xtr, Ytr, kernel_type=kernel, search_space=(0.01, 100.0))
    return perm_test(Xte, Yte, res['optimal_sigma'], kernel=kernel, seed=seed)


def run_mmdagg(X, Y, seed=None):
    if not HAS_AGG:
        return None
    return int(mmdagg_fn(X, Y, alpha=ALPHA, kernel='laplace_gaussian',
                         number_bandwidths=10,
                         permutations_same_sample_size=True, seed=seed))


def main():
    log("=" * 70)
    log("KURTOSIS PILLAR: polynomial features vs. bandwidth aggregation")
    log(f"d={D}, n={N_SAMPLES}, reps={N_REPS}")
    log(f"Device: {_DEVICE}")
    log("=" * 70)

    df_values = [None, 5, 8, 12, 20]  # None = H0
    methods = [
        ('Median', run_median),
        ('BW_select', run_bw_select),
        ('Lap+poly2', lambda X, Y, seed=None: run_poly_select(X, Y, 2, seed=seed)),
        ('Lap+poly4', lambda X, Y, seed=None: run_poly_select(X, Y, 4, seed=seed)),
    ]
    if HAS_AGG:
        methods.append(('MMDAgg', run_mmdagg))

    all_rows = []
    for df in df_values:
        is_null = (df is None)
        label = "TYPE_I" if is_null else f"POWER (df={df})"
        log(f"\n--- {label} ---")

        for mname, mfn in methods:
            t0 = time.time()
            count, valid = 0, 0
            for rep in range(N_REPS):
                seed = rep * 1234 + (0 if df is None else df * 100)
                n_total = N_SAMPLES * 2  # for split methods
                X, Y = generate_kurtosis_data(n_total, D, df=df, seed=seed)
                try:
                    r = mfn(X, Y, seed=seed)
                    if r is not None:
                        count += int(r)
                        valid += 1
                except Exception:
                    pass
            dt = time.time() - t0
            rate = count / valid if valid > 0 else 0
            log(f"  {mname:15s}: {rate:.3f} ({valid}/{N_REPS} valid, {dt:.1f}s)")
            all_rows.append({
                'df': 'None' if df is None else df,
                'hypothesis': 'H0' if is_null else 'H1',
                'method': mname,
                'rate': rate,
                'n_valid': valid,
                'n_reps': N_REPS,
            })

    out = os.path.join(RESULTS_DIR, 'kurtosis_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    log(f"\nSaved {out}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
