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
import torch
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
N_REPS = int(os.environ.get('N_REPS', 200))
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


# ----------------------------------------------------------------------
# CP-MMD on the polynomial class h_sigma(x) = Psi_p(x) / sigma.
# J_CP(sigma) = gamma_hat^2(sigma) - C1_hat * G(sigma), with
# G(sigma) = ||Psi_p(D_tr)||_F / (sigma * N_tr). C1_hat is calibrated on the
# training half only: for each of N_CAL permutations of the pooled training
# half, take the largest ratio gamma_hat^2(sigma) / G(sigma) over the sigma grid,
# then the 0.95 quantile of these ratios (same rule as run_linear_pillar.py).
# The kernel matrix is computed once per half and re-indexed for permutations.
# ----------------------------------------------------------------------
# Search interval [0.01, 100] as stated in the paper. L1 distances between
# degree-4 feature vectors are ~1e3 (median) at d = 10, so the selected sigma
# sits at the upper end of this interval; SIGMA_MAX / SIGMA_POINTS widen it.
SIGMA_GRID_POLY = np.geomspace(0.01, float(os.environ.get('SIGMA_MAX', 100.0)),
                                int(os.environ.get('SIGMA_POINTS', 48)))
N_CAL = 10
# CAL_RULE='sup': ratio = max over the grid of gamma_hat^2(sigma) / G(sigma).
# CAL_RULE='plainmax': ratio at the plain-MMD maximizer sigma_pi, as in Eq. (11).
CAL_RULE = os.environ.get('CAL_RULE', 'sup')


def _mmd_u_from_K(K, ix, iy):
    """Unbiased MMD^2 for every kernel in the stack K of shape (S, N, N)."""
    Kxx = K[:, ix][:, :, ix]
    Kyy = K[:, iy][:, :, iy]
    Kxy = K[:, ix][:, :, iy]
    m, n = len(ix), len(iy)
    xx = (Kxx.sum((1, 2)) - Kxx.diagonal(dim1=1, dim2=2).sum(1)) / (m * (m - 1))
    yy = (Kyy.sum((1, 2)) - Kyy.diagonal(dim1=1, dim2=2).sum(1)) / (n * (n - 1))
    xy = Kxy.sum((1, 2)) / (m * n)
    return xx + yy - 2.0 * xy


def _laplacian_stack(P, sigmas):
    """Laplacian kernels exp(-||x - y||_1 / sigma) for all sigmas, shape (S, N, N)."""
    D1 = torch.cdist(P, P, p=1)
    return torch.exp(-D1[None] / sigmas[:, None, None])


def run_poly_jcp(X, Y, degree=4, seed=None):
    Xp = poly_features(X, degree)
    Yp = poly_features(Y, degree)
    n = len(Xp) // 2
    Xtr, Ytr = Xp[:n], Yp[:n]
    Xte, Yte = Xp[n:], Yp[n:]

    # Calibrate C1_hat on the training half.
    P_tr = torch.as_tensor(np.vstack([Xtr, Ytr]), dtype=torch.float32, device=_DEVICE)
    N_tr = P_tr.shape[0]
    sigmas = torch.as_tensor(SIGMA_GRID_POLY, dtype=torch.float32, device=_DEVICE)
    K_tr = _laplacian_stack(P_tr, sigmas)
    G = float(torch.linalg.norm(P_tr)) / (sigmas * N_tr)
    rng_cal = np.random.default_rng(None if seed is None else seed + 1)
    ratios = []
    for _ in range(N_CAL):
        perm = torch.as_tensor(rng_cal.permutation(N_tr), device=_DEVICE)
        mmd = _mmd_u_from_K(K_tr, perm[:n], perm[n:]).clamp(min=0.0)
        if CAL_RULE == 'plainmax':
            k = int(torch.argmax(mmd))
            ratios.append(float(mmd[k] / G[k]))
        else:
            ratios.append(float((mmd / G).max()))
    C1 = float(np.quantile(ratios, 0.95))

    # Select sigma by J_CP on the training half.
    idx = torch.arange(N_tr, device=_DEVICE)
    J = _mmd_u_from_K(K_tr, idx[:n], idx[n:]) - C1 * G
    sigma_sel = float(SIGMA_GRID_POLY[int(torch.argmax(J))])

    # Permutation test on the held-out half (same rule as perm_test).
    P_te = torch.as_tensor(np.vstack([Xte, Yte]), dtype=torch.float32, device=_DEVICE)
    m_te = len(Xte)
    K_te = _laplacian_stack(P_te, torch.tensor([sigma_sel], device=_DEVICE))
    idx_te = torch.arange(P_te.shape[0], device=_DEVICE)
    obs = float(_mmd_u_from_K(K_te, idx_te[:m_te], idx_te[m_te:])[0])
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        perm = torch.as_tensor(rng.permutation(P_te.shape[0]), device=_DEVICE)
        null[k] = float(_mmd_u_from_K(K_te, perm[:m_te], perm[m_te:])[0])
    thr = float(np.percentile(null, (1 - ALPHA) * 100))
    JCP_DIAG.append((C1, sigma_sel))
    return int(obs > thr)


JCP_DIAG = []


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

    # DFS="None,5" restricts the sweep (for running dfs in parallel processes).
    df_env = os.environ.get('DFS')
    df_values = ([None if v == 'None' else int(v) for v in df_env.split(',')]
                 if df_env else [None, 5, 8, 12, 20])  # None = H0
    # Every method receives the same N_SAMPLES per group. CP-MMD selects on
    # the first half and tests on the second half; Median and MMDAgg use all.
    methods = [
        ('Median', run_median),
        ('CP-MMD (Lap+poly4)', lambda X, Y, seed=None: run_poly_jcp(X, Y, 4, seed=seed)),
    ]
    if HAS_AGG:
        methods.append(('MMDAgg', run_mmdagg))
    # METHODS="CP-MMD" runs only the methods whose names start with a listed prefix.
    if os.environ.get('METHODS'):
        keep = os.environ['METHODS'].split(',')
        methods = [(nm, fn) for nm, fn in methods if any(nm.startswith(k) for k in keep)]

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
                X, Y = generate_kurtosis_data(N_SAMPLES, D, df=df, seed=seed)
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
            if mname.startswith('CP-MMD') and JCP_DIAG:
                c1s, sigs = zip(*JCP_DIAG)
                log(f"    median C1_hat {np.median(c1s):.3g}, median sigma_sel {np.median(sigs):.3g}")
                JCP_DIAG.clear()
            all_rows.append({
                'df': 'None' if df is None else df,
                'hypothesis': 'H0' if is_null else 'H1',
                'method': mname,
                'rate': rate,
                'n_valid': valid,
                'n_reps': N_REPS,
            })

    out = os.path.join(RESULTS_DIR, f"kurtosis_results{os.environ.get('OUT_SUFFIX', '')}.csv")
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    log(f"\nSaved {out}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
