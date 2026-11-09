"""Higgs real-data, v2: linear/poly2/poly3 with proper J_CP penalty, GPU.

v2 changes vs run_higgs_extra_gpu.py:
  - Adds CPMMD-poly3 (degree-3 monomial cross features, top-20 by variance).
  - Adds *both* PermSplit and full-sample variants for linear/poly2/poly3, so we
    can isolate whether PermSplit's half-sample cost is the dominant driver of
    the gap to Median/MMDAgg on Higgs.
  - n_cal raised from 10 to 30 for stable Ĉ₁ (95th percentile from 30 ratios).
  - Diagnostic logging per rep for the FIRST 5 H1 reps of every method:
      - calibrated Ĉ₁, selected σ, J_CP value at selection, mmd² at selection,
        whether σ_sel hits the grid boundary (selection-saturation flag).
  - Sigma grids unchanged (linear [2,20], poly [10,100]); 24-point geometric.

Camera-ready changes:
  - Inputs are standardized on the pooled loaded rows, exactly as in run_higgs.py,
    so every Higgs method in Fig. 3(a) sees the same features. The polynomial
    ansatze use the 20 columns with the largest raw variance, fixed once.
  - Each (n, method) has its own seed and each repetition passes its generator to
    the split, the calibration and the permutation test, so runs are reproducible
    and METHODS can be split across processes without changing the numbers.
  - MODES (default permsplit) selects the protocol; the paper uses permsplit.

Usage: [MODES=permsplit] [METHODS=CPMMD-poly3] [OUT_SUFFIX=_poly3] python run_higgs_extra_gpu.py
"""
from __future__ import annotations
import os
import csv
import time
import gzip
import urllib.request

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
RESULTS_DIR = os.path.join(PROJECT, 'results', 'higgs_extra_v2')
DATA_DIR = os.path.join(PROJECT, 'data', 'higgs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

ALPHA = 0.05
N_REPS = 50
N_PERM = 200
N_CAL = 30  # was 10
SAMPLE_SIZES = [200, 500]
MAX_ROWS = 100000
SEED = 42
N_DIAG_REPS = 5  # log diagnostics for first 5 H1 reps per method
MODES = os.environ.get('MODES', 'permsplit').split(',')
METHODS = [m for m in os.environ.get('METHODS', '').split(',') if m]
OUT_SUFFIX = os.environ.get('OUT_SUFFIX', '')
VAR_IDX = None  # columns of the polynomial ansatze, set in main()

LOG_PATH = os.path.join(RESULTS_DIR, f'run_log_gpu{OUT_SUFFIX}.txt')
DIAG_PATH = os.path.join(RESULTS_DIR, f'diagnostics{OUT_SUFFIX}.csv')


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


HIGGS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'


def download_higgs():
    gz = os.path.join(DATA_DIR, 'HIGGS.csv.gz')
    if not os.path.exists(gz):
        log(f'  Downloading HIGGS to {gz} ...')
        urllib.request.urlretrieve(HIGGS_URL, gz)
    return gz


def load_higgs(max_rows=MAX_ROWS):
    cache = os.path.join(DATA_DIR, f'higgs_{max_rows}.npz')
    if os.path.exists(cache):
        log(f'  Loading cached {cache}')
        d = np.load(cache)
        return d['signal'], d['background']
    gz = download_higgs()
    log(f'  Parsing {gz} ...')
    sig, bg = [], []
    with gzip.open(gz, 'rt') as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            parts = line.strip().split(',')
            label = float(parts[0])
            feats = np.array([float(x) for x in parts[1:29]])
            (sig if label == 1.0 else bg).append(feats)
    X_sig = np.array(sig)
    X_bg = np.array(bg)
    np.savez(cache, signal=X_sig, background=X_bg)
    log(f'  Loaded {len(X_sig)} signal, {len(X_bg)} background')
    return X_sig, X_bg


# ----------------------------------------------------------------------
# GPU kernels
# ----------------------------------------------------------------------
def sq_dists_torch(X, Y):
    x2 = (X * X).sum(1).unsqueeze(1)
    y2 = (Y * Y).sum(1).unsqueeze(0)
    return (x2 + y2 - 2 * X @ Y.t()).clamp_min(0.0)


def gaussian_kernel_torch(X, Y, sigma):
    return torch.exp(-sq_dists_torch(X, Y) / (2.0 * sigma * sigma))


def laplacian_kernel_torch(X, Y, sigma):
    return torch.exp(-torch.sqrt(sq_dists_torch(X, Y)) / sigma)


def mmd2_unbiased_torch(K_xx, K_yy, K_xy):
    m = K_xx.shape[0]
    n = K_yy.shape[0]
    sum_xx = (K_xx.sum() - K_xx.diagonal().sum()) / (m * (m - 1))
    sum_yy = (K_yy.sum() - K_yy.diagonal().sum()) / (n * (n - 1))
    sum_xy = K_xy.sum() / (m * n)
    return sum_xx + sum_yy - 2 * sum_xy


def mmd2_at_torch(X, Y, kernel_fn, sigma):
    K_xx = kernel_fn(X, X, sigma)
    K_yy = kernel_fn(Y, Y, sigma)
    K_xy = kernel_fn(X, Y, sigma)
    return mmd2_unbiased_torch(K_xx, K_yy, K_xy)


# ----------------------------------------------------------------------
# Polynomial-feature transforms with FIXED column indices
# ----------------------------------------------------------------------
def poly2_features_np(X, var_idx):
    d = X.shape[1]
    cross = []
    for i in var_idx:
        for j in range(d):
            if j != i:
                cross.append(X[:, i] * X[:, j])
    cross = np.stack(cross, axis=1) if cross else np.zeros((X.shape[0], 0))
    return np.hstack([X, cross]).astype(np.float32)


def poly3_features_np(X, var_idx):
    """Degree-3 monomials: original d features + degree-2 cross + degree-3 cross.

    Degree-3 cross: for each (i, j, k) in var_idx triples (no repeats), x_i x_j x_k.
    With |var_idx|=20 and triples chosen from var_idx itself, we get C(20,3)=1140
    triple terms, on top of the ~568 degree-2 features. Total ~1736 features.
    """
    d = X.shape[1]
    feats = [X]
    # degree-2 cross
    deg2 = []
    for i in var_idx:
        for j in range(d):
            if j != i:
                deg2.append(X[:, i] * X[:, j])
    if deg2:
        feats.append(np.stack(deg2, axis=1))
    # degree-3 cross over triples in var_idx
    deg3 = []
    var_list = list(var_idx)
    for ai in range(len(var_list)):
        for bi in range(ai + 1, len(var_list)):
            for ci in range(bi + 1, len(var_list)):
                a, b, c = var_list[ai], var_list[bi], var_list[ci]
                deg3.append(X[:, a] * X[:, b] * X[:, c])
    if deg3:
        feats.append(np.stack(deg3, axis=1))
    return np.hstack(feats).astype(np.float32)


# ----------------------------------------------------------------------
# CP-MMD calibration / selection / perm test
# ----------------------------------------------------------------------
def calibrate_C1_torch(X_train, Y_train, kernel_fn, sigma_grid, G_fn, n_perm=N_CAL, rng=None):
    """C1 = 95th percentile of (max_sigma mmd^2/G) over null permutations."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    m = X_train.shape[0]
    n = Y_train.shape[0]
    pooled = torch.cat([X_train, Y_train], dim=0)
    ratios = []
    for _ in range(n_perm):
        perm = torch.tensor(rng.permutation(m + n), device=DEVICE, dtype=torch.long)
        Xp = pooled[perm[:m]]
        Yp = pooled[perm[m:]]
        best = -float('inf')
        for s in sigma_grid:
            mmd2 = max(mmd2_at_torch(Xp, Yp, kernel_fn, s).item(), 0.0)
            G = G_fn(s)
            r = mmd2 / G if G > 0 else 0.0
            if r > best:
                best = r
        ratios.append(max(best, 0.0))
    return float(np.quantile(ratios, 0.95))


def cpmmd_select_torch(X_train, Y_train, kernel_fn, sigma_grid, C1, G_fn):
    """Returns (best_sigma, best_J, mmd2_at_best)."""
    best_sigma = sigma_grid[0]
    best_J = -float('inf')
    best_mmd2 = 0.0
    for s in sigma_grid:
        mmd2 = mmd2_at_torch(X_train, Y_train, kernel_fn, s).item()
        J = mmd2 - C1 * G_fn(s)
        if J > best_J:
            best_J = J
            best_sigma = s
            best_mmd2 = mmd2
    return best_sigma, best_J, best_mmd2


def perm_test_torch(X, Y, kernel_fn, sigma, n_perm=N_PERM, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    m, n = X.shape[0], Y.shape[0]
    pooled = torch.cat([X, Y], dim=0)
    test_stat = mmd2_at_torch(X, Y, kernel_fn, sigma).item()
    null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = torch.tensor(rng.permutation(m + n), device=DEVICE, dtype=torch.long)
        null[i] = mmd2_at_torch(pooled[perm[:m]], pooled[perm[m:]], kernel_fn, sigma).item()
    threshold = float(np.quantile(null, 1 - ALPHA))
    return int(test_stat > threshold)


# ----------------------------------------------------------------------
# Method drivers
# ----------------------------------------------------------------------
def run_linear(X_np, Y_np, sigma_grid, mode='permsplit', diag_writer=None,
               method_label=None, hypothesis=None, rep=None, n_samples=None, rng=None):
    """mode in {'permsplit', 'fullsample'}."""
    rng = rng if rng is not None else np.random.default_rng()
    if mode == 'permsplit':
        n = X_np.shape[0]
        half = n // 2
        perm_x = rng.permutation(n)
        perm_y = rng.permutation(n)
        X_tr = torch.tensor(X_np[perm_x[:half]], device=DEVICE, dtype=DTYPE)
        Y_tr = torch.tensor(Y_np[perm_y[:half]], device=DEVICE, dtype=DTYPE)
        X_te = torch.tensor(X_np[perm_x[half:]], device=DEVICE, dtype=DTYPE)
        Y_te = torch.tensor(Y_np[perm_y[half:]], device=DEVICE, dtype=DTYPE)
    else:  # fullsample
        X_tr = X_te = torch.tensor(X_np, device=DEVICE, dtype=DTYPE)
        Y_tr = Y_te = torch.tensor(Y_np, device=DEVICE, dtype=DTYPE)
    pooled_tr = torch.cat([X_tr, Y_tr], dim=0)
    DF_tr = float(torch.linalg.norm(pooled_tr).item())
    N_tr = pooled_tr.shape[0]
    G_lin = lambda s: DF_tr / (s * N_tr)
    C1 = calibrate_C1_torch(X_tr, Y_tr, gaussian_kernel_torch, sigma_grid, G_lin,
                            n_perm=N_CAL, rng=rng)
    sigma_sel, J_sel, mmd2_sel = cpmmd_select_torch(X_tr, Y_tr, gaussian_kernel_torch,
                                                    sigma_grid, C1, G_lin)
    if diag_writer is not None and rep is not None and rep < N_DIAG_REPS:
        boundary = (sigma_sel == sigma_grid[0]) or (sigma_sel == sigma_grid[-1])
        diag_writer.writerow([method_label, hypothesis, n_samples, rep, mode,
                              float(C1), float(sigma_sel), float(J_sel),
                              float(mmd2_sel), int(boundary)])
    return perm_test_torch(X_te, Y_te, gaussian_kernel_torch, sigma_sel, rng=rng)


def run_poly(X_np, Y_np, sigma_grid, degree, mode='permsplit', diag_writer=None,
             method_label=None, hypothesis=None, rep=None, n_samples=None, rng=None):
    """degree in {2, 3}. Uses Laplacian kernel on standardised polynomial features."""
    rng = rng if rng is not None else np.random.default_rng()
    transform_fn = poly2_features_np if degree == 2 else poly3_features_np

    if mode == 'permsplit':
        n = X_np.shape[0]
        half = n // 2
        perm_x = rng.permutation(n)
        perm_y = rng.permutation(n)
        Xtr_np = X_np[perm_x[:half]]; Ytr_np = Y_np[perm_y[:half]]
        Xte_np = X_np[perm_x[half:]]; Yte_np = Y_np[perm_y[half:]]
    else:
        Xtr_np = Xte_np = X_np
        Ytr_np = Yte_np = Y_np

    pool_tr = np.vstack([Xtr_np, Ytr_np])
    var_idx = VAR_IDX if VAR_IDX is not None else np.argsort(-pool_tr.var(axis=0))[:20]
    Xtr_f = transform_fn(Xtr_np, var_idx); Ytr_f = transform_fn(Ytr_np, var_idx)
    Xte_f = transform_fn(Xte_np, var_idx); Yte_f = transform_fn(Yte_np, var_idx)
    pool_tr_f = np.vstack([Xtr_f, Ytr_f])
    mu, sd = pool_tr_f.mean(0), pool_tr_f.std(0) + 1e-6
    Xtr_f = ((Xtr_f - mu) / sd).astype(np.float32)
    Ytr_f = ((Ytr_f - mu) / sd).astype(np.float32)
    Xte_f = ((Xte_f - mu) / sd).astype(np.float32)
    Yte_f = ((Yte_f - mu) / sd).astype(np.float32)

    X_tr = torch.tensor(Xtr_f, device=DEVICE, dtype=DTYPE)
    Y_tr = torch.tensor(Ytr_f, device=DEVICE, dtype=DTYPE)
    X_te = torch.tensor(Xte_f, device=DEVICE, dtype=DTYPE)
    Y_te = torch.tensor(Yte_f, device=DEVICE, dtype=DTYPE)
    pooled_tr = torch.cat([X_tr, Y_tr], dim=0)
    DF_tr = float(torch.linalg.norm(pooled_tr).item())
    N_tr = pooled_tr.shape[0]
    G_poly = lambda s: DF_tr / (s * N_tr)
    C1 = calibrate_C1_torch(X_tr, Y_tr, laplacian_kernel_torch, sigma_grid, G_poly,
                            n_perm=N_CAL, rng=rng)
    sigma_sel, J_sel, mmd2_sel = cpmmd_select_torch(X_tr, Y_tr, laplacian_kernel_torch,
                                                    sigma_grid, C1, G_poly)
    if diag_writer is not None and rep is not None and rep < N_DIAG_REPS:
        boundary = (sigma_sel == sigma_grid[0]) or (sigma_sel == sigma_grid[-1])
        diag_writer.writerow([method_label, hypothesis, n_samples, rep, mode,
                              float(C1), float(sigma_sel), float(J_sel),
                              float(mmd2_sel), int(boundary)])
    return perm_test_torch(X_te, Y_te, laplacian_kernel_torch, sigma_sel, rng=rng)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    open(LOG_PATH, 'w').close()
    log(f'=== Higgs extra GPU v2 ({DEVICE}) ===')
    log(f'N_REPS={N_REPS}, N_PERM={N_PERM}, N_CAL={N_CAL}, alpha={ALPHA}')

    global VAR_IDX
    sig, bg = load_higgs()
    pooled = np.concatenate([bg, sig], axis=0)
    VAR_IDX = np.argsort(-pooled.var(axis=0))[:20]   # 20 largest raw variances
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    std[std < 1e-10] = 1.0
    sig = ((sig - mean) / std).astype(np.float32)    # as run_higgs.load_higgs
    bg = ((bg - mean) / std).astype(np.float32)

    sigma_lin = list(np.geomspace(2.0, 20.0, 24).astype(np.float32))
    sigma_poly = list(np.geomspace(10.0, 100.0, 24).astype(np.float32))

    out_path = os.path.join(RESULTS_DIR, f'higgs_extra_v2_results{OUT_SUFFIX}.csv')
    rows = [['experiment', 'n_samples', 'method', 'mode', 'metric', 'value', 'n_reps']]

    diag_file = open(DIAG_PATH, 'w', newline='')
    diag_writer = csv.writer(diag_file)
    diag_writer.writerow(['method', 'hypothesis', 'n_samples', 'rep', 'mode',
                          'C1_calibrated', 'sigma_sel', 'J_at_sel', 'mmd2_at_sel',
                          'boundary_hit'])

    methods = [
        ('CPMMD-linear', 'permsplit', lambda X, Y, dw, hyp, r, n, g: run_linear(
            X, Y, sigma_lin, 'permsplit', dw, 'CPMMD-linear', hyp, r, n, g)),
        ('CPMMD-linear', 'fullsample', lambda X, Y, dw, hyp, r, n, g: run_linear(
            X, Y, sigma_lin, 'fullsample', dw, 'CPMMD-linear', hyp, r, n, g)),
        ('CPMMD-poly2', 'permsplit', lambda X, Y, dw, hyp, r, n, g: run_poly(
            X, Y, sigma_poly, 2, 'permsplit', dw, 'CPMMD-poly2', hyp, r, n, g)),
        ('CPMMD-poly2', 'fullsample', lambda X, Y, dw, hyp, r, n, g: run_poly(
            X, Y, sigma_poly, 2, 'fullsample', dw, 'CPMMD-poly2', hyp, r, n, g)),
        ('CPMMD-poly3', 'permsplit', lambda X, Y, dw, hyp, r, n, g: run_poly(
            X, Y, sigma_poly, 3, 'permsplit', dw, 'CPMMD-poly3', hyp, r, n, g)),
        ('CPMMD-poly3', 'fullsample', lambda X, Y, dw, hyp, r, n, g: run_poly(
            X, Y, sigma_poly, 3, 'fullsample', dw, 'CPMMD-poly3', hyp, r, n, g)),
    ]

    for n in SAMPLE_SIZES:
        log(f'\n--- n = {n} ---')
        for mi, (method, mode, fn) in enumerate(methods):
            if mode not in MODES or (METHODS and method not in METHODS):
                continue
            rng_master = np.random.default_rng([SEED, n, mi])
            label = f'{method}-{mode}'
            t0 = time.time()
            decisions = []
            for r in range(N_REPS):
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                idx = rng.choice(len(bg), 2 * n, replace=False)
                X = bg[idx[:n]]
                Y = bg[idx[n:]]
                decisions.append(fn(X, Y, diag_writer, 'H0', r, n, rng))
            type_i = float(np.mean(decisions))
            log(f'  {label} TypeI@n={n}: {type_i:.3f}  ({time.time() - t0:.1f}s)')
            rows.append(['H0_Higgs', n, method, mode, 'type_I', type_i, N_REPS])

            t0 = time.time()
            decisions = []
            for r in range(N_REPS):
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                idx_x = rng.choice(len(sig), n, replace=False)
                idx_y = rng.choice(len(bg), n, replace=False)
                X = sig[idx_x]
                Y = bg[idx_y]
                decisions.append(fn(X, Y, diag_writer, 'H1', r, n, rng))
            power = float(np.mean(decisions))
            log(f'  {label} Power@n={n}: {power:.3f}  ({time.time() - t0:.1f}s)')
            rows.append(['H1_Higgs', n, method, mode, 'power', power, N_REPS])

            with open(out_path, 'w') as f:
                csv.writer(f).writerows(rows)
            diag_file.flush()

    diag_file.close()
    log(f'\nWrote {out_path}')
    log(f'Diagnostics: {DIAG_PATH}')
    log('=== Done ===')


if __name__ == '__main__':
    main()
