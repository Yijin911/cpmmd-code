"""
Higgs boson two-sample testing experiment.

P = background events (label=0)
Q = signal events (label=1)

Dataset: UCI Higgs (https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)
11M rows, first column is label, remaining 28 columns are features.

This experiment compares kernel SELECTION criteria (Step 1) on real physics data.
All learned-kernel methods use the same Step 2 (PermSplit_2) to isolate
the effect of the selection criterion.

Methods compared:
  Deep kernel selection (Step 1) + PermSplit_2 (Step 2):
    CPMMD_1:  J_CP = MMD^2 - C1*G_h (complexity-penalised)
    Liu_1:    J_Liu  = MMD^2 / sigma_hat (variance-normalised)
    Plain_1:  max MMD^2 (no regularisation)

  Fixed-kernel baselines (permutation on full data):
    Median_1: median heuristic + permutation test
    MMDAgg:   kernel aggregation + permutation test
"""

import os, sys, time, csv, warnings, gzip
import numpy as np
import torch
import torch.nn as nn
import urllib.request

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, optimize_bandwidth,
    DeepDiscriminator, mmd_loss_torch, _to_tensor,
    _sq_euclidean_distances, _DEVICE
)
from calibration_engine import compute_S

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'higgs')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50          # repetitions per setting (override in main if desired)
N_PERM = 200         # permutations for threshold
SAMPLE_SIZES = [200, 500]
MAX_ROWS = 100000    # only load first 100K rows to save memory

# Deep kernel hyperparameters (matching kernel_criterion.tex)
INPUT_DIM = 28
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005

LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt')
log_fh = None


def log(msg):
    print(msg)
    if log_fh is not None:
        log_fh.write(msg + '\n')
        log_fh.flush()


# ======================================================================
# Data loading
# ======================================================================
HIGGS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'


def download_higgs():
    """Download HIGGS.csv.gz if not already present."""
    gz_path = os.path.join(DATA_DIR, 'HIGGS.csv.gz')
    if os.path.exists(gz_path):
        log(f"  Found cached {gz_path}")
        return gz_path
    log(f"  Downloading HIGGS dataset from UCI...")
    log(f"  URL: {HIGGS_URL}")
    log(f"  This is ~2.6 GB, may take a while...")
    urllib.request.urlretrieve(HIGGS_URL, gz_path)
    log(f"  Download complete: {gz_path}")
    return gz_path


def load_higgs(max_rows=MAX_ROWS):
    """Load Higgs dataset, returning background (P) and signal (Q) arrays.

    Only loads first max_rows rows to save memory (50K per class is plenty).
    All 28 features are standardized (zero mean, unit variance on pooled data).

    Returns:
        P: ndarray of shape (n_background, 28) -- background events (label=0)
        Q: ndarray of shape (n_signal, 28) -- signal events (label=1)
    """
    gz_path = download_higgs()

    log(f"  Loading first {max_rows} rows from {gz_path}...")
    rows = []
    with gzip.open(gz_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(line.strip().split(','))

    data = np.array(rows, dtype=np.float32)
    labels = data[:, 0].astype(int)   # first column is label
    features = data[:, 1:]            # remaining 28 columns

    # Separate by class
    bg_mask = labels == 0
    sig_mask = labels == 1
    P_raw = features[bg_mask]
    Q_raw = features[sig_mask]

    log(f"  Background (P): {P_raw.shape[0]} events")
    log(f"  Signal    (Q): {Q_raw.shape[0]} events")

    # Standardize: zero mean, unit variance on pooled data
    pooled = np.concatenate([P_raw, Q_raw], axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < 1e-10] = 1.0  # avoid division by zero
    P = (P_raw - mean) / std
    Q = (Q_raw - mean) / std

    log(f"  Standardized features: mean ~ {P.mean():.4f}, std ~ {P.std():.4f}")
    return P, Q


# ======================================================================
# Helpers
# ======================================================================
def median_heuristic(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(Z)
    dists = _sq_euclidean_distances(Z_t, Z_t)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    return max(torch.sqrt(dists[mask]).median().item(), 1e-6)


def compute_G_spectral(model, X, Y, sigma):
    """Spectral norm bound on Gaussian complexity for deep kernel."""
    spectral_product = 1.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                spectral_product *= torch.linalg.matrix_norm(param, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return spectral_product * np.linalg.norm(Z) / (np.sqrt(N) * sigma)


def permutation_test(X, Y, sigma, n_perm=N_PERM):
    """Full permutation test at fixed bandwidth."""
    m, n = len(X), len(Y)
    test_stat = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
    pooled = np.concatenate([X, Y], axis=0)
    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m + n)
        null_mmds[i] = compute_mmd_u_statistic(
            pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma)
    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return test_stat > threshold


# ======================================================================
# Step 1: Deep kernel selection criteria
# ======================================================================
def step1_cpmmd(X, Y, dim, C1):
    """J_CP = MMD^2 - C1*G_h (complexity-penalised deep kernel)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    Z = np.concatenate([X, Y], axis=0)
    data_geom = np.linalg.norm(Z) / np.sqrt(len(Z))
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        G_h = spec_prod * data_geom / sigma
        J = mmd2 - C1 * G_h
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


def step1_liu(X, Y, dim):
    """J_Liu = MMD^2 / sigma_hat (variance-normalised deep kernel)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        m = f_x.shape[0]
        K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) / (sigma**2))
        K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) / (sigma**2))
        h_x = K_xx.mean(dim=1) - K_xy.mean(dim=1)
        var_h = torch.var(h_x).clamp(min=1e-10)
        sigma_hat = torch.sqrt(4 * var_h / m)
        J = mmd2 / sigma_hat
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


def step1_plain(X, Y, dim):
    """Plain: max MMD^2 (no regularisation)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        (-mmd2).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


# ======================================================================
# Step 2: PermSplit_2 for deep kernel methods
# ======================================================================
def step2_perm_split_deep(X, Y, step1_fn, dim, n_perm=N_PERM):
    """Train deep kernel on first half, permutation test on second half.

    step1_fn: callable(X_tr, Y_tr, dim) -> {'model', 'sigma', 'mmd'}
    """
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_tr, Y_tr = X[:m_tr], Y[:n_tr]
    X_te, Y_te = X[m_tr:], Y[n_tr:]

    result = step1_fn(X_tr, Y_tr, dim)
    model, sigma = result['model'], result['sigma']

    # Compute features for test half
    te_all = np.concatenate([X_te, Y_te], axis=0)
    te_t = _to_tensor(te_all).to(_DEVICE)
    with torch.no_grad():
        f_all = model(te_t)

    m_te, n_te = len(X_te), len(Y_te)
    test_stat = mmd_loss_torch(f_all[:m_te], f_all[m_te:], sigma).item()

    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m_te + n_te)
        null_mmds[i] = mmd_loss_torch(f_all[perm[:m_te]], f_all[perm[m_te:]], sigma).item()

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return test_stat > threshold


# ======================================================================
# Calibration for C1 (deep kernel, from null permutations)
# ======================================================================
def calibrate_C1(dim, X_null, Y_null, n_reps=10):
    """Calibrate C1 = 95th percentile of MMD/G_ref over null permutations.

    Journal procedure (shared G_ref): G is computed ONCE on the original
    un-permuted (X_null, Y_null) by training a fresh deep kernel on it, and
    reused across all permutations. Only the numerator (max-MMD trained
    on the permuted data) varies. This makes the calibration ratio
    exchangeable under the null permutation distribution
    (cf. Prop 4.23 in the JMLR draft).
    """
    m, n = len(X_null), len(Y_null)
    N = m + n
    pooled = np.concatenate([X_null, Y_null], axis=0)

    # Shared G_ref: train once on un-permuted null data
    res_ref = step1_plain(X_null, Y_null, dim)
    G_ref = compute_G_spectral(res_ref['model'], X_null, Y_null, res_ref['sigma'])
    if abs(G_ref) <= 1e-12:
        G_ref = 1e-12

    ratios = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        res = step1_plain(X_p, Y_p, dim)
        ratios.append(res['mmd'] / G_ref)

    return float(np.percentile(ratios, 95)) if ratios else 0.01


# ======================================================================
# MMDAgg fallback (bandwidth aggregation with Bonferroni)
# ======================================================================
def mmdagg_perm_test(X, Y, n_perm=N_PERM):
    """MMDAgg-style bandwidth aggregation with permutation test.

    Tries to import mmdagg; falls back to Bonferroni aggregation.
    """
    try:
        from mmdagg import mmdagg as mmdagg_fn
        result = mmdagg_fn(
            X, Y, alpha=ALPHA,
            kernel='laplace_gaussian',
            number_bandwidths=10,
            permutations_same_sample_size=True,
        )
        return int(result)
    except ImportError:
        # Fallback: Bonferroni-corrected bandwidth aggregation
        sigmas = np.logspace(-1, 1, 10)
        m, n = len(X), len(Y)
        pooled = np.concatenate([X, Y], axis=0)
        N = m + n
        p_values = []
        for sigma in sigmas:
            mmd_obs = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
            null_mmds = []
            for _ in range(n_perm):
                perm = np.random.permutation(N)
                null_mmds.append(compute_mmd_u_statistic(
                    pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))
            p_val = np.mean(np.array(null_mmds) >= mmd_obs)
            p_values.append(p_val)
        min_p = min(p_values) * len(sigmas)  # Bonferroni
        return int(min_p < ALPHA)


# ======================================================================
# Main experiment
# ======================================================================
def run_higgs_experiment(P, Q, n_samples, n_reps=N_REPS):
    """Run all methods on Higgs background vs signal at given sample size.

    P: background events (standardized features)
    Q: signal events (standardized features)
    n_samples: m = n = n_samples
    """
    dim = INPUT_DIM  # 28 features

    log(f"\n{'='*70}")
    log(f"Higgs experiment: n_samples={n_samples}, n_reps={n_reps}")
    log(f"{'='*70}")

    # Calibrate C1 from null (both samples from background P)
    log("  Calibrating C1 from Higgs background null (deep kernel)...")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(P))
    X_cal = P[idx[:n_samples]]
    Y_cal = P[idx[n_samples:2*n_samples]]
    C1 = calibrate_C1(dim, X_cal, Y_cal, n_reps=10)
    log(f"  C1={C1:.6f}")

    methods = [
        'CPMMD1+PermSplit2',
        'Liu1+PermSplit2',
        'Plain1+PermSplit2',
        'Median+Perm',
        'MMDAgg+Perm',
    ]
    results = []

    for hypothesis, label in [('H1', 'Background vs Signal (power)'),
                               ('H0', 'Background vs Background (type I)')]:
        log(f"\n  --- {hypothesis}: {label} (n={n_samples}) ---")
        counts = {k: 0 for k in methods}

        for rep in range(n_reps):
            seed = rep + (1000 if hypothesis == 'H1' else 5000)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if hypothesis == 'H1':
                # Power: X from P (background), Y from Q (signal)
                idx_p = np.random.choice(len(P), n_samples, replace=False)
                idx_q = np.random.choice(len(Q), n_samples, replace=False)
                X = P[idx_p]
                Y = Q[idx_q]
            else:
                # Type I error: both X and Y from P (background)
                idx_all = np.random.permutation(len(P))
                X = P[idx_all[:n_samples]]
                Y = P[idx_all[n_samples:2*n_samples]]

            # Deep kernel methods (Step 1) + PermSplit_2 (Step 2)
            try:
                reject = step2_perm_split_deep(
                    X, Y, lambda Xt, Yt, d: step1_cpmmd(Xt, Yt, d, C1), dim)
                counts['CPMMD1+PermSplit2'] += int(reject)
            except Exception as e:
                log(f"    [CPMMD1] rep {rep} failed: {e}")

            try:
                reject = step2_perm_split_deep(X, Y, step1_liu, dim)
                counts['Liu1+PermSplit2'] += int(reject)
            except Exception as e:
                log(f"    [Liu1] rep {rep} failed: {e}")

            try:
                reject = step2_perm_split_deep(X, Y, step1_plain, dim)
                counts['Plain1+PermSplit2'] += int(reject)
            except Exception as e:
                log(f"    [Plain1] rep {rep} failed: {e}")

            # Median + full permutation test
            sigma_med = median_heuristic(X, Y)
            reject = permutation_test(X, Y, sigma_med)
            counts['Median+Perm'] += int(reject)

            # MMDAgg + permutation test
            try:
                reject = mmdagg_perm_test(X, Y)
                counts['MMDAgg+Perm'] += int(reject)
            except Exception as e:
                log(f"    [MMDAgg] rep {rep} failed: {e}")

            if (rep + 1) % 10 == 0:
                log(f"    rep {rep+1}/{n_reps}")

        metric = 'power' if hypothesis == 'H1' else 'type_I'
        for method in methods:
            val = counts[method] / n_reps
            log(f"    {method:25s}: {metric.upper()} = {val:.3f} ({counts[method]}/{n_reps})")
            results.append({
                'experiment': f'{hypothesis}_Higgs',
                'n_samples': n_samples,
                'method': method,
                'metric': metric,
                'value': val,
                'n_reps': n_reps,
            })

    return results


# ======================================================================
# Entry point
# ======================================================================
def main(n_reps=N_REPS):
    global log_fh
    log_fh = open(LOG_FILE, 'w', buffering=1)

    log(f"Higgs Boson Two-Sample Testing Experiment")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")

    # Load data
    log("\nLoading Higgs dataset...")
    P, Q = load_higgs(max_rows=MAX_ROWS)
    log(f"  Background (P): {P.shape}")
    log(f"  Signal    (Q): {Q.shape}")

    all_results = []
    for n in SAMPLE_SIZES:
        results = run_higgs_experiment(P, Q, n, n_reps=n_reps)
        all_results.extend(results)

    # Save results to CSV
    csv_path = os.path.join(RESULTS_DIR, 'higgs_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'n_samples', 'method',
                                                'metric', 'value', 'n_reps'])
        writer.writeheader()
        writer.writerows(all_results)

    log(f"\nResults saved to {csv_path}")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main(n_reps=N_REPS)
