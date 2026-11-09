"""
Tennessee Eastman Process (TEP) two-sample testing experiment for fault detection.

Synthetic TEP-like data:
  P = Normal operating conditions: N(0, Sigma_normal) with sensor correlations
  Q = Fault conditions: modified covariance structure
    - Fault 3:  Increase correlation between sensors 1-5 and 6-10 (0.1 -> 0.6)
    - Fault 9:  Small mean shift (0.1) on sensors 1-10 + covariance change
    - Fault 15: Increase variance of sensors 20-30 by factor 1.5

d = 52 features (sensors), m = n = 200, 50 repetitions per setting.

Methods:
  Deep kernel (Step 1) + PermSplit_2 (Step 2):
    CPMMD_1:  J_CP = MMD^2 - C1*G_h
    Liu_1:    J_Liu  = MMD^2 / sigma_hat
    Plain_1:  max MMD^2

  Fixed-kernel baselines:
    Median+Perm:  median heuristic + permutation test
    MMDAgg+Perm:  bandwidth aggregation + permutation test

  Polynomial feature method:
    Lap+poly2:  Laplacian kernel on degree-2 polynomial features
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations_with_replacement

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, optimize_bandwidth,
    DeepDiscriminator, mmd_loss_torch, _to_tensor,
    _sq_euclidean_distances, _DEVICE, BestIterate
)
from calibration_engine import compute_S

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tep')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50
N_PERM = 200
N_SAMPLES = 200
INPUT_DIM = 52
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
# Synthetic TEP data generation
# ======================================================================
def make_normal_covariance(d=52, rng=None):
    """Build a 52x52 covariance matrix for normal operating conditions.

    Off-diagonal correlations are in [0.1, 0.3] to simulate sensor coupling.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # Start with identity
    Sigma = np.eye(d)
    # Add random off-diagonal correlations
    for i in range(d):
        for j in range(i + 1, d):
            rho = rng.uniform(0.1, 0.3)
            Sigma[i, j] = rho
            Sigma[j, i] = rho
    # Ensure positive definiteness via nearest PD projection
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.01)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Symmetrize
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def generate_tep_data(n, fault_type, rng=None):
    """Generate synthetic TEP data for normal and fault conditions.

    Parameters
    ----------
    n : int
        Number of samples per distribution (m = n).
    fault_type : str
        One of 'none', 'fault3', 'fault9', 'fault15'.
    rng : np.random.Generator or None

    Returns
    -------
    X : ndarray (n, 52) -- normal operating conditions
    Y : ndarray (n, 52) -- fault conditions (or normal if fault_type='none')
    """
    if rng is None:
        rng = np.random.default_rng()

    d = INPUT_DIM
    Sigma_normal = make_normal_covariance(d, rng=np.random.default_rng(0))

    # X always from normal distribution
    X = rng.multivariate_normal(np.zeros(d), Sigma_normal, size=n)

    if fault_type == 'none':
        Y = rng.multivariate_normal(np.zeros(d), Sigma_normal, size=n)
        return X.astype(np.float32), Y.astype(np.float32)

    # Build fault covariance
    Sigma_fault = Sigma_normal.copy()
    mean_fault = np.zeros(d)

    if fault_type == 'fault3':
        # Increase correlation between sensors 1-5 (idx 0-4) and 6-10 (idx 5-9)
        for i in range(5):
            for j in range(5, 10):
                Sigma_fault[i, j] = 0.6
                Sigma_fault[j, i] = 0.6

    elif fault_type == 'fault9':
        # Mean shift on sensors 1-10 plus covariance change
        mean_fault[:10] = 0.1
        for i in range(5):
            for j in range(5, 10):
                Sigma_fault[i, j] = 0.4
                Sigma_fault[j, i] = 0.4

    elif fault_type == 'fault15':
        # Increase variance of sensors 20-30 (idx 19-29) by factor 1.5
        for i in range(19, 30):
            Sigma_fault[i, i] *= 1.5
            # Also scale cross-covariances to maintain correlation structure
            for j in range(d):
                if j != i:
                    Sigma_fault[i, j] *= np.sqrt(1.5)
                    Sigma_fault[j, i] *= np.sqrt(1.5)

    # Ensure positive definiteness
    eigvals, eigvecs = np.linalg.eigh(Sigma_fault)
    eigvals = np.maximum(eigvals, 0.01)
    Sigma_fault = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Sigma_fault = (Sigma_fault + Sigma_fault.T) / 2.0

    Y = rng.multivariate_normal(mean_fault, Sigma_fault, size=n)
    return X.astype(np.float32), Y.astype(np.float32)


# ======================================================================
# Polynomial features
# ======================================================================
def poly2_features(X):
    """Compute degree-2 polynomial features.

    For x in R^d, output (x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2).
    Output dimension: d + d*(d+1)/2.
    """
    n, d = X.shape
    # Linear features
    linear = X
    # Quadratic features: all pairs (i, j) with i <= j
    pairs = list(combinations_with_replacement(range(d), 2))
    quad = np.empty((n, len(pairs)), dtype=X.dtype)
    for idx, (i, j) in enumerate(pairs):
        quad[:, idx] = X[:, i] * X[:, j]
    return np.concatenate([linear, quad], axis=1)


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


def permutation_test(X, Y, sigma, kernel_type='gaussian', n_perm=N_PERM):
    """Full permutation test at fixed bandwidth."""
    m, n = len(X), len(Y)
    test_stat = compute_mmd_u_statistic(X, Y, kernel_type, sigma=sigma)
    pooled = np.concatenate([X, Y], axis=0)
    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m + n)
        null_mmds[i] = compute_mmd_u_statistic(
            pooled[perm[:m]], pooled[perm[m:]], kernel_type, sigma=sigma)
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
    best = BestIterate()  # h_hat = argmax_{0<=t<=T} J_CP(h^(t))
    for step in range(N_STEPS + 1):
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
        best.update(J.item(), D, log_sigma)
        if step == N_STEPS:
            break
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    best.restore(D, log_sigma)
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
    """Train deep kernel on first half, permutation test on second half."""
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_tr, Y_tr = X[:m_tr], Y[:n_tr]
    X_te, Y_te = X[m_tr:], Y[n_tr:]

    result = step1_fn(X_tr, Y_tr, dim)
    model, sigma = result['model'], result['sigma']

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
    """Calibrate C1 = 95th percentile of MMD/G over null permutations."""
    m, n = len(X_null), len(Y_null)
    N = m + n
    pooled = np.concatenate([X_null, Y_null], axis=0)

    ratios = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        res = step1_plain(X_p, Y_p, dim)
        G = compute_G_spectral(res['model'], X_p, Y_p, res['sigma'])
        if abs(G) > 1e-12:
            ratios.append(res['mmd'] / G)

    return float(np.percentile(ratios, 95)) if ratios else 0.01


# ======================================================================
# MMDAgg fallback (bandwidth aggregation with Bonferroni)
# ======================================================================
def mmdagg_perm_test(X, Y, n_perm=N_PERM):
    """MMDAgg-style bandwidth aggregation with permutation test."""
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
# Laplacian + poly2 method
# ======================================================================
def lap_poly2_test(X, Y, n_perm=N_PERM):
    """Laplacian kernel on degree-2 polynomial features with adaptive calibration.

    1. Compute poly2 features: d' = d + d*(d+1)/2 = 52 + 1378 = 1430
    2. Optimize Laplacian bandwidth over 20 log-spaced values in [0.1, 100]
    3. Use adaptive calibration quantile: q = 1 - alpha * d / d'
    """
    d = X.shape[1]
    d_prime = d + d * (d + 1) // 2  # 1430 for d=52

    # Compute polynomial features
    X_poly = poly2_features(X)
    Y_poly = poly2_features(Y)

    # Standardize polynomial features (important for high-dim kernel methods)
    pooled_poly = np.concatenate([X_poly, Y_poly], axis=0)
    mean_poly = pooled_poly.mean(axis=0)
    std_poly = pooled_poly.std(axis=0)
    std_poly[std_poly < 1e-10] = 1.0
    X_poly = (X_poly - mean_poly) / std_poly
    Y_poly = (Y_poly - mean_poly) / std_poly

    # Optimize bandwidth: find sigma maximizing MMD^2 with Laplacian kernel
    search_sigmas = np.logspace(np.log10(0.1), np.log10(100.0), 20)
    best_mmd = -float('inf')
    best_sigma = 1.0
    for sigma in search_sigmas:
        mmd_val = compute_mmd_u_statistic(X_poly, Y_poly, 'laplacian', sigma=sigma)
        if mmd_val > best_mmd:
            best_mmd = mmd_val
            best_sigma = sigma

    # Permutation test with adaptive calibration quantile
    sigma = best_sigma
    m, n = len(X_poly), len(Y_poly)
    test_stat = compute_mmd_u_statistic(X_poly, Y_poly, 'laplacian', sigma=sigma)

    pooled = np.concatenate([X_poly, Y_poly], axis=0)
    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m + n)
        null_mmds[i] = compute_mmd_u_statistic(
            pooled[perm[:m]], pooled[perm[m:]], 'laplacian', sigma=sigma)

    # Adaptive quantile: q = 1 - alpha * d / d'
    adaptive_alpha = ALPHA * d / d_prime
    threshold = np.percentile(null_mmds, (1 - adaptive_alpha) * 100)
    return test_stat > threshold


# ======================================================================
# Main experiment
# ======================================================================
def run_tep_experiment(n_reps=N_REPS):
    """Run all methods on all TEP fault types."""
    dim = INPUT_DIM  # 52 features
    n = N_SAMPLES    # 200

    log(f"\n{'='*70}")
    log(f"TEP experiment: d={dim}, n={n}, n_reps={n_reps}")
    log(f"{'='*70}")

    # Calibrate C1 from null (both samples from normal conditions)
    log("  Calibrating C1 from TEP normal null (deep kernel)...")
    rng_cal = np.random.default_rng(42)
    X_cal, Y_cal = generate_tep_data(n, 'none', rng=rng_cal)
    C1 = calibrate_C1(dim, X_cal, Y_cal, n_reps=10)
    log(f"  C1={C1:.6f}")

    methods = [
        'CPMMD1+PermSplit2',
        'Liu1+PermSplit2',
        'Plain1+PermSplit2',
        'Median+Perm',
        'MMDAgg+Perm',
        'Lap+poly2',
    ]
    all_results = []

    fault_types = [
        ('none', 'H0: Normal vs Normal (type I)'),
        ('fault3', 'H1: Fault 3 - correlation shift sensors 1-10'),
        ('fault9', 'H1: Fault 9 - mean+cov shift sensors 1-10'),
        ('fault15', 'H1: Fault 15 - variance shift sensors 20-30'),
    ]

    for fault_type, description in fault_types:
        hypothesis = 'H0' if fault_type == 'none' else 'H1'
        metric = 'type_I' if fault_type == 'none' else 'power'
        log(f"\n  --- {description} (n={n}) ---")
        counts = {k: 0 for k in methods}

        for rep in range(n_reps):
            seed = rep + (0 if fault_type == 'none' else
                          1000 if fault_type == 'fault3' else
                          2000 if fault_type == 'fault9' else 3000)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)

            X, Y = generate_tep_data(n, fault_type, rng=rng)

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
            try:
                sigma_med = median_heuristic(X, Y)
                reject = permutation_test(X, Y, sigma_med)
                counts['Median+Perm'] += int(reject)
            except Exception as e:
                log(f"    [Median] rep {rep} failed: {e}")

            # MMDAgg + permutation test
            try:
                reject = mmdagg_perm_test(X, Y)
                counts['MMDAgg+Perm'] += int(reject)
            except Exception as e:
                log(f"    [MMDAgg] rep {rep} failed: {e}")

            # Laplacian + poly2
            try:
                reject = lap_poly2_test(X, Y)
                counts['Lap+poly2'] += int(reject)
            except Exception as e:
                log(f"    [Lap+poly2] rep {rep} failed: {e}")

            if (rep + 1) % 10 == 0:
                log(f"    rep {rep+1}/{n_reps}")

        for method in methods:
            val = counts[method] / n_reps
            log(f"    {method:25s}: {metric.upper()} = {val:.3f} ({counts[method]}/{n_reps})")
            all_results.append({
                'experiment': f'{hypothesis}_{fault_type}',
                'fault_type': fault_type,
                'n_samples': n,
                'method': method,
                'metric': metric,
                'value': val,
                'n_reps': n_reps,
            })

    return all_results


# ======================================================================
# Entry point
# ======================================================================
def main():
    global log_fh
    log_fh = open(LOG_FILE, 'w', buffering=1)

    log(f"Tennessee Eastman Process (TEP) Two-Sample Testing Experiment")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")
    log(f"Sensors: {INPUT_DIM}, Samples: m=n={N_SAMPLES}, Reps: {N_REPS}")

    all_results = run_tep_experiment(n_reps=N_REPS)

    # Save results to CSV
    csv_path = os.path.join(RESULTS_DIR, 'tep_results.csv')
    fieldnames = ['experiment', 'fault_type', 'n_samples', 'method',
                  'metric', 'value', 'n_reps']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    log(f"\nResults saved to {csv_path}")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
