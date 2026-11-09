"""
Phase 2a: Fix Type I inflation in the HDGM experiment.

Re-runs the HDGM (High-Dimensional Gaussian Model) experiment with adaptive calibration.

Setup (same as paper):
  P = N(0, I_d), Q differs in covariance: Cov(Y_1, Y_2) = 0.5
  d in {10, 20}
  m = n = 2000
  50 repetitions

Methods:
  - Lap+poly1: Laplacian kernel on raw features, optimize bandwidth
  - Lap+poly2: Laplacian kernel on degree-2 polynomial features, optimize bandwidth
  - Lap+poly2 (adaptive): Same but with adaptive quantile q = 1 - alpha * d/d'
  - MMDAgg

Adaptive quantile fix: when calibrating C1, use quantile level
  q = 1 - alpha * d / max(d, d')
instead of 0.95.

For hypothesis testing: PermSplit_2 (train on first half, permutation test on second half).

Saves to results/hdgm/hdgm_revised_results.csv.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
from itertools import combinations_with_replacement
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, _to_tensor, _l1_distances, _DEVICE
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'hdgm')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50
N_PERM = 200
N_SAMPLES = 2000

LOG_FILE = os.path.join(RESULTS_DIR, 'hdgm_log.txt')
log_fh = open(LOG_FILE, 'w', buffering=1)


def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ======================================================================
# HDGM data generation
# ======================================================================
def generate_hdgm_data(n_samples, dim, rho=0.0):
    """Generate HDGM data.

    P = N(0, I_d)
    Q = N(0, Sigma) where Sigma has Cov(Y_1, Y_2) = rho, rest diagonal.

    Parameters
    ----------
    n_samples : int
        Number of samples per dataset.
    dim : int
        Dimensionality.
    rho : float
        Covariance between first two dimensions of Q. rho=0 means H0 (same dist).

    Returns
    -------
    X, Y : ndarray of shape (n_samples, dim)
    """
    X = np.random.standard_normal((n_samples, dim))

    # Build covariance matrix for Q
    Sigma = np.eye(dim)
    if rho != 0.0 and dim >= 2:
        Sigma[0, 1] = rho
        Sigma[1, 0] = rho

    # Sample from N(0, Sigma)
    L = np.linalg.cholesky(Sigma)
    Z = np.random.standard_normal((n_samples, dim))
    Y = Z @ L.T

    return X, Y


# ======================================================================
# Polynomial feature expansion
# ======================================================================
def poly_features_degree2(X):
    """Compute degree-2 polynomial features.

    Input: X of shape (n, d)
    Output: (x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2)
    Dimension: d + d*(d+1)/2

    Parameters
    ----------
    X : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n, d + d*(d+1)/2)
    """
    n, d = X.shape
    # Linear features
    features = [X]

    # Quadratic features: all products x_i * x_j for i <= j
    quad_features = []
    for i, j in combinations_with_replacement(range(d), 2):
        quad_features.append((X[:, i] * X[:, j]).reshape(-1, 1))

    features.append(np.concatenate(quad_features, axis=1))
    return np.concatenate(features, axis=1)


def get_poly_dim(d, degree=2):
    """Get output dimension of polynomial features."""
    if degree == 1:
        return d
    elif degree == 2:
        return d + d * (d + 1) // 2
    else:
        raise ValueError(f"Unsupported degree: {degree}")


# ======================================================================
# Laplacian kernel MMD computation
# ======================================================================
def compute_mmd_laplacian(X, Y, sigma):
    """Compute unbiased MMD^2 U-statistic with Laplacian kernel."""
    return compute_mmd_u_statistic(X, Y, kernel_type='laplacian', sigma=sigma)


# ======================================================================
# Bandwidth optimization for Laplacian kernel
# ======================================================================
def optimize_bandwidth_laplacian(X, Y, n_sigmas=20, sigma_range=(0.1, 100.0)):
    """Optimize bandwidth for Laplacian kernel over log-spaced grid.

    Parameters
    ----------
    X, Y : ndarray
    n_sigmas : int
        Number of bandwidths to try.
    sigma_range : tuple
        Range for log-spaced bandwidth search.

    Returns
    -------
    dict with 'optimal_sigma' and 'max_mmd'
    """
    search_sigmas = np.logspace(np.log10(sigma_range[0]),
                                np.log10(sigma_range[1]),
                                n_sigmas)
    best_mmd = -float('inf')
    best_sigma = search_sigmas[0]

    for sigma in search_sigmas:
        mmd_val = compute_mmd_laplacian(X, Y, sigma)
        if mmd_val > best_mmd:
            best_mmd = mmd_val
            best_sigma = sigma

    return {'optimal_sigma': best_sigma, 'max_mmd': best_mmd}


# ======================================================================
# Calibration with standard and adaptive quantile
# ======================================================================
def calibrate_C1_laplacian(X, Y, n_reps=10, quantile_level=0.95,
                           n_sigmas=20, sigma_range=(0.1, 100.0)):
    """Calibrate C1 for Laplacian bandwidth optimization under the null.

    Computes the quantile_level-th quantile of MMD/G ratios from null permutations.

    Parameters
    ----------
    X, Y : ndarray
        Original data (used for pooled permutations).
    n_reps : int
        Number of null permutations.
    quantile_level : float
        Quantile level for C1. Standard is 0.95, adaptive adjusts this.
    n_sigmas : int
        Number of bandwidths in grid.
    sigma_range : tuple
        Bandwidth search range.

    Returns
    -------
    C1 : float
    """
    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)

    ratios = []
    mmd_vals = []

    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]

        opt = optimize_bandwidth_laplacian(X_p, Y_p, n_sigmas=n_sigmas,
                                           sigma_range=sigma_range)
        mmd_val = opt['max_mmd']
        sigma = opt['optimal_sigma']

        # G for bandwidth-only: ||Z|| / (sqrt(N) * sigma)
        Z = np.concatenate([X_p, Y_p], axis=0)
        G = np.linalg.norm(Z) / (np.sqrt(N) * sigma)

        mmd_vals.append(mmd_val)
        if abs(G) > 1e-12:
            ratios.append(mmd_val / G)

    if ratios:
        C1 = float(np.percentile(ratios, quantile_level * 100))
    else:
        C1 = 0.01

    return C1


# ======================================================================
# Step 2: PermSplit_2 for Laplacian kernel
# ======================================================================
def perm_split_test_laplacian(X, Y, sigma):
    """Permutation test on held-out second half with Laplacian kernel.

    Parameters
    ----------
    X, Y : ndarray
        Full data. First half is training, second half is testing.
    sigma : float
        Selected bandwidth.

    Returns
    -------
    dict with 'reject', 'test_stat', 'threshold'
    """
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_te, Y_te = X[m_tr:], Y[n_tr:]
    m_te, n_te = len(X_te), len(Y_te)

    test_stat = compute_mmd_laplacian(X_te, Y_te, sigma)

    pooled_te = np.concatenate([X_te, Y_te], axis=0)
    N_te = m_te + n_te
    null_mmds = []
    for _ in range(N_PERM):
        perm = np.random.permutation(N_te)
        null_mmds.append(compute_mmd_laplacian(
            pooled_te[perm[:m_te]], pooled_te[perm[m_te:]], sigma))

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    reject = test_stat > threshold
    return {'reject': reject, 'test_stat': test_stat, 'threshold': threshold}


# ======================================================================
# MMDAgg fallback (bandwidth aggregation with Laplacian kernel)
# ======================================================================
def mmdagg_laplacian(X, Y, n_sigmas=10, sigma_range=(0.1, 100.0)):
    """MMDAgg with Laplacian kernel (Bonferroni aggregation fallback).

    Parameters
    ----------
    X, Y : ndarray
    n_sigmas : int
    sigma_range : tuple

    Returns
    -------
    dict with 'reject'
    """
    try:
        from mmdagg import mmdagg as mmdagg_fn
        result = mmdagg_fn(X, Y, alpha=ALPHA, kernel_type='laplacian',
                           return_dictionary=True)
        if isinstance(result, dict):
            return {'reject': bool(result.get('reject', False))}
        return {'reject': bool(result)}
    except (ImportError, TypeError):
        pass

    # Fallback: Bonferroni over bandwidths
    sigmas = np.logspace(np.log10(sigma_range[0]),
                         np.log10(sigma_range[1]),
                         n_sigmas)
    m, n = len(X), len(Y)
    pooled = np.concatenate([X, Y], axis=0)
    N = m + n

    p_values = []
    for sigma in sigmas:
        mmd_obs = compute_mmd_laplacian(X, Y, sigma)
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(N)
            null_mmds.append(compute_mmd_laplacian(
                pooled[perm[:m]], pooled[perm[m:]], sigma))
        p_val = np.mean(np.array(null_mmds) >= mmd_obs)
        p_values.append(p_val)

    min_p = min(p_values) * len(sigmas)
    reject = min_p < ALPHA
    return {'reject': reject}


# ======================================================================
# Method runners
# ======================================================================
def run_lap_poly1(X, Y, dim):
    """Lap+poly1: Laplacian kernel on raw features, optimize bandwidth, PermSplit_2."""
    m_tr, n_tr = len(X) // 2, len(Y) // 2
    X_tr, Y_tr = X[:m_tr], Y[:n_tr]

    # Optimize bandwidth on training half
    opt = optimize_bandwidth_laplacian(X_tr, Y_tr)
    sigma = opt['optimal_sigma']

    # Test on held-out half
    return perm_split_test_laplacian(X, Y, sigma)


def run_lap_poly2(X, Y, dim):
    """Lap+poly2: Laplacian kernel on degree-2 polynomial features, optimize bandwidth, PermSplit_2."""
    # Expand to polynomial features
    X_poly = poly_features_degree2(X)
    Y_poly = poly_features_degree2(Y)

    m_tr, n_tr = len(X_poly) // 2, len(Y_poly) // 2
    X_tr, Y_tr = X_poly[:m_tr], Y_poly[:n_tr]

    # Optimize bandwidth on training half
    opt = optimize_bandwidth_laplacian(X_tr, Y_tr)
    sigma = opt['optimal_sigma']

    # Test on held-out half
    return perm_split_test_laplacian(X_poly, Y_poly, sigma)


def run_lap_poly2_adaptive(X, Y, dim):
    """Lap+poly2 (adaptive): Same as poly2 but with adaptive quantile for C1 calibration.

    Adaptive quantile: q = 1 - alpha * d / max(d, d')
    where d is original dimension, d' is polynomial feature dimension.

    This reduces the calibration quantile when d' >> d, making the penalty
    less aggressive and preventing Type I inflation from feature expansion.
    """
    d = dim
    d_prime = get_poly_dim(d, degree=2)

    # Adaptive quantile level
    q_level = 1.0 - ALPHA * d / max(d, d_prime)

    # Expand to polynomial features
    X_poly = poly_features_degree2(X)
    Y_poly = poly_features_degree2(Y)

    m_tr, n_tr = len(X_poly) // 2, len(Y_poly) // 2
    X_tr, Y_tr = X_poly[:m_tr], Y_poly[:n_tr]

    # Calibrate C1 with adaptive quantile on training data
    C1 = calibrate_C1_laplacian(X_tr, Y_tr, n_reps=10, quantile_level=q_level)

    # Optimize bandwidth on training half with complexity penalty
    search_sigmas = np.logspace(np.log10(0.1), np.log10(100.0), 20)
    best_J = -float('inf')
    best_sigma = search_sigmas[0]

    for sigma in search_sigmas:
        mmd_val = compute_mmd_laplacian(X_tr, Y_tr, sigma)
        Z = np.concatenate([X_tr, Y_tr], axis=0)
        G = np.linalg.norm(Z) / (np.sqrt(len(Z)) * sigma)
        J = mmd_val - C1 * G
        if J > best_J:
            best_J = J
            best_sigma = sigma

    # Test on held-out half
    return perm_split_test_laplacian(X_poly, Y_poly, best_sigma)


def run_mmdagg(X, Y, dim):
    """MMDAgg with Laplacian kernel on raw features."""
    return mmdagg_laplacian(X, Y)


# ======================================================================
# Experiment runner
# ======================================================================
def run_hdgm_experiment(dim, rho_values):
    """Run the HDGM experiment for one dimension.

    Parameters
    ----------
    dim : int
        Dimensionality.
    rho_values : list of float
        Covariance values for Q. rho=0 is H0 (Type I), rho>0 is H1 (power).

    Returns
    -------
    list of dict
        Results for each method x rho combination.
    """
    methods = {
        'Lap+poly1': run_lap_poly1,
        'Lap+poly2': run_lap_poly2,
        'Lap+poly2 (adaptive)': run_lap_poly2_adaptive,
        'MMDAgg': run_mmdagg,
    }

    all_results = []

    for rho in rho_values:
        is_null = (rho == 0.0)
        label = "TYPE_I" if is_null else "POWER"
        log(f"\n  rho={rho} ({label})")

        for method_name, method_fn in methods.items():
            rejects = 0
            valid = 0
            t0 = time.time()

            for rep in range(N_REPS):
                np.random.seed(rep * 5555 + int(rho * 1000) + dim)
                torch.manual_seed(rep * 5555 + int(rho * 1000) + dim)

                X, Y = generate_hdgm_data(N_SAMPLES, dim, rho=rho)

                try:
                    result = method_fn(X, Y, dim)
                    rejects += int(result['reject'])
                    valid += 1
                except Exception as e:
                    log(f"      {method_name} rep {rep} failed: {e}")

            dt = time.time() - t0
            rate = rejects / max(valid, 1)
            log(f"    {method_name:25s}: {label}={rate:.3f} ({valid}/{N_REPS} valid, {dt:.1f}s)")

            all_results.append({
                'dim': dim,
                'rho': rho,
                'hypothesis': 'H0' if is_null else 'H1',
                'method': method_name,
                'rejection_rate': rate,
                'n_valid': valid,
                'n_reps': N_REPS,
                'n_samples': N_SAMPLES,
                'time_s': dt,
            })

    return all_results


def save_results(results, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not results:
        log(f"  No results to save for {filename}")
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    log(f"  Saved {filepath}")


def main():
    log(f"{'='*70}")
    log(f"Phase 2a: HDGM Experiment (revised with adaptive calibration)")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")
    log(f"N_REPS={N_REPS}, N_PERM={N_PERM}, N_SAMPLES={N_SAMPLES}")
    log(f"ALPHA={ALPHA}")
    log(f"{'='*70}")

    all_results = []

    # rho=0 is H0 (Type I error), rho=0.5 is H1 (power)
    rho_values = [0.0, 0.5]

    for dim in [10, 20]:
        d_prime = get_poly_dim(dim, degree=2)
        adaptive_q = 1.0 - ALPHA * dim / max(dim, d_prime)
        log(f"\n{'='*70}")
        log(f"Dimension d={dim}, poly2 dim d'={d_prime}")
        log(f"Standard quantile: 0.95, Adaptive quantile: {adaptive_q:.4f}")
        log(f"{'='*70}")

        results = run_hdgm_experiment(dim, rho_values)
        all_results.extend(results)

        # Save incremental results
        save_results(all_results, f'hdgm_revised_d{dim}.csv')

    # Save final combined results
    save_results(all_results, 'hdgm_revised_results.csv')

    # Print summary table
    log(f"\n{'='*70}")
    log(f"Summary")
    log(f"{'='*70}")
    log(f"{'Method':30s} {'dim':>4s} {'rho':>5s} {'Rate':>8s}")
    log(f"{'-'*50}")
    for r in all_results:
        log(f"{r['method']:30s} {r['dim']:4d} {r['rho']:5.1f} {r['rejection_rate']:8.3f}")

    log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
